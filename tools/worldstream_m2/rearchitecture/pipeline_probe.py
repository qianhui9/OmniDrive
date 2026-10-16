"""Separate projection/assembly profile on the actual Peft base model, no M3 edits."""
import copy
from time import perf_counter
import torch
from .common import ROOT,write,stats
from .integration import load,PipelineTimer


@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    model,dataset=load()
    from .repair_language_oracle import repair
    if not (ROOT/'language_oracle_recheck.json').exists():repair(model,dataset)
    from worldstream.m2.cost_model import block_fingerprint
    torch.cuda.synchronize();tick=perf_counter()
    fingerprint=block_fingerprint(model.img_backbone)
    torch.cuda.synchronize()
    write(ROOT/'camera_signature_initialization.json',dict(wall_ms=(perf_counter()-tick)*1000,
        model_sha256=fingerprint,scope='first content hash of this loaded EVA; device-to-host reads included, separate from warmed service',
        cache_cold_is_not_process_cold=True))
    from .refresh_camera_cost import refresh
    refresh(model.img_backbone)
    from .balanced_camera_probe import run as balanced_camera
    if not (ROOT/'balanced_camera_results.json').exists():balanced_camera(model,dataset)
    if not (ROOT/'balanced_real_camera_results.json').exists():balanced_camera(model,dataset,real=True)
    from .visual_service_probe import run as visual_service
    if not (ROOT/'visual_service_results.json').exists():visual_service(model,dataset)
    timer=PipelineTimer()
    for name,module in [('detection_head',model.pts_bbox_head),('map_head',model.map_head),
        ('det_carrier_output_projection',model.pts_bbox_head.output_projection),
        ('map_carrier_output_projection',model.map_head.output_projection)]:timer.wrap(module,name)
    base=model.lm_head.get_base_model() if hasattr(model.lm_head,'get_base_model') else model.lm_head
    original=base.prepare_inputs_labels_for_multimodal
    def assembly(*a,**kw):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        tick=perf_counter();start.record()
        with torch.autograd.profiler.record_function('carrier_query_assembly'):result=original(*a,**kw)
        end.record();timer.records.append(('carrier_query_assembly',start,end,(perf_counter()-tick)*1000))
        return result
    base.prepare_inputs_labels_for_multimodal=assembly
    from mmcv.parallel import scatter,collate
    import json
    protocol=json.load(open(ROOT/'protocol.json'))
    ids=protocol['real_scenes'][0]['indices'];samples={i:dataset[i] for i in ids};rows=[]
    for rep in range(3):
        model.test_flag=False;model.save_path=str(ROOT/f'carrier_profile_{rep}')+'/'
        for i in ids:
            batch=scatter(collate([copy.deepcopy(samples[i])],samples_per_gpu=1),[torch.cuda.current_device()])[0]
            timer.records.clear()
            model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
            rows.append(dict(repeat=rep,index=i,warmup=rep==0,components=timer.collect()))
    keys={k for r in rows for k in r['components']}
    summary={k:dict(gpu_ms=stats([r['components'][k]['gpu_ms'] for r in rows if not r['warmup']]),
                    cpu_enqueue_wall_ms=stats([r['components'][k]['cpu_enqueue_wall_ms'] for r in rows if not r['warmup']])) for k in keys}
    # Trace with these same correct base-model wrappers after all warmups.
    model.save_path=str(ROOT/'carrier_profile_trace')+'/'
    batch=scatter(collate([copy.deepcopy(samples[ids[0]])],samples_per_gpu=1),[torch.cuda.current_device()])[0]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
    prof.export_chrome_trace(str(ROOT/'traces/carrier_profile.json'))
    write(ROOT/'carrier_profile.json',dict(rows=rows,summary=summary,
        note='Separate 4-frame original-model sequence; one warmup replay plus two measurements. Projection and multimodal assembly measured on actual Peft base. Query decoder shares head computation; projection timings are nested in heads, not additive.'))

if __name__=='__main__':main()
