"""Continuous nuScenes, deployed full OmniDrive, M1+M2 and deterministic actual LLM generation."""
import argparse
from pathlib import Path
from time import perf_counter
from dataclasses import replace
import random
import numpy as np
import torch
from .common import write
from tools.worldstream_m1.model_smoke import digest


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', default='outputs/module2/integration')
    p.add_argument('--frames', type=int, default=4)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--scene', default='01452fbfbf4543af8acdfd3e8a1ee806')
    args=p.parse_args(); root=Path(args.output); root.mkdir(parents=True, exist_ok=True)
    if (root/'runs.json').exists(): raise ValueError('Choose fresh output to preserve prior inference')
    write(root/'status.json', dict(status='loading', args=vars(args)))
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    from worldstream.m1 import M1Config, WorldStateManager
    from worldstream.m2 import IncrementalVisionRuntime
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_coverage_observe.py')
    cfg.model.m1_enable=False; cfg.model.train_cfg=None; cfg.model.pretrained=None; cfg.data.test.test_mode=True
    dataset=build_dataset(cfg.data.test)
    indices=[i for i,info in enumerate(dataset.data_infos) if info['scene_token']==args.scene][:args.frames]
    if len(indices)<2: raise ValueError('Need at least 2 continuous frames')
    samples=[dataset[i] for i in indices]
    model=build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'): wrap_fp16_model(model)
    load_checkpoint(model,'ckpts/iter_10548.pth',map_location='cpu')
    model.cuda().eval()
    # Only the experiment fixes generation; production generation settings are untouched.
    original_generate=model.lm_head.generate
    def deterministic_generate(*a, **kw):
        kw.update(do_sample=False)
        kw.pop('temperature', None); kw.pop('top_p', None)
        return original_generate(*a, **kw)
    model.lm_head.generate=deterministic_generate
    settings=M1Config.load('configs/worldstream_m1_coverage.yaml')
    baseline=None; runs=[]; comparisons=[]
    events={}; features=[]
    def start_backbone(m, inputs):
        event=torch.cuda.Event(enable_timing=True); event.record(); events['start']=event
    def end_backbone(m, inputs, output):
        event=torch.cuda.Event(enable_timing=True); event.record(); events['end']=event
    handles=[model.img_backbone.patch_embed.register_forward_pre_hook(start_backbone),
             model.img_backbone.blocks[-1].register_forward_hook(end_backbone)]
    original_extract=model.extract_img_feat
    def extract(img):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        tick=perf_counter(); start.record()
        value=original_extract(img); end.record()
        events['vision']=(start,end,(perf_counter()-tick)*1000)
        features.append(value.detach()); return value
    model.extract_img_feat=extract
    for mode in ['baseline','m1_observe','m1_m2_exact','m1_m2_auto']:
        for rep in range(args.warmup+args.repeats):
            random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            model.test_flag=False
            model.m1_manager=None if mode=='baseline' else WorldStateManager(replace(settings,
                log_path=str(root/f'{mode}_{rep}.jsonl')))
            model.m2_runtime=(IncrementalVisionRuntime(model.img_backbone, dict(
                mode='exact' if mode.endswith('exact') else 'auto',
                cost_table='outputs/module2/break_even_results.json')) if mode.startswith('m1_m2') else None)
            if model.m2_runtime:
                runtime_forward=model.m2_runtime.forward
                def measured_runtime(*a,**kw):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    tick=perf_counter(); start.record()
                    result=runtime_forward(*a,**kw); end.record()
                    events['runtime']=(start,end,(perf_counter()-tick)*1000)
                    return result
                model.m2_runtime.forward=measured_runtime
            model.save_path=str(root/f'{mode}_{rep}'/'language')+'/'
            outputs=[]; memories=[]; backbones=[]; texts=[]; frame_rows=[]
            torch.cuda.reset_peak_memory_stats()
            for index, sample in zip(indices, samples):
                batch=scatter(collate([sample],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                torch.cuda.synchronize(); tick=perf_counter()
                result=model(return_loss=False,rescale=True,**batch)
                torch.cuda.synchronize(); elapsed=(perf_counter()-tick)*1000
                backbone_ms=events['start'].elapsed_time(events['end'])
                vs,ve,vcpu=events['vision']
                vision_ms=vs.elapsed_time(ve)
                if model.m2_runtime:
                    rs,re,rcpu=events['runtime']; runtime_ms=rs.elapsed_time(re)
                else:
                    runtime_ms=backbone_ms; rcpu=0.
                m1_ms=(model.m1_manager.last_commit_diagnostics['m1_total_including_adapters_ms']
                       if model.m1_manager else 0.)
                diag=getattr(model,'m2_diagnostics',{}) if model.m2_runtime else {}
                feature=features.pop()
                backbones.append(digest(feature)); outputs.append(digest(result))
                memories.append({name:{k:digest(v) for k,v in vars(head).items()
                    if k.startswith('memory_') or k=='sample_time'}
                    for name,head in [('det',model.pts_bbox_head),('map',model.map_head)]})
                texts.append(result[0].get('text_out'))
                m1_candidates={}
                if model.m1_manager:
                    for cid, mask in model.m1_manager.last_masks.items():
                        h,w=mask.shape
                        patch=(~mask[:h//16*16,:w//16*16]).reshape(h//16,16,w//16,16).all(axis=(1,3))
                        candidates=0
                        for y in range(0,patch.shape[0],16):
                            for x in range(0,patch.shape[1],16):
                                candidates+=int(patch[y:y+16,x:x+16].all())
                        m1_candidates[cid]=dict(stable_pixels=int((~mask).sum()),
                            complete_patch_candidates=int(patch.sum()), complete_attention_windows=candidates)
                frame_rows.append(dict(frame_id=dataset.data_infos[index]['token'], cold=len(frame_rows)==0,
                    complete_wall_ms=elapsed, backbone_gpu_ms=backbone_ms, m1_total_ms=m1_ms,
                    vision_gpu_interval_ms=vision_ms, vision_cpu_dispatch_ms=vcpu,
                    m2_runtime_gpu_interval_ms=runtime_ms, m2_cpu_dispatch_ms=rcpu,
                    m2_overhead_gpu_interval_ms=max(0.,runtime_ms-backbone_ms),
                    m2=diag, m1_candidates=m1_candidates))
            current=dict(outputs=outputs, memory=memories, backbone=backbones, text=texts)
            if baseline is None:
                baseline=current; write(root/'baseline_digests.json',baseline)
            equality={k:current[k]==baseline[k] for k in baseline}
            comparisons.append(dict(mode=mode, repeat=rep, **equality))
            record=dict(mode=mode, repeat=rep, warmup=rep<args.warmup, frames=frame_rows,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            runs.append(record); write(root/'runs.json', runs); write(root/'comparisons.json',comparisons)
            print(mode,rep,'equality',equality,'wall ms',[round(f['complete_wall_ms'],2) for f in frame_rows],flush=True)
    for h in handles: h.remove()
    summary={}
    for mode in ['baseline','m1_observe','m1_m2_exact','m1_m2_auto']:
        frames=[f for r in runs if r['mode']==mode and not r['warmup'] for f in r['frames']]
        summary[mode]={k:np.percentile([f[k] for f in frames],[50,95]).tolist()
                       for k in ('complete_wall_ms','backbone_gpu_ms','m1_total_ms')}
        summary[mode]['cold_wall_ms']=[f['complete_wall_ms'] for f in frames if f['cold']]
        summary[mode]['steady_p50_p95_ms']=np.percentile(
            [f['complete_wall_ms'] for f in frames if not f['cold']],[50,95]).tolist()
    write(root/'status.json',dict(status='completed',args=vars(args)))
    write(root.parent/'e2e_latency.json',dict(summary=summary, runs=runs, comparisons=comparisons,
        sample_ids=[dataset.data_infos[i]['token'] for i in indices], unique_frames=len(indices),
        generation='greedy, seed 0, original max_new_tokens=320; actual LLM executed every frame',
        protocol='preloaded CPU samples; scatter before timing; synchronized complete forward includes all M1 adapters/commits; digests outside timer; fixed mode order'))
    write(root.parent/'real_m1_results.json', dict(status='newly_measured',
        frames=[dict(mode=r['mode'],repeat=r['repeat'],**f) for r in runs for f in r['frames']],
        all_equal=all(all(c[k] for k in ('outputs','memory','backbone','text')) for c in comparisons),
        transformer_certificate=False, approximate_path=False))


if __name__=='__main__': main()
