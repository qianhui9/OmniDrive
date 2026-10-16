"""Full original OmniDrive, current temporal heads and real greedy language generation."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import resource
from time import perf_counter
import numpy as np
import torch
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.camera_executor import camera_cost_key
from worldstream.m2.profiler import measure
from tools.worldstream_m2.refine_integration import planning_metric
from tools.worldstream_m2.refinement_common import window_statistics
from .common import ROOT,write,arrays,memory,compare,quality_pass,reference,stats
from .phase_ab import PackedDense


class PipelineTimer:
    def __init__(self):self.records=[]
    def wrap(self,module,name,dynamic=None):
        original=module.forward
        def forward(*args,**kw):
            label=dynamic(args,kw) if dynamic else name
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            tick=perf_counter();a.record()
            with torch.autograd.profiler.record_function(label):result=original(*args,**kw)
            b.record();self.records.append((label,a,b,(perf_counter()-tick)*1000))
            return result
        module.forward=forward
    def collect(self):
        result={}
        for name,a,b,cpu in self.records:
            r=result.setdefault(name,dict(gpu_ms=0.,cpu_enqueue_wall_ms=0.,calls=0))
            r['gpu_ms']+=a.elapsed_time(b);r['cpu_enqueue_wall_ms']+=cpu;r['calls']+=1
        self.records.clear();return result


def load():
    from mmcv import Config
    from mmcv.runner import load_checkpoint,wrap_fp16_model
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_coverage_observe.py')
    cfg.model.m1_enable=False;cfg.model.train_cfg=None;cfg.model.pretrained=None;cfg.data.test.test_mode=True
    dataset=build_dataset(cfg.data.test)
    model=build_model(cfg.model,test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):wrap_fp16_model(model)
    load_checkpoint(model,'ckpts/iter_10548.pth',map_location='cpu');model.cuda().eval()
    generate=model.lm_head.generate
    def greedy(*a,**kw):
        kw.update(do_sample=False);kw.pop('temperature',None);kw.pop('top_p',None)
        return generate(*a,**kw)
    model.lm_head.generate=greedy
    return model,dataset


def protocol(infos):
    old=json.load(open('outputs/module2_refinement/protocol.json'))['scenes']
    chosen=copy.deepcopy(old);used={s['scene'] for s in chosen}
    groups={}
    for i,info in enumerate(infos):groups.setdefault(info['scene_token'],[]).append(i)
    for category,term in [('pedestrians','ped'),('low_light','night')]:
        scene=next(s for s,ids in sorted(groups.items()) if s not in used and term in infos[ids[0]]['description'].lower() and len(ids)>8)
        ids=sorted(groups[scene],key=lambda i:infos[i]['timestamp']);indices=[ids[k] for k in (0,1,2,4)]
        chosen.append(dict(scene=scene,scene_name=infos[indices[0]]['scene_name'],category=category,indices=indices,
            description=infos[indices[0]]['description'],sample_ids=[infos[i]['token'] for i in indices],
            gaps_s=np.diff([infos[i]['timestamp']/1e6 for i in indices]).tolist()))
        used.add(scene)
    return chosen,groups


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=2);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--output',default=str(ROOT/'integration'));args=p.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    if any(dest.iterdir()):raise ValueError('Choose a completely fresh output directory, including language files')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    import cv2
    cv2.setNumThreads(1)
    model,dataset=load();scenes,groups=protocol(dataset.data_infos)
    controlled=[]
    for spec in scenes[:2]:
        ids=sorted(groups[spec['scene']],key=lambda i:dataset.data_infos[i]['timestamp'])
        start=ids.index(spec['indices'][0]);picked=ids[start:start+16]
        if len(picked)<16:picked=ids[:16]
        controlled.append(dict(spec,indices=picked))
    write(ROOT/'protocol.json',dict(real_scenes=scenes,controlled_scenes=controlled,
        quality_targets=json.load(open('outputs/module2_refinement/quality_targets.json')),
        warmup=args.warmup,repeats=args.repeats,order='Latin-square rotation over scenes/replays; state reset per sequence',
        language='actual greedy, seed 0, max_new_tokens 320; unique save paths',
        service_contracts=dict(maintain='full M1 history',reset_on_reject='explicit reset plus current precheck/commit',
                              output_only='explicitly discard M1 state; heads/memory remain live')))
    samples={i:dataset[i] for s in scenes+controlled for i in s['indices']}
    from mmcv.parallel import collate,scatter
    timer=PipelineTimer();timer.wrap(model.img_backbone.patch_embed,'patch_embedding')
    for i,b in enumerate(model.img_backbone.blocks):timer.wrap(b,f'block_{i}')
    timer.wrap(model.pts_bbox_head,'detection_head');timer.wrap(model.map_head,'map_head')
    def lm_stage(a,kw):
        t=kw.get('inputs_embeds');t=t if t is not None else kw.get('input_ids')
        if t is None and a:t=a[0]
        return 'llm_prefill' if t is not None and t.shape[1]>1 else 'llm_decode'
    timer.wrap(model.lm_head.get_model(),'llm',lm_stage)
    multimodal=model.lm_head.prepare_inputs_labels_for_multimodal
    def carrier(*a,**kw):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        tick=perf_counter();start.record()
        with torch.autograd.profiler.record_function('carrier_query_assembly'):out=multimodal(*a,**kw)
        end.record();timer.records.append(('carrier_query_assembly',start,end,(perf_counter()-tick)*1000));return out
    model.lm_head.prepare_inputs_labels_for_multimodal=carrier
    captured={};features=[];stages={}
    model.img_backbone.register_forward_hook(lambda m,i,o:captured.update(backbone=o[0]))
    for j in (0,1,2,5,11,23):
        model.img_backbone.blocks[j].register_forward_hook(lambda m,i,o,j=j:stages.update({j:o.detach()}))
    original_extract=model.extract_img_feat
    def extract(img):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();tick=perf_counter()
        result=original_extract(img);b.record();timer.records.append(('visual_service',a,b,(perf_counter()-tick)*1000))
        features.append(result.detach());return result
    model.extract_img_feat=extract
    settings=replace(M1Config.load('configs/worldstream_m1_coverage.yaml'),logging=False)
    runs=[];quality=[];refs={};old=reference()
    original_forward=model.img_backbone.forward
    frozen_table='outputs/module2_refinement/break_even_results.json'
    camera_table=str(ROOT/'camera_costs.json')
    # Before quality evidence there is deliberately no formal camera permission.
    write(ROOT/'status.json',dict(status='running',stage='controlled_quality'))
    def configure(mode):
        model.img_backbone.forward=original_forward
        model.m1_manager=None
        if mode=='baseline':model.m2_runtime=None
        elif mode=='packed_dense':
            packed=PackedDense(model.img_backbone)
            model.img_backbone.forward=lambda x:packed(x)
            model.m2_runtime=None
        elif mode=='original':
            model.m1_manager=WorldStateManager(settings)
            model.m2_runtime=old.IncrementalVisionRuntime(model.img_backbone,dict(mode='exact'))
        else:
            service={'gate_maintain':'maintain','gate_reset':'reset_on_reject'}.get(mode,'output_only')
            is_camera=mode.startswith('camera')
            controlled_mode=mode in ('window_exact','camera_exact')
            if not controlled_mode:model.m1_manager=WorldStateManager(settings)
            model.m2_runtime=IncrementalVisionRuntime(model.img_backbone,dict(
                mode='exact' if controlled_mode else 'auto',reuse_boundary='camera' if is_camera else 'window',
                cost_table=frozen_table,camera_cost_table=camera_table if Path(camera_table).exists() else '',
                cheap_gate=not controlled_mode,m1_service=service if not controlled_mode else 'maintain'))
    def run(spec,mode,rep,case='real'):
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        configure(mode);model.test_flag=False
        model.save_path=str(dest/f'{case}_{mode}_{rep}_{spec["scene"]}'/'language')+'/'
        frame_rows=[];base=None;torch.cuda.reset_peak_memory_stats()
        for ordinal,index in enumerate(spec['indices']):
            batch=scatter(collate([copy.deepcopy(samples[index])],samples_per_gpu=1),[torch.cuda.current_device()])[0]
            image=batch['img'][0]
            if case!='real':
                if base is None:base=image.clone()
                image.copy_(base)
                if case=='mixed':
                    # Counts and positions are protocol-defined, never selected for favourable errors.
                    k=ordinal%8
                    if k==2:image[0,0,:,:16,:16]+=.03
                    elif k==3:image[0,:3]+= .03
                    elif k==4:image[0,:5]+= .03
                    elif k==5:image+=.03
                    elif k==6:image[0,0,:,:256,:256]+=.03
                    elif k==7:image[0,:2,:,:256,:256]+=.03
            timer.records.clear();stages.clear();features.clear()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize();tick=perf_counter();start.record()
            out=model(return_loss=False,rescale=True,**batch)
            end.record();torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
            component=timer.collect();feat=features.pop()
            diag=copy.deepcopy(getattr(model,'m2_diagnostics',{})) if model.m2_runtime else {}
            admission=copy.deepcopy(getattr(model,'m1_admission_diagnostics',{})) if model.m2_runtime else {}
            m1=model.m1_manager.last_commit_diagnostics.get('m1_total_including_adapters_ms',0.) if model.m1_manager else 0.
            current=dict(output=arrays(out),feature=arrays(feat),memory=memory(model),text=out[0].get('text_out'),
                planning=planning_metric(out[0].get('text_out'),dataset.data_infos[index]),
                stages={j:arrays(v) for j,v in stages.items()})
            key=(case,spec['scene'],ordinal)
            if mode=='baseline' and key not in refs:refs[key]=current
            if mode!='baseline':
                q=compare(current,refs[key]);q['frozen_targets_pass']=quality_pass(q)
                q.update(case=case,scene=spec['scene'],ordinal=ordinal,index=index,mode=mode,repeat=rep,
                    actual_reused_cameras=diag.get('actual_reused_cameras',0),actual_skipped_windows=diag.get('actual_skipped_windows',0),
                    stages={str(j):error_location_cpu(v,refs[key]['stages'][j]) for j,v in current['stages'].items() if j in refs[key]['stages']})
                quality.append(q)
            state={}
            if model.m1_manager:
                mgr=model.m1_manager
                state=dict(epoch=mgr.store.generation,last_timestamp=mgr.store.last_timestamp,
                    pending=getattr(mgr,'_pending',None),feature_versions=mgr.store.feature_versions(),
                    history_camera_count=len(mgr._previous_cameras),anchor_count=len(mgr.store.anchors),
                    rejection_reason=admission.get('reason'),world_history_preserved=mode in ('original','gate_maintain'))
            frame_rows.append(dict(index=index,ordinal=ordinal,frame_id=dataset.data_infos[index]['token'],cold=ordinal==0,
                complete_wall_ms=wall,gpu_interval_ms=start.elapsed_time(end),components=component,m1_ms=m1,
                runtime=diag,admission=admission,m1_state=state,
                windows=window_statistics(model.m1_manager,model.img_backbone.patch_embed.proj) if mode=='original' else {}))
            # Retain native dtype temporal state without putting diagnostic I/O in timers.
            if case!='real' and rep==0:
                raw={f'{branch}.{name}':v for branch,values in current['memory'].items() for name,v in values.items() if isinstance(v,np.ndarray)}
                np.savez_compressed(dest/f'{case}_{mode}_{spec["scene"]}_{ordinal}_memory.npz',**raw)
        rec=dict(case=case,scene=spec['scene'],category=spec['category'],mode=mode,repeat=rep,
            warmup=rep<args.warmup if case=='real' else rep==0,frames=frame_rows,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            cpu_rss_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        runs.append(rec);write(dest/'runs.json',runs);write(dest/'quality.json',quality)
        print(case,mode,spec['scene'][:8],rep,'P50',round(np.median([f['complete_wall_ms'] for f in frame_rows]),2),
              'camera',sum(f['runtime'].get('actual_reused_cameras',0) for f in frame_rows),flush=True)
    # Numerical A/B/C and partial camera full-task diagnostics. Long sequence is 16 frames.
    for spec in controlled:
        for mode in ('baseline','packed_dense','window_exact','camera_exact'):
            run(spec,mode,0,'mixed')
    write(ROOT/'correctness_results.json',dict(quality=quality,notes='mixed sequence with real chronological poses/time; synthetic local/camera input changes'))
    # Full repeated image requests keep pose/time/heads/LLM changing. Warm replay then two measured replays.
    for spec in controlled:
        for mode in ('baseline','camera_exact'):
            for rep in range(3):run(spec,mode,rep,'repeat')
    # Camera table is restricted by actual downstream checks; no partial count is
    # promoted by merely passing a whole-feature norm. Only repeated full-batch equality.
    repeated=[q for q in quality if q['case']=='repeat' and q['mode']=='camera_exact' and q['ordinal']>0]
    repeat_ok=bool(repeated) and all(q['frozen_targets_pass'] and q['feature']['max_abs']==0 and
        q['memory']['max_abs']==0 and q['output']['max_abs']==0 and not q['memory']['shape_mismatches']
        and not q['output']['shape_mismatches'] for q in repeated)
    image=torch.randn(6,3,640,640,device='cuda');ctx=dict(scene_id='cost',frame_id='0',camera_ids=list(range(6)))
    records=[]
    for changed in (0,1,3,5,6):
        current=image.clone();current[:changed]+=.03
        executor=IncrementalVisionRuntime(model.img_backbone,dict(mode='exact',reuse_boundary='camera'))
        executor.forward(image,execution_context=ctx)
        cache=executor.camera_executor.cache
        saved=copy.copy(vars(cache))
        def execute():
            cache.__dict__.update(saved)
            return executor.forward(current,execution_context=dict(ctx,frame_id='1'))
        reuse=measure(execute,warmup=3,repeats=10);full=measure(lambda:model.img_backbone(current),warmup=3,repeats=10)
        records.append(dict(key=camera_cost_key(model.img_backbone,image),changed_cameras=changed,
            reuse_wall_ms=reuse['wall_ms'],full_wall_ms=full['wall_ms'],reuse_gpu_ms=reuse['gpu_ms'],full_gpu_ms=full['gpu_ms'],
            quality_certified=changed==0 and repeat_ok,reuse=reuse,full=full))
    write(ROOT/'camera_costs.json',dict(schema_version=1,records=records,repeat_full_task_exact=repeat_ok,
        qualification='only all-camera identical requests approved; partial camera shape variation is experimental'))
    # Actual formal Auto full-model runs after publishing quality/cost evidence.
    for spec in controlled:
        run(spec,'camera_auto',0,'repeat')
    write(ROOT/'status.json',dict(status='running',stage='real_scenes'))
    # Baseline oracle first; timing runs use balanced rotations (including baseline reruns).
    modes=['baseline','original','gate_maintain','gate_reset','gate_output','camera_auto']
    for spec in scenes:run(spec,'baseline',0)
    for rep in range(args.warmup+args.repeats):
        for si,spec in enumerate(scenes):
            order=modes[(rep+si)%len(modes):]+modes[:(rep+si)%len(modes)]
            for mode in order:
                if rep==0 and mode=='baseline':continue
                run(spec,mode,rep)
    # Trace actual full model after prior warmups; one real request, diagnostic only.
    spec=scenes[0];configure('baseline');model.test_flag=False
    model.save_path=str(dest/'profile_language')+'/'
    batch=scatter(collate([copy.deepcopy(samples[spec['indices'][0]])],samples_per_gpu=1),[torch.cuda.current_device()])[0]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
    prof.export_chrome_trace(str(ROOT/'traces/full_pipeline.json'))
    write(ROOT/'latency_results.json',dict(runs=runs,quality=quality,profiling='identical wrapper overhead across modes; diagnostics/scatter outside wall timer'))
    write(ROOT/'status.json',dict(status='completed'))


def error_location_cpu(a,b):
    from .common import error_location
    if a.shape!=b.shape:
        return dict(computed_shape=list(a.shape),full_shape=list(b.shape),
            comparison='camera_subbatch_intermediate_not_full_batch; final assembled features compared separately')
    return error_location(torch.from_numpy(a),torch.from_numpy(b))

if __name__=='__main__':main()
