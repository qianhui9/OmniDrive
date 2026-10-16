"""Balanced multi-scene actual OmniDrive plus nonzero controlled sparse quality checks."""
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import copy
import json
import random
import re
import resource
import numpy as np
import torch
from .common import write
from .refinement_common import ROOT,reference,select_protocol,arrays,numeric_diff,detection_diff,error_location,window_statistics,memory_alignment,map_alignment
from tools.worldstream_m1.model_smoke import digest
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.profiler import measure
from worldstream.m2.cost_model import cost_key
from worldstream.m2.packing import partition,unpartition
from worldstream.m2.window_executor import WindowExecutor


def memory(model):
    return arrays({name:{k:v for k,v in vars(head).items() if k.startswith('memory_') or k=='sample_time'}
                   for name,head in [('det',model.pts_bbox_head),('map',model.map_head)]})


def planning_metric(text,info):
    answers=' '.join(str(a) for qa in (text or []) for a in qa.get('A',[]))
    pairs=re.findall(r'\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)',answers)
    if len(pairs)<6:return dict(parsed=False)
    xy=np.asarray(pairs[:6],float);gt=np.asarray(info['gt_planning'])[0,:,:2]
    mask=np.asarray(info['gt_planning_mask'])[0].all(axis=-1).astype(bool)
    distance=np.linalg.norm(xy-gt,axis=1)
    return dict(parsed=True,trajectory=xy.tolist(),ade_m=float(distance[mask].mean()) if mask.any() else None,
        fde_m=float(distance[-1]) if mask[-1] else None)


@torch.no_grad()
def supplement_costs(model):
    # Block 0 full output and packed reconstruction can have different strides.
    # Profile the second actual layout independently instead of sharing/interpolating costs.
    path=ROOT/'break_even_results.json';table=json.loads(path.read_text())
    block=model.img_backbone.blocks[1];x=torch.randn(6,40,40,1024,device='cuda')
    reference()
    from m2_refinement_reference.window_executor import WindowExecutor as OriginalExecutor
    original=OriginalExecutor()
    ex=WindowExecutor();cache=partition(block(x),16)[0]
    prior_path=ROOT/'layout_cost_supplement.json'
    rows=json.loads(prior_path.read_text()) if prior_path.exists() else []
    for count in (0,1,6,36,54):
        key=cost_key(block,x,1)
        if any(r['key']==key and r['updated']==count for r in table['records']):continue
        ids=torch.randperm(54,device='cuda')[:count]
        wins,pad=partition(x,16);wins[ids]+=.03;current=unpartition(wins,16,pad,(40,40))
        full=measure(lambda:block(current),warmup=5,repeats=20)
        packed=measure(lambda:ex.execute(block,current,cache,ids),warmup=5,repeats=20)
        prior=measure(lambda:original.execute(block,current,cache,ids),warmup=5,repeats=20)
        spatial=packed['gpu_ms']<prior['gpu_ms'] and packed['wall_ms']<prior['wall_ms']
        chosen=packed if spatial else prior
        err=error_location(ex.execute(block,current,cache,ids)[0],block(current))
        qualified=table['whole_backbone_counts'].get(str(count),False) and err['max_abs']<=.02 and err['relative_l2']<=.002
        table['records'].append(dict(key=key,block=1,updated=count,incremental_gpu_ms=chosen['gpu_ms'],
            packing_backend='spatial' if spatial else 'original',
            incremental_wall_ms=chosen['wall_ms'],full_gpu_ms=full['gpu_ms'],full_wall_ms=full['wall_ms'],
            backbone_quality_certified=qualified,
            quality_certified=qualified and count==0 and bool(table.get('formal_quality_policy'))))
        rows.append(dict(block=1,updated=count,full=full,packed=packed,original=prior,error=err))
    write(ROOT/'layout_cost_supplement.json',rows);write(path,table)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--frames',type=int,default=4)
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--output',default=str(ROOT/'integration'))
    p.add_argument('--controlled-only',action='store_true')
    p.add_argument('--repeat-only',action='store_true')
    p.add_argument('--run-repeat-after',action='store_true')
    p.add_argument('--run-auto-after',action='store_true');args=p.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'runs.json').exists():raise ValueError('Fresh output required')
    write(dest/'status.json',dict(status='loading',args=vars(args)))
    if not (ROOT/'quality_targets.json').exists():
        from datetime import datetime,timezone
        write(ROOT/'quality_targets.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),
            purpose='Pre-run regression targets, not road-safety certification',
            detection=dict(unmatched=0,max_center_difference_m=.01),map_max_control_point_difference_m=.01,
            greedy_text_exact=True,planning_output_exact=True,
            memory='Report raw and jointly aligned fields; no task certification from whole-feature L2.'))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    import cv2
    cv2.setNumThreads(1)
    from mmcv import Config
    from mmcv.parallel import collate,scatter
    from mmcv.runner import load_checkpoint,wrap_fp16_model
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    from worldstream.m1 import M1Config,WorldStateManager
    old=reference()
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_coverage_observe.py')
    cfg.model.m1_enable=False;cfg.model.train_cfg=None;cfg.model.pretrained=None;cfg.data.test.test_mode=True
    dataset=build_dataset(cfg.data.test);specs=select_protocol(dataset.data_infos)
    # Selection uses ONLY offline metadata, frozen before model outputs or timings.
    write((dest if args.controlled_only else ROOT)/'protocol.json',dict(scenes=specs,warmup=args.warmup,repeats=args.repeats,
        order='Latin-square rotation over scene/replay; full state reset each run',
        controlled_scenes=[s['scene'] for s in specs[:2]],generation='greedy, seed=0, original max_new_tokens=320',
        quality='new independent random inputs checked against frozen prior per-layer envelope; real nonzero sparse task checks separately'))
    samples={i:dataset[i] for s in specs for i in s['indices']}
    model=build_model(cfg.model,test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):wrap_fp16_model(model)
    load_checkpoint(model,'ckpts/iter_10548.pth',map_location='cpu');model.cuda().eval()
    generate=model.lm_head.generate
    def deterministic(*a,**kw):
        kw.update(do_sample=False);kw.pop('temperature',None);kw.pop('top_p',None)
        return generate(*a,**kw)
    model.lm_head.generate=deterministic
    supplement_costs(model)
    settings=M1Config.load('configs/worldstream_m1_coverage.yaml')
    events={};stage_outputs={};features=[];inputs=[]
    def pre(m,i):
        start=torch.cuda.Event(enable_timing=True);start.record();events['backbone_start']=start
    def post(m,i,o):
        end=torch.cuda.Event(enable_timing=True);end.record();events['backbone_end']=end
    handles=[model.img_backbone.patch_embed.register_forward_pre_hook(pre),
             model.img_backbone.blocks[-1].register_forward_hook(post)]
    # Capturing references is lightweight; diagnostic CPU copies happen after timing.
    hooks=[model.img_backbone.blocks[j].register_forward_hook(lambda m,i,o,j=j:stage_outputs.update({j:o.detach()})) for j in (0,1)]
    original_extract=model.extract_img_feat
    def extract(img):
        inputs.append(img.detach())
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();value=original_extract(img);end.record();events['vision']=(start,end)
        features.append(value.detach());return value
    model.extract_img_feat=extract
    runs=[];quality=[];reference_real={};reference_control={}
    modes=['baseline','original','optimized','auto']
    def run_sequence(spec,mode,rep,controlled=False):
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        model.test_flag=False
        model.m1_manager=(None if mode=='baseline' or controlled else WorldStateManager(replace(settings,
            logging=False,log_path=str(dest/f'{mode}_{rep}_{spec["scene"]}.jsonl'))))
        if mode=='original':model.m2_runtime=old.IncrementalVisionRuntime(model.img_backbone,dict(mode='exact'))
        elif mode in ('optimized','auto'):
            model.m2_runtime=IncrementalVisionRuntime(model.img_backbone,dict(mode='auto' if mode=='auto' else 'exact',
                cost_table=str(ROOT/'break_even_results.json')))
        else:model.m2_runtime=None
        if model.m2_runtime:
            forward=model.m2_runtime.forward
            def wrapped(*a,**kw):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                tick=perf_counter();start.record();value=forward(*a,**kw);end.record()
                events['runtime']=(start,end,(perf_counter()-tick)*1000)
                return value
            model.m2_runtime.forward=wrapped
            if controlled:
                model.m2_runtime.stage_observer=lambda j,out:stage_outputs.update({j:out.detach()}) if j in (0,1) else None
        case='repeat' if args.repeat_only and controlled else 'controlled' if controlled else 'real'
        model.save_path=str(dest/f'{case}_{mode}_{rep}_{spec["scene"]}'/'language')+'/'
        frames=[];torch.cuda.reset_peak_memory_stats();base_image=None;previous_input=None
        for ordinal,index in enumerate(spec['indices'][:args.frames]):
            batch=scatter(collate([copy.deepcopy(samples[index])],samples_per_gpu=1),[torch.cuda.current_device()])[0]
            if controlled:
                image=batch['img'][0]
                if base_image is None:base_image=image.detach().clone()
                # Always compare the SAME synthetic sequence against its full baseline.
                image.copy_(base_image)
                if not args.repeat_only:
                    if ordinal in (1,2):image[0,0,:,:16,:16]+=.03*ordinal
                    if ordinal==3:image[0,0,:,-16:,-16:]+=.03
            torch.cuda.synchronize();tick=perf_counter()
            output=model(return_loss=False,rescale=True,**batch)
            torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
            diag=copy.deepcopy(getattr(model,'m2_diagnostics',{})) if model.m2_runtime else {}
            vs,ve=events['vision'];backbone_ms=events['backbone_start'].elapsed_time(events['backbone_end'])
            runtime_ms=events['runtime'][0].elapsed_time(events['runtime'][1]) if model.m2_runtime else backbone_ms
            m1=(model.m1_manager.last_commit_diagnostics['m1_total_including_adapters_ms'] if model.m1_manager else 0.)
            feat=features.pop();actual_input=inputs.pop()
            if actual_input.ndim==5:actual_input=actual_input.flatten(0,1)
            # All following oracle reads and metrics are outside the synchronized timer.
            current=dict(output=arrays(output),memory=memory(model),feature=arrays(feat),
                stages={j:arrays(stage_outputs[j]) for j in (0,1)},text=output[0].get('text_out'))
            stats=window_statistics(model.m1_manager,model.img_backbone.patch_embed.proj) if model.m1_manager else {}
            if stats:
                normalized=arrays(actual_input)
                for ci,cid in enumerate(stats):
                    equal=(np.zeros((640,640),bool) if previous_input is None else
                           (normalized[ci]==previous_input[ci]).all(axis=0)&np.isfinite(normalized[ci]).all(axis=0))
                    same=equal.reshape(40,16,40,16).all(axis=(1,3))
                    stable=np.asarray(stats[cid]['stable_patch_mask'],bool)
                    stats[cid]['exact_same_coordinate_patches']=int(same.sum())
                    stats[cid]['supported_and_exact_patches']=int((same&stable).sum())
                previous_input=normalized
            key=(spec['scene'],ordinal,args.repeat_only if controlled else False)
            refs=reference_control if controlled else reference_real
            if mode=='baseline' and key not in refs:refs[key]=current
            comparison=None
            if key in refs:
                ref=refs[key]
                comparison=dict(detection=detection_diff(current['output'],ref['output']),
                    output_numeric=numeric_diff(current['output'],ref['output']),
                    map_numeric=numeric_diff(current['output'][0]['lane_results'],ref['output'][0]['lane_results']),
                    memory_numeric=numeric_diff(current['memory'],ref['memory']),
                    feature=error_location(torch.from_numpy(current['feature']),torch.from_numpy(ref['feature'])),
                    stages={j:error_location(torch.from_numpy(current['stages'][j]),torch.from_numpy(ref['stages'][j])) for j in (0,1)},
                    text_equal=current['text']==ref['text'],
                    planning=planning_metric(current['text'],dataset.data_infos[index]),
                    baseline_planning=planning_metric(ref['text'],dataset.data_infos[index]))
                quality.append(dict(scene=spec['scene'],frame=index,ordinal=ordinal,mode=mode,repeat=rep,
                    input_case='repeat' if args.repeat_only and controlled else 'changed' if controlled else 'real',
                    controlled=controlled,actual_skipped_windows=diag.get('actual_skipped_windows',0),**comparison))
                if args.controlled_only:
                    quality[-1]['slot_alignment']=memory_alignment(current['memory'],ref['memory'])
                    quality[-1]['map_alignment']=map_alignment(current['output'][0]['lane_results'],ref['output'][0]['lane_results'])
                    raw={f'{branch}.{name}':v for branch,values in current['memory'].items() for name,v in values.items() if isinstance(v,np.ndarray)}
                    np.savez_compressed(dest/f'{case}_memory_{mode}_{spec["scene"]}_{ordinal}.npz',**raw)
            frames.append(dict(index=index,frame_id=dataset.data_infos[index]['token'],cold=ordinal==0,
                complete_wall_ms=wall,backbone_gpu_interval_ms=backbone_ms,vision_gpu_interval_ms=vs.elapsed_time(ve),
                m1_ms=m1,m2_extra_interval_ms=max(0.,runtime_ms-backbone_ms),m2=diag,m1_windows=stats))
        record=dict(scene=spec['scene'],category=spec['category'],mode=mode,repeat=rep,warmup=rep<args.warmup,
            input_case='repeat' if args.repeat_only and controlled else 'changed' if controlled else 'real',
            controlled=controlled,frames=frames,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        runs.append(record);write(dest/'runs.json',runs);write(dest/'quality.json',quality)
        print('controlled' if controlled else 'real',spec['category'],mode,rep,
            'skip',sum(f['m2'].get('actual_skipped_windows',0) for f in frames),
            'ms',[round(f['complete_wall_ms'],1) for f in frames],flush=True)
    # Reference warmup first populates quality oracles even when measured modes rotate.
    if not args.controlled_only:
        for spec in specs:
            run_sequence(spec,'baseline',0)
        for rep in range(args.warmup+args.repeats):
            for si,spec in enumerate(specs):
                order=modes[(rep+si)%4:]+modes[:(rep+si)%4]
                for mode in order:
                    if rep==0 and mode=='baseline':continue
                    run_sequence(spec,mode,rep)
    # Two independent scenes, actual full detector/map/memory/LLM, nonzero local reuse.
    for spec in specs[:2]:run_sequence(spec,'baseline',0,controlled=True)
    for rep in range(args.warmup+args.repeats):
        for si,spec in enumerate(specs[:2]):
            order=['baseline','optimized'] if (rep+si)%2==0 else ['optimized','baseline']
            for mode in order:
                if rep==0 and mode=='baseline':continue
                run_sequence(spec,mode,rep,controlled=True)
    if args.run_repeat_after:
        args.repeat_only=True
        for spec in specs[:2]:
            for mode in ('baseline','optimized'):run_sequence(spec,mode,0,controlled=True)
    for h in handles+hooks:h.remove()
    write(dest/'status.json',dict(status='completed',args=vars(args)))
    write(ROOT/('quality_audit_results.json' if args.controlled_only else 'e2e_latency.json'),dict(runs=runs,quality=quality,protocol=json.loads(((dest if args.controlled_only else ROOT)/'protocol.json').read_text()),
        output_layout_preserved=True,
        limitations='timings exclude scatter, diagnostics, and reference CPU reads; include all online M1/M2, detection/map, memory and actual greedy LLM. Synthetic repeated/local images are controlled, not real M1 gains.'))
    if args.run_auto_after:
        from .validate_quality_policy import main as publish_quality_policy
        publish_quality_policy()
        from .refine_auto import run
        run(model.img_backbone)


if __name__=='__main__':main()
