"""Offline real-weight calibration. Never grants production or task approval."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .common import backbone, write
from worldstream.m2.cost_model import cost_key, SCHEMA_VERSION, block_fingerprint
from worldstream.m2.dependency_planner import (DependencyPlanner, block0_input,
    exact_pixel_changes, exact_token_changes, patch_updates)
from worldstream.m2.packing import partition
from worldstream.m2.state_cache import StateCache
from worldstream.m2.window_executor import WindowExecutor, NUMERICAL_POLICY


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(samples):
    return dict(wall_ms=float(np.median([s[0] for s in samples])),
        gpu_span_ms=float(np.median([s[1] for s in samples])),repeats=len(samples),
        statistic='median',wall_samples_ms=[s[0] for s in samples],
        gpu_span_samples_ms=[s[1] for s in samples])


def paired_measure(incremental, full, repeats, warmup):
    for _ in range(warmup):
        incremental();full()
    torch.cuda.synchronize()
    samples=[[],[]];results=[None,None]
    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(2)]
    for repeat in range(repeats):
        for index in ((0,1) if repeat%2==0 else (1,0)):
            torch.cuda.synchronize()
            a,b=events[index];tick=perf_counter();a.record()
            results[index]=(incremental,full)[index]()
            b.record();torch.cuda.synchronize()
            samples[index].append(((perf_counter()-tick)*1000,a.elapsed_time(b)))
    return results,summary(samples[0]),summary(samples[1])


def difference(actual, expected):
    error=(actual.float()-expected.float()).abs()
    return dict(exact=torch.equal(actual,expected),max_abs=float(error.max()),
        mean_abs=float(error.mean()))


def change_metrics(image, previous, current_input, previous_input, projection, plans):
    pixels=exact_pixel_changes(image,previous)
    old_tokens=patch_updates(pixels,projection)
    tokens=exact_token_changes(current_input,previous_input)
    return dict(pixel_changed_ratio=float(pixels.float().mean()),
        pixel_derived_token_update_ratio=float(old_tokens.float().mean()),
        feature_changed_token_ratio=float(tokens.float().mean()),
        token_update_ratio=float(tokens.float().mean()),
        window_update_ratios={p.block:float(p.updated.float().mean()) for p in plans})


@torch.no_grad()
def run(args):
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py');cfg.data.test.test_mode=True
    cfg.data.test.pipeline=[node for node in cfg.data.test.pipeline if node['type'] in
        ('LoadMultiViewImageFromFiles','ResizeCropFlipRotImage','ResizeMultiview3D','NormalizeMultiviewImage','PadMultiViewImage')]
    dataset=build_dataset(cfg.data.test);dataset.data_infos.sort(key=lambda i:(i['scene_token'],i['timestamp']))
    model=backbone(args.checkpoint);planner=DependencyPlanner(model);prefix=planner.cacheable_prefix()
    if planner.rejection_reason or not prefix:raise ValueError('No verified cacheable prefix')
    executor=WindowExecutor();records=[]
    def native(image):
        cache={};handles=[]
        for i in prefix:
            handles.append(model.blocks[i].register_forward_hook(lambda block,inputs,output,i=i:
                cache.update({i:partition(output,block.window_size)[0].detach().clone()})))
        try:output=model(image)[0]
        finally:
            for h in handles:h.remove()
        return output,cache
    def execute(x,plans,cache,retain=False):
        inputs={};outputs={};stages=[]
        for i,block in enumerate(model.blocks):
            if i in prefix:
                ids=plans[i].updated.nonzero().flatten();total=plans[i].updated.numel()
                if retain:inputs[i]=x
                x,outputs[i]=executor.execute(block,x,cache[i],ids)
                stages.append(dict(block=i,total_windows=total,updated_windows=len(ids),
                    executed_attention_windows=len(ids),reused_output_windows=total-len(ids),
                    pointwise_window_work=total if len(ids) else 0,
                    pointwise_tokens=len(x)*x.shape[1]*x.shape[2] if len(ids) else 0))
            else:x=block(x)
        return x.permute(0,3,1,2),stages,inputs,outputs
    def pipeline(image,source_input,cache,retain=False):
        x=block0_input(model,image)
        plans=planner.plan(exact_token_changes(x,source_input),stop_after=len(prefix))
        return execute(x,plans,cache,retain)
    def evaluate(image,previous,source_input,cache,name,kind,frame_id):
        x0=block0_input(model,image)
        plans=planner.plan(exact_token_changes(x0,source_input),stop_after=len(prefix))
        actual,stages,inputs,outputs=execute(x0,plans,cache,True)
        reference=model(image)[0];delta=difference(actual,reference)
        _,sparse,dense=paired_measure(lambda:pipeline(image,source_input,cache),
            lambda:model(image)[0],args.repeats,args.warmup)
        # Publication is separately synchronized and timed, never hidden in pure
        # block costs. This excludes M1, loading and downstream tasks explicitly.
        state=StateCache(4*1024**3)
        context=dict(frame_id=frame_id,camera_ids=[str(i) for i in range(len(image))],timestamp=1.)
        def publish():
            state.commit(image,outputs,context,'offline',block0_input=x0)
        _,publication,_=paired_measure(publish,publish,args.repeats,args.warmup)
        row=dict(case=name,kind=kind,frame_id=frame_id,difference=delta,stages=stages,
            actual_reused_windows=sum(s['reused_output_windows'] for s in stages),
            changes=change_metrics(image,previous,x0,source_input,model.patch_embed.proj,plans),
            incremental=sparse,native=dense,cache_publication=publication,cache_bytes=state.bytes,
            scope='Embedding + exact feature comparison + closure + executor + full suffix; publication separately measured; excludes M1/loading/tasks')
        for i in prefix:
            x=inputs[i];ids=plans[i].updated.nonzero().flatten();block=model.blocks[i]
            (selected,full),a,b=paired_measure(lambda:executor.execute(block,x,cache[i],ids),
                lambda:block(x),args.repeats,args.warmup)
            block_delta=difference(selected[0],full)
            records.append(dict(key=cost_key(block,x,i,model=model),block=i,updated=len(ids),
                total_windows=plans[i].updated.numel(),reused_output_windows=plans[i].updated.numel()-len(ids),
                executed_attention_windows=len(ids),pointwise_window_work=plans[i].updated.numel() if len(ids) else 0,
                pointwise_tokens=len(x)*x.shape[1]*x.shape[2] if len(ids) else 0,
                numerical_verified=delta['exact'] and block_delta['exact'],difference=block_delta,
                quality_certified=False,quality_rejection_reason='downstream_quality_unverified',
                incremental_gpu_ms=a['gpu_span_ms'],full_gpu_ms=b['gpu_span_ms'],
                incremental_wall_ms=a['wall_ms'],full_wall_ms=b['wall_ms'],
                incremental_samples=a,full_samples=b,evidence_cases=[name],
                numerical_evidence=dict(path='results.json',case=name,kind=kind)))
        print(name,delta,row['changes'],sparse['wall_ms'],dense['wall_ms'],flush=True)
        return row
    natural=[];previous=None;source_input=None;cache=None;first=None;first_cache=None;first_input=None;old_info=None
    for index in range(min(args.frames,len(dataset))):
        sample=dataset[index];info=dataset.data_infos[index]
        image=torch.from_numpy(np.stack(sample['img']).transpose(0,3,1,2).copy()).cuda()
        reference,next_cache=native(image)
        if previous is not None and info['scene_token']==old_info['scene_token'] and info['timestamp']>old_info['timestamp']:
            natural.append(evaluate(image,previous,source_input,cache,'natural_'+str(index),
                'natural_continuous_visual_operator_validation',info['token']))
        else:
            natural.append(dict(frame_id=info['token'],case='cold_or_scene_boundary',compared=False,
                reason='No ordered history in this scene',actual_reused_windows=0))
        previous=image;source_input=block0_input(model,image);cache=next_cache;old_info=info
        if first is None:first=image.clone();first_cache=next_cache;first_input=source_input.clone()
    cases=[('identical',[]),('one_window',[(0,0,0)]),
        ('padded_edge_window',[(0,first.shape[-2]-1,first.shape[-1]-1)]),
        ('one_window_per_camera',[(c,1,1) for c in range(len(first))])]
    # Sweep real image regions inside each input footprint, across cameras and
    # padding boundaries. Dependency closure determines actual counts per block.
    footprints=planner.input_footprints(*first.shape[-2:])[0]
    total=len(first)*len(footprints)
    counts=args.counts if args.counts is not None else [2,4,8,total//2,total-1,total]
    for count in counts:
        if not 0<=count<=total:raise ValueError('Requested count exceeds actual windows')
        points=[]
        for unit in range(count):
            x0,y0,x1,y1=footprints[unit%len(footprints)]['input_region']
            points.append((unit//len(footprints),(y0+y1-1)//2,(x0+x1-1)//2))
        cases.append(('window_scale_'+str(count),points))
    controlled=[]
    for name,points in cases:
        image=first.clone()
        for camera,y,x in points:image[camera,:,y,x]+=.125
        controlled.append(evaluate(image,first,first_input,first_cache,name,
            'controlled_visual_operator_validation','controlled_'+name))
    result=dict(checkpoint=str(Path(args.checkpoint).resolve()),backbone_fingerprint=block_fingerprint(model),
        numerical_policy=NUMERICAL_POLICY,cacheable_blocks=prefix,natural=natural,
        controlled_microbenchmarks=controlled,warmup=args.warmup,repeats=args.repeats,
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),cpu_affinity=sorted(__import__('os').sched_getaffinity(0))),
        production_approvals_added=0,full_end_to_end_metrics='NOT_MEASURED: no paired full downstream service evaluation',
        downstream_quality='PENDING independent Detection, Map and full VLM assessment',
        pointwise_execution='Native full shapes for any attention updates; zero-update blocks skip all work')
    write(out/'results.json',result)
    # All evidence remains available; select the conservative median-cost case
    # per identity/count, and require exactness in every covered case.
    groups={}
    for record in records:groups.setdefault((record['key'],record['updated']),[]).append(record)
    unique=[]
    for group in groups.values():
        row=dict(max(group,key=lambda r:max(r['incremental_gpu_ms']/max(r['full_gpu_ms'],1e-9),
            r['incremental_wall_ms']/max(r['full_wall_ms'],1e-9))))
        row['numerical_verified']=all(r['numerical_verified'] for r in group)
        row['evidence_cases']=[case for r in group for case in r['evidence_cases']]
        row['measurements']=group
        row['numerical_evidence']['sha256']=sha256(out/'results.json')
        unique.append(row)
    write(out/'window_costs_provisional.json',dict(schema_version=SCHEMA_VERSION,records=unique,
        formal_quality_policy=dict(approved_counts=[]),production_ready=False,
        validation_sha256=sha256(out/'results.json'),backbone_fingerprint=block_fingerprint(model)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint',default='ckpts/iter_10548.pth')
    parser.add_argument('--frames',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--counts',type=int,nargs='+')
    args=parser.parse_args()
    if args.frames<2 or args.repeats<3 or args.warmup<1:parser.error('Need frames>=2, repeats>=3, warmup>=1')
    run(args)
