"""Real-weight window execution checks; controlled inputs never count as natural reuse.

Produces provisional measurements only. Does not approve or overwrite cost tables.
"""
import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .common import backbone, write
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.cost_model import cost_key, SCHEMA_VERSION
from worldstream.m2.window_executor import WindowExecutor
from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos


def measure(fn, repeats):
    fn()
    values=[]
    result=None
    for _ in range(repeats):
        torch.cuda.synchronize()
        a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
        tick=perf_counter();a.record()
        result=fn()
        b.record();torch.cuda.synchronize()
        values.append(((perf_counter()-tick)*1000,a.elapsed_time(b)))
    return result,dict(wall_ms=float(np.mean([x[0] for x in values])),
        gpu_span_ms=float(np.mean([x[1] for x in values])), repeats=repeats,
        wall_samples_ms=[x[0] for x in values],gpu_span_samples_ms=[x[1] for x in values])


def difference(actual, expected):
    error=(actual.float()-expected.float()).abs()
    return dict(exact=torch.equal(actual,expected),max_abs=float(error.max()),
        mean_abs=float(error.mean()),p95_abs=float(torch.quantile(error.flatten(),.95)),
        atol=1e-5,rtol=1e-5,pass_tolerance=torch.allclose(actual,expected,atol=1e-5,rtol=1e-5))


@torch.no_grad()
def run(args):
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin  # noqa
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    cfg.data.test.test_mode=True
    # Same image operations as the production pipeline; no language request is
    # constructed in this explicitly visual-only, bounded validation tool.
    cfg.data.test.pipeline=[node for node in cfg.data.test.pipeline
        if node['type'] in ('LoadMultiViewImageFromFiles','ResizeCropFlipRotImage',
            'ResizeMultiview3D','NormalizeMultiviewImage','PadMultiViewImage')]
    dataset=build_dataset(cfg.data.test)
    dataset.data_infos.sort(key=lambda i:(i['scene_token'],i['timestamp']))
    model=backbone(args.checkpoint)
    runtime=IncrementalVisionRuntime(model,dict(mode='exact',cache_capture='always',reuse_boundary='window'))
    natural=[];base=None;base_context=None
    for i in range(args.frames):
        info=dataset.data_infos[i];sample=dataset[i]
        images=torch.from_numpy(np.stack(sample['img']).transpose(0,3,1,2).copy()).cuda()
        context=dict(scene_id=info['scene_token'],frame_id=info['token'],timestamp=info['timestamp']/1e6,
            camera_ids=list(info['cams']),coordinate_space='preprocessed_image',execution_signature='real_checkpoint_window_probe')
        actual,d=runtime.forward(images,execution_context=context)
        reference=model(images)[0];torch.cuda.synchronize()
        natural.append(dict(frame_id=info['token'],scene_id=info['scene_token'],
            kind='natural_continuous_visual_stream',diagnostics=d,difference=difference(actual[0],reference)))
        if base is None:base,base_context=images.clone(),context
        print('NATURAL',i,d['actual_skipped_windows'],natural[-1]['difference'],flush=True)
    controlled=[];records=[]
    cases=[('identical',[]),('one_window',[(0,0,0)]),
           ('padded_edge_window',[(0,base.shape[-2]-1,base.shape[-1]-1)]),
           ('one_window_per_camera',[(c,1,1) for c in range(len(base))])]
    for ordinal,(name,changes) in enumerate(cases):
        rt=IncrementalVisionRuntime(model,dict(mode='exact',cache_capture='always',reuse_boundary='window'))
        rt.forward(base,execution_context=base_context)
        saved=copy.copy(vars(rt.cache))
        image=base.clone()
        for camera,y,x in changes:image[camera,:,y,x]+=.125
        context=dict(base_context,frame_id='controlled_'+name,timestamp=base_context['timestamp']+.1)
        stage_inputs={}
        x0=model.patch_embed(image)
        if model.pos_embed is not None:x0=x0+get_abs_pos(model.pos_embed,model.pretrain_use_cls_token,x0.shape[1:3])
        stage_inputs[0]=x0
        def capture(index,value):
            if index+1 in rt.prefix:stage_inputs[index+1]=value.detach()
        rt.stage_observer=capture
        def incremental():
            rt.cache.__dict__.update(saved)
            return rt.forward(image,execution_context=context)
        (actual,d),timing=measure(incremental,args.repeats)
        reference,native_timing=measure(lambda:model(image)[0],args.repeats)
        delta=difference(actual[0],reference)
        controlled.append(dict(kind='controlled_microbenchmark',case=name,modified_pixels=changes,
            diagnostics=d,difference=delta,incremental=timing,native=native_timing,
            net_wall_saving_ms=native_timing['wall_ms']-timing['wall_ms']))
        for stage in d['stages'][:len(rt.prefix)]:
            index=stage['block'];block=model.blocks[index];x=stage_inputs[index]
            from worldstream.m2.dependency_planner import exact_pixel_changes,patch_updates
            plans=rt.planner.plan(patch_updates(exact_pixel_changes(image,base),model.patch_embed.proj),stop_after=len(rt.prefix))
            ids=plans[index].updated.nonzero(as_tuple=False).flatten()
            executor=WindowExecutor()
            _,sparse=measure(lambda:executor.execute(block,x,saved['outputs'][index],ids),args.repeats)
            _,dense=measure(lambda:block(x),args.repeats)
            records.append(dict(key=cost_key(block,x,index),block=index,updated=len(ids),
                incremental_gpu_ms=sparse['gpu_span_ms'],full_gpu_ms=dense['gpu_span_ms'],
                incremental_wall_ms=sparse['wall_ms'],full_wall_ms=dense['wall_ms'],
                packing_backend='spatial',quality_certified=False,
                quality_rejection_reason='Controlled backbone check is not downstream detection/map/language qualification'))
        print('CONTROLLED',name,d['actual_skipped_windows'],delta,'net_ms',controlled[-1]['net_wall_saving_ms'],flush=True)
    write(out/'results.json',dict(checkpoint=args.checkpoint,mode='exact',backbone_only=True,
        frames=args.frames,repeats=args.repeats,cacheable_blocks=runtime.prefix,
        global_barrier_block=len(runtime.prefix),natural=natural,controlled_microbenchmarks=controlled,
        full_end_to_end_metrics='NOT_MEASURED',production_auto_approvals_added=0,
        cuda_event_scope='stream spans, not summed active kernel durations'))
    # Different actual counts/keys can appear more than once across cases.
    unique={(r['key'],r['updated']):r for r in records}
    write(out/'window_costs_provisional.json',dict(schema_version=SCHEMA_VERSION,records=list(unique.values()),
        formal_quality_policy=dict(approved_counts=[]),production_ready=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint',default='ckpts/iter_10548.pth')
    parser.add_argument('--frames',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.frames<1 or args.repeats<1:parser.error('frames/repeats must be positive')
    run(args)
