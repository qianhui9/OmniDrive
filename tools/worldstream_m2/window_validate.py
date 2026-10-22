"""Real-weight window execution checks; controlled inputs never count as natural reuse.

Produces provisional measurements only. Does not approve or overwrite cost tables.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .common import backbone, write
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
    from worldstream.m2.dependency_planner import DependencyPlanner,exact_pixel_changes,patch_updates
    from worldstream.m2.packing import partition
    from worldstream.m2.window_executor import NUMERICAL_POLICY
    import projects.mmdet3d_plugin
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py');cfg.data.test.test_mode=True
    cfg.data.test.pipeline=[node for node in cfg.data.test.pipeline if node['type'] in
        ('LoadMultiViewImageFromFiles','ResizeCropFlipRotImage','ResizeMultiview3D','NormalizeMultiviewImage','PadMultiViewImage')]
    dataset=build_dataset(cfg.data.test);dataset.data_infos.sort(key=lambda i:(i['scene_token'],i['timestamp']))
    model=backbone(args.checkpoint);planner=DependencyPlanner(model);prefix=planner.cacheable_prefix()
    executor=WindowExecutor()
    def native(image):
        cache={};handles=[]
        for i in prefix:
            handles.append(model.blocks[i].register_forward_hook(lambda block,inputs,output,i=i:
                cache.update({i:partition(output,block.window_size)[0].detach().clone()})))
        try:output=model(image)[0]
        finally:
            for h in handles:h.remove()
        return output,cache
    def incremental(image,source,cache,retain_inputs=False):
        plans=planner.plan(patch_updates(exact_pixel_changes(image,source),model.patch_embed.proj),stop_after=len(prefix))
        x=model.patch_embed(image)
        if model.pos_embed is not None:x=x+get_abs_pos(model.pos_embed,model.pretrain_use_cls_token,x.shape[1:3])
        stages=[];inputs={}
        for i,block in enumerate(model.blocks):
            if i in prefix:
                ids=plans[i].updated.nonzero().flatten();total=plans[i].updated.numel()
                if retain_inputs:inputs[i]=x
                x,_=executor.execute(block,x,cache[i],ids)
                stages.append(dict(block=i,total_windows=total,executed_attention_windows=len(ids),
                    reused_output_windows=total-len(ids),pointwise_window_work=total if len(ids) else 0))
            else:x=block(x)
        return x.permute(0,3,1,2),stages,inputs
    natural=[];previous=None;cache=None;first=None;first_cache=None
    for i in range(args.frames):
        sample=dataset[i];image=torch.from_numpy(np.stack(sample['img']).transpose(0,3,1,2).copy()).cuda()
        reference,next_cache=native(image)
        if previous is None:actual=reference;stages=[]
        else:actual,stages,_=incremental(image,previous,cache)
        row=dict(frame_id=dataset.data_infos[i]['token'],kind='natural_continuous_visual_operator_validation',
            difference=difference(actual,reference),actual_reused_windows=sum(s['reused_output_windows'] for s in stages),stages=stages)
        natural.append(row);print('NATURAL',row['frame_id'],row['actual_reused_windows'],row['difference'],flush=True)
        previous=image;cache=next_cache
        if first is None:first=image.clone();first_cache={k:v.clone() for k,v in cache.items()}
    cases=[('identical',[]),('one_window',[(0,0,0)]),('padded_edge_window',[(0,first.shape[-2]-1,first.shape[-1]-1)]),
           ('one_window_per_camera',[(c,1,1) for c in range(len(first))])]
    controlled=[];records=[]
    for name,changes in cases:
        image=first.clone()
        for c,y,x in changes:image[c,:,y,x]+=.125
        (actual,stages,inputs),sparse=measure(lambda:incremental(image,first,first_cache,True),args.repeats)
        reference,dense=measure(lambda:model(image)[0],args.repeats)
        delta=difference(actual,reference)
        controlled.append(dict(case=name,kind='controlled_visual_operator_validation',difference=delta,
            actual_reused_windows=sum(s['reused_output_windows'] for s in stages),stages=stages,
            incremental=sparse,native=dense,
            scope='Dependency planning, native-shaped pointwise, selected attention and full suffix; cache publication excluded'))
        plans=planner.plan(patch_updates(exact_pixel_changes(image,first),model.patch_embed.proj),stop_after=len(prefix))
        for i in prefix:
            x=inputs[i];ids=plans[i].updated.nonzero().flatten();block=model.blocks[i]
            _,a=measure(lambda:executor.execute(block,x,first_cache[i],ids),args.repeats)
            _,b=measure(lambda:block(x),args.repeats)
            records.append(dict(key=cost_key(block,x,i),block=i,updated=len(ids),numerical_verified=delta['exact'],
                quality_certified=False,incremental_gpu_ms=a['gpu_span_ms'],full_gpu_ms=b['gpu_span_ms'],
                incremental_wall_ms=a['wall_ms'],full_wall_ms=b['wall_ms'],
                quality_rejection_reason='Operator validation alone is not an independently qualified downstream task/cost certificate'))
        print('CONTROLLED',name,controlled[-1]['actual_reused_windows'],delta,flush=True)
    write(out/'results.json',dict(checkpoint=args.checkpoint,numerical_policy=NUMERICAL_POLICY,cacheable_blocks=prefix,
        natural=natural,controlled_microbenchmarks=controlled,production_approvals_added=0,
        full_end_to_end_metrics='NOT_MEASURED; see separately timed mainline service run',
        pointwise_execution='Full native shapes whenever a block has attention updates; fully identical blocks skip all pointwise work'))
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
