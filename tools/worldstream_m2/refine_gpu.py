"""Production-weight per-block profiling, independent correctness, and measured v2 costs."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from .common import backbone,write
from .refinement_common import ROOT,reference,error_location
from .benchmark import kernel_profile
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.window_executor import WindowExecutor
from worldstream.m2.packing import partition,unpartition
from worldstream.m2.cost_model import cost_key,SCHEMA_VERSION
from worldstream.m2.profiler import measure,errors


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--blocks-only',action='store_true',help='Refresh actual-layout block measurements; retain independent full-EVA evidence')
    args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(314);torch.backends.cuda.matmul.allow_tf32=False
    old=reference();from m2_refinement_reference.window_executor import WindowExecutor as OriginalExecutor
    model=backbone();old_exec=OriginalExecutor();optimized=WindowExecutor()
    images=torch.randn(6,3,640,640,device='cuda')
    from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
    x=model.patch_embed(images);x=x+get_abs_pos(model.pos_embed,model.pretrain_use_cls_token,x.shape[1:3])
    rows=[];records=[];profiles={}
    counts=[0,1,2,4,6,8,16,27,36,40,44,48,54]
    for index in (0,1):
        block=model.blocks[index];cache=partition(block(x),16)[0]
        for count in counts:
            ids=torch.randperm(54,device='cuda')[:count]
            wins,padded=partition(x,16);wins[ids]+=.03125
            reconstructed=unpartition(wins,16,padded,(40,40))
            current=torch.empty_strided(x.shape,x.stride(),device=x.device,dtype=x.dtype)
            current.copy_(reconstructed);oracle=block(current)
            funcs=dict(full=lambda:block(current),original=lambda:old_exec.execute(block,current,cache,ids),
                optimized=lambda:optimized.execute(block,current,cache,ids),
                naive=lambda:old_exec.execute(block,current,cache,ids,naive=True))
            timings={}
            # Reverse method order across neighbouring counts and blocks.
            order=list(funcs) if (count+index)%2==0 else list(funcs)[::-1]
            for method in order:timings[method]=measure(funcs[method],warmup=5,repeats=args.repeats)
            actual=funcs['optimized']()[0];err=error_location(actual,oracle)
            row=dict(block=index,updated=count,total=54,input_stride=list(current.stride()),ids=ids.cpu().tolist(),timing=timings,error=err,
                block_correct=err['max_abs']<=.02 and err['relative_l2']<=.002)
            rows.append(row)
            backend='optimized' if (timings['optimized']['gpu_ms']<timings['original']['gpu_ms'] and
                                    timings['optimized']['wall_ms']<timings['original']['wall_ms']) else 'original'
            records.append(dict(key=cost_key(block,current,index),block=index,updated=count,
                packing_backend='spatial' if backend=='optimized' else 'original',
                incremental_gpu_ms=timings[backend]['gpu_ms'],full_gpu_ms=timings['full']['gpu_ms'],
                incremental_wall_ms=timings[backend]['wall_ms'],full_wall_ms=timings['full']['wall_ms'],
                quality_certified=False))
            if count in (1,27,48):
                profiles[f'{index}_{count}']={m:kernel_profile(funcs[m],ROOT/'traces'/f'block{index}_{count}_{m}.json')
                    for m in ('full','original','optimized')}
            print('block',index,'count',count,'old/new',timings['original']['gpu_ms'],timings['optimized']['gpu_ms'],flush=True)
        x=block(x)
    write(ROOT/'block_benchmark.json',dict(rows=rows,repeats=args.repeats,warmup=5,checkpoint='ckpts/iter_10548.pth',seed=314))
    write(ROOT/'packing_profile.json',profiles)
    if args.blocks_only:
        prior=json.loads((ROOT/'break_even_results.json').read_text())
        certified=prior['whole_backbone_counts']
        for record in records:
            record['backbone_quality_certified']=bool(certified.get(str(record['updated']),False)) and next(
                r['block_correct'] for r in rows if r['block']==record['block'] and r['updated']==record['updated'])
            record['quality_rejection_reason']='downstream_quality_unverified'
        measured={(r['key'],r['updated']) for r in records}
        # Preserve separately measured contiguous Block-1 reconstruction costs.
        prior['records']=[r for r in prior['records'] if (r['key'],r['updated']) not in measured]+records
        write(ROOT/'break_even_results.json',prior)
        from .validate_quality_policy import main as publish_quality_policy
        publish_quality_policy()
        return
    # Freeze prior precision limits; new independent inputs may FAIL them, never widen them.
    limits=json.load(open('outputs/module2/precision_calibration.json'))['tolerance']
    evaluations=[];backbone_rows=[];certified={k:k in (0,1,6,36,54) for k in counts}
    for seed in (811,1207):
        torch.manual_seed(seed);base=torch.randn(6,3,640,640,device='cuda')
        dense={};hooks=[b.register_forward_hook(lambda m,i,o,j=j:dense.update({j:o.detach().clone()}))
                        for j,b in enumerate(model.blocks)]
        baseline=model(base)[0]
        for h in hooks:h.remove()
        micro=torch.cat([model(cam[None])[0] for cam in base])
        independent=dict(seed=seed,dense_microbatch_error=error_location(micro,baseline),cases=[])
        for count in (0,1,6,36,54):
            current=base.clone()
            if count==1:current[0,:,:16,:16]+=.03
            elif count==6:current[:,:,:16,:16]+=.03
            elif count==36:current[:,:,:512,:]+=.03
            elif count==54:current+=.03
            layer_oracle={}
            hooks=[b.register_forward_hook(lambda m,i,o,j=j:layer_oracle.update({j:o.detach().clone()}))
                   for j,b in enumerate(model.blocks)]
            oracle=model(current)[0]
            for h in hooks:h.remove()
            runtime=IncrementalVisionRuntime(model,dict(mode='exact'))
            context=dict(scene_id=f'independent-{seed}',frame_id='0',camera_ids=list(range(6)))
            runtime.forward(base,execution_context=context)
            layer_errors={}
            runtime.stage_observer=lambda j,out:layer_errors.update({j:error_location(out,layer_oracle[j])})
            y,d=runtime.forward(current,execution_context=dict(context,frame_id='1'))
            runtime.stage_observer=None
            passed=all(e['max_abs']<=max(limits[str(j)]['max_abs'],1e-6) and
                       e['relative_l2']<=max(limits[str(j)]['relative_l2'],1e-7) for j,e in layer_errors.items())
            certified[count] &= passed
            independent['cases'].append(dict(updated=count,error=error_location(y[0],oracle),layers=layer_errors,
                within_frozen_envelope=passed,diagnostics=d))
            if seed==811:
                timings={}
                timings['full']=measure(lambda:model(current),warmup=3,repeats=7)
                for name,cls in [('original',old.IncrementalVisionRuntime),('optimized',IncrementalVisionRuntime)]:
                    run=cls(model,dict(mode='exact'));run.forward(base,execution_context=context)
                    saved=(run.cache.images,run.cache.outputs,run.cache.context,run.cache.signature,run.cache.version)
                    def execute():
                        run.cache.images,run.cache.outputs,run.cache.context,run.cache.signature,run.cache.version=saved
                        return run.forward(current,execution_context=dict(context,frame_id='1'))
                    timings[name]=measure(execute,warmup=3,repeats=7)
                backbone_rows.append(dict(updated=count,timing=timings))
            print('independent',seed,count,'within frozen',passed,flush=True)
        evaluations.append(independent)
    for record in records:
        record['backbone_quality_certified']=certified.get(record['updated'],False) and next(
            r['block_correct'] for r in rows if r['block']==record['block'] and r['updated']==record['updated'])
        record['quality_certified']=False  # Downstream admission is published separately.
        record['quality_rejection_reason']='downstream_quality_unverified'
    write(ROOT/'break_even_results.json',dict(schema_version=SCHEMA_VERSION,records=records,
        whole_backbone_counts=certified,quality_protocol='both new independent seeds must pass frozen previous per-layer limits; no interpolation'))
    write(ROOT/'correctness_report.json',dict(frozen_limits=limits,independent_inputs=evaluations))
    write(ROOT/'backbone_benchmark.json',backbone_rows)
    write(ROOT/'memory_profile.json',dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope='controlled profiling including full-layer independent oracles'))


if __name__=='__main__':main()
