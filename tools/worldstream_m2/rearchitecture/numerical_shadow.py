"""Isolate cache substitution from GEMM batch shape with a fixed-shape shadow oracle."""
import torch
import copy
import json
from time import perf_counter
from worldstream.m2.packing import partition,unpartition
from worldstream.m2.dependency_planner import exact_pixel_changes,patch_updates,DependencyPlanner
from worldstream.m2.window_executor import WindowExecutor
from .common import ROOT,write,backbone,error_location


@torch.no_grad()
def main():
    from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    model=backbone();rows=[]
    for seed in (2801,3509):
        torch.manual_seed(seed);base=torch.randn(6,3,640,640,device='cuda')
        old={};hs=[b.register_forward_hook(lambda m,i,o,j=j:old.update({j:o.detach().clone()})) for j,b in enumerate(model.blocks[:2])]
        model(base)
        for h in hs:h.remove()
        current=base.clone();current[0,:,:16,:16]+=.03
        oracle={};hs=[b.register_forward_hook(lambda m,i,o,j=j:oracle.update({j:o.detach().clone()})) for j,b in enumerate(model.blocks)]
        y=model(current)[0]
        for h in hs:h.remove()
        plans=DependencyPlanner(model).plan(patch_updates(exact_pixel_changes(current,base),model.patch_embed.proj),stop_after=2)
        x=model.patch_embed(current)+get_abs_pos(model.pos_embed,model.pretrain_use_cls_token,(40,40))
        layers={}
        for i,b in enumerate(model.blocks):
            if i<2:
                # Compute all windows first, then substitute only closure-proven unchanged
                # windows with their previous original-full values. No runtime speed claim.
                all_windows,pad=partition(x,b.window_size)
                ids=torch.arange(len(all_windows),device=x.device)
                dense,_=WindowExecutor().execute(b,x,all_windows,ids)
                packed,_=partition(dense,b.window_size)
                unchanged=(~plans[i].updated).nonzero(as_tuple=False).flatten()
                old_packed,_=partition(old[i],b.window_size)
                packed.index_copy_(0,unchanged,old_packed.index_select(0,unchanged))
                x=unpartition(packed,b.window_size,pad,(40,40))
            else:x=b(x)
            layers[str(i)]=error_location(x,oracle[i])
        rows.append(dict(seed=seed,layers=layers,final=error_location(x.permute(0,3,1,2),y),
            kind='fixed_full_batch_shadow_no_GPU_saving'))
    write(ROOT/'cache_substitution_shadow.json',dict(rows=rows))
    from .component_audit import run as component_audit
    component_audit(model,base)
    from worldstream.m2 import IncrementalVisionRuntime
    from tools.worldstream_m2.benchmark import kernel_profile
    runtime=IncrementalVisionRuntime(model,dict(mode='exact',reuse_boundary='camera'))
    ctx=dict(scene_id='kernel_work',frame_id='0',camera_ids=list(range(6)))
    runtime.forward(base,execution_context=ctx);cache=runtime.camera_executor.cache
    saved=copy.copy(vars(cache))
    def repeat():
        cache.__dict__.update(saved)
        return runtime.forward(base,execution_context=dict(ctx,frame_id='1'))
    profiles={}
    for name,fn in [('full',lambda:model(base)),('all_camera_reuse',repeat)]:
        trace=ROOT/'traces'/f'camera_{name}.json'
        values=kernel_profile(fn,trace)
        kernels=[e for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel']
        values['kernel_count']=len(kernels)
        values['kernel_active_ms']=sum(e.get('dur',0) for e in kernels)/1000
        profiles[name]=values
    write(ROOT/'camera_kernel_work.json',profiles)
    from worldstream.m2.profiler import measure
    partial=[]
    for count in (1,6,36):
        current=base.clone()
        if count==1:current[0,:,:16,:16]+=.03
        elif count==6:current[:,:,:16,:16]+=.03
        else:current[:,:,:512,:]+=.03
        window=IncrementalVisionRuntime(model,dict(mode='exact'))
        window.forward(base,execution_context=ctx);saved=copy.copy(vars(window.cache))
        def execute():
            window.cache.__dict__.update(saved)
            return window.forward(current,execution_context=dict(ctx,frame_id='1'))
        reuse=measure(execute,warmup=3,repeats=10)
        full=measure(lambda:model(current),warmup=3,repeats=10)
        partial.append(dict(updated_windows_per_block=count,full=full,reuse=reuse,
            saving_wall_ms=full['wall_ms']-reuse['wall_ms'],quality_status='experimental_only; downstream frozen targets not approved'))
    write(ROOT/'partial_window_costs.json',dict(rows=partial))
    from .common import stats
    metadata_wall=[]
    for _ in range(30):
        tick=perf_counter()
        for __ in range(1000):x.permute(0,3,1,2).view(1,6,1024,40,40)
        metadata_wall.append((perf_counter()-tick))  # seconds/1000 -> milliseconds/operation
    metadata_trace=ROOT/'traces/output_layout_metadata.json'
    kernel_profile(lambda:x.permute(0,3,1,2).view(1,6,1024,40,40),metadata_trace)
    metadata_kernels=[e for e in json.loads(metadata_trace.read_text())['traceEvents'] if e.get('cat')=='kernel']
    write(ROOT/'output_layout_profile.json',dict(cpu_wall_ms_per_operation=stats(metadata_wall),
        batches=30,operations_per_batch=1000,gpu_kernel_count=len(metadata_kernels),
        scope='original EVA permute plus detector reshape; metadata-only, no GPU copy. Camera assembly copies are included in camera runtime costs.'))

if __name__=='__main__':main()
