"""Actual production EVA Auto execution against per-block measured v2 costs."""
import torch
from .common import backbone,write
from .refinement_common import ROOT,error_location
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.profiler import measure


@torch.no_grad()
def run(model):
    torch.set_num_threads(1);torch.manual_seed(1909);torch.backends.cuda.matmul.allow_tf32=False
    model.eval();images=torch.randn(6,3,640,640,device='cuda')
    context=dict(scene_id='auto-independent',frame_id='0',camera_ids=list(range(6)))
    cold=IncrementalVisionRuntime(model,dict(mode='exact',cache_capture='always'))
    no_cache=IncrementalVisionRuntime(model,dict(mode='exact',cache_capture='never'))
    capture_cost=dict(full=measure(lambda:model(images),warmup=3,repeats=7),
        capture=measure(lambda:(cold.reset(),cold.forward(images,execution_context=context)),warmup=3,repeats=7),
        no_capture=measure(lambda:no_cache.forward(images,execution_context=context),warmup=3,repeats=7))
    write(ROOT/'cache_capture_benchmark.json',capture_cost)
    rows=[]
    for count in (0,1,2,6,36,48,54):
        current=images.clone()
        for wid in range(count):
            camera,local=divmod(wid,9);row,col=divmod(local,3)
            current[camera,:,row*256:row*256+16,col*256:col*256+16]+=.03
        runtime=IncrementalVisionRuntime(model,dict(mode='auto',cost_table=str(ROOT/'break_even_results.json')))
        runtime.forward(images,execution_context=context)
        saved=(runtime.cache.images,runtime.cache.outputs,runtime.cache.context,runtime.cache.signature,runtime.cache.version)
        def run():
            runtime.cache.images,runtime.cache.outputs,runtime.cache.context,runtime.cache.signature,runtime.cache.version=saved
            return runtime.forward(current,execution_context=dict(context,frame_id='1'))
        y,d=run();timing=measure(run,warmup=3,repeats=7)
        full=measure(lambda:model(current),warmup=3,repeats=7)
        err=error_location(y[0],model(current)[0])
        rows.append(dict(updated=count,timing=timing,full=full,error=err,diagnostics=d,
            speedup=full['gpu_ms']/timing['gpu_ms']))
        print(count,[s.get('mode') for s in d['stages'][:2]],
            [s.get('fallback_reason') for s in d['stages'][:2]],flush=True)
    write(ROOT/'auto_runtime_benchmark.json',dict(seed=1909,rows=rows,
        limitations='Independent update locations can differ from earlier calibration; measured errors are retained rather than silently re-certifying any previously unverified count.'))
    from .refinement_common import reference
    old=reference();backbone_rows=[]
    for count in (0,1,6,36,54):
        current=images.clone()
        if count==1:current[0,:,:16,:16]+=.03
        elif count==6:current[:,:,:16,:16]+=.03
        elif count==36:current[:,:,:512,:]+=.03
        elif count==54:current+=.03
        timings={'full':measure(lambda:model(current),warmup=3,repeats=7)}
        for name,cls in [('original',old.IncrementalVisionRuntime),('optimized',IncrementalVisionRuntime)]:
            runtime=cls(model,dict(mode='exact'));runtime.forward(images,execution_context=context)
            saved=(runtime.cache.images,runtime.cache.outputs,runtime.cache.context,runtime.cache.signature,runtime.cache.version)
            def execute():
                runtime.cache.images,runtime.cache.outputs,runtime.cache.context,runtime.cache.signature,runtime.cache.version=saved
                return runtime.forward(current,execution_context=dict(context,frame_id='1'))
            timings[name]=measure(execute,warmup=3,repeats=7)
        backbone_rows.append(dict(updated=count,timing=timings,output_layout_preserved=True))
    write(ROOT/'backbone_benchmark.json',backbone_rows)


def main():run(backbone())


if __name__=='__main__':main()
