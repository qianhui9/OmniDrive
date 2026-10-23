"""Offline-only attribution. Profiling calls are excluded from latency samples."""
from contextlib import contextmanager, ExitStack
import cProfile
import io
import pstats
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
from .common import write


@contextmanager
def instrumentation(runtime):
    import worldstream.m2.runtime as module
    from worldstream.m2.state_cache import StateCache
    handles=[]
    def wrapped(fn,name):
        def call(*args,**kwargs):
            with torch.profiler.record_function(name):return fn(*args,**kwargs)
        return call
    def hooks(obj,name):
        active=[]
        def before(*args):
            ctx=torch.profiler.record_function(name);ctx.__enter__();active.append(ctx)
        def after(*args):active.pop().__exit__(None,None,None)
        handles.extend([obj.register_forward_pre_hook(before),obj.register_forward_hook(after)])
    weights={}
    for i,b in enumerate(runtime.model.blocks):
        if i in runtime.prefix:
            for name in ('q_proj','k_proj','v_proj'):weights[id(getattr(b.attn,name).weight)]=f'prefix/{i}/qkv'
            hooks(b.attn.proj,f'prefix/{i}/output_projection')
            hooks(b.mlp,f'prefix/{i}/mlp')
            if b.attn.flash_attn:hooks(b.attn.inner_attn,f'prefix/{i}/attention')
        else:hooks(b,f'suffix/{i}')
    linear=F.linear
    def measured_linear(input,weight,*args,**kwargs):
        name=weights.get(id(weight))
        if name:
            with torch.profiler.record_function(name):return linear(input,weight,*args,**kwargs)
        return linear(input,weight,*args,**kwargs)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(F,'linear',measured_linear))
            for name in ('block0_input','select_regions','computation_admission'):
                stack.enter_context(patch.object(module,name,wrapped(getattr(module,name),'m2/'+name)))
            for obj,name,label in ((StateCache,'commit','cache/commit'),
                (runtime.planner,'plan','m2/dependency_plan'),(runtime.planner,'cacheable_prefix','m2/topology_check'),
                (runtime,'signature','m2/signature'),(runtime,'_execute','m2/execution_and_lineage')):
                stack.enter_context(patch.object(obj,name,wrapped(getattr(obj,name),label)))
            yield
    finally:
        for handle in handles:handle.remove()


def profile_call(runtime,call,directory):
    """CPU time, kernel-attributed CUDA time and explicit sync events, not sums of nested spans."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    with instrumentation(runtime):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,profile_memory=True,with_stack=False) as profile:
            call();torch.cuda.synchronize()
        profile.export_chrome_trace(str(directory/'trace.json'))
        averages=profile.key_averages()
        rows=[dict(name=e.key,calls=e.count,cpu_total_us=e.cpu_time_total,self_cpu_us=e.self_cpu_time_total,
            cuda_total_us=e.cuda_time_total,self_cuda_us=e.self_cuda_time_total,
            self_cpu_memory_bytes=e.self_cpu_memory_usage,self_cuda_memory_bytes=e.self_cuda_memory_usage) for e in averages]
        write(directory/'operators.json',dict(rows=rows,scope='Instrumented attribution only; CPU sync waits are not CUDA compute. Nested totals overlap.'))
        (directory/'operators.txt').write_text(averages.table(sort_by='self_cuda_time_total',row_limit=80))
    cpu=cProfile.Profile();cpu.enable();call();torch.cuda.synchronize();cpu.disable()
    cpu.dump_stats(str(directory/'cpu.prof'))
    stream=io.StringIO();pstats.Stats(cpu,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(60)
    (directory/'cpu.txt').write_text(stream.getvalue())
