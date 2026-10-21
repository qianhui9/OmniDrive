"""Production-weight CUDA block/backbone benchmark, correctness oracle and profiler traces."""
import argparse
from pathlib import Path
import torch
import numpy as np
from time import perf_counter
from .common import backbone, write
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.cost_model import cost_key
from worldstream.m2.packing import partition, unpartition
from worldstream.m2.window_executor import WindowExecutor, packed_forward
from worldstream.m2.profiler import measure, errors


def kernel_profile(fn, path):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                          torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        fn(); torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return dict(kernel_count=len(events), kernel_active_ms=sum(e.self_cuda_time_total for e in events)/1000,
                h2d_d2h_events=[e.name for e in events if 'Memcpy' in e.name],
                linear_input_shapes=[e.input_shapes for e in prof.events() if e.name=='aten::linear'],
                cpu_cuda_calls=[dict(name=e.key, calls=e.count, cpu_us=e.self_cpu_time_total)
                                for e in prof.key_averages() if 'cuda' in e.key.lower()])


@torch.no_grad()
def breakdown(block, x, cache, ids, executor, repeats=20):
    samples=[]
    for _ in range(repeats):
        ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
        tick=perf_counter(); ev[0].record()
        windows,padded=partition(x,block.window_size)
        valid,_=partition(torch.ones_like(x[...,:1]),block.window_size)
        packed=executor.buffer.gather(windows,ids); valid=valid.index_select(0,ids)
        ev[1].record()
        output=packed_forward(block,packed,valid)
        ev[2].record()
        target=cache.clone(); target.index_copy_(0,ids,output)
        target=unpartition(target,block.window_size,padded,x.shape[1:3])
        ev[3].record(); ev[3].synchronize()
        parts=[ev[i].elapsed_time(ev[i+1]) for i in range(3)]
        samples.append(dict(pack_ms=parts[0], sparse_ms=parts[1], scatter_ms=parts[2],
            synchronization_and_host_residual_ms=max(0.,(perf_counter()-tick)*1000-sum(parts))))
    return {k:float(np.median([r[k] for r in samples])) for k in samples[0]}
