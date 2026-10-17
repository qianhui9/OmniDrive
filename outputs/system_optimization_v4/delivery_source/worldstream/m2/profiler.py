"""CUDA-event intervals include launch gaps; profiler kernel sum is reported separately."""
from time import perf_counter
import numpy as np
import torch


@torch.no_grad()
def measure(fn, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gpu, wall = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        tick = perf_counter(); start.record(); fn(); end.record()
        end.synchronize()
        gpu.append(start.elapsed_time(end)); wall.append((perf_counter()-tick)*1000)
    return dict(gpu_ms=float(np.median(gpu)), wall_ms=float(np.median(wall)),
                gpu_p95_ms=float(np.percentile(gpu, 95)), wall_p95_ms=float(np.percentile(wall, 95)),
                gpu_samples_ms=gpu, wall_samples_ms=wall)


def errors(actual, reference):
    a, b = actual.float().flatten(), reference.float().flatten()
    return dict(max_abs=float((a-b).abs().max()),
                relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-12)),
                cosine=float(torch.nn.functional.cosine_similarity(a[None], b[None])))
