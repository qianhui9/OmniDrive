"""Non-overlapping kernel totals and separate wall/stream timeline attribution."""
import json
from pathlib import Path
import numpy as np
from tools.worldstream_m1.io_utils import write_json
from .profile import ROOT


def stats(values):
    return dict(count=len(values),mean=float(np.mean(values)),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),min=float(min(values)),max=float(max(values))) if values else None


def main():
    p=ROOT/'profiling'
    operations=json.loads((p/'operators.json').read_text())
    trace=json.loads((p/'decode_only_trace.json').read_text())['traceEvents']
    # Actual leaf kernels are separate from attributed CPU operator CUDA time.
    kernels=[e for e in trace if e.get('cat')=='kernel' and 'dur' in e]
    span=max(e['ts']+e['dur'] for e in kernels)-min(e['ts'] for e in kernels)
    # Union on the measured stream timeline handles overlapping intervals.
    intervals=sorted((e['ts'],e['ts']+e['dur']) for e in kernels)
    union=0.;start,end=intervals[0]
    for lo,hi in intervals[1:]:
        if lo>end:union+=end-start;start,end=lo,hi
        else:end=max(end,hi)
    union+=end-start
    device_events=[e for e in trace if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and 'dur' in e]
    device_intervals=sorted((e['ts'],e['ts']+e['dur']) for e in device_events)
    device_union=0.;lo,hi=device_intervals[0]
    for a,b in device_intervals[1:]:
        if a>hi:device_union+=hi-lo;lo,hi=a,b
        else:hi=max(hi,b)
    device_union+=hi-lo
    device_span=max(e['ts']+e['dur'] for e in device_events)-min(e['ts'] for e in device_events)
    grouped={}
    for e in kernels:
        name=e['name'];d=grouped.setdefault(name,dict(calls=0,total_ms=0.));d['calls']+=1;d['total_ms']+=e['dur']/1000
    micro=[r for r in json.loads((p/'single_token.json').read_text()) if not r['warmup']]
    full=[r for r in json.loads((p/'native_generation.json').read_text()) if not r['warmup']]
    result=dict(decode_calls=operations['decode_calls'],leaf_kernel_calls=len(kernels),leaf_kernel_sum_ms=sum(e['dur'] for e in kernels)/1000,
        kernel_union_ms=union/1000,first_to_last_kernel_span_ms=span/1000,kernel_timeline_gap_ms=(span-union)/1000,
        device_memcpy_ms=sum(e['dur'] for e in device_events if e['cat']=='gpu_memcpy')/1000,
        device_memset_ms=sum(e['dur'] for e in device_events if e['cat']=='gpu_memset')/1000,
        all_device_activity_union_ms=device_union/1000,device_activity_span_ms=device_span/1000,
        unattributed_timeline_gap_ms=(device_span-device_union)/1000,
        native_kernel_launch_calls=sum(1 for e in trace if e.get('cat')=='cuda_runtime' and e.get('name')=='cudaLaunchKernel'),
        kernel_timeline_occupied_fraction=union/span,top_leaf_kernels=sorted([dict(name=k,**v) for k,v in grouped.items()],key=lambda x:-x['total_ms'])[:30],
        operators=operations['operators'],single_token={k:stats([r[k] for r in micro]) for k in ('wall_ms','gpu_interval_ms','cpu_submission_ms')},
        native_generation_wall_ms=stats([r['wall_ms'] for r in full]),native_decode_cuda_interval_ms=stats([sum(t['gpu_interval_ms'] for t in r['steps'] if t['stage']=='decode') for r in full]),
        note='CUDA intervals include host gaps. Kernel-only gaps also contain memcpy/memset; all-device union separates those. Unattributed gaps are not proof of CPU-launch causality. No P50 subtraction or overlapping time addition.',
        optimized_operator_trace_available=(ROOT/'micro/optimized_decode_trace.json').exists() if (ROOT/'micro/status.json').exists() else None,
        optimized_trace_failure=json.loads((ROOT/'micro/status.json').read_text()).get('trace_status') if (ROOT/'micro/status.json').exists() else None)
    write_json(ROOT/'decode_operator_breakdown.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('operators','top_leaf_kernels')},indent=2))
    print(json.dumps(result['top_leaf_kernels'][:8],indent=2))


if __name__=='__main__':main()
