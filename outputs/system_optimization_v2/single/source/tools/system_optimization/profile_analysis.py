"""Attribute each real GPU kernel once via profiler external IDs and CPU scopes."""
from bisect import bisect_right
from collections import defaultdict,Counter
import json
import heapq
import re
from pathlib import Path
from .audit import write


def union_ms(intervals):
    end=-float('inf');total=0.
    for a,b in sorted(intervals):
        if b>end:total+=b-max(a,end);end=b
    return total/1000


def analyze(path,output=None,worker=None):
    events=json.loads(Path(path).read_text())['traceEvents']
    cpu={e['args']['External id']:e for e in events if e.get('cat')=='cpu_op' and 'External id' in e.get('args',{})}
    runtime={e['args']['correlation']:e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and e.get('ph')=='X' and 'correlation' in e.get('args',{})}
    phases=sorted([e for e in events if e.get('name') in ('stream/llm_prefill','stream/llm_decode')],key=lambda e:e['ts'])
    markers=sorted([e for e in events if e.get('name','').startswith('decodeop/') and e.get('ph')=='X'],key=lambda e:e['ts'])
    starts=[e['ts'] for e in phases]
    by_phase=defaultdict(lambda:dict(kernel_count=0,kernel_sum_ms=0.,intervals=[]))
    operators=defaultdict(lambda:dict(kernel_count=0,kernel_ms=0.))
    kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
    # PyTorch 2.0/CUPTI can stop supplying ATen external IDs after the first
    # profiler session. Correlation still links every kernel to its actual
    # CUDA launch API. Its host timestamp is inside the owning CPU scopes.
    linked={};linkage=Counter()
    for i,kernel in enumerate(kernels):
        op=runtime.get(kernel.get('args',{}).get('correlation'))
        if op is not None:linkage['cuda_correlation']+=1
        else:
            op=cpu.get(kernel.get('args',{}).get('External id'))
            linkage['cpu_external_id' if op is not None else 'unresolved']+=1
        if op is not None:linked[i]=op
    components={};active=[];cursor=0
    for identifier,op in sorted(linked.items(),key=lambda item:item[1]['ts']):
        t=op['ts']
        while cursor<len(markers) and markers[cursor]['ts']<=t:
            e=markers[cursor];heapq.heappush(active,(-e['ts'],cursor,e['ts']+e['dur'],e['name']));cursor+=1
        while active and active[0][2]<t:heapq.heappop(active)
        if active:components[identifier]=active[0][3].replace('decodeop/','')
    names=Counter()
    for i,kernel in enumerate(kernels):
        op=linked.get(i);phase='outside_llm';component='unattributed'
        if op is not None:
            j=bisect_right(starts,op['ts'])-1
            if j>=0 and op['ts']<=phases[j]['ts']+phases[j]['dur']:phase=phases[j]['name'].replace('stream/llm_','')
            component=components.get(i,'unattributed')
        bucket=by_phase[phase];bucket['kernel_count']+=1;bucket['kernel_sum_ms']+=kernel['dur']/1000
        bucket['intervals'].append((kernel['ts'],kernel['ts']+kernel['dur']))
        if phase=='decode':
            operators[component]['kernel_count']+=1;operators[component]['kernel_ms']+=kernel['dur']/1000
            names[kernel['name']]+=kernel['dur']/1000
    for bucket in by_phase.values():bucket['kernel_active_union_ms']=union_ms(bucket.pop('intervals'))
    decode_calls=sum(e['name']=='stream/llm_decode' for e in phases)
    layers=defaultdict(lambda:dict(kernel_ms=0.,kernel_count=0))
    for name,values in operators.items():
        match=re.search(r'model.layers\.(\d+)',name)
        if match:
            row=layers[match.group(1)]
            row['kernel_ms']+=values['kernel_ms'];row['kernel_count']+=values['kernel_count']
    for row in layers.values():row['per_decode_token_kernel_ms']=row['kernel_ms']/decode_calls if decode_calls else None
    result=dict(source=str(path),evidence='actual profiler GPU kernels, exclusive component attribution via External id',
        phases=dict(by_phase),kernel_host_linkage=dict(linkage),decode_forward_calls=decode_calls,
        per_decode_token_kernel_ms=by_phase['decode']['kernel_sum_ms']/decode_calls if decode_calls and 'decode' in by_phase else None,
        decode_operators=dict(operators),decode_layers=dict(layers),decode_top_kernel_duration_ms=dict(names.most_common(20)),
        cpu_cuda_synchronize_ms=sum(e['dur'] for e in events if e.get('cat')=='cuda_runtime' and 'synchroniz' in e.get('name','').lower())/1000,
        cpu_cuda_allocation_api_ms=sum(e['dur'] for e in events if e.get('cat')=='cuda_runtime' and ('malloc' in e.get('name','').lower() or 'free' in e.get('name','').lower()))/1000,
        gpu_copy_ms={kind:sum(e['dur'] for e in events if e.get('cat')=='gpu_memcpy' and kind.lower() in e.get('name','').lower())/1000 for kind in ('HtoD','DtoH','DtoD')},
        launch_gap_attribution_ms=None,
        note='Kernel sums and active unions are distinguished; profiler runs are excluded from E2E tables')
    if worker:
        sync=next((e for e in events if e.get('name')=='m1_overlap_clock_sync'),None)
        if sync:
            offset=sync['ts']-worker['clock_sync_s']*1e6
            # Conservative intersection trims clock-marker uncertainty.
            a=worker['worker_start_s']*1e6+offset+sync['dur'];b=worker['worker_end_s']*1e6+offset-sync['dur']
            overlap=[(max(a,e['ts']),min(b,e['ts']+e['dur'])) for e in kernels if min(b,e['ts']+e['dur'])>max(a,e['ts'])]
            result['m1_worker_gpu_kernel_overlap_lower_ms']=union_ms(overlap)
            result['clock_alignment_uncertainty_us']=sync['dur']
            result['m1_worker']=worker
    if output is not None:write(output,result)
    return result
