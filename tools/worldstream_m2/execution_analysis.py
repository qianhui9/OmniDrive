"""Rebuild optimization evidence; profiler kernels and event spans stay distinct."""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import numpy as np
from tools.worldstream_m123.benchmark.metrics import distribution
from tools.worldstream_m123.benchmark.quality_calibration import paired_quality
from .common import write


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def visual_summary(path):
    values=read(Path(path)/'results.json')['natural'];result={}
    for name,get in [('native',lambda r:r['native']),('before',lambda r:r['reference_pair']['before']),('optimized',lambda r:r['reference_pair']['optimized'])]:
        result[name]={key:distribution(get(r)[key] for r in values) for key in ('wall_ms','gpu_span_ms')}
    result.update(frames=len(values),reused_windows=sum(r['actual_skipped_windows'] for r in values),
        total_windows=sum(s['total_windows'] for r in values for s in r['stages'] if 'total_windows' in s),
        continuous_total_windows=sum(s['total_windows'] for r in values[1:] for s in r['stages'] if 'total_windows' in s),
        all_equal=all(all(r['reference_pair'][key] for key in ('window_ids_equal','packed_outputs_equal','ages_equal','cache_versions_equal','provenance_equal')) and r['reference_pair']['features']['exact'] for r in values),
        peak_allocated_bytes=max(r['peak_gpu_allocated_bytes'] for r in values),
        peak_reserved_bytes=max(r['peak_gpu_reserved_bytes'] for r in values))
    result['optimized_vs_before_speedup']=result['before']['wall_ms']['mean']/result['optimized']['wall_ms']['mean']
    result['optimized_vs_native_speedup']=result['native']['wall_ms']['mean']/result['optimized']['wall_ms']['mean']
    return result


def trace_summary(directory):
    events=read(Path(directory)/'trace.json')['traceEvents']
    cpu=[e for e in events if e.get('cat')=='cpu_op']
    launches={e['args']['correlation']:e for e in events if e.get('cat')=='cuda_runtime' and 'correlation' in e.get('args',{})}
    annotations=[e for e in events if e.get('cat')=='user_annotation']
    special=[e for e in cpu if e['name'] in ('aten::clone','aten::isfinite')]
    inside=lambda a,b:b['ts']<=a['ts']<=b['ts']+b['dur']
    kernels=[e for e in events if e.get('cat')=='kernel']
    categories=defaultdict(float);counts=Counter()
    for e in kernels:
        launch=launches.get(e.get('args',{}).get('correlation'))
        ancestors=[a for a in cpu if launch and inside(launch,a)]
        op=min(ancestors,key=lambda a:(a['dur'],-a['ts'])) if ancestors else None
        names=[a['name'] for a in annotations if launch and inside(launch,a)]
        parents=[a['name'] for a in special if launch and inside(launch,a)]
        if 'cache/commit' in names:
            group='cache_clone' if 'aten::clone' in parents else 'cache_finite' if 'aten::isfinite' in parents else 'cache_other'
        elif any(n.startswith('suffix/') for n in names):group='suffix_22_blocks'
        elif any(n.endswith('/qkv') for n in names):group='prefix_qkv'
        elif any(n.endswith('/attention') for n in names):group='prefix_attention'
        elif any(n.endswith('/mlp') for n in names):group='prefix_mlp'
        elif 'window/_grid' in names:group='window_grid_gather'
        elif any(n.startswith('prefix/') and n.endswith('/execute') for n in names):
            opname=op['name'] if op else ''
            group='window_gather' if opname in ('aten::index','aten::index_select') else 'window_scatter_clone' if opname in ('aten::index_copy_','aten::copy_','aten::zero_','aten::fill_') else 'prefix_norm_residual_other'
        elif 'm2/select_regions' in names:group='region_scoring'
        elif 'm2/block0_input' in names:group='embedding'
        elif 'm2/dependency_plan' in names:group='dependency_masks'
        else:group='other'
        categories[group]+=e['dur']/1000;counts[group]+=1
    memcpy=[e for e in events if e.get('cat')=='gpu_memcpy']
    runtime=[e for e in events if e.get('cat')=='cuda_runtime']
    ops=read(Path(directory)/'operators.json')['rows']
    selected={name:dict(calls=v['calls'],cpu_total_ms=v['cpu_total_us']/1000,self_cuda_ms=v['self_cuda_us']/1000)
        for v in ops for name in [v['name']] if name in ('aten::nonzero','aten::item','aten::clone','aten::copy_','aten::index_select','aten::index_copy_','cudaStreamSynchronize')}
    cache_clones=[o for o in cpu if o['name']=='aten::clone' and any(a['name']=='cache/commit' and inside(o,a) for a in annotations)]
    clone_bytes=0
    for op in cache_clones:
        shape=op['args']['Input Dims'][0];dtype=op['args']['Input type'][0]
        clone_bytes+=int(np.prod(shape))*(8 if dtype in ('long int','long','int64') else 4)
    return dict(kernel_active_sum_ms=sum(e['dur'] for e in kernels)/1000,kernel_count=len(kernels),
        streams=sorted({e['args'].get('stream') for e in kernels}),kernel_ms_by_stage=dict(categories),kernel_count_by_stage=dict(counts),
        memcpy={kind:dict(calls=sum(kind in e['name'] for e in memcpy),bytes=sum(e['args'].get('bytes',0) for e in memcpy if kind in e['name']),
            gpu_ms=sum(e['dur'] for e in memcpy if kind in e['name'])/1000) for kind in ('DtoH','HtoD','DtoD')},
        synchronize_calls=sum(e['name']=='cudaStreamSynchronize' for e in runtime),
        synchronize_host_wait_ms=sum(e['dur'] for e in runtime if e['name']=='cudaStreamSynchronize')/1000,
        cache_clone_calls=len(cache_clones),cache_clone_bytes=clone_bytes,operators=selected,
        scope='Each kernel counted once. CUPTI correlation -> CUDA launch CPU timestamp -> CPU annotation nesting. Torch 1.13 External id counters cannot reliably join later profiling sessions. CPU waits and nested operator totals are not active kernel time')


def service_stats(frames,questions,label):
    return dict(frames=len(frames),request_ms=distribution(f['request_ms'] for f in frames),
        visual_cuda_span_ms=distribution(f['stages'][label]['gpu_span_ms'] for f in frames),
        prefill_cuda_ms=distribution(f['stages']['prefill']['gpu_span_ms'] for f in frames),
        decode_forward_cuda_ms=distribution(f['stages'].get('decode',{}).get('gpu_span_ms',0.) for f in frames),
        m1_cpu_ms=distribution(f.get('m1_commit_timings_ms',{}).get('m1_total_including_adapters_ms',0.) for f in frames),
        m2_host_ms=distribution((f.get('m2') or {}).get('M2_total_ms',0.) for f in frames),
        m3_host_ms=distribution((f.get('m3') or {}).get('runtime_host_elapsed_ms',0.) for f in frames),
        peak_allocated_bytes=max(f['peak_gpu_allocated_bytes'] for f in frames),
        peak_reserved_bytes=max(f['peak_gpu_reserved_bytes'] for f in frames),
        throughput_fps=1000*len(frames)/sum(f['request_ms'] for f in frames),
        generated_tokens=sum(q['generated_tokens'] for q in questions),
        reused_tokens=sum((q.get('reuse_diagnostics') or {}).get('reused_tokens',0) for q in questions))


def run(root,prior):
    root=Path(root);prior=Path(prior)
    output=dict(formal_boundary=visual_summary(root/'paired40'),snapshot_boundary=visual_summary(root/'snapshot40'),
        profiler_before=trace_summary(root/'isolated_profile/reference_profile_output'),
        profiler_optimized=trace_summary(root/'isolated_profile/profile_output'),
        publication_before=read(root/'isolated_profile/reference_synchronized_actual_publication.json'),
        publication_optimized=read(root/'isolated_profile/synchronized_actual_publication.json'))
    all_frames=defaultdict(list);all_questions=defaultdict(list);rounds=[];pairs=[]
    oldf=rows(prior/'formal40/per_frame_metrics.jsonl');oldq=rows(prior/'formal40/per_question_metrics.jsonl')
    for repeat in range(3):
        record={};data={}
        for name,label in [('native','eva_native'),('formal','m2_total')]:
            path=root/'service'/('r'+str(repeat)+'_'+name)
            ff=rows(path/'per_frame_metrics.jsonl');qq=rows(path/'per_question_metrics.jsonl');data[name]=(ff,qq)
            record[name]=service_stats(ff,qq,label);all_frames[name]+=ff;all_questions[name]+=qq
        nf,nq=data['native'];ff,fq=data['formal']
        assert [(q['frame_id'],q['question']) for q in nq]==[(q['frame_id'],q['question']) for q in fq]
        quality=[dict(frame_id=q['frame_id'],question=q['question'],**paired_quality(q['answer'],n['answer'],q['reference'])) for n,q in zip(nq,fq)]
        old_equal=sum(a['answer']==b['answer'] for a,b in zip(oldq,fq))
        windows_equal=all([s['executed_window_ids'] for s in a['m2']['per_block']]==[s['executed_window_ids'] for s in b['m2']['per_block']] for a,b in zip(oldf,ff))
        record.update(window_reused=sum(f['m2']['reused_windows'] for f in ff),total_windows=sum(f['m2']['total_windows'] for f in ff),
            reuse_frames=sum(f['m2']['reused_windows']>0 for f in ff),old_window_ids_equal=windows_equal,
            old_task_digests_equal=all(a['output_digest']==b['output_digest'] for a,b in zip(oldf,ff)),
            old_answers_equal=old_equal,
            native_f1=float(np.mean([q['baseline_f1'] for q in quality])),formal_f1=float(np.mean([q['f1'] for q in quality])),
            critical_regression_questions=sum(q['critical_regressions']>0 for q in quality),
            critical_regression_checks=sum(q['critical_regressions'] for q in quality))
        rounds.append(record);pairs.append(dict(repeat=repeat,rows=quality))
    output['service_rounds']=rounds
    output['service_aggregate']={name:service_stats(all_frames[name],all_questions[name],label) for name,label in [('native','eva_native'),('formal','m2_total')]}
    output['official_subset']={name:read(root/'service'/('r0_'+name+'_official_subset')/'summary.json') for name in ('native','formal')}
    critical_ids={'88a6b5e35eb44deba6b1065d74b0a3c4','fd612e8660e74196af7da7e6487e5d53'}
    output['retained_critical_cases']=[dict(repeat=p['repeat'],**q) for p in pairs for q in p['rows'] if q['frame_id'] in critical_ids]
    write(root/'vlm_pairs.json',pairs);write(root/'comparison_summary.json',output)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--prior',required=True)
    args=parser.parse_args();result=run(args.root,args.prior)
    print(json.dumps(dict(formal_boundary=result['formal_boundary'],snapshot_boundary=result['snapshot_boundary'],service_rounds=result['service_rounds']),indent=2))
