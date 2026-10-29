"""Offline analysis and predeclared finite experiment scheduling."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import json
import gzip
import shutil

import numpy as np

from .benchmark.common import write,sha256
from .continuous_stream import lines
from .continuous_stream_report import events,paired_quality,mean
from .continuous_stream_analysis import performance,dist,cluster_performance,quality_summary

ROOT=Path('outputs/m123_critical_path_and_joint_tuning_run01')


def read(root,name):return json.loads((root/name).read_text())


def stage_rows(root,stage):
    rows=lines(root/stage/'requests.jsonl')
    reuse=root/(stage+'_receipt_reuse.json')
    if reuse.exists():
        for item in read(root,reuse.name)['mappings']:
            source=stage_rows(root,item['source_stage'])
            for r in source:
                if r['job_spec']['label']!=item['source_label']:continue
                if (r['workload'],r['frequency'])!=(item['workload'],item['frequency']):continue
                rows.append(dict(r,reused_receipt=True,origin_stage=item['source_stage'],
                    job_spec=dict(r['job_spec'],label=item['new_label'])))
    return rows


def job(root,scene,label,condition='C',frequency='three',workload='addressable',m1=None,m2=None,policy=None,**extra):
    return dict(scene=scene,label=label,condition=condition,frequency=frequency,workload=workload,
        mode='online' if condition=='C' else 'off',policy=str(root/'reference_policy.json') if policy is None and condition=='C' else policy,
        m1=m1 or {},m2=m2 or {},**extra)


def dev_scenes(root):return [s for s in read(root,'extended_frame_selection.json')['selected'] if s['role']=='extended_DEV']


def per_label_quality(root,rows):
    relabeled=[dict(r,condition=r['job_spec']['label']) for r in rows]
    es=events(root,relabeled)
    return es,{label:quality_summary([e for e in es if e['condition']==label]) for label in sorted({e['condition'] for e in es})}


def diagnostic_performance(pairs):
    result=cluster_performance(pairs)
    if result.get('independent_logs',0)<2:
        result.update(speedup_log_CI95=None,delta_log_CI95_ms=None,
            interval_scope='Population/log uncertainty not estimable from one independent log; paired means are diagnostic only')
    return result


def valid_feature_witness(row):
    if 'feature' not in row.get('visual_quality',{}):return False
    return row.get('feature_receipt_version')=='returned_m2_nchw_v2' or not (row.get('m2') or {}).get('global_suffix_skipped_blocks',0)


def feature_difference_receipt(row):
    valid=valid_feature_witness(row)
    return dict(valid=valid,metrics=row['visual_quality'].get('full_feature_difference') if valid else None,
        reason=None if valid else 'Legacy last-block observer did not fire on global residual reuse; use corrected post-final audit')


def summarize_profile(root):
    rows=stage_rows(root,'profile');summaries=[]
    for label in ('M2-Full','M2-Incremental'):
        for phase in ('cold','transition','steady'):
            rs=[r for r in rows if r['job_spec']['label']==label and r['phase']==phase]
            tags=sorted({k for r in rs for k in r['probe']})
            summaries.append(dict(label=label,phase=phase,frames=len(rs),
                M1_including_adapters_ms=dist([r['m1_including_adapters_ms'] for r in rs]),
                phases={k:{field:dist([r['probe'].get(k,{}).get(field,0.) for r in rs])
                    for field in ('host_ms','process_cpu_ms','thread_cpu_ms')} for k in tags},
                historical_anchors=dist([len(r['m1_precheck'].get('anchors',{})) for r in rs]),
                historical_object_anchors=dist([sum(a.get('anchor_type')=='object' for a in r['m1_precheck'].get('anchors',{}).values()) for r in rs]),
                projected_history_candidates=dist([sum(any(v.get('predicted_region') is not None for v in a.get('cameras',{}).values()) for a in r['m1_precheck'].get('anchors',{}).values()) for r in rs]),
                association_pair_matrix_slots=dist([len(r['m1_precheck'].get('anchors',{}))*len(r['m1_commit']['selected_prediction_indices']) for r in rs]),
                accepted_association_pairs=dist([len(r['m1_commit']['association_pairs']) for r in rs]),
                current_anchors=dist([r['m1_commit']['anchor_count'] for r in rs]),
                association_ms=dist([r['m1_commit']['association_ms'] for r in rs]),
                precheck_timings={k:dist([r['m1_precheck']['timings_ms'].get(k,0.) for r in rs])
                    for k in sorted({k for r in rs for k in r['m1_precheck']['timings_ms']})}))
    profiles={p.stem:read(root,str(p.relative_to(root))) for p in sorted((root/'profile/profile_output').glob('*.json'))}
    write(root/'m1_cpu_profile.json',dict(groups=summaries,heavy_profiler_frames=[0,6,7],
        nested_times_not_additive=True,cprofile={k:v['cprofile'] for k,v in profiles.items()},
        association_implementation='Current native WorldStateManager class/geometry association and SciPy unmatched assignment; not a separate DQTrack network',
        source='profile/requests.jsonl and profile/profile_output/*.json'))
    write(root/'m2_cuda_profile.json',dict(profiles=profiles,heavy_profiles_not_formal_timings=True))
    clean=[r for r in rows if not r['heavy_profiler']]
    pairs=[]
    full={r['frame_id']:r for r in clean if r['job_spec']['label']=='M2-Full'}
    for r in clean:
        if r['job_spec']['label']!='M2-Incremental':continue
        ref=full[r['frame_id']]
        pairs.append(dict(log_id=r['log_id'],frame_id=r['frame_id'],from_ms=ref['stages']['m2_total']['gpu_span_ms'],
            to_ms=r['stages']['m2_total']['gpu_span_ms'],full_e2e_ms=ref['e2e_wall_ms'],incremental_e2e_ms=r['e2e_wall_ms'],
            feature_difference=feature_difference_receipt(r),
            skipped_windows=r['m2']['actual_skipped_windows'],suffix_skipped_blocks=r['m2']['global_suffix_skipped_blocks']))
    es,quality=per_label_quality(root,rows)
    write(root/'m2_full_vs_incremental.json',dict(scope='Retrospective one-stream diagnostic; profiling wrappers enabled, heavy frames excluded from timing pairs',
        pairs=pairs,performance=diagnostic_performance(pairs),quality=quality,
        paired_answers=paired_quality(es,'M2-Full','M2-Incremental'),
        no_second_visual_runtime=True,feature_differences_on_all_frames=[dict(frame_id=r['frame_id'],**feature_difference_receipt(r)) for r in rows if r['job_spec']['label']=='M2-Incremental']))
    print(json.dumps(dict(status='PROFILE_SUMMARIZED',clean_pairs=len(pairs))))


def archive(root):
    receipts=[]
    for raw in sorted(root.glob('*/requests.jsonl')):
        dest=raw.with_suffix('.jsonl.gz')
        with raw.open('rb') as source,dest.open('wb') as target:
            with gzip.GzipFile(filename='',fileobj=target,mode='wb',mtime=0) as stream:shutil.copyfileobj(source,stream)
        with gzip.open(dest,'rb') as stream:
            import hashlib
            h=hashlib.sha256()
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        assert h.hexdigest()==sha256(raw)
        receipts.append(dict(raw=str(raw.relative_to(root)),archive=str(dest.relative_to(root)),raw_sha256=sha256(raw),archive_sha256=sha256(dest)))
    write(root/'receipt_archive.json',dict(files=receipts))


def candidate_jobs(root,stage,m1=None):
    protocol=read(root,'tuning_protocol.json')
    candidates=protocol['M1_candidates'] if stage=='tune_m1' else protocol['M2_candidates']
    if stage=='tune_m2' and (root/'m2_candidates_freeze.json').exists():candidates=read(root,'m2_candidates_freeze.json')['candidates']
    jobs=[]
    for ci,parameters in enumerate(candidates):
        if stage=='tune_m2' and ci==0 and (root/'m1_parameter_results.json').exists():
            chosen=read(root,'m1_parameter_results.json')['selected']['index']
            write(root/'tune_m2_receipt_reuse.json',dict(no_new_GPU_calls=True,mappings=[dict(source_stage='tune_m1',source_label=f'tune_m1_{chosen}_{c}',
                new_label=f'tune_m2_0_{c}',workload='addressable',frequency='three') for c in ('B','C')]))
            continue
        for si,s in enumerate(dev_scenes(root)):
            order=['B','C'] if (ci+si)%2==0 else ['C','B']
            for cond in order:
                jobs.append(job(root,s,f'{stage}_{ci}_{cond}',condition=cond,
                    m1=parameters if stage=='tune_m1' else m1,
                    m2=parameters if stage=='tune_m2' else {}))
    write(root/(stage+'_jobs.json'),jobs)
    return jobs


def select_stage(root,stage):
    rows=[r for r in stage_rows(root,stage) if r['job_spec']['label'].startswith(stage+'_')];es,quality=per_label_quality(root,rows)
    module=visual_task_quality(root,[r for r in rows if r['condition']=='C'])
    write(root/(stage+'_visual_quality.json'),module)
    rates={}
    for label in sorted({g['label'] for g in module['groups']}):
        counts=Counter()
        for g in module['groups']:
            if g['label']==label:counts.update(g['counts'])
        rates[label]=dict(counts=dict(counts),**{k:counts[n]/counts[d] if counts[d] else None for k,n,d in (
            ('M1_class','M1_class_correct','M1_scored'),('visual_class','visual_class_correct','visual_scored'),
            ('association','correct_links','scored_links'))})
    records=[]
    for index in sorted({int(r['job_spec']['label'].split('_')[-2]) for r in rows}):
        a=f'{stage}_{index}_B';b=f'{stage}_{index}_C'
        paired=paired_quality(es,a,b);part=[r for r in rows if r['job_spec']['label']==b]
        records.append(dict(index=index,m1=part[0]['job_spec']['m1'],m2=part[0]['job_spec']['m2'],module_quality=rates[b],
            E2E_ms=dist([r['e2e_wall_ms'] for r in part]),quality=quality[b],paired=paired,
            actual_skips=sum(q['llm_skipped'] for r in part for q in r['requests']),
            M1_ms=mean([r['m1_including_adapters_ms'] for r in part]),
            M2_cuda_ms=mean([r['stages']['m2_total']['gpu_span_ms'] for r in part])))
    base=next(r for r in records if r['index']==0)['module_quality']
    for r in records:r['known_module_budget_pass']=all(base[k] is None or r['module_quality'][k] is None or base[k]-r['module_quality'][k]<=.02 for k in ('M1_class','visual_class','association'))
    eligible=[r for r in records if r['paired']['loss'] is not None and r['paired']['loss']<=.02 and r['paired']['new_severe']==0 and r['known_module_budget_pass']]
    assert eligible,'No DEV candidate satisfies the declared known-label/severe gate'
    chosen=min(eligible,key=lambda r:r['E2E_ms']['mean'])
    write(root/('m1_parameter_results.json' if stage=='tune_m1' else 'm2_parameter_results.json'),dict(
        stage=stage,records=records,selected=chosen,
        scope='Three-question DEV screening only; final joint choice requires all five weighted workloads, no global-optimality claim',
        unknowns_prevent_all_task_budget=True,executor_frozen=read(root,'optimized_executor_freeze.json')))
    print(json.dumps(dict(stage=stage,selected=chosen['index'],m1=chosen['m1'],m2=chosen['m2'],mean_ms=chosen['E2E_ms']['mean'])))
    return chosen


def m3_screen(root):
    from .target_complete_report import world,context
    from worldstream.m3.answer_memory import AnswerMemory
    from worldstream.m3.answer_validity import propose_fact_answer
    selected=read(root,'m2_parameter_results.json')['selected'];rows=stage_rows(root,'tune_m2')
    rows=[r for r in rows if r['job_spec']['label']==f"tune_m2_{selected['index']}_C"]
    base=read(root,'reference_policy.json');candidates=read(root,'tuning_protocol.json')['M3_candidates'];records=[]
    for index,override in enumerate(candidates):
        policy=dict(base,**override);differences=[]
        for r in rows:
            w=world(r['committed_world'])
            for q in r['requests']:
                proposal=propose_fact_answer(q['question'],w,context(w,r['frame_index'],r['camera_ids']),AnswerMemory(),('critical_DEV',),policy,target_binding_enabled=True)
                old=q['fast_answer']
                if any(proposal.get(k)!=old.get(k) for k in ('supported','admitted','answer_text')):
                    differences.append(dict(frame_id=r['frame_id'],question=q['question'],before=old['admitted'],after=proposal['admitted']))
        write(root/f'm3_candidate_{index}.json',policy)
        records.append(dict(index=index,override=override,requests=sum(len(r['requests']) for r in rows),
            differences=differences,changed=len(differences),GPU_required=bool(differences),quality_not_counterfactually_assumed=True))
    write(root/'m3_parameter_results.json',dict(records=records,unchanged_candidates_skip_GPU=True,
        source='Selected M1/M2 DEV current world receipts only; no GT used for admission replay'))
    return records


def executor_comparison(root,stage='executor_check'):
    import re
    rows=[r for r in stage_rows(root,stage) if not r['job_spec'].get('profile')]
    def normalized(value):
        if isinstance(value,str):return re.sub(r':g\d+:',':gRESET:',value)
        if isinstance(value,dict):return {k:normalized(v) for k,v in value.items() if k not in ('epoch','state_version')}
        if isinstance(value,list):return [normalized(v) for v in value]
        return value
    comparisons=[];checks=[]
    for frequency in ('every4','three'):
        old={r['frame_id']:r for r in rows if r['job_spec']['label']=='baseline_paired' and r['frequency']==frequency}
        if not old:continue
        pairs=[];m1=[]
        for r in rows:
            if r['job_spec']['label']!='optimized_paired' or r['frequency']!=frequency:continue
            ref=old[r['frame_id']]
            pairs.append(dict(log_id=r['log_id'],frame_id=r['frame_id'],from_ms=ref['e2e_wall_ms'],to_ms=r['e2e_wall_ms']))
            m1.append(dict(log_id=r['log_id'],from_ms=ref['m1_including_adapters_ms'],to_ms=r['m1_including_adapters_ms']))
            checks.append(dict(frame_id=r['frame_id'],frequency=frequency,
                world_equal=normalized(ref['committed_world'])==normalized(r['committed_world']),
                visual_feature_equal=ref['visual_quality']['feature']==r['visual_quality']['feature'] if valid_feature_witness(ref) and valid_feature_witness(r) else None,
                current_feature_witness_valid=valid_feature_witness(ref) and valid_feature_witness(r),
                detection_outputs_equal=all(ref['visual_quality'][k]==r['visual_quality'][k] for k in ('boxes','labels','scores')),
                IDs_and_associations_equal=normalized(ref['visual_quality']['association_pairs'])==normalized(r['visual_quality']['association_pairs']) and normalized(ref['visual_quality']['committed_anchor_ids'])==normalized(r['visual_quality']['committed_anchor_ids']),
                commit_count_equal=ref['m1_commit_count']==r['m1_commit_count'],
                M2_window_decisions_equal=ref['m2']['stages']==r['m2']['stages'],
                answer_tokens_equal=[q['tokens'] for q in ref['requests']]==[q['tokens'] for q in r['requests']],
                actual_skips_equal=[q['llm_skipped'] for q in ref['requests']]==[q['llm_skipped'] for q in r['requests']]))
        comparisons.append(dict(frequency=frequency,E2E=cluster_performance(pairs),M1=cluster_performance(m1)))
    all_equal=all(all(v is not False for k,v in c.items() if k.endswith('_equal')) for c in checks)
    dest='before_after_executor_comparison.json' if stage=='executor_clean' else 'executor_pilot_comparison.json'
    write(root/dest,dict(stage=stage,comparisons=comparisons,checks=checks,all_equal=all_equal,
        equivalence_scope='Scene reset epoch and derived geometry-version hash normalized only. World/detection outputs/IDs/associations/answers/refresh checked on all pairs; legacy feature bytes counted only when the last-block hook actually fired. Post-final return-value feature audit covers reuse frames.',
        valid_feature_pairs=sum(c['current_feature_witness_valid'] for c in checks),
        local_speedup_is_not_automatically_system_speedup=True,heavy_profiler=False))
    assert all_equal,'Executor semantic mismatch; investigate before freezing'
    print(json.dumps(dict(status='EXECUTOR_COMPARISON',all_equal=all_equal,comparisons=comparisons)))


def visual_task_quality(root,rows):
    from .fact_bypass_identity import match_instances
    from worldstream.m1.omnidrive_adapter import CLASS_NAMES
    objects=read(root,'current_gt_annotations.json');groups=[]
    by=defaultdict(list)
    for r in rows:by[(r['job_spec']['label'],r['workload'],r['frequency'],r['scene_id'])].append(r)
    for key,part in by.items():
        counts=Counter();previous={};previous_instances={}
        for r in sorted(part,key=lambda x:x['frame_index']):
            raw=r['visual_quality'];anchors=[]
            for i,(box,label,score) in enumerate(zip(raw['boxes'],raw['labels'],raw['scores'])):
                if score<.3:continue
                center=list(box[:3]);center[2]+=box[5]/2
                anchors.append(dict(anchor_type='object',anchor_id=str(i),observed=True,source_frame=r['frame_id'],
                    source_timestamp=r['timestamp'],evidence_origin='actual_perception',center_ego=center,semantic_class=CLASS_NAMES[label]))
            matches=match_instances(anchors,objects[r['frame_id']],r['frame_id'],r['timestamp'])
            known=[m for m in matches.values() if m['known']]
            counts.update(visual_predictions=len(matches),visual_scored=len(known),visual_class_correct=sum(m['class_correct'] for m in known))
            world=(r['committed_world'] or {}).get('anchors',[])
            matches=match_instances(world,objects[r['frame_id']],r['frame_id'],r['timestamp'])
            known=[m for m in matches.values() if m['known']]
            counts.update(M1_predictions=len(matches),M1_scored=len(known),M1_class_correct=sum(m['class_correct'] for m in known))
            current={aid:m['instance_id'] for aid,m in matches.items() if m['known']}
            for aid,_ in raw['association_pairs']:
                if aid not in matches:continue
                counts['accepted_object_links']+=1
                if aid in previous and aid in current:
                    counts['scored_links']+=1;counts['correct_links']+=int(previous[aid]==current[aid])
                else:counts['UNKNOWN_links']+=1
            instances={v:k for k,v in current.items()}
            common=set(instances)&set(previous_instances)
            counts['consecutive_GT_instances']+=len(common)
            counts['ID_switches']+=sum(instances[i]!=previous_instances[i] for i in common)
            previous=current;previous_instances=instances
        groups.append(dict(label=key[0],workload=key[1],frequency=key[2],scene_id=key[3],counts=dict(counts),
            visual_class_accuracy=counts['visual_class_correct']/counts['visual_scored'] if counts['visual_scored'] else None,
            M1_class_accuracy=counts['M1_class_correct']/counts['M1_scored'] if counts['M1_scored'] else None,
            association_precision=counts['correct_links']/counts['scored_links'] if counts['scored_links'] else None))
    return dict(groups=groups,scope='Partial independent GT matching, fixed raw detection threshold .3, unchanged <2m/.5m unique class-independent matching. Not official nuScenes mAP/NDS or standard MOT.',
        map_quality='No independent lane-label denominator retained by this worker; map semantic quality is NOT_CERTIFIED. Native map head still executes; executor equivalence and detection/answer quality are checked.',
        UNKNOWN_never_correct=True)


def profile_reports(root):
    profile=read(root,'m1_cpu_profile.json');m2=read(root,'m2_full_vs_incremental.json')
    detail={p.stem:read(root,str(p.relative_to(root))) for p in (root/'executor_check/profile_output').glob('*.json')}
    write(root/'m2_detailed_profile.json',dict(profiles=detail,raw_traces_not_committed=True,profiler_disabled_for_formal_E2E=True))
    (root/'m1_state_cost_analysis.md').write_text('''# M1 measured state cost

M1 is the current WorldStateManager geometry/class association and bounded state maintenance. This checkout does not execute a separate DQTrack neural network in this path; no tracker weights were changed.

The prior230ms steady-state mean came from different evaluation scenes. This run separates input/output adapters, optical flow/observation support, projection, matching, immutable transaction forks, complete live-state hashing and state publication; see m1_cpu_profile.json and raw per-frame m1_precheck/m1_commit. CPU process and calling-thread times are separate from host elapsed. Nested spans must not be summed. Heavy profiler results identify hotspots and are not formal latency claims.

Representative profiled steady frames show roughly47ms temporal correspondences +8ms geometric visibility +5ms patch support,23-25ms observation checking,10-12ms precheck projection,21-26ms matching, about42ms across three complete source-digest calls under cProfile, and19ms across two exact frame digests. These operations depend on anchor count, frame contents, immutable transaction metadata and retained history. Input D2H was about2.2ms; output heads had already produced CPU scalars. No evidence supports attributing the entire cold/steady gap to redundant GPU transfer.

Only redundant scalar reflection and temporary byte materialization were changed. Complete source-state traversal, frame hashing, object association, optical flow and immutable publication remain. No cached hash bypasses live mutation checks and no synchronization was relocated. The before/after source copies and canonical-hash regression tests preserve byte compatibility. Actual paired M1/E2E results are in before_after_executor_comparison.json and executor_pilot_comparison.json. An outlier in the pilot is retained; its lower-frequency mean is inconclusive.
''')
    table='| Probe | Prefix kernels / ms | Global kernels / ms | Grid kernel ms | Cache kernel ms |\n|---|---:|---:|---:|---:|\n'
    for name,d in sorted(detail.items()):
        v=d['visual_ranges'];prefix=v.get('critical/M2/window_execute',{});suffix=v.get('critical/M2/_update_global',{})
        table+=f"| {name} | {prefix.get('kernel_count')} / {prefix.get('kernel_ms',0):.3f} | {suffix.get('kernel_count')} / {suffix.get('kernel_ms',0):.3f} | {v.get('critical/M2/window__grid',{}).get('kernel_ms',0):.3f} | {v.get('critical/M2/cache_commit',{}).get('kernel_ms',0):.3f} |\n"
    (root/'m2_kernel_breakdown.md').write_text('# Actual CUDA work\n\n'+table+'''
000 is same-executor Full and001 is Incremental, frames01 and06 of the same8-frame prefix. On frame01 the incremental global suffix is reused; on06 it refreshes. Full and Incremental share weights, dtype, resolution, token layout, native block functions and synchronization. Only the private M2 refresh hint changes in Full; M1 processing is preserved. Each ordered condition has its own continuous closed-loop state and cache; downstream state differences can arise from actual approximate outputs.

On frame06 the cached prefix falls from38.366ms to31.994ms of actual kernels, while the global suffix remains432.468ms versus434.159ms. On frame01 the global update falls from430.344ms/939 kernels to0.905ms/11 kernels. Partial-window GEMMs/attention perform less work even though gather/scatter can add kernel launches; a kernel-count reduction is not proportional to a window-count reduction. There is no full-prefix recompute followed by cosmetic cache replacement in the partial path. The patch embedding always executes once. Cache finite validation/ownership and global attention on refresh frames are retained correctness costs.

The prefix packing costs about0.7ms in Full; partial output-grid reconstruction about0.35ms; refresh selection about0.62ms kernels and cache publication about1.6ms kernels. Detailed copy/runtime-launch totals and kernel names are in m2_detailed_profile.json. Correlation IDs map GPU events to CPU launch ranges. Nested ranges overlap; kernel durations measure work, CUDA events measure elapsed stream spans including launch gaps/waits, and synchronized outer E2E measures the full model. Their differences are not a hardware SM-idle measurement. CPU adapter D2H and cuda_runtime synchronize calls are separately retained in raw profiles.

In the initial16-frame Incremental sequence,12 frames refresh the global suffix because drift exceeds the existing threshold. This dominates the observed limited benefit despite many prefix windows reused. Remaining executor overhead is small compared with required suffix computation; stop M2 implementation edits and test a finite set of existing refresh parameters. No architecture, window geometry, new risk selector or kernel library is introduced.
''')
    (root/'critical_path_report.md').write_text('# Critical path evidence\n\n'+
        f"Same-executor Full/Incremental initial unheavy pairs: {m2['performance']['from_ms']:.3f}→{m2['performance']['to_ms']:.3f}ms visual CUDA span, {m2['performance']['speedup']:.4f}× on13 frames/one log. This is a diagnostic with lightweight observers, not an independent multi-log performance result. Known answers {m2['quality']['M2-Full']['scored']}/48, same correctness in both conditions; UNKNOWN retained. Continuous output feature deviations are stored per frame.\n\n"+
        'M1 source-digest reflection and frame byte copies were low-complexity implementation opportunities; necessary state work is retained. M2 global-suffix refresh is the dominant remaining cost, so changing executor packing further has little justified headroom. Final improvements from implementation and parameters must be separated. Native host enqueue is never compared against synchronized M2 wall; native/incremental CUDA stream spans and full outer synchronized E2E are separately reported.\n\n'+
        'Instrumentation: critical_path_probe.py adds external CPU/process/thread timing wrappers, record_function ranges, cProfile and Kineto correlation summaries; critical_path.py reuses the continuous-stream full-model worker and existing Observer, captures raw predictions and frozen job manifests. Production runtime instrumentation was not added. Raw giant Chrome traces are deleted after compact summaries; temporary visual tensors remain outside Git. No per-kernel claim uses a theoretical FLOP/window estimate.\n')
    (root/'m2_optimization_report.md').write_text('# M2 executor stop decision\n\nNo serving M2 implementation changed. P0 verified real selective QKV/attention/MLP execution, small gather/scatter/cache costs and dominant necessary global suffix computation. An executor rewrite would add complexity for small remaining overhead. Full/Incremental is an algorithmic reuse ablation, not a before/after code-optimization speedup. Subsequent changes to existing refresh parameters are reported as parameter effects. Source hashes preserve the unchanged M2.\n')
    (root/'m1_optimization_report.md').write_text('# M1 implementation change\n\nsource_digest.py: fast-path Python scalar leaves and hash contiguous numeric arrays directly. input_contract.py: hash exact contiguous buffers without allocating temporary tobytes copies. Canonical bytes and live mutable-state coverage are unchanged. No association, age, coordinate, image or descriptor rule changed. Legacy functions are confined to an external paired benchmark control; serving has one implementation. See before_after_executor_comparison.json for fresh-process results and executor_pilot_comparison.json for all pilot results including its outlier. Two logs do not establish a population-wide speedup. Stop further M1 edits after this bounded change.\n')


def prepare_joint(root):
    chosen=read(root,'m2_parameter_results.json')['selected']
    m3=read(root,'m3_parameter_results.json')
    assert 'selected_index' in m3,'Complete necessary M3 GPU decisions before joint freeze'
    selected_policy=str(root/f"m3_candidate_{m3['selected_index']}.json")
    candidates=[dict(m1={},m2={},policy=str(root/'reference_policy.json'),label='reference_parameters')]
    if chosen['m1'] or chosen['m2'] or m3['selected_index']:
        candidates.append(dict(m1=chosen['m1'],m2=chosen['m2'],policy=selected_policy,label='sequential_best'))
    jobs=[];reuse=[]
    for ci,c in enumerate(candidates):
        for cond in ('B','C'):
            reuse.append(dict(source_stage='tune_m1' if ci==0 else 'tune_m3',
                source_label=f'tune_m1_0_{cond}' if ci==0 else f"tune_m3_{m3['selected_index']}_{cond}",
                new_label=f'joint_{ci}_{cond}',workload='addressable',frequency='three'))
        for wi,cell in enumerate(read(root,'tuning_protocol.json')['workload_weights']):
            w,f=cell.split('/')
            if f=='three' and w=='addressable':continue
            for si,s in enumerate(dev_scenes(root)):
                order=['B','C'] if (ci+wi+si)%2==0 else ['C','B']
                for cond in order:jobs.append(job(root,s,f'joint_{ci}_{cond}',condition=cond,workload=w,frequency=f,m1=c['m1'],m2=c['m2'],policy=c['policy'] if cond=='C' else None))
    write(root/'joint_candidates.json',dict(candidates=candidates,selection_rule='Min equal-weight five-cell mean E2E among known-quality-budget/severe-gated candidates; no global optimum claim'))
    write(root/'joint_receipt_reuse.json',dict(mappings=reuse,no_new_GPU_calls_for_reused_three_question_receipts=True))
    write(root/'joint_jobs.json',jobs)
    return jobs


def select_joint(root):
    rows=stage_rows(root,'joint');es,quality=per_label_quality(root,rows);weights=read(root,'tuning_protocol.json')['workload_weights']
    candidates=read(root,'joint_candidates.json')['candidates'];results=[]
    for ci,c in enumerate(candidates):
        cells=[]
        for cell,weight in weights.items():
            w,f=cell.split('/');part=[r for r in rows if (r['workload'],r['frequency'])==(w,f)]
            qs=[e for e in es if (e['workload'],e['frequency'])==(w,f)]
            p=paired_quality(qs,f'joint_{ci}_B',f'joint_{ci}_C')
            selected=[r for r in part if r['job_spec']['label']==f'joint_{ci}_C']
            baseline=[r for r in part if r['job_spec']['label']==f'joint_{ci}_B']
            assert len(selected)==len(baseline)==32
            cells.append(dict(workload=w,frequency=f,weight=weight,C_ms=mean([r['e2e_wall_ms'] for r in selected]),
                B_ms=mean([r['e2e_wall_ms'] for r in baseline]),paired=p))
        results.append(dict(index=ci,parameters=c,cells=cells,weighted_C_ms=sum(s['weight']*s['C_ms'] for s in cells)))
    bybudget={}
    for eps in (.01,.02,.03):
        eligible=[r for r in results if all(c['paired']['loss'] is not None and c['paired']['loss']<=eps and c['paired']['new_severe']==0 for c in r['cells'])]
        bybudget[str(eps)]=min(eligible,key=lambda r:r['weighted_C_ms'])['index'] if eligible else None
    assert bybudget['0.02'] is not None,'No jointly confirmed DEV configuration meets the declared known-label gate'
    selected=results[bybudget['0.02']]
    write(root/'joint_quality_latency_pareto.json',dict(results=results,best_index_by_budget=bybudget,
        qualification='Known paired labels in each preset cell only; UNKNOWN/all-request bounds and severe errors remain separate',
        not_exhaustive_global_optimum=True,reused_receipts_not_new_calls=True))
    policy=json.loads(Path(selected['parameters']['policy']).read_text())
    write(root/'selected_joint_policy.json',dict(status='FROZEN_RETROSPECTIVE_POLICY',scope='RETROSPECTIVE_EVALUATION',
        executor='M123_optimized_executor',executor_freeze_sha256=sha256(root/'optimized_executor_freeze.json'),
        selected_index=selected['index'],m1=selected['parameters']['m1'],m2=selected['parameters']['m2'],m3=policy,
        weighted_DEV_E2E_ms=selected['weighted_C_ms'],default_budget=.02,automatic_deployment=False,
        selection_sha256=sha256(root/'joint_quality_latency_pareto.json')))
    final=[];scenes=[s for s in read(root,'extended_frame_selection.json')['selected'] if s['role']=='extended_evaluation']
    for wi,cell in enumerate(weights):
        w,f=cell.split('/')
        for si,s in enumerate(scenes):
            order=['A','B','C'];offset=(wi+si)%3;order=order[offset:]+order[:offset]
            for cond in order:final.append(job(root,s,cond,condition=cond,workload=w,frequency=f,
                m1=selected['parameters']['m1'] if cond!='A' else {},m2=selected['parameters']['m2'] if cond!='A' else {},
                policy=selected['parameters']['policy'] if cond=='C' else None,native_without_m1=cond=='A'))
    # Only this finite3Q companion retains the previous A-with-M1 accounting.
    final += [job(root,s,'A_with_M1',condition='A') for s in scenes]
    write(root/'final_jobs.json',final)
    print(json.dumps(dict(selected=selected['index'],m1=selected['parameters']['m1'],m2=selected['parameters']['m2'],weighted_ms=selected['weighted_C_ms'])))
    return final


def prepare_m3(root):
    records=m3_screen(root);selected=read(root,'m2_parameter_results.json')['selected'];jobs=[];reuse=[]
    for r in records:
        ci=r['index']
        reuse.append(dict(source_stage='tune_m2',source_label=f"tune_m2_{selected['index']}_B",new_label=f'tune_m3_{ci}_B',workload='addressable',frequency='three'))
        if not r['changed']:
            reuse.append(dict(source_stage='tune_m2',source_label=f"tune_m2_{selected['index']}_C",new_label=f'tune_m3_{ci}_C',workload='addressable',frequency='three'))
        else:
            for s in dev_scenes(root):jobs.append(job(root,s,f'tune_m3_{ci}_C',m1=selected['m1'],m2=selected['m2'],policy=str(root/f'm3_candidate_{ci}.json')))
    write(root/'tune_m3_jobs.json',jobs);write(root/'tune_m3_receipt_reuse.json',dict(mappings=reuse,no_new_GPU_calls_for_equivalent_decisions=True))
    if not jobs:
        out=root/'tune_m3';out.mkdir(exist_ok=True);(out/'requests.jsonl').write_text('')
        write(out/'run_status.json',dict(status='NOT_RUN_ALL_CANDIDATES_DECISION_EQUIVALENT',full_M123_frame_calls=0))
    return jobs


def select_m3(root):
    rows=stage_rows(root,'tune_m3');es,quality=per_label_quality(root,rows);screen=read(root,'m3_parameter_results.json');records=[]
    for r in screen['records']:
        ci=r['index'];qs=paired_quality(es,f'tune_m3_{ci}_B',f'tune_m3_{ci}_C')
        part=[x for x in rows if x['job_spec']['label']==f'tune_m3_{ci}_C']
        records.append(dict(**r,E2E_ms=mean([x['e2e_wall_ms'] for x in part]),paired=qs,
            actual_new_GPU_calls=sum(not x.get('reused_receipt',False) for x in part)))
    eligible=[r for r in records if r['paired']['loss'] is not None and r['paired']['loss']<=.02 and r['paired']['new_severe']==0]
    assert eligible
    chosen=min(eligible,key=lambda r:r['E2E_ms'])
    write(root/'m3_parameter_results.json',dict(records=records,selected_index=chosen['index'],same_facts_CPU_replay='Unchanged admission/text means no separate GPU experiment; changed cases measured with causal history evolution'))
    print(json.dumps(dict(stage='M3',selected_index=chosen['index'],mean_ms=chosen['E2E_ms'])))


def final_report(root):
    from .fact_bypass_identity import coverage,UNKNOWN_REASONS
    all_final=stage_rows(root,'final')
    rows=[r for r in all_final if r['job_spec']['label']!='A_with_M1']
    assert read(root,'final/run_status.json')['status']=='COMPLETED'
    assert not any(r['heavy_profiler'] for r in rows)
    perf=performance(rows)
    # Preserve the raw worker receipts, but give legacy convenience aliases
    # their actual measurement names in the new report. Decode-loop CUDA span
    # is not CPU wall; prefill CPU enqueue is not a synchronized prefill wall.
    for g in perf['groups']:
        if g['condition']=='A':
            g['prefill_forward_host_elapsed_ms']=g.pop('prefill_wall_ms')
            g['decode_forward_host_elapsed_ms']=g.pop('decode_wall_ms')
            g['prefill_forward_cuda_sum_ms']=g.pop('prefill_cuda_ms')
        else:
            g['prefill_cpu_enqueue_ms']=g.pop('prefill_wall_ms')
            g['decode_loop_cuda_span_ms']=g.pop('decode_wall_ms')
            g['prefill_including_cache_cuda_span_ms']=g.pop('prefill_cuda_ms')
        g['decode_forward_cuda_sum_ms']=g.pop('decode_cuda_ms')
        part=[r for r in rows if (r['condition'],r['workload'],r['frequency'])==(g['condition'],g['workload'],g['frequency'])]
        g['synchronized_LLM_generation_wall_ms']=dist([q.get('llm_wall_ms',0. if q['llm_skipped'] else None) for r in part for q in r['requests']])
        g['language_timing_scope']='Native forward-only host spans exclude sampler management. B/C prefill is CPU enqueue and CUDA event span; decode_loop is CUDA event span, decode_forward is sum of forward CUDA spans. Synchronized LLM generation and M3 runtime host wall are separate. No separately synchronized pure-prefill/decode-host measurement is invented.'
    write(root/'abc_comparison.json',perf)
    es=events(root,rows)
    with (root/'quality_requests.jsonl').open('w') as handle:
        for event in es:handle.write(json.dumps(event)+'\n')
    primary=[e for e in es if (e['workload'],e['frequency'])==('addressable','three')]
    c=[e for e in primary if e['condition']=='C'];fast=[e for e in c if e['llm_skipped']]
    quality=paired_quality(primary,'B','C');native=paired_quality(primary,'A','C')
    tasks=[quality_summary([e for e in primary if e['condition']==cond and e['task']==task],condition=cond,task=task)
        for cond in ('A','B','C') for task in ('category','existence','position')]
    cells=sorted({(e['workload'],e['frequency']) for e in es})
    cell_quality=[dict(workload=w,frequency=f,**paired_quality([e for e in es if (e['workload'],e['frequency'])==(w,f)],'B','C')) for w,f in cells]
    write(root/'quality_by_task.json',dict(scope='RETROSPECTIVE_EVALUATION',groups=tasks))
    module=visual_task_quality(root,rows)
    aggregates={}
    for condition in ('A','B','C'):
        counter=Counter()
        for item in module['groups']:
            if (item['label'],item['workload'],item['frequency'])==(condition,'addressable','three'):counter.update(item['counts'])
        aggregates[condition]=dict(counts=dict(counter),**{name:counter[n]/counter[d] if counter[d] else None for name,n,d in (
            ('M1_class_accuracy','M1_class_correct','M1_scored'),('visual_class_accuracy','visual_class_correct','visual_scored'),
            ('association_precision','correct_links','scored_links'),('consecutive_identity_retention','correct_links','consecutive_GT_instances'))})
    module['primary_three_question_aggregates']=aggregates
    write(root/'module_task_quality.json',module)
    write(root/'task_quality.json',dict(primary_cell='addressable/three',paired=quality,native_to_C=native,
        C=quality_summary(c),actual_fast=quality_summary(fast),fallback=quality_summary([e for e in c if not e['llm_skipped']]),
        object=coverage(c,'object'),target=coverage(c,'target'),
        UNKNOWN_reasons={k:sum(e['target_reference'].get('reason')==k for e in c if not e['quality']['known']) for k in UNKNOWN_REASONS},
        module_quality=module,
        all_cells=cell_quality,
        native_all_cells=[dict(workload=w,frequency=f,**paired_quality([e for e in es if (e['workload'],e['frequency'])==(w,f)],'A','C')) for w,f in cells]))
    position_cases=[]
    for e in primary:
        if e['task']!='position':continue
        dirs=set(e['quality'].get('extracted',{}).get('directions',()))
        reference=set(e['quality'].get('reference',{}).get('directions',()))
        position_cases.append(dict(frame_id=e['frame_id'],question=e['question'],condition=e['condition'],known=e['quality']['known'],
            fast=e['llm_skipped'],left_right_flip=bool(('left' in dirs and 'right' in reference) or ('right' in dirs and 'left' in reference)),
            front_rear_flip=bool(('front' in dirs and 'rear' in reference) or ('rear' in dirs and 'front' in reference)),
            target_known=e['target']['known'],wrong_nearest_target=e['target']['known'] and not e['target'].get('correct',False)))
    write(root/'severe_error_audit.json',dict(primary=quality,
        new_cases=[p for p in quality['pairs'] if p['new_severe']],
        position=[g for g in tasks if g['task']=='position'],
        position_cases=position_cases,
        all_cell_pairs=cell_quality,
        unique_new_severe_frame_questions=sorted({(p['frame_id'],p['question']) for cell in cell_quality for p in cell['pairs'] if p['new_severe']}),
        fast_position=quality_summary([e for e in fast if e['task']=='position']),
        fallback_position=quality_summary([e for e in c if not e['llm_skipped'] and e['task']=='position']),
        strict_rare_error_2percent_certification=False,
        caveat='New severe versus matched B is separate from absolute severe count; UNKNOWN never correct. Three exposed logs cannot certify rare-risk or independent generalization.'))
    studies=[]
    for pair in quality['pairs']:
        if not pair['new_severe']:continue
        studies.append(dict(pair=pair,events=[e for e in primary if (e['frame_id'],e['question'])==(pair['frame_id'],pair['question'])]))
    write(root/'severe_case_study.json',dict(primary_new_cases=studies,no_threshold_refit_after_evaluation=True))
    paragraphs=[]
    for study in studies:
        b=next(e for e in study['events'] if e['condition']=='B');c=next(e for e in study['events'] if e['condition']=='C')
        paragraphs.append(f"Frame `{c['frame_id']}`, log `{c['log_id']}`. GT required directions={c['quality']['reference']['directions']}. B: {b['answer']} C: {c['answer']} C fast={c['llm_skipped']}. B score={b['quality']['score']}, severe={b['quality']['severe_error']}; C score={c['quality']['score']}, severe={c['quality']['severe_error']}. C object={c['object']}; C target={c['target']}. Both incomplete and explicitly reversed answers can score zero, while only the latter is a severe direction error. The average accuracy budget cannot hide this difference.")
    (root/'severe_case_study.md').write_text('# Independent severe-error case study\n\n'+('\n\n'.join(paragraphs) if paragraphs else 'No new primary severe case observed.')+'\n\nThe final policy remains frozen; no frame-specific rule, GT admission, semantic relabeling or threshold retuning was performed. A nonzero new severe count fails the independent gate used in this study, regardless of improved mean accuracy.\n')
    write(root/'question_frequency_analysis.json',dict(groups=[g for g in perf['groups'] if g['workload']=='addressable'],
        comparisons=[p for p in perf['comparisons'] if p['workload']=='addressable']))
    contributions=dict(M1=read(root,'before_after_executor_comparison.json'),M2=read(root,'m2_full_vs_incremental.json'),
        M3=[p for p in perf['comparisons'] if p['comparison']=='B->C'],
        corrected_feature_audit=read(root,'corrected_feature_audit.json'),
        legacy_feature_validity='legacy_feature_witness_validity.json',
        parameter_DEV=read(root,'joint_quality_latency_pareto.json'),
        attribution='M1 exact implementation pairs; same-executor M2 reuse ablation; frozen same-M1/M2 B/C language-policy contrast. Parameter effects measured separately from code changes, no additive speedup assumption.')
    book=[r for r in all_final if r['job_spec']['label']=='A_with_M1']
    if book:
        native_rows={r['frame_id']:r for r in rows if (r['condition'],r['workload'],r['frequency'])==('A','addressable','three')}
        chosen_rows={r['frame_id']:r for r in rows if (r['condition'],r['workload'],r['frequency'])==('C','addressable','three')}
        contributions['old_A_with_M1_to_C']=cluster_performance([dict(log_id=r['log_id'],from_ms=r['e2e_wall_ms'],to_ms=chosen_rows[r['frame_id']]['e2e_wall_ms']) for r in book])
        contributions['true_native_to_old_A_with_M1']=cluster_performance([dict(log_id=r['log_id'],from_ms=native_rows[r['frame_id']]['e2e_wall_ms'],to_ms=r['e2e_wall_ms']) for r in book])
        contributions['old_A_caveat']='Previous1.9405x used native vision/generation PLUS M1 bookkeeping. Main A now removes M1, so compare old1.9405x only with this explicit companion, never attribute the accounting change to optimization.'
    write(root/'module_contribution.json',contributions)
    stages={};actual=[]
    for p in sorted(root.glob('*/run_status.json')):
        stage=p.parent.name
        raw=p.parent/'requests.jsonl'
        part=lines(raw) if raw.exists() else []
        stages[stage]=dict(status=read(root,str(p.relative_to(root)))['status'],full_model_frame_calls=len(part),
            actual_skips=sum(q['llm_skipped'] for r in part for q in r['requests']),
            QA_frames=sum(bool(r['questions']) for r in part),idle_frames=sum(not r['questions'] for r in part))
        actual.extend(part)
    qs=[q for r in actual for q in r['requests']]
    counts=dict(full_model_frame_calls=len(actual),stateful_M123_frame_calls=sum(r['condition']!='A' for r in actual),
        native_no_M1_frame_calls=sum(r['job_spec'].get('native_without_m1',False) for r in actual),
        native_plus_M1_frame_calls=sum(r['condition']=='A' and not r['job_spec'].get('native_without_m1',False) for r in actual),
        with_QA=sum(bool(r['questions']) for r in actual),no_QA=sum(not r['questions'] for r in actual),
        question_requests=len(qs),actual_LLM_forwards=sum(q['llm_forward_calls'] for q in qs),actual_prefill_skips=sum(q['llm_skipped'] for q in qs),
        actual_decode_skips=sum(q['llm_skipped'] for q in qs),distinct_frames=len({r['frame_id'] for r in actual}),
        distinct_scenes=len({r['scene_id'] for r in actual}),distinct_logs=len({r['log_id'] for r in actual}),
        stages=stages,reused_receipts_count_as_new_GPU_calls=False,regression_full_model_calls=0)
    write(root/'gpu_execution_counts.json',counts)
    selected=read(root,'selected_joint_policy.json');pareto=read(root,'joint_quality_latency_pareto.json')
    gains=lambda w,f,comp='B->C':next(p for p in perf['comparisons'] if (p['workload'],p['frequency'],p['comparison'])==(w,f,comp))
    main_gain=gains('addressable','three');mixed=gains('mixed','three')
    verdict=dict(executor_status='EXECUTOR_OPTIMIZED_AND_RETROSPECTIVELY_VALIDATED',
        independent_generalization='INDEPENDENT_GENERALIZATION_INCOMPLETE',
        scope='RETROSPECTIVE_EVALUATION_WITHIN_HISTORICALLY_EXPOSED_LOGS_AND_FRAMES',
        main_B_to_C=main_gain,main_A_to_C=gains('addressable','three','A->C'),mixed=mixed,
        known_primary_2pp_pass=quality['loss'] is not None and quality['loss']<=.02,
        new_severe=quality['new_severe'],strict_rare_risk_certification=False,
        per_cell_known_quality_or_severe_gate_failures=[dict(workload=p['workload'],frequency=p['frequency'],loss=p['loss'],new_severe=p['new_severe'],scored=p['scored'])
            for p in cell_quality if p['loss'] is None or p['loss']>.02 or p['new_severe']>0],
        full_task_quality_budget_certified=False,independent_TEST=False,
        historical_v3_NO_GO='UNCHANGED',historical_v4_NO_GO='UNCHANGED',previous_INCOMPLETE='UNCHANGED',
        production_enabled=False,not_exhaustive_global_optimum=True)
    verdict['selected_policy_retrospective_quality_gate']='FAILED' if verdict['per_cell_known_quality_or_severe_gate_failures'] else 'PASS_ON_KNOWN_LABELS_ONLY'
    write(root/'research_verdict.json',verdict)
    write(root/'policy_qualification.json',dict(selected_policy_sha256=sha256(root/'selected_joint_policy.json'),
        status=verdict['selected_policy_retrospective_quality_gate'],new_primary_severe=quality['new_severe'],
        known_answer_2pp_pass=verdict['known_primary_2pp_pass'],full_task_budget_certified=False,
        production_enabled=False,no_policy_refit_after_evaluation=True,
        frozen_configuration_preserved=True,qualification_is_separate_from_executor_regressions=True))
    table='| 预设负载 | A原生 ms/帧 | B ms/帧 | C ms/帧 | B→C | A→C | B→C log CI95 |\n|---|---:|---:|---:|---:|---:|---|\n'
    for w,f in [('addressable',f) for f in ('every4','every2','one','three')]+[('mixed','three')]:
        p=gains(w,f);a=gains(w,f,'A->C')
        table+=f"| {w}/{f} | {a['from_ms']:.3f} | {p['from_ms']:.3f} | {p['to_ms']:.3f} | {p['speedup']:.4f}× | {a['speedup']:.4f}× | {p['speedup_log_CI95']} |\n"
    (root/'e2e_performance_report.md').write_text('# Frozen retrospective full-system evaluation\n\n'+
        f"Selected policy risk gate: {verdict['selected_policy_retrospective_quality_gate']}; primary new severe errors={quality['new_severe']}. Improved mean accuracy does not override this failure. Independent generalization remains INCOMPLETE; production stays off.\n\n"+table+
        '\nEach cell uses all72 continuous frames/3 exposed logs, synchronized full-model wall including answer export and excluding input preparation/audit serialization equally. No-QA frames execute vision and, for B/C, M1+M2+heads. Heavy profiler never runs in the final process. A executes native vision/generation with no M1/M2/M3 computation; the limited old-A companion is separately reported. Per-stage wall/CUDA, prefill/decode, actual forwards/skips, cold/steady, P50/P95, FPS/QPS and peak memory are in abc_comparison.json. Memory is the harness process allocation peak and can retain inactive buffers from prior conditions; it is not a minimum standalone Native deployment footprint. FPS/QPS are compute service throughput, not sensor-arrival queue throughput; no real-time deadline claim is made. P99 and3-log intervals are descriptive. The legacy feature-witness observation defect does not change these measured timings or head/QA outputs; corrected return-value comparisons are in feature_audit_report.md.\n')
    (root/'parameter_sensitivity_report.md').write_text('# Finite sequential DEV search\n\n'+
        'Protocol was frozen before hyperparameter DEV, including equal weights across five workloads and1/2/3pp known-label budgets. P0 justified one additional pre-DEV drift-threshold point, recorded in m2_candidates_freeze.json. M1 candidates vary existing detection filtering and association gates; M2 candidates vary existing ordinary budget, global anchor interval and drift threshold. M3 candidates vary existing confidence/binding-risk gates only. No Cartesian search, new risk model or changed sampling.\n\n'+
        f"Selected default2pp: M1={selected['m1']}, M2={selected['m2']}, M3 min_confidence={selected['m3']['min_confidence']}, max_binding_risk={selected['m3'].get('max_binding_risk')}. The1/2/3pp DEV choices are {pareto['best_index_by_budget']}. These are DEV measured operating points, not three separately retested final policies.\n\n"+
        'Candidate module and task quality include separate known denominators, UNKNOWN, partial association continuity and severe errors. The M1 .4 threshold removes lower-score states, so conditional accuracy changes must be read together with retained-state/label counts; it does not prove improved full-object recall. Same-decision candidates and exact already-executed B/reference paths reuse receipts without counting extra GPU calls. Joint confirmation measures the other four cells; the existing3Q cell is reused. The final selected combination is tested on the three retrospective evaluation streams after freeze. No exhaustive or population-wide optimum is established.\n')
    selection=read(root,'extended_frame_selection.json')
    profiler_scenes={r['scene_id'] for r in actual if r['job_spec'].get('profile',r['stage']=='profile')}
    write(root/'data_roles.json',dict(independent_TEST=False,scenes=[dict(scene_id=s['scene_id'],log_id=s['log_id'],frames=len(s['tokens']),
        roles=(([ 'Profiler'] if s['scene_id'] in profiler_scenes else [])+['Executor Development','Hyperparameter DEV']) if s['role']=='extended_DEV' else ['Retrospective Evaluation'],
        prior_M123_execution=True,prior_method_and_log_exposure=True) for s in selection['selected']]))
    (root/'data_independence_report.md').write_text('# Data roles and independence\n\nAll104 selected frames/5 scenes/5 logs were executed in the previous study and their logs have method-development exposure. Two16-frame streams supply profiling/executor DEV/hyperparameter DEV; three24-frame streams supply retrospective frozen-policy evaluation. None is a new independent TEST or a new unexecuted-frame extension. Official val is outside the configured nuScenes training split, while exact pretraining exposure is unknown. GT is read only by offline grading. See data_roles.json and the previous retained frame-exposure audit. Historical v3/v4 NO_GO, previous INCOMPLETE, dataset split and checkpoint are preserved.\n\nThe raw worker/policy scope enum WITHIN_LOG_EXTENDED_EVALUATION is inherited from the prior runtime interface. This run reuses the already-executed selection and is classified retrospectively regardless of that legacy enum. Original selection role names and the copied historical QA protocol are reference semantics, not fresh-data claims. See raw_scope_addendum.json.\n')
    (root/'implementation_summary.md').write_text('# Delivery\n\nTwo minimal M1 hashing efficiency edits, no serving M2/M3 algorithm edit. External profiling/benchmark code reuses the existing full model, state manager, M2 WindowExecutor and M3Runtime. The executor was frozen before staged hyperparameter DEV. All experiments use the existing environment, GPU lock and CPU affinity excluding4/5. No training/download/installation.\n\n'+
        f"Executor regression/equivalence validation passed, but the final selected policy risk gate is {verdict['selected_policy_retrospective_quality_gate']}: {quality['new_severe']} new severe primary error(s). The configuration remains a frozen retrospective research candidate and is not enabled for production.\n\n"+
        f"Actual full-model frame calls={counts['full_model_frame_calls']}, real prefill/decode skips={counts['actual_prefill_skips']}, unique frames={counts['distinct_frames']}. Main B→C={main_gain['speedup']:.6f}×; pure Native A→C={gains('addressable','three','A->C')['speedup']:.6f}×.\n\n"+
        'Instrumentation files: critical_path_probe.py (CPU/CUDA observers), critical_path.py (existing-worker adapter and explicit comparison controls), critical_path_analysis.py (offline scheduling/statistics/reporting). Production changes: source_digest.py and input_contract.py only. Complete live mutation and byte-compatible hash tests pass. Profiler traces/tensors are not versioned; byte-identical raw request receipts are gzip archived with hashes.\n')
    return dict(rows=rows,perf=perf,events=es,primary=primary,quality=quality,selected=selected,counts=counts,verdict=verdict)


def complete_report(root):
    result=final_report(root);perf=result['perf'];selected=result['selected'];q=result['quality'];counts=result['counts']
    before=read(root,'before_after_executor_comparison.json')['comparisons'][0]
    m1=read(root,'m1_parameter_results.json');m2=read(root,'m2_parameter_results.json');m3=read(root,'m3_parameter_results.json')
    pareto=read(root,'joint_quality_latency_pareto.json');module=read(root,'module_contribution.json')
    selected_m2=read(root,'selected_m2_full_vs_incremental.json')
    per_workload=[]
    for cell in read(root,'tuning_protocol.json')['workload_weights']:
        w,f=cell.split('/');points=[]
        for item in pareto['results']:
            measured=next(c for c in item['cells'] if (c['workload'],c['frequency'])==(w,f))
            points.append(dict(candidate=item['index'],C_ms=measured['C_ms'],C_correctness=measured['paired']['C_score'],
                scored=measured['paired']['scored'],UNKNOWN=measured['paired']['UNKNOWN'],loss_vs_matched_B=measured['paired']['loss'],new_severe=measured['paired']['new_severe']))
        front=[p['candidate'] for p in points if p['C_correctness'] is not None and not any(
            other['C_correctness'] is not None and other['C_ms']<=p['C_ms'] and other['C_correctness']>=p['C_correctness']
            and (other['C_ms']<p['C_ms'] or other['C_correctness']>p['C_correctness']) for other in points)]
        best={}
        for eps in (.01,.02,.03):
            admissible=[p for p in points if p['loss_vs_matched_B'] is not None and p['loss_vs_matched_B']<=eps and p['new_severe']==0]
            best[str(eps)]=min(admissible,key=lambda p:p['C_ms'])['candidate'] if admissible else None
        per_workload.append(dict(workload=w,frequency=f,points=points,known_quality_latency_frontier=front,best_index_by_budget=best))
    write(root/'workload_quality_latency_pareto.json',dict(cells=per_workload,descriptive_only=True,
        no_new_selection_or_GPU_experiments=True,global_policy_uses_predeclared_equal_weights=True))
    speed=lambda w='addressable',f='three',comp='B->C':next(p for p in perf['comparisons'] if (p['workload'],p['frequency'],p['comparison'])==(w,f,comp))
    group=lambda w='addressable',f='three',cond='C':next(g for g in perf['groups'] if (g['workload'],g['frequency'],g['condition'])==(w,f,cond))
    base2=next(r for r in m2['records'] if r['index']==0);chosen2=m2['selected']
    task=read(root,'task_quality.json');primary=group()
    module_quality=read(root,'module_task_quality.json')['primary_three_question_aggregates']['C']
    budgets=[]
    for budget,index in pareto['best_index_by_budget'].items():
        point=next((r for r in pareto['results'] if r['index']==index),None)
        budgets.append(f"{float(budget):.0%}: "+(f"DEV加权{point['weighted_C_ms']:.3f}ms/帧，候选{index}" if point else '无已评分预算内候选'))
    answers=[
        '光流与观测验证、投影/描述符、完整可变状态摘要、事务快照和对象关联共同构成。230ms不是固定常数，随场景、对象数量、历史规模及系统波动变化。分阶段Host/CPU、矩阵候选槽位、对象数见m1_cpu_profile.json。当前路径不是独立DQTrack网络。',
        '移除了每个标量上的重复NumPy/dataclass反射，以及为哈希构造临时tobytes的复制。没有取消完整状态检查、对象关联、合法观测、坐标变换或状态提交。',
        f"新进程主确认M1 {before['M1']['from_ms']:.3f}→{before['M1']['to_ms']:.3f}ms，平均减少{-before['M1']['delta_ms']:.3f}ms；三问pilot减少3.646ms。两日志和跨轮波动限制普遍性，不将所有差异归因于纯CPU代码。",
        f"主确认完整E2E {before['E2E']['from_ms']:.3f}→{before['E2E']['to_ms']:.3f}ms，差值CI={before['E2E']['delta_log_CI95_ms']}；pilot三问减少4.491ms。低频pilot保留一次1386ms precheck离群值，区间跨零，未宣称低频已可靠获益。没有移走同步来制造局部加速。",
        '窗口复用仅减少两个可缓存前缀块的部分工作；代表帧全局后缀仍需430–434ms。初始16帧中12帧因漂移触发后缀全量刷新，故1518个历史窗口跳过不能等价为整段视觉等比例加速。',
        '选中窗口才执行QKV、窗口注意力、投影和有效Token MLP；复用窗口使用受完整性约束的缓存。全局残差合法复用时，后缀从939个Kernel约430ms减到11个约0.905ms；这些是实际Profiler记录。',
        '代表帧完整前缀packing约0.7ms、部分输出重组约0.35ms、刷新评分Kernel约0.62ms、缓存发布Kernel约1.6ms；Copy和同步/Launch调用在m2_detailed_profile.json。嵌套时间不能相加，等待与GPU执行跨度不是Kernel累计时间。',
        '没有发现已跳过前缀仍全部重算后再替换的路径。Patch Embedding始终执行一次；必须刷新时全局后缀仍执行，这是当前依赖/漂移策略的必要工作。',
        f"同一M2执行器初始13对非重型Profiler帧的CUDA跨度493.175→459.060ms，1.0743×。冻结最优联合参数后的另一次对照为{selected_m2['performance']['from_ms']:.3f}→{selected_m2['performance']['to_ms']:.3f}ms，{selected_m2['performance']['speedup']:.4f}×，{selected_m2['performance']['paired_frames']}对；均是一日志诊断。输出特征偏差、Kernel和独立GT评分见m2_full_vs_incremental.json及selected_m2_full_vs_incremental.json，不能当作独立总体因果效应。",
        f"M2实现未修改，不能宣称代码优化收益。固定所选M1/M3后的DEV参数对照：M2 CUDA均值{base2['M2_cuda_ms']:.3f}→{chosen2['M2_cuda_ms']:.3f}ms；C完整E2E {base2['E2E_ms']['mean']:.3f}→{chosen2['E2E_ms']['mean']:.3f}ms。最终三日志联合系统结果另报，不把局部差异线性相加。",
        '更多前缀复用只影响前缀计算；全局后缀是否刷新支配大部分GPU时间。实际Kernel形状、计算时长、缓存开销和最终E2E比窗口数更有解释力。',
        '本轮不进一步改M2算法或窗口划分；保留单一执行器，有限调整现有预算/年龄/漂移参数后停止。',
        f"M1候选结果见m1_parameter_results.json。三问初筛选择{m1['selected']['m1']}；检测阈值改变保留状态数量和后续区域/事实，关联阈值改变链接。类别准确率与关联精度只覆盖可评分对象，删减状态后的覆盖下降一并报告。",
        f"测试了现有ordinary_budget、global_anchor_interval及P0预注册的global_drift_threshold。三问初筛选择{chosen2['m2']}；实际CUDA、刷新与跳过窗口由请求级收据记录，未用理论FLOPs代替。",
        f"仅考察现有min_confidence=.5及max_binding_risk=.5相对原配置的变化。M3选中候选{m3['selected_index']}；相同决策CPU核对后复用收据，变化决策执行GPU。未添加Temporal Risk、Completeness或答案缓存。",
        f"顺序筛选结果为M1={chosen2['m1']}、M2={chosen2['m2']}、M3候选{m3['selected_index']}。五负载等权联合确认最终选候选{selected['selected_index']}，M1={selected['m1']}、M2={selected['m2']}。以联合实测选择，未假定局部最优组合就是全局最优。",
        f"所选配置主三问C的M1已评分类别准确率={module_quality['M1_class_accuracy']}，关联精度={module_quality['association_precision']}；分子/分母、UNKNOWN链接、连续GT实例上的ID变化见module_task_quality.json。它是保守的部分GT匹配评价，不是标准MOT或完整身份认证；配置未通过新增严重错误门槛。",
        f"M2固定.3检测阈值的已评分类别准确率={module_quality['visual_class_accuracy']}，使用独立当前GT和原<2m/.5m唯一匹配标准，同时报告匹配覆盖。不能替代官方全量mAP/NDS。该worker没有独立车道评分分母，Map语义质量未认证。未将缺失标签记为正确。",
        f"正式对象三问C真实跳过{primary['actual_prefill_skips']}/{primary['questions']}次Prefill和Decode，比例{primary['actual_prefill_skips']/primary['questions']:.2%}。所有命中输入展开为0且LLM Forward为0；各任务及未命中质量分开报告。",
        '；'.join(budgets)+'。这是DEV的已评分条件工作点；正式GPU只评价冻结的默认2pp配置，UNKNOWN使全任务预算仍无法保证。',
        '；'.join(f"{f}: B={speed(f=f)['from_ms']:.3f}、C={speed(f=f)['to_ms']:.3f}ms/视觉帧，{speed(f=f)['speedup']:.4f}×" for f in ('every4','every2','one','three'))+'。所有无QA帧仍处理。',
        f"预设1/3结构化插槽mixed三问B→C={speed('mixed')['speedup']:.4f}×，log CI={speed('mixed')['speedup_log_CI95']}；前轮点估计1.0908×仅作跨轮描述性参照，不代表真实自然问题分布。",
        f"新对象三问C={primary['e2e_ms']['mean']:.3f}ms/帧，相比前轮1287.076ms的点估计变化为{primary['e2e_ms']['mean']-1287.076446:.3f}ms；本轮B→C={speed()['speedup']:.4f}×，前轮1.4078×。M3算法不变，改动涉及视觉固定成本与参数，不能称为新语言算法收益。",
        f"本轮纯Native A→C={speed(comp='A->C')['speedup']:.4f}×。旧A含M1成本的同口径伴随对照为{module.get('old_A_with_M1_to_C',{}).get('speedup')}×；与前轮1.9405×的比较必须使用后者，不能混淆基线口径。",
        f"主已评分B={q['B_score']:.2%}、C={q['C_score']:.2%}，标签{q['scored']}/{q['requests']}，loss={q['loss']:.6f}，新增严重={q['new_severe']}，最终独立门槛={result['verdict']['selected_policy_retrospective_quality_gate']}。全请求loss界={q['conservative_all_request_loss_bounds']}；绝对严重错误、UNKNOWN和其他负载另列。严重案例是快速回答把合法最近目标的前左说成后左；B只回答左侧，虽同为必要事实不完整的0分，却没有明确反向。不能据均值保证全任务或严格2%风险。",
        '代码收益由同输入/同参数的原M1与优化M1配对识别；M2执行器没有代码变化。M1/M2/M3参数效应由冻结执行器的DEV对照、五负载联合确认和最终ABC记录识别，module_contribution.json保留各自证据，未假定可加性。',
        '仅能称为本轮有限候选和预设等权目标下的已评分预算内最优工作点。未穷尽参数空间，DEV仅两日志，时延存在波动、标签不完备，不能证明现有实现的全局最优。',
        '可支持真实连续六相机状态维护、实际增量计算、零LLM Forward旁路及限定负载内完整系统质量—时延的系统论文证据；独立泛化TEST、自然业务分布、全量感知质量和严格罕见严重风险仍缺证据。历史NO_GO/INCOMPLETE不变，生产默认关闭。']
    assert len(answers)==28
    (root/'result_answers.md').write_text('# 28项结果回答\n\nRETROSPECTIVE_EVALUATION / INDEPENDENT_GENERALIZATION_INCOMPLETE\n\n'+'\n\n'.join(f'{i}. {s}' for i,s in enumerate(answers,1))+'\n')
    py='taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python'
    (root/'REPRODUCE.md').write_text('# Reproduce\n\nExisting OmniDrive environment, RTX3090, original checkpoint; no installs, download or training. GPU jobs acquire /tmp/omnidrive_m123_latency_gpu.lock and wait for occupancy. All invocations exclude CPU4/5.\n\nVerify retained results (restores byte-identical compressed receipts if needed, no inference):\n\n```sh\n'+py+' -m tools.worldstream_m123.critical_path_analysis verify\n```\n\n'+
        'For a fixed-input/configuration inference repetition, copy the frozen inputs/configuration records to a NEW output root: baseline_sources/, optimized_sources/, benchmark_v2_sources/, selection/protocol, GT annotation cache for OFFLINE analysis, policies, parameter-result/selection/freezes, historical manifest, candidate/job manifests and receipt-reuse mappings. Omit completed stage directories, execution logs, validation directories, receipt_archive.json and generated presentation reports. Replace policy path prefixes in copied job manifests with the new root and rebind rewritten manifests before execution; keep the original artifacts unchanged. Repetition of these104 already-used frames is regression/performance repetition, never new independent evaluation. Full must precede Incremental to regenerate temporary comparison tensors.\n\n'+
        'Stages in order: profile, baseline, executor_check, executor_clean, tune_m1, tune_m2, tune_m3, joint, selected_m2_ablation, final, feature_audit. Set baseline_m1=true in the initial profile/baseline replay jobs to invoke the original archived digest implementation; later before/after controls already carry this flag. For the original run tune_m3 had one changed candidate and32 new calls. Each stage uses its retained <stage>_jobs.json; execute with:\n\n```sh\n'+py+' -m tools.worldstream_m123.critical_path <stage> --root /tmp/omnidrive-critical-replay\n```\n\n'+
        'Worker source versions are retained: baseline_sources/.../critical_path.py for original profile/baseline; optimized_sources/.../critical_path.py for executor/M1 DEV; benchmark_v2_sources/.../critical_path.py for M2/M3 DEV, joint confirmation, selected M2 profiling and final measurements. Current worker corrects feature capture for all-mode returned tensors and was used in the48-call feature_audit. Replaying current worker reproduces inference/configuration with corrected observation, not the invalid legacy feature-witness fields. For historical controller reconstruction use the archived source versions in an isolated checkout and their execution_sources.json fingerprints. Serving M1/M2/M3 semantics and frozen policy are unchanged. Legacy digest functions remain external benchmark controls, never deployment fallbacks. See feature_audit_report.md and the explicit548-row legacy invalidity ledger.\n\n'+
        'For a fresh retrospective parameter search, retain the predeclared protocol and run select_m1/select_m2, prepare_m3/select_m3, prepare_joint/select_joint only after the corresponding DEV stage completes. Candidate0 reference B/C receipts already executed with identical configuration are explicitly reused rather than rerun. Do not choose new thresholds using final results. The report records the actual frozen policy and all unknowns.\n\n'+
        'Regression invocations:\n\n```sh\n'+py+' -m tools.worldstream_m123.answer_validity_validate --output /tmp/omnidrive-critical-replay/validation_cpu\n'+py+' -m tools.worldstream_m123.answer_validity_validate --gpu --output /tmp/omnidrive-critical-replay/validation_gpu\n```\n\n'+
        'After all stages, run analysis profile, executor, executor_clean, selected_m2_ablation and feature_audit, followed by report, archive, verify. Existing result reanalysis uses the same retained raw data and grader; inference replay timings vary with clocks/order/system load. Native A performs no M1/M2/M3 computation; the3Q A_with_M1 companion retains old accounting. Process peak memory can include inactive harness buffers and is not a minimal standalone footprint. Timing is compute service latency, not a real-time arrival/queue simulation. The final selected candidate FAILED its new-severe-error gate; do not retune from that case or enable production. Historical conclusions remain unchanged.\n')
    write(root/'report_sources.json',dict(analysis_sha256=sha256(__file__),grading_sha256=sha256('tools/worldstream_m123/continuous_stream_report.py')))


def restore_receipts(root):
    manifest=root/'receipt_archive.json'
    if not manifest.exists():return
    for item in read(root,manifest.name)['files']:
        raw=root/item['raw'];compressed=root/item['archive']
        assert sha256(compressed)==item['archive_sha256']
        if not raw.exists():
            temp=raw.with_suffix('.restoring')
            with gzip.open(compressed,'rb') as source,temp.open('wb') as target:shutil.copyfileobj(source,target)
            assert sha256(temp)==item['raw_sha256'];temp.replace(raw)
        assert sha256(raw)==item['raw_sha256']


def selected_m2_ablation(root):
    rows=stage_rows(root,'selected_m2_ablation');pairs=[]
    ref={r['frame_id']:r for r in rows if r['job_spec']['label']=='selected_M2_Full' and not r['heavy_profiler']}
    for r in rows:
        if r['job_spec']['label']!='selected_M2_Incremental' or r['heavy_profiler']:continue
        full=ref[r['frame_id']]
        pairs.append(dict(log_id=r['log_id'],frame_id=r['frame_id'],from_ms=full['stages']['m2_total']['gpu_span_ms'],to_ms=r['stages']['m2_total']['gpu_span_ms'],
            full_E2E_ms=full['e2e_wall_ms'],incremental_E2E_ms=r['e2e_wall_ms'],
            output_difference=feature_difference_receipt(r),skipped_windows=r['m2']['actual_skipped_windows'],
            skipped_suffix_blocks=r['m2']['global_suffix_skipped_blocks']))
    es,quality=per_label_quality(root,rows)
    profiles={p.stem:json.loads(p.read_text()) for p in (root/'selected_m2_ablation/profile_output').glob('*.json')}
    result=dict(selected_policy_sha256=sha256(root/'selected_joint_policy.json'),
        scope='Selected frozen joint policy; one16-frame DEV stream. Heavy frames1/6 excluded from lightweight timing pairs; no population claim.',
        performance=diagnostic_performance(pairs),pairs=pairs,profiles=profiles,quality=quality,
        paired_answers=paired_quality(es,'selected_M2_Full','selected_M2_Incremental'),visual_quality=visual_task_quality(root,rows),
        expensive_work_scope='Profile range critical/M2/_execute has comparable actual prefix+global kernel work; other ranges include patch/selection/cache. No kernel time inferred from window counts.',
        all_feature_differences=[dict(frame_id=r['frame_id'],**feature_difference_receipt(r)) for r in rows if r['job_spec']['label']=='selected_M2_Incremental'])
    write(root/'selected_m2_full_vs_incremental.json',result)
    initial=read(root,'m2_full_vs_incremental.json')
    if initial['performance'].get('independent_logs',0)<2:
        initial['performance'].update(speedup_log_CI95=None,delta_log_CI95_ms=None,interval_scope='One-log diagnostic; between-log uncertainty is not estimable')
    initial['selected_joint_policy_ablation']=dict(report='selected_m2_full_vs_incremental.json',
        selected_policy_sha256=result['selected_policy_sha256'],performance=result['performance'],quality=quality,paired_answers=result['paired_answers'])
    write(root/'m2_full_vs_incremental.json',initial)
    print(json.dumps(dict(stage='selected_M2_ablation',performance=result['performance'],quality=quality)))


def feature_audit(root):
    import re
    rows=stage_rows(root,'feature_audit')
    assert all(r.get('feature_receipt_version')=='returned_m2_nchw_v2' for r in rows)
    full={r['frame_id']:r for r in rows if r['job_spec']['label']=='audit_M2_Full'}
    incremental={r['frame_id']:r for r in rows if r['job_spec']['label']=='audit_M2_Incremental'}
    reference={r['frame_id']:r for r in rows if r['job_spec']['label']=='audit_M1_reference'}
    def normalized(value):
        if isinstance(value,str):return re.sub(r':g\d+:',':gRESET:',value)
        if isinstance(value,list):return [normalized(v) for v in value]
        if isinstance(value,dict):return {k:normalized(v) for k,v in value.items() if k not in ('epoch','state_version')}
        return value
    checks=[];pairs=[]
    for token,inc in incremental.items():
        ref=reference[token];baseline=full[token]
        checks.append(dict(frame_id=token,global_reuse=inc['m2']['global_suffix_skipped_blocks']>0,
            exact_returned_features_equal=inc['visual_quality']['feature']==ref['visual_quality']['feature'],
            feature_difference=ref['visual_quality']['equivalence_feature_difference'],
            world_equal=normalized(inc['committed_world'])==normalized(ref['committed_world']),
            detections_equal=all(inc['visual_quality'][k]==ref['visual_quality'][k] for k in ('boxes','labels','scores')),
            IDs_equal=normalized(inc['visual_quality']['committed_anchor_ids'])==normalized(ref['visual_quality']['committed_anchor_ids']),
            decisions_equal=inc['m2']['stages']==ref['m2']['stages'],
            answers_equal=[q['tokens'] for q in inc['requests']]==[q['tokens'] for q in ref['requests']]))
        pairs.append(dict(frame_id=token,log_id=inc['log_id'],from_ms=baseline['stages']['m2_total']['gpu_span_ms'],to_ms=inc['stages']['m2_total']['gpu_span_ms'],
            actual_returned_feature_difference=inc['visual_quality']['full_feature_difference'],global_reuse=inc['m2']['global_suffix_skipped_blocks']>0))
    assert len(checks)==16 and any(c['global_reuse'] for c in checks)
    assert all(all(v for k,v in c.items() if k.endswith('_equal')) for c in checks),'Return-value feature audit failed'
    es,quality=per_label_quality(root,rows)
    write(root/'corrected_feature_audit.json',dict(status='PASS',capture='Actual returned M2 NCHW tensor for every frame, holder cleared and asserted current',
        exact_M1_feature_and_state_checks=checks,all_M1_checks_equal=True,
        Full_vs_Incremental_actual_features=pairs,diagnostic_performance=diagnostic_performance(pairs),quality=quality,
        paired_answers=paired_quality(es,'audit_M2_Full','audit_M2_Incremental'),
        reuse_Incremental_as_optimized_M1_arm=True,additional_model_calls=48,
        earlier_main_timings_predictions_answers_and_forward_counts_unchanged=True))
    invalid=[]
    for path in sorted(root.glob('*/requests.jsonl')):
        for r in lines(path):
            if r['condition']!='A' and not valid_feature_witness(r):
                invalid.append(dict(stage=r['stage'],job=r['job'],frame_id=r['frame_id'],frame_index=r['frame_index']))
    write(root/'legacy_feature_witness_validity.json',dict(invalid_legacy_feature_rows=len(invalid),invalid_rows=invalid,
        only_fields_invalid=['visual_quality.feature','visual_quality.full_feature_difference'],
        timing_detections_QA_quality_forwards_skips_valid=True,raw_receipts_never_rewritten=True,
        replacement='corrected_feature_audit.json'))
    (root/'feature_audit_report.md').write_text('# Corrected feature witness\n\nThe legacy worker attached its feature witness to the final global-block stage callback. That callback is absent when a global suffix is reused, so those feature-only receipts could refer to the last full-suffix frame. This was an external observation defect; the actual M2 return still fed the native heads and M3 correctly. Timings, detections, independent GT scores and LLM forward/skip counts remain valid as measured.\n\n'+
        f"{len(invalid)} legacy feature rows are explicitly marked invalid in legacy_feature_witness_validity.json; raw records are preserved. A48-call continuous audit captures the actual returned NCHW tensor every frame, validates sixteen complete Full/Incremental feature comparisons, and reuses the Incremental arm to compare original/optimized M1. All sixteen returned-feature, state, ID, detection, refresh and answer pairs are exactly equal for the M1 implementation control, including residual-reuse frames. No new parameter selection or serving-runtime change occurred.\n\n"+
        'See corrected_feature_audit.json for the complete current-feature errors versus Full and the partial independent QA score. Main A/B/C data were not rerun or replaced; the corrected observation was isolated after their frozen source guard passed.\n')
    print(json.dumps(dict(status='PASS',M1_exact_pairs=16,additional_calls=48,invalid_legacy_feature_rows=len(invalid))))


def verify(root):
    restore_receipts(root)
    protocol=read(root,'protocol_freeze.json')
    assert sha256(root/'extended_frame_selection.json')==protocol['selection_sha256']
    assert sha256(root/'tuning_protocol.json')==protocol['protocol_sha256']
    archived=['baseline_sources','optimized_sources','benchmark_v2_sources']
    checked=0;calls=0;skips=0
    for status in root.glob('*/run_status.json'):
        record=json.loads(status.read_text());stage=status.parent
        rows=lines(stage/'requests.jsonl')
        assert len(rows)==record['full_M123_frame_calls']
        assert record['status']=='COMPLETED' or (record['status']=='NOT_RUN_ALL_CANDIDATES_DECISION_EQUIVALENT' and not rows)
        sources=stage/'execution_sources.json'
        if sources.exists():
            for p,h in json.loads(sources.read_text()).items():
                assert sha256(p)==h or any((root/a/p).exists() and sha256(root/a/p)==h for a in archived),p
        jobs=defaultdict(list)
        for r in rows:
            assert r['execution_completed'];jobs[r['job']].append(r)
            assert r['observed_llm_forward_calls']==sum(q['llm_forward_calls'] for q in r['requests'])
            if r['condition']!='A':assert r['m1_commit_completed'] and r['m2']['execution_completed']
            if r['job_spec'].get('native_without_m1'):assert not r['m1_commit_completed'] and not r['m2_completed'] and r['committed_world'] is None
            if not r['questions']:assert not r['answers'] and r['observed_llm_forward_calls']==0
            for q in r['requests']:
                if q['llm_skipped']:
                    assert q['llm_forward_calls']==0 and q['layout_preparation_ms']==0
                    assert not q['prefill_completed'] and not q['decode_completed'];skips+=1
            calls+=1
        for rs in jobs.values():
            expected=rs[0]['job_spec']['scene']['tokens']
            assert [r['frame_id'] for r in rs]==expected
            assert [r['frame_index'] for r in rs]==list(range(len(expected)))
            assert all(a['timestamp']<b['timestamp'] for a,b in zip(rs,rs[1:]))
            if rs[0]['condition']!='A':assert [r['m1_commit_count'] for r in rs]==list(range(1,len(rs)+1))
        checked+=1
    freeze=read(root,'optimized_executor_freeze.json')
    for p,h in freeze['sources'].items():
        assert sha256(p)==h or (p=='tools/worldstream_m123/critical_path.py' and sha256(root/'optimized_sources'/p)==h),p
    for name in ('validation_cpu','validation_gpu'):
        assert read(root,name+'/summary.json')['exit_code']==0
    assert read(root,'before_after_executor_comparison.json')['all_equal']
    assert read(root,'executor_pilot_comparison.json')['all_equal']
    assert read(root,'corrected_feature_audit.json')['status']=='PASS'
    assert calls==read(root,'gpu_execution_counts.json')['full_model_frame_calls']
    assert skips==read(root,'gpu_execution_counts.json')['actual_prefill_skips']
    policy=read(root,'selected_joint_policy.json')
    assert policy['status']=='FROZEN_RETROSPECTIVE_POLICY' and not policy['automatic_deployment']
    assert sha256(root/'joint_quality_latency_pareto.json')==policy['selection_sha256']
    assert sha256(root/'optimized_executor_freeze.json')==policy['executor_freeze_sha256']
    final_freeze=read(root,'final_protocol_freeze.json')
    assert sha256(root/'selected_joint_policy.json')==final_freeze['policy_sha256']
    assert sha256(root/'final_jobs.json')==final_freeze['final_jobs_sha256']
    assert sha256(root/'selected_m2_ablation_jobs.json')==final_freeze['selected_m2_jobs_sha256']
    joint_freeze=read(root,'joint_jobs_freeze.json')
    assert sha256(root/'joint_jobs.json')==joint_freeze['jobs_sha256']
    assert sha256(root/'joint_candidates.json')==joint_freeze['candidate_sha256']
    assert sha256(root/'selection_sources/selector_before_joint.py')==joint_freeze['selector_source_sha256']
    final=[r for r in stage_rows(root,'final') if r['job_spec']['label']!='A_with_M1'];matched=defaultdict(dict)
    for r in final:matched[(r['workload'],r['frequency'],r['frame_id'])][r['condition']]=r
    assert len(matched)==5*72
    for cell in matched.values():
        assert set(cell)=={'A','B','C'} and cell['A']['questions']==cell['B']['questions']==cell['C']['questions']
        for key in ('m1','m2'):assert cell['B']['job_spec'][key]==cell['C']['job_spec'][key]
        assert cell['C']['job_spec']['m1']==policy['m1'] and cell['C']['job_spec']['m2']==policy['m2']
        assert json.loads(Path(cell['C']['policy_file']).read_text())==policy['m3']
        assert all(not r['heavy_profiler'] for r in cell.values())
    sources=read(root,'report_sources.json');assert sha256(__file__)==sources['analysis_sha256']
    assert sha256('tools/worldstream_m123/continuous_stream_report.py')==sources['grading_sha256']
    historical=read(root,'historical_manifest.json')
    assert all(sha256(p)==h for p,h in historical.items())
    result=dict(status='PASS',stages_checked=checked,actual_full_model_calls=calls,actual_prefill_decode_skips=skips,
        historical_files_unchanged=len(historical),CPU=read(root,'validation_cpu/summary.json')['counts'],CUDA=read(root,'validation_gpu/summary.json')['counts'],
        independent_generalization='INDEPENDENT_GENERALIZATION_INCOMPLETE')
    write(root/'integrity_verification.json',result);print(json.dumps(result))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['profile','archive','select_m1','select_m2','m3_screen','executor','executor_clean','profile_reports','prepare_joint','select_joint','prepare_m3','select_m3','report','verify','selected_m2_ablation','feature_audit']);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    if a.stage in ('select_m1','select_m2'):select_stage(a.root,'tune_'+a.stage.split('_')[1])
    elif a.stage=='executor_clean':executor_comparison(a.root,'executor_clean')
    else:{'profile':summarize_profile,'archive':archive,'m3_screen':m3_screen,'executor':executor_comparison,'profile_reports':profile_reports,'prepare_joint':prepare_joint,'select_joint':select_joint,'prepare_m3':prepare_m3,'select_m3':select_m3,'report':complete_report,'verify':verify,'selected_m2_ablation':selected_m2_ablation,'feature_audit':feature_audit}[a.stage](a.root)


if __name__=='__main__':main()
