"""Third-round source-derived quality and performance report; missing stays missing."""
import hashlib,json,re,gzip,pickle
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .audit import write,stats
from .v2_checks import rows
from .report import summarize,prefill_token_accounting
from .v2_report import paired
from .profile_analysis import analyze,union_ms
from .v3 import ROOT,quality
RUNS=('main','long','auto','multi','cost_support','auto_supported')


def m2_bound(root,data):
    from worldstream.m2.dependency_planner import DependencyPlanner,patch_updates
    spec=json.loads((root/'eva_structure.json').read_text())
    blocks=[SimpleNamespace(window_size=b['window_size'],use_residual_block=b['residual']) for b in spec['blocks']]
    planner=DependencyPlanner(SimpleNamespace(blocks=blocks))
    patch=spec['patch'];projection=SimpleNamespace(kernel_size=tuple(patch['kernel_size']),stride=tuple(patch['stride']),padding=tuple(patch['padding']),dilation=tuple(patch['dilation']))
    samples=[]
    for path in sorted(root.glob('*/oracle_frames/*.pkl.gz')):
        with gzip.open(path,'rb') as f:snapshot=pickle.load(f)
        masks=torch.from_numpy(np.stack(list(snapshot['masks'].values())))
        token=patch_updates(masks,projection);plans=planner.plan(token)
        samples.append(dict(frame=snapshot['frame'].frame_id,run=path.parents[1].name,
            invalid_pixel_fraction=float(masks.float().mean()),invalid_patch_fraction=float(token.float().mean()),
            blocks=[dict(block=p.block,reason=p.reason,window_size=p.window_size,total_tokens=p.tokens.numel(),
                optimistic_unchanged_tokens=int((~p.tokens).sum()),active_units=int(p.updated.sum()),units=p.updated.numel()) for p in plans]))
    measured=[r for r in data if r['mode'] in ('m12','m123')]
    result=dict(eva=spec,cacheable_exact_prefix=planner.cacheable_prefix(),samples=samples,
        actual_reused_cameras=sum(r['frame']['m2'].get('actual_reused_cameras',0) for r in measured),
        actual_skipped_windows=sum(r['frame']['m2'].get('actual_skipped_windows',0) for r in measured),
        m1_costs=stats([r['frame']['m1'].get('m1_total_including_adapters_ms',0) for r in measured]),
        optimistic_skip_fraction=float(np.mean([sum(b['optimistic_unchanged_tokens'] for b in s['blocks'])/sum(b['total_tokens'] for b in s['blocks']) for s in samples])) if samples else None,
        exact_saved_gpu_ms=0. if all(r['frame']['m2'].get('actual_reused_cameras',0)==0 and r['frame']['m2'].get('actual_skipped_windows',0)==0 for r in measured) else None,
        scope='M1 masks provide optimistic physical support only. Neural bitwise input, positions and temporal dependencies still required; unchanged tokens here are not legal reuse approvals',
        decision='Full EVA; no end-to-end exact skip benefit' if all(r['frame']['m2'].get('actual_reused_cameras',0)==0 for r in measured) else 'Inspect actual dependency evidence')
    write(root/'m2_feasibility_bound.json',result);return result


def task_metrics(root,data):
    from worldstream.m3.post_perception import direct_query
    samples=[r for r in data if r['mode'] in ('task_full','task_compact')]
    records=[];funnel=Counter();reasons=Counter();queries=activations=consumed=0;reductions=[]
    references={(r['run'],r['index']):r for r in data if (r['run']=='main' and r['mode']=='certified_serial') or (r['run']=='multi' and r['mode']=='original')}
    for row in samples:
        ref=references[(row['run'],row['index'])]
        before=prefill_token_accounting(ref['frame'])['physical_forward_tokens'];after=prefill_token_accounting(row['frame'])['physical_forward_tokens']
        def prefill_wall(r):return sum(s['wall_ms'] for s in r['frame']['stages'] if s['name']=='llm_prefill')
        reductions.append(dict(run=row['run'],mode=row['mode'],token=row['frame']['identity']['frame_id'],
            original_effective_prefill_tokens=before,candidate_effective_prefill_tokens=after,
            effective_prefill_tokens_reduced=before-after if before is not None and after is not None else None,
            prefill_wall_ms_reduced=prefill_wall(ref)-prefill_wall(row),
            e2e_wall_ms_reduced=ref['frame']['e2e_wall_ms']-row['frame']['e2e_wall_ms'],
            e2e_service_matched=row['frame']['service']==ref['frame']['service']))
        post=row.get('post_perception',{});d=post.get('diagnostics') or {}
        for field in ('historical_candidate_count','physical_verified_count','task_verified_candidates'):funnel[field]+=d.get(field,0)
        for task,q in zip(row['tasks'],row['frame']['m3'].get('requests',[])):
            state=q.get('state_retrieval')
            if state is None:continue
            queries+=1;activations+=bool(state.get('admitted'));consumed+=len(state.get('records',[])) if state.get('admitted') else 0
            if not state.get('admitted'):reasons[state.get('reason','unknown')]+=1
            records.append(dict(run=row['run'],mode=row['mode'],scene=row['frame']['identity']['scene_id'],token=row['frame']['identity']['frame_id'],
                question=task['question'],retrieval=state,task_quality=task,post_diagnostics=d,
                per_frame_prefill=prefill_token_accounting(row['frame']),verification_wall_ms=post.get('verification_wall_ms')))
    long_quality=None;long_tokens_equal=True
    if (root/'long_task_supply/queries.jsonl').exists():
        from .v2_state_quality import main as score_long
        long_quality=score_long(root,'long_task_supply')
        for row in rows(root/'long_task_supply/queries.jsonl'):
            long_tokens_equal=long_tokens_equal and all(row['complete_question_token_equality'])
            d=row['verified_state_diagnostics']
            for field in ('historical_candidate_count','physical_verified_count','task_verified_candidates'):funnel[field]+=d.get(field,0)
            for question,q in zip(row['questions'],row['diagnostics'].get('requests',[])):
                state=q.get('state_retrieval')
                if state is None:continue
                queries+=1;activations+=bool(state.get('admitted'));consumed+=len(state.get('records',[])) if state.get('admitted') else 0
                if not state.get('admitted'):reasons[state.get('reason','unknown')]+=1
                records.append(dict(run='long_task_supply',mode=row['mode'],scene=row['scene'],token=row['token'],question=question,
                    retrieval=state,post_diagnostics=d,per_frame_prefill=None,verification_wall_ms=None,
                    scope='Supplemental native checkpoints, excluded from ordinary E2E and independent-scene counts'))
    token_quality=[r for r in samples if r['frame']['quality']['exact_token_equality']]
    funnel['task_relevant_record_occurrences']=sum(len(r['retrieval'].get('records',[])) for r in records)
    funnel['actually_consumed_record_occurrences']=consumed
    common=dict(actual_queries=queries,actual_query_activations=activations,actual_consumed_record_occurrences=consumed,
        task_query_hit_rate=activations/queries if queries else None,funnel=dict(funnel),rejection_reasons=dict(reasons),
        post_verification_wall_ms=stats([r.get('post_perception',{}).get('verification_wall_ms',0) for r in samples]),
        measured_input_compute_reductions=reductions,
        source='Actual original Dataset questions and real M1/head/LLM executions',question_replacement=False,
        current_head_is_prediction=True,independent_physical_certificate=False,records=records)
    write(root/'two_stage_state_results.json',common);write(root/'task_consumption_results.json',common)
    quality_result=dict(measured_method_frames=len(samples),complete_token_equal_frames=len(token_quality),
        exact_quality_pass=bool(samples) and len(token_quality)==len(samples) and long_tokens_equal,
        state_replacement_quality_pass=bool(activations) and len(token_quality)==len(samples) and long_tokens_equal,
        task_quality_fallback_only=not bool(activations),
        qa_metrics=[dict(run=r['run'],mode=r['mode'],tasks=r['tasks']) for r in samples],long_history_task_quality=long_quality,
        direct_queries=dict(actual_eligible_requests=0 if not activations else None,actual_executions=0,
            numeric_error_m=None,latency_ms=None,reason='No independently certified original local requests; no synthetic questions/answers introduced'),
        interpretation='Full fallback agreement is not evidence that state replacement is accurate; token F1 is descriptive, not semantic correctness')
    write(root/'task_quality_results.json',quality_result);return common,quality_result


def profile(root):
    profiles=[]
    for row in json.loads((root/'probes/profile_manifest.json').read_text()):
        result=analyze(row['path'],root/'traces'/(row['mode']+'_summary.json'),row.get('worker'))
        result['mode']=row['mode'];profiles.append(result)
    return profiles


def critical_timeline(root):
    record=next(r for r in json.loads((root/'probes/profile_manifest.json').read_text()) if r['mode']=='refined_overlap')
    events=json.loads(Path(record['path']).read_text())['traceEvents']
    names=('stream/m1_input_adapter_and_d2h','stream/eva','stream/detection_head','stream/map_head','stream/m1_commit','stream/llm_prefill','stream/llm_decode','stream/post_perception_task_verification')
    selected=[e for e in events if e.get('name') in names and e.get('ph')=='X']
    origin=min(e['ts'] for e in selected)
    cpu=[dict(name=e['name'],start_ms=(e['ts']-origin)/1000,duration_ms=e['dur']/1000,thread=e['tid']) for e in selected]
    gpu=[dict(start_ms=(e['ts']-origin)/1000,duration_ms=e['dur']/1000) for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
    launches={e['args']['correlation']:e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})}
    stage_gpu={}
    for name in names:
        scopes=[e for e in selected if e['name']==name];kernels=[]
        for e in events:
            if e.get('cat')!='kernel' or e.get('ph')!='X':continue
            host=launches.get(e.get('args',{}).get('correlation'))
            if host and any(s['ts']<=host['ts']<=s['ts']+s['dur'] for s in scopes):kernels.append(e)
        stage_gpu[name]=dict(actual_kernel_sum_ms=sum(e['dur'] for e in kernels)/1000,
            actual_kernel_union_ms=union_ms([(e['ts'],e['ts']+e['dur']) for e in kernels]),kernel_count=len(kernels))
    sync=next(e for e in events if e.get('name')=='m1_overlap_clock_sync');w=record['worker'];offset=sync['ts']-w['clock_sync_s']*1e6
    worker=dict(start_ms=(w['worker_start_s']*1e6+offset-origin)/1000,duration_ms=w['worker_ms'],queue_delay_ms=w['queue_delay_ms'],clock_uncertainty_us=sync['dur'])
    result=dict(source=record['path'],cpu=cpu,gpu_kernels=gpu,worker=worker,stage_gpu=stage_gpu,
        scope='Actual profiler CPU scopes and GPU kernels; worker monotonic-clock alignment, uncertainty preserved')
    write(root/'traces/critical_path_timeline.json',result)
    return result


def figures(root,summary,pairs,risk,task,m2,profiles,timeline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=root/'figures';created=[]
    def save(name,draw):
        fig,ax=plt.subplots(figsize=(10,4.7));draw(ax);fig.tight_layout();fig.savefig(dest/(name+'.png'),dpi=160);fig.savefig(dest/(name+'.pdf'));plt.close(fig);created.append(name)
    for run in RUNS:
        selected=[r for r in summary if r['run']==run]
        def draw(ax):
            ax.bar(range(len(selected)),[s['e2e_wall_ms']['p50'] for s in selected],label='P50')
            ax.scatter(range(len(selected)),[s['e2e_wall_ms']['p95'] for s in selected],c='black',label='P95')
            ax.set_xticks(range(len(selected)));ax.set_xticklabels([s['mode']+'\n'+s['service'] for s in selected],rotation=55,ha='right');ax.set_ylabel('Complete service wall ms');ax.legend();ax.set_title(run)
        save(run+'_wall',draw)
    selected=[p for p in pairs if p['mode'] in ('overlap','refined_overlap','refined_serial','unified_auto')]
    def draw(ax):
        for i,p in enumerate(selected):
            y=p['mean_scene_increment_ms'];lo,hi=p['scene_cluster_bootstrap_increment_95ci_ms'];ax.errorbar(i,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o')
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(len(selected)));ax.set_xticklabels([p['run']+'/'+p['mode']+' vs '+p['reference'] for p in selected],rotation=55,ha='right');ax.set_ylabel('Scene-paired increment ms; bootstrap 95% CI')
    save('scene_paired_increment',draw)
    def draw(ax):
        for candidate in risk['candidates']:
            for split,marker in (('calibration','o'),('heldout','x')):
                s=candidate['splits'][split]
                if s['physical_false_acceptance_rate'] is not None:
                    y=s['physical_false_acceptance_rate'];ci=s['scene_cluster_bootstrap_far_95ci'] or s['false_acceptance_95ci_trial_descriptive']
                    ax.errorbar(s['valid_state_recovery_rate'],y,yerr=[[max(0,y-ci[0])],[max(0,ci[1]-y)]],marker=marker,
                        label=candidate['candidate']['name']+'/'+split)
        ax.axhline(.05,c='red',ls='--',label='Frozen FAR budget');ax.set_xlabel('Valid state recovery');ax.set_ylabel('Physical false acceptance fraction');ax.legend(fontsize=8)
        ax.text(.02,.95,'Zero accepted support is unknown risk, not a zero-FAR point',transform=ax.transAxes,va='top',fontsize=9)
    save('risk_coverage',draw)
    def draw(ax):
        counts=task['funnel'];ax.bar(range(len(counts)),list(counts.values()));ax.set_xticks(range(len(counts)));ax.set_xticklabels(list(counts),rotation=35,ha='right');ax.set_ylabel('Actual task-mode record occurrences')
    save('task_supply_funnel',draw)
    def draw(ax):
        for p in profiles:
            ordered=sorted(p['decode_layers'].items(),key=lambda r:int(r[0]));ax.plot([int(i) for i,v in ordered],[v['per_decode_token_kernel_ms'] for i,v in ordered],label=p['mode'])
        ax.set_xlabel('Actual decoder layer');ax.set_ylabel('GPU kernel sum per decode token ms');ax.legend()
    save('decode_actual_kernel_layers',draw)
    def draw(ax):
        for run in ('main','long'):
            a=[s for s in m2['samples'] if s['run']==run]
            if a:
                b=a[0]['blocks'];ax.plot([r['block'] for r in b],[np.mean([s['blocks'][i]['optimistic_unchanged_tokens']/s['blocks'][i]['total_tokens'] for s in a]) for i in range(len(b))],label=run)
        ax.set_xlabel('Actual EVA block');ax.set_ylabel('Optimistic unchanged token fraction');ax.legend();ax.set_title('Physical support is not a neural dependency certificate')
    save('eva_dependency_bound',draw)
    def draw(ax):
        mapping={'stream/m1_input_adapter_and_d2h':0,'stream/eva':1,'stream/detection_head':3,'stream/map_head':3,'stream/m1_commit':4,'stream/llm_prefill':5,'stream/llm_decode':6}
        for e in timeline['cpu']:
            if e['name'] in mapping:ax.broken_barh([(e['start_ms'],e['duration_ms'])],(mapping[e['name']]-.3,.6),facecolors='grey')
        w=timeline['worker'];ax.broken_barh([(w['start_ms'],w['duration_ms'])],(1.7,.6),facecolors='royalblue')
        ax.broken_barh([(e['start_ms'],e['duration_ms']) for e in timeline['gpu_kernels']],(6.7,.6),facecolors='green')
        ax.set_yticks(range(8));ax.set_yticklabels(['Input snapshot','EVA CPU scope','M1 worker','Heads CPU scope','M1 commit','Prefill CPU scope','Decode CPU scopes','Actual GPU kernels']);ax.set_xlabel('Profiler-aligned wall ms');ax.set_title('Separate refined-overlap control; not an E2E timing sample')
    save('critical_path_timeline',draw)
    resources=[s for s in summary if s['run']=='main']
    def draw(ax):
        ax.bar(range(len(resources)),[s['memory']['gpu_peak_allocated']/1024**3 for s in resources]);ax.set_xticks(range(len(resources)));ax.set_xticklabels([s['mode'] for s in resources],rotation=55,ha='right');ax.set_ylabel('Peak GPU allocated GiB')
    # Actual summary schema is checked before plotting; no fabricated memory zeros.
    if resources and all('gpu_peak_allocated' in s['memory'] for s in resources):save('gpu_memory',draw)
    write(root/'figure_manifest.json',dict(figures=created,formats=['png','pdf'],source='Actual v3 records only'))
    return created


def main():
    root=ROOT;data=[];checks=[]
    for name in RUNS:
        status=json.loads((root/name/'status.json').read_text());assert status['status']=='completed' and status['weights_unchanged']
        selected=rows(root/name/'frames.jsonl')
        for r in selected:r['run']=name
        data.extend(selected);result=quality(root/name)
        baseline={r['index']:r for r in selected if r['mode']=='m1'}
        for row in selected:
            if row['mode'] not in ('m12','m123') or row['index'] not in baseline:continue
            ref=baseline[row['index']];q=row['frame']['quality']
            p=dict(mode=row['mode'],reference='m1',index=row['index'],scene=row['frame']['identity']['scene_id'],
                state_equal=q['m1_state_digest']==ref['frame']['quality']['m1_state_digest'],token_equal=row['generated_ids']==ref['generated_ids'],
                feature_memory_equal=q['exact_perception_memory_digest'] and all(v['max_abs']==0 and not v['shape_mismatches'] for v in q['numerical'].values()))
            p['passed']=p['state_equal'] and p['token_equal'] and p['feature_memory_equal'];result['pairs'].append(p)
        result['passed']=bool(result['pairs']) and all(p['passed'] for p in result['pairs'])
        write(root/name/'extended_quality.json',result);checks.append(dict(run=name,**result))
    from .identity import token_index,resolve_sequence
    from collections import defaultdict
    with open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb') as handle:infos=token_index(pickle.load(handle)['infos'])
    executed=defaultdict(list)
    for r in data:executed[r['frame']['identity']['frame_id']].append(r['frame']['identity'])
    coverage=[]
    for run in RUNS:
        for spec in json.loads((root/run/'protocol.json').read_text())['scenes']:
            _,resolved=resolve_sequence(spec,infos,executed);coverage.append(dict(run=run,scene=spec['scene'],**resolved))
    write(root/'sequence_coverage.json',dict(sequences=coverage,stable_token_scene_timestamp_join=True))
    del infos
    summary=summarize(data);pairs=[]
    for s in summary:
        selected=[r for r in data if r['run']==s['run'] and r['mode']==s['mode']]
        s['exact_typed_feature_memory']=all(r['frame']['quality']['exact_perception_memory_digest'] for r in selected)
        s['exact_full_output_equality']=s['exact_full_output_equality'] and s['exact_typed_feature_memory']
    references=dict(refined_serial='m1',overlap='m1',refined_overlap='m1',certified_overlap='certified_serial',unified_auto='m1',m12='m1',m123='m1',task_full='certified_serial',task_compact='certified_serial',native_decode='original',prefix='original',batch='original',adaptive='original')
    for run in RUNS:
        selected=[r for r in data if r['run']==run]
        for mode,reference in references.items():
            p=paired(selected,mode,reference)
            if p:pairs.append(dict(run=run,**p))
        p=paired(selected,'refined_overlap','overlap')
        if p:pairs.append(dict(run=run,**p))
    write(root/'overlap_performance.json',dict(methods=summary,paired=pairs,strict_quality=checks,all_cpu_affinity='0-3,6-31'))
    constrained=[]
    for p in pairs:
        s=next(s for s in summary if s['run']==p['run'] and s['mode']==p['mode'])
        state_pairs=[x for check in checks if check['run']==p['run'] for x in check['pairs'] if x['mode']==p['mode']]
        passed=s['exact_full_output_equality'] and (all(x['passed'] for x in state_pairs) if state_pairs else s['service']=='current_output')
        if p['mode'] in ('task_full','task_compact'):passed=False  # task replacements require independent physical/task support
        constrained.append(dict(p,strict_quality_pass=passed,quality_constrained_time_ratio=p['scene_mean_time_ratio'] if passed else None,
            independently_positive=passed and p['scene_cluster_bootstrap_ratio_95ci'][0]>1))
    write(root/'quality_constrained_speedup.json',dict(comparisons=constrained,hardware_scope='Only new healthy-core v3 measurements',
        quality_scope='Complete output/temporal and same-service state-maintenance equality on fixed cohorts; absolute physical risk and semantic replacement certification are separate'))
    task,tquality=task_metrics(root,data);m2=m2_bound(root,data);profiles=profile(root);timeline=critical_timeline(root)
    m2['actual_eva_kernel_control']=timeline['stage_gpu']['stream/eva']
    m2['ideal_remove_entire_eva_upper_ms']=timeline['stage_gpu']['stream/eva']['actual_kernel_sum_ms']
    m2['upper_bound_scope']='Removing all measured EVA kernels is an intentionally loose ideal control, not achievable savings or E2E speedup. Actual legal skipped work remains independently counted.'
    write(root/'m2_feasibility_bound.json',m2)
    workers=[dict(run=r['run'],mode=r['mode'],identity=r['frame']['identity'],worker=r['frame']['m1']['worker']) for r in data if r['frame']['m1'].get('worker')]
    critical=dict(profiles=profiles,worker_samples=workers,
        worker_queue_delay_ms=stats([s['worker']['queue_delay_ms'] for s in workers]),
        actual_precheck_worker_wall_ms=stats([s['worker']['worker_ms'] for s in workers]),
        main_join_wait_ms=stats([s['worker']['join_wait_ms'] for s in workers]),
        post_perception_task_verification_wall_ms=task['post_verification_wall_ms'],
        stage_wall=[dict(run=s['run'],mode=s['mode'],wall=s['components_wall_ms'],cpu=s['components_cpu_process_ms']) for s in summary],
        scope='Real kernel duration/union from separate traces; complete E2E samples excluded from profiler controls. Queue/worker/join use monotonic clocks, not sums of stage P50')
    write(root/'overlap_critical_path.json',critical)
    decode=json.loads((root/'probes/decode_backend_results.json').read_text());write(root/'decode_backend_results.json',decode)
    resources=json.loads((root/'probes/resource_controls/resource_controls.json').read_text())
    thread_resources=json.loads((root/'thread_resource_controls.json').read_text())
    write(root/'resource_controls.json',dict(basic=resources,thread_variations=thread_resources,
        memory_scope='20ms RSS controls and allocator GPU peaks; ordinary CPU RSS is process lifetime high-water, not per-mode incremental allocation'))
    batch=json.loads((root/'batch_probes/batch_quality_results.json').read_text())
    counterexamples=[dict(count=c['count'],maximum=c['maximum'],adaptive=c['adaptive'],question=q['question'],first_divergent_token=q['first_divergent_token'])
        for c in batch['cases'] for q in c['quality'] if not q['equal']]
    write(root/'multi_question_quality.json',dict(methods=[s for s in summary if s['run']=='multi'],paired=[p for p in pairs if p['run']=='multi'],bounded_failed_carrier_diagnosis=batch,
        known_complete_token_counterexamples=counterexamples,global_exact_approved=False,
        scope='Fresh fixed-cohort equality does not eliminate retained known-counterexample failures or grant general deployment approval'))
    auto=[r for r in data if r['run']=='auto_supported' and r['mode']=='unified_auto'];cost=json.loads((root/'cost_calibration.json').read_text())
    admission=dict(calibration=cost,actual_method_frames=len(auto),actual_admitted_frames=sum(r['frame']['admission'].get('allowed') is True for r in auto),
        actual_overlap_frames=sum(r['frame']['m1'].get('overlap_active') is True for r in auto),
        strict_quality_pass=next(c['passed'] for c in checks if c['run']=='auto_supported'),
        incorrect_quality_admissions=sum(r['frame']['admission'].get('allowed') is True and not r['frame']['quality']['exact_token_equality'] for r in auto),
        fallback_reasons=dict(Counter(r['frame']['admission'].get('reason','unknown') for r in auto)),
        actual_lookup_wall_ms=stats([r['frame']['admission']['lookup_wall_ms'] for r in auto]),
        future_request_fallback_only=True,statistical_bound_not_realtime=True)
    from worldstream.runtime import UnifiedRuntime
    costs=json.loads((root/'cost_certificate.json').read_text())['records'];qualities=json.loads((root/'quality_certificate.json').read_text())['records']
    admission['incorrect_scope_admission_count']=sum(r['frame']['admission'].get('allowed') is True and not any(
        UnifiedRuntime.admit(r['frame']['admission']['workload_key'],c,q)[0] for c in costs for q in qualities) for r in auto)
    admission['incorrect_admission_count']=sum(r['frame']['admission'].get('allowed') is True and
        (not r['frame']['quality']['exact_token_equality'] or not any(UnifiedRuntime.admit(r['frame']['admission']['workload_key'],c,q)[0] for c in costs for q in qualities)) for r in auto)
    admitted_indices={r['index'] for r in auto if r['frame']['admission'].get('allowed') is True}
    admitted_pairs=[r for r in data if r['run']=='auto_supported' and r['index'] in admitted_indices]
    admission['admitted_workload_paired_performance']=paired(admitted_pairs,'unified_auto','m1')
    native_auto={r['index']:r for r in data if r['run']=='auto_supported' and r['mode']=='m1'}
    admission['admitted_frame_wall_increments_ms']=[r['frame']['e2e_wall_ms']-native_auto[r['index']]['frame']['e2e_wall_ms'] for r in auto if r['frame']['admission']['allowed']]
    admission['observed_admitted_positive_increment_frames']=sum(v>0 for v in admission['admitted_frame_wall_increments_ms'])
    admission['overall_with_fallback_paired_performance']=next(p for p in pairs if p['run']=='auto_supported' and p['mode']=='unified_auto')
    admission['performance_scope']='Report admitted-key conditional benefit and whole workload with fallbacks separately. One admitted scene cannot establish independent-scene generalization.'
    initial_auto=[r for r in data if r['run']=='auto' and r['mode']=='unified_auto']
    admission['initial_validation']=dict(actual_frames=len(initial_auto),actual_admitted_frames=sum(r['frame']['admission'].get('allowed') is True for r in initial_auto),
        calibration='initial_cost_calibration/cost_calibration.json',scope='Original No-Go and full fallback preserved; never retrospectively treated as admitted')
    write(root/'auto_admission_results.json',admission)
    risk=json.loads((root/'m1_risk_coverage_results.json').read_text());diagnosis=json.loads((root/'m1_false_accept_diagnosis.json').read_text())
    made=figures(root,summary,pairs,risk,task,m2,profiles,timeline)
    refined=[p for p in constrained if p['mode']=='refined_overlap' and p['reference']=='overlap']
    decisions=dict(m1=dict(decision=risk['decision'],independent_risk_pass=risk['independent_quality_pass']),
        two_stage=dict(actual_activations=task['actual_query_activations'],quality_certified=tquality['state_replacement_quality_pass'] and risk['independent_quality_pass'] and risk['selected']=='bidirectional_depth',
            physical_candidate_configuration_matches=risk['selected']=='bidirectional_depth',
            decision='No-Go' if not task['actual_query_activations'] or not risk['independent_quality_pass'] else 'Needs task certification'),
        refined_overlap=dict(comparisons=refined,default_enabled=False),auto=admission,
        m2=dict(exact_reused_cameras=m2['actual_reused_cameras'],skipped_windows=m2['actual_skipped_windows']),
        decode=dict(probe_tokens_equal=all(q['complete_token_equality'] for q in decode['quality']),default_enabled=False),
        batch=dict(exact_required=True,default_enabled=False,known_counterexamples=counterexamples,global_exact_approved=False),
        complete_system_closed_loop_approved=bool(risk['independent_quality_pass'] and risk['selected']=='bidirectional_depth' and tquality['state_replacement_quality_pass'] and m2['actual_reused_cameras']>0),old_results_overwritten=False)
    write(root/'go_nogo_decisions.json',decisions)
    lines=['# OmniDrive 第三轮系统优化实验报告','',
        f"本轮实际测量 {len(data)} 个方法帧，{len({r['frame']['identity']['frame_id'] for r in data})} 个唯一真实帧，{len({r['frame']['identity']['scene_id'] for r in data})} 个场景。全部使用 `taskset -c 0-3,6-31` 与 OmniDrive；不混入第二轮未绑定 CPU 的时延。所有模型/LoRA、原始问题、Sampling/Greedy 合同保持不变。",
        '主实验按冻结场景顺序切分 calibration/heldout；新 Auto 场景不参与校准。长流用于独立持续维护验证，绝不重复同帧增加独立样本数。风险阈值和候选网格在 protocol.json 中先冻结。','',
        '主实验的精确问题长度分层各只有 2 个校准/2 个验证场景，未达到冻结的 3/3 支持门槛，初次校准与 Auto 完整回退单独保留。cost_support_protocol.json 在附加 GPU 输出前冻结新场景 136–143 与新 Auto 144–147，不替换原场景或降低门槛。新成本分析同时执行原先冻结的 [.49,.51] 暖帧间隔域；初次零批准结果归档于 initial_cost_calibration。额外性能场景不进入原物理风险选择集。','',
        '| Run | 方法 | 服务 | 帧 | P50/P95 ms | 完整输出 Exact |','|---|---|---|---:|---:|---|']
    for s in summary:lines.append(f"| {s['run']} | {s['mode']} | {s['service']} | {s['measured_frames']} | {s['e2e_wall_ms']['p50']:.2f}/{s['e2e_wall_ms']['p95']:.2f} | {s['exact_full_output_equality']} |")
    lines+=['','## 同服务场景配对','',
        '场景 bootstrap 10,000 次。相同输出不代表相同历史职责；certified M1 与原 M1 的状态变化另报，任务替代候选不属于 Exact KV。','',
        '| Run / 方法 / 参考 | 平均配对增量 ms | 95% CI | 时间比 |','|---|---:|---|---:|']
    for p in pairs:lines.append(f"| {p['run']} / {p['mode']} / {p['reference']} | {p['mean_scene_increment_ms']:.2f} | {p['scene_cluster_bootstrap_increment_95ci_ms']} | {p['scene_mean_time_ratio']:.4f} |")
    lines+=['','## 物理风险与覆盖','',
        '单侧 95% FAR 上界为描述性 Clopper–Pearson；重复目标/帧相关，另外报告场景支持。缺少可判定身份不算真负，零接受的 FAR 和覆盖不足不能算成功。','',
        '| 候选 / split | Accepted scorable | 错误 | FAR | FAR upper95 | Valid recovery | 支持场景 | 风险通过 |','|---|---:|---:|---|---|---|---:|---|']
    for candidate in risk['candidates']:
        for split,s in candidate['splits'].items():lines.append(f"| {candidate['candidate']['name']} / {split} | {s['accepted_labeled']} | {s['physical_errors']} | {s['physical_false_acceptance_rate']} | {s['far_upper95']} | {s['valid_state_recovery_rate']} | {s['accepted_support_scenes']} | {s['physical_risk_pass']} |")
    lines+=['','## 十二项验收','',
        f"1. 旧 29 例误接受的逐例来源/相机/对应点/残差/Oracle 与图片已保存。主要机制分类为 {diagnosis['primary_mechanisms']}；属于两个旧场景的探索性诊断。相似纹理不能定位三维中心；Identity Switch/前景归属没有独立证据时保留未证实。",
        f"2. 独立风险结论：{risk['decision']}。当前批准候选={risk['selected']}，不以零接受充当零风险成功；静态/动态差异、真实分母和附加验证成本见 m1_risk_coverage_results.json。",
        f"3. Post-Perception 只核对真实当前 Head 预测与已冻结物理历史，不更新 Precheck/M2 证据。实际候选漏斗={dict(task['funnel'])}；预测一致不是独立物理真值。",
        f"4. 原始任务查询 {task['actual_queries']}，实际激活 {task['actual_query_activations']}，消费记录出现数 {task['actual_consumed_record_occurrences']}。不替换原问题以制造成功。",
        '5. Prompt/Prefill 与验证开销逐请求记录。没有实际状态激活时不声称减少 LLM 计算；Direct Query 没有独立认证支持就不执行，数值误差和延迟保持未测。',
        '6. Refined Overlap 对 Current Overlap 的真实净收益与区间见上表。遮挡表向量化复现原逐对几何判断，完整状态摘要、来源、感知、时序记忆与 Token 逐帧验证；新物理门则分别与自己的串行版本比较。',
        f"7. 成本校准批准 {cost['approved_keys']} 键；新场景实际准入 {admission['actual_admitted_frames']}/{len(auto)}，重叠 {admission['actual_overlap_frames']}，严格质量={admission['strict_quality_pass']}，错误质量准入={admission['incorrect_quality_admissions']}。只在 calibration 实际观测的帧间隔范围准入；统计均值界不是实时截止保证。",
        f"8. M2 实际 Camera 复用 {m2['actual_reused_cameras']}、Window 跳算 {m2['actual_skipped_windows']}；准确 EVA Block 依赖范围与物理掩码的乐观上限见 m2_feasibility_bound.json。没有缩小 Window 或绕过 Global Attention。",
        '9. 新单/多问题真实 Exact 与速度见质量约束表；2/3/5 问题的原失败 Carrier 逐层/同历史 Logits 诊断重新执行。仅几毫秒的通用 Decode 不归因于状态缓存；质量不同仍 No-Go，未添加无证据的大型推理后端。',
        '10. Overlap 的调度贡献与状态消费/视觉跳算贡献分别记录。若物理风险或实际任务消费尚未通过，三个模块仍未形成经质量认证的完整缓存闭环。',
        '11. 原生自回归 Decode 的小 GEMM、动态 KV 和 Launch、EVA Global Attention 的依赖扩散，以及全局语义问题与局部几何状态的输入需求差异，是主要结构边界。RGB 背景一致性无法解决历史定位偏差。',
        '12. 保留已证明有效的 Overlap 与严格依赖执行边界；后续应围绕具有明确可验证输入需求的任务/对象抽象推进，而不是继续优化零命中的 Window Packing 或降低风险阈值。','',
        '## 测量与限制','',
        'Full Carrier、History+Full、Task Compact 均以原始实际问题运行；不合格输入保持 Full Carrier。Full 回退的输出等价不证明状态替代质量。QA Token F1 是描述性重叠，不当作语义或安全证书。',
        '`certified_serial/certified_overlap` 是开启额外几何拒绝门的实验名称，是否取得物理风险认证以 m1_risk_coverage_results.json 为准；名称本身不表示风险已通过。新几何门的实际有效 Anchor 集与离线 Replay 另作逐帧对照。',
        '性能表的 Exact 字段只表示该固定输入集合上的完整输出检查；多问题的新场景即使一致，也不能抹去原失败 Carrier 的数值/完整 Token 反例。已知反例单列于 multi_question_quality.json，未授予 Prefix/Batch 通用生产许可。',
        '同服务时延包含输入准备、GPU 同步、实际输出与 Token 捕获；定时后的大张量审计和物理快照、独立 Oracle、原始 Logits 与 Profiler 单独运行。Snapshot/观测等基础成本与所有共同插桩均计费；不相加组件 P50。',
        'Cold 指同一个已加载模型上的冷流/缓存首帧，Warm 指该连续流后续帧；checkpoint 加载和实验配置安装不计入逐帧 E2E，也不当作冷进程启动性能。各场景采用固定轮换方法顺序，额外 Native Reference 不增加独立数据数。',
        'CPU RSS 的逐帧字段是进程终生高水位；独立 20ms RSS 采样和线程变化的控制实验见 resource_controls.json，不把共同加载模型的内存当作各优化的净增量。长历史第 8/16 帧的五问题补充请求和 Direct Query 可行性单列，不计入普通 E2E 或新增独立场景。',
        '最坏观测增量、场景均值上界和预先固定分位数独立报告；Auto 缓存只缓存匹配候选，每次重新计入准入开销，模型/执行/CPU/证书内容变化都会改变键。已推进时序 Head 后不进行同帧重放，回退只对后续请求或尚未开始的优化生效。',
        f"生成 {len(made)} 组 PNG/PDF 图；原始 profiler traces、线程亲和性、温度频率和共享进程记录在 traces/、probes/ 与 logs/environment_monitor.jsonl。权重参数版本/类型每个运行均检查，最终全文件哈希另检。默认不启用未获证书的新研究选项。"]
    (root/'experiment_report.md').write_text('\n'.join(lines)+'\n')
    write(root/'report_manifest.json',dict(status='completed',measured_method_frames=len(data),unique_frames=len({r['frame']['identity']['frame_id'] for r in data}),figures=made,
        raw_sha256={str(root/name/'frames.jsonl'):hashlib.sha256((root/name/'frames.jsonl').read_bytes()).hexdigest() for name in RUNS}))


if __name__=='__main__':main()
