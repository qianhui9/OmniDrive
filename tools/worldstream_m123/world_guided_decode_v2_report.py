"""Measured v2 A/B/C comparisons with scene pairing and explicit qualification."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from tools.worldstream_m123.benchmark.common import write,sha256
from tools.worldstream_m123.world_guided_decode_v2_quality import task_quality,world_accuracy,qa_pairs,compare_facts,facts


def lines(path):return [json.loads(s) for s in path.read_text().splitlines()]
def average(v):return float(np.mean(v)) if len(v) else None


def scene_ci(pairs,field):
    groups=defaultdict(list)
    for p in pairs:groups[p['scene_id']].append(p[field])
    means=np.array([np.mean(v) for v in groups.values()])
    if len(means)<2:return None
    rng=np.random.RandomState(20261028)
    boot=np.mean(rng.choice(means,(2000,len(means))),axis=1)
    return np.percentile(boot,[2.5,97.5]).tolist()


def compare_system(trials,counts=(1,3)):
    summary=[];paired=[]
    for workload in ('natural','fixed'):
        for count in counts:
            selected=[r for r in trials if r['workload']==workload and r['question_count']==count]
            grouped=defaultdict(dict)
            for row in selected:grouped[(row['round'],row['scene_id'],row['frame_id'])][row['condition']]=row
            assert len(grouped)==36 and all(set(v)=={'A','B','C'} for v in grouped.values())
            for a,b in (('A','B'),('B','C'),('A','C')):
                ps=[]
                for key,g in grouped.items():
                    x,y=g[a],g[b]
                    ps.append(dict(workload=workload,question_count=count,round=key[0],scene_id=key[1],frame_id=key[2],
                        from_condition=a,to_condition=b,cold=x['cold_scene_frame'],
                        delta_e2e_ms=y['e2e_wall_ms']-x['e2e_wall_ms'],
                        from_e2e_ms=x['e2e_wall_ms'],to_e2e_ms=y['e2e_wall_ms'],
                        from_m3_ms=x['m3_wall_ms'],to_m3_ms=y['m3_wall_ms'],
                        from_decode_ms=sum(q['decode_wall_ms'] for q in x['requests']),
                        to_decode_ms=sum(q['decode_wall_ms'] for q in y['requests']),
                        from_output_tokens=sum(len(q['tokens']) for q in x['requests']),
                        to_output_tokens=sum(len(q['tokens']) for q in y['requests'])))
                paired.extend(ps)
                per_round=[]
                for r in range(3):
                    part=[p for p in ps if p['round']==r]
                    per_round.append(sum(p['from_e2e_ms'] for p in part)/sum(p['to_e2e_ms'] for p in part))
                per_scene={}
                for scene in sorted({p['scene_id'] for p in ps}):
                    part=[p for p in ps if p['scene_id']==scene]
                    per_scene[scene]=sum(p['from_e2e_ms'] for p in part)/sum(p['to_e2e_ms'] for p in part)
                summary.append(dict(workload=workload,question_count=count,comparison=a+'->'+b,pairs=len(ps),
                    e2e_speedup=sum(p['from_e2e_ms'] for p in ps)/sum(p['to_e2e_ms'] for p in ps),
                    m3_speedup=sum(p['from_m3_ms'] for p in ps)/sum(p['to_m3_ms'] for p in ps),
                    decode_wall_speedup=sum(p['from_decode_ms'] for p in ps)/sum(p['to_decode_ms'] for p in ps),
                    delta_e2e_ms=average([p['delta_e2e_ms'] for p in ps]),delta_e2e_ci95_scene_ms=scene_ci(ps,'delta_e2e_ms'),
                    round_e2e_speedups=per_round,scene_e2e_speedups=per_scene,
                    source_tokens=sum(p['from_output_tokens'] for p in ps),target_tokens=sum(p['to_output_tokens'] for p in ps),
                    steady_e2e_speedup=sum(p['from_e2e_ms'] for p in ps if not p['cold'])/sum(p['to_e2e_ms'] for p in ps if not p['cold'])))
    modules=[]
    for workload in ('natural','fixed'):
        for count in counts:
            for condition in ('A','B','C'):
                rs=[r for r in trials if (r['workload'],r['question_count'],r['condition'])==(workload,count,condition)]
                qs=[q for r in rs for q in r['requests']]
                modules.append(dict(workload=workload,question_count=count,condition=condition,frames=len(rs),
                    e2e_wall_ms=average([r['e2e_wall_ms'] for r in rs]),m3_wall_ms=average([r['m3_wall_ms'] for r in rs]),
                    m1_wall_ms=average([r['m1_wall_ms'] for r in rs]),m2_wall_ms=average([r['m2_wall_ms'] for r in rs]),
                    m1_including_adapters_ms=average([r['m1_including_adapters_ms'] for r in rs]),
                    heads_wall_ms=average([r['heads_wall_ms'] for r in rs]),
                    committed_snapshot_wall_ms=average([r['snapshot_wall_ms'] for r in rs]),
                    prefill_cuda_ms_per_question=average([q['prefill_cuda_ms'] for q in qs]),
                    decode_cuda_ms_per_question=average([q['decode_cuda_ms'] for q in qs]),
                    decode_wall_ms_per_question=average([q['decode_wall_ms'] for q in qs]),
                    prefill_host_ms_per_question=average([q['prefill_host_ms'] for q in qs]),
                    decode_host_ms_per_question=average([q['decode_host_ms'] for q in qs]),
                    fps=len(rs)*1000/sum(r['e2e_wall_ms'] for r in rs),qps=len(qs)*1000/sum(r['e2e_wall_ms'] for r in rs),
                    peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in rs),
                    cold_scene_frame_wall_ms=average([r['e2e_wall_ms'] for r in rs if r['cold_scene_frame']]),
                    steady_frame_wall_ms=average([r['e2e_wall_ms'] for r in rs if not r['cold_scene_frame']])))
    return summary,paired,modules


def draft_analysis(root,trials,development_root=None):
    results=[]
    for workload in ('natural','fixed'):
        for count in sorted({r['question_count'] for r in trials}):
            for condition in ('B','C'):
                rs=[r for r in trials if (r['workload'],r['question_count'],r['condition'])==(workload,count,condition)]
                qs=[q for r in rs for q in r['requests']]
                active=[q for q in qs if q['draft_status']=='target_verification']
                logs=[l for q in active for l in q['verification_log']]
                accepted=sum(q.get('draft_accepted_tokens',0) for q in active)
                positive=[q for q in active if q.get('draft_accepted_tokens',0)>=2]
                results.append(dict(workload=workload,question_count=count,condition=condition,requests=len(qs),
                    exact_available=sum(q['draft_exact_available'] for q in qs),
                    semantic_candidates=sum(q['draft_semantic_available'] for q in qs),
                    semantic_only_candidates=sum(q['draft_semantic_available'] and not q['draft_exact_available'] for q in qs),
                    verification_started=len(active),verification_start_rate=len(active)/len(qs),
                    candidate_tokens=sum(q.get('draft_candidate_tokens',0) for q in qs),accepted_tokens=accepted,
                    accepted_per_verification=accepted/len(logs) if logs else None,
                    accepted_at_least_two_requests=len(positive),token_effective_coverage=len(positive)/len(qs),
                    target_forward_calls=sum(q['target_forward_calls'] for q in qs),verification_calls=len(logs),
                    first_rejection_positions=dict(Counter(str(l['rejection_position']) for l in logs if l['rejection_position'] is not None)),
                    reasons=dict(Counter(q['draft_decision_reason'] for q in qs)),
                    scheduler_wall_ms=average([q['draft_selection_wall_ms'] for q in qs]),
                    world_diff_wall_ms=average([q['world_diff_wall_ms'] for q in qs]),
                    metadata_wall_ms=average([q['metadata_wall_ms'] for q in qs]),
                    answer_management_wall_ms=average([q['draft_management_wall_ms'] for q in qs]),
                    crop_host_ms=average([q.get('kv_crop_host_ms',0.) for q in qs])))
    dev=lines((development_root or root)/'development_coverage.jsonl');qs=[q for r in dev for q in r['diagnostics']['requests']]
    ds=dict(requests=len(qs),exact_available=sum(q['draft_exact_available'] for q in qs),
        semantic_candidates=sum(q['draft_semantic_available'] for q in qs),
        verification_started=sum(q['draft_status']=='target_verification' for q in qs),
        accepted_tokens=sum(q.get('draft_accepted_tokens',0) for q in qs),
        accepted_at_least_two_requests=sum(q.get('draft_accepted_tokens',0)>=2 for q in qs),
        reasons=dict(Counter(q['draft_decision_reason'] for q in qs)))
    ds['token_effective_coverage']=ds['accepted_at_least_two_requests']/len(qs)
    ds['economic_local_coverage']='NOT_RUN: development replay has no paired native per-question cost; token acceptance alone is not net-cost coverage'
    write(root/'draft_retrieval_analysis.json',dict(status='MEASURED',same_48_qa_development=ds,heldout_summaries=results,
        note='Semantic candidates include exact matches. Candidate retrieval is not effective draft reuse. Token-effective means >=2 accepted draft tokens; economic coverage is separately measured against paired native full-request wall, never assumed from a hit.'))
    write(root/'draft_scheduler_analysis.json',dict(status='MEASURED',summaries=results,
        policy='One K8 verifier: exact B; C preserves exact-history proposals without relabeling UNKNOWN as stable, adds semantic proposals using committed dependencies; unqualified positive class overlap allows only a 4-token prefix, incomplete-scene semantic candidates remain disabled. Zero-acceptance cooldown uses past target evidence only.'))
    write(root/'draft_acceptance_analysis.json',dict(status='MEASURED',summaries=results))
    return results,ds


def report(root):
    status=json.loads((root/'run_status.json').read_text());assert status['status']=='COMPLETED'
    if (root/'p3/run_status.json').exists():
        p2trials=lines(root/'e2e_trials.jsonl');assert len(p2trials)==432
        p3=root/'p3'
        assert json.loads((p3/'run_status.json').read_text())['status']=='COMPLETED'
        trials=lines(p3/'e2e_trials.jsonl')
    else:
        p2trials=[];p3=root;trials=lines(root/'e2e_trials.jsonl')
    assert len(trials)==216
    assert all(r['execution_completed'] for r in trials)
    initial_summary,initial_pairs,initial_modules=compare_system(p2trials) if p2trials else ([],[],[])
    summary,pairs,modules=compare_system(trials,counts=(3,))
    drafts,dev=draft_analysis(root,trials,p3)
    evaluated,quality,critical=task_quality(root,trials)
    world_accuracy(root,trials,development_root=p3)
    # Net economic coverage needs paired current-question generation cost and
    # accepted tokens, with output length explicitly kept alongside the ratio.
    by={}
    for r in trials:by[(r['workload'],r['question_count'],r['round'],r['frame_id'],r['condition'])]=r
    economic=[]
    for r in trials:
        if r['condition']=='A':continue
        base=by[(r['workload'],r['question_count'],r['round'],r['frame_id'],'A')]
        for q,a in zip(r['requests'],base['requests']):
            if q['draft_status']!='target_verification':continue
            # CPU overhead is counted; native per-Q wall is not saved, so use
            # the fair post-prefill decode span, not an asymmetric M3 proxy.
            gain=(a['decode_wall_ms']-q['decode_wall_ms']-q['draft_selection_wall_ms']-q['world_diff_wall_ms']
                  -q['draft_management_wall_ms']-q['metadata_wall_ms']-r['snapshot_wall_ms']/r['question_count'])
            economic.append(dict(condition=r['condition'],workload=r['workload'],question_count=r['question_count'],
                round=r['round'],frame_id=r['frame_id'],question=q['question'],accepted=q.get('draft_accepted_tokens',0),
                net_decode_gain_ms=gain,native_tokens=len(a['tokens']),generated_tokens=len(q['tokens']),
                effective=q.get('draft_accepted_tokens',0)>=2 and gain>0))
    economic_summary=[]
    for d in drafts:
        events=[e for e in economic if (e['workload'],e['question_count'],e['condition'])==
                (d['workload'],d['question_count'],d['condition'])]
        economic_summary.append(dict(condition=d['condition'],workload=d['workload'],question_count=d['question_count'],
            total_requests=d['requests'],net_positive_accepted_requests=sum(e['effective'] for e in events),
            economic_local_coverage=sum(e['effective'] for e in events)/d['requests']))
    write(root/'effective_draft_coverage.json',dict(status='MEASURED_LOCAL_NET_COST',events=economic,summary=economic_summary,
        scope='Paired decode wall minus measured retrieval/diff/management/metadata and per-question committed snapshot share; full-request acceleration is independently measured in abc_comparison.json. Different sampling output lengths remain explicitly reported.'))
    write(root/'abc_comparison.json',dict(status='MEASURED',summary=summary,paired_frames=pairs,module_breakdown=modules,
        rounds=3,independent_scenes=3,full_model_frames=len(trials)+len(p2trials),primary_phase='P3 final, three questions/frame',
        initial_p2_summary=initial_summary,initial_p2_pairs=initial_pairs,initial_p2_module_breakdown=initial_modules,
        numerical_contract='APPROXIMATE_FP16',
        timing='All A/B/C start before the original model request, including M1/M2/heads/current visual assembly/generation/current-state/draft management/answer serialization; synchronize at both boundaries. Input collation and output analysis are outside equally.'))
    qmap=defaultdict(dict)
    for r in evaluated:qmap[(r['workload'],r['round'],r['scene_id'],r['frame_id'],r['question'])][r['condition']]=r
    qloss=[]
    for key,g in qmap.items():
        if set(g)=={'A','B','C'}:
            for condition in ('B','C'):
                qloss.append(dict(workload=key[0],scene_id=key[2],condition=condition,
                    loss=g['A']['score']-g[condition]['score']))
    quality_delta=[]
    for workload in ('natural','fixed'):
        for condition in ('B','C'):
            part=[q for q in qloss if q['workload']==workload and q['condition']==condition]
            quality_delta.append(dict(workload=workload,condition=condition,paired_scored_answers=len(part),
                fact_score_loss=average([q['loss'] for q in part]),scene_ci95_loss=scene_ci(part,'loss')))
    new_critical=[]
    for key,g in qmap.items():
        if 'A' not in g:continue
        native={f['fact']:f['correct'] for f in g['A']['facts']}
        for c in ('B','C'):
            if c not in g:continue
            for f in g[c]['facts']:
                critical_fact=f['fact'] in ('signal_color','relative_direction','motion_state') or g[c]['task']=='safety'
                if critical_fact and not f['correct'] and native.get(f['fact']) is True:
                    new_critical.append(dict(condition=c,workload=key[0],round=key[1],scene_id=key[2],frame_id=key[3],question=key[4],
                        fact=f,answer=g[c]['answer'],native_answer=g['A']['answer'],reference=g[c]['reference']))
    write(root/'quality_comparison.json',dict(status='MEASURED_PARTIAL',summary=quality_delta,
        quality_budget=.02,critical_findings=len(critical),new_critical_vs_native=new_critical,complete_driving_quality_qualified=False))
    # Actual first-token frequencies are diagnostics, not a tiny-sample proof
    # of distribution equivalence (only three rounds per question).
    frequency=defaultdict(lambda:defaultdict(Counter))
    for row in trials:
        if row['workload']!='natural':continue
        for q in row['requests']:
            frequency[(row['frame_id'],q['question'])][row['condition']][str(q['tokens'][0])]+=1
    write(root/'sampling_output_diagnostics.json',dict(status='MEASURED_SMALL_N',
        first_token_frequencies=[dict(frame_id=k[0],question=k[1],conditions={c:dict(counts) for c,counts in v.items()}) for k,v in frequency.items()],
        strict_target_distribution_equivalence=False,draws_per_question_per_condition=3,
        note='Original sampling params, raw output IDs and lengths retained; n=3 frequencies do not establish a law-equivalence test. Residual transition logic has separate real small-model evidence/regression.'))
    main=[s for s in summary if s['comparison']=='A->C' and s['question_count']==3]
    increment=[s for s in summary if s['comparison']=='B->C' and s['question_count']==3]
    e2e_positive=all(s['delta_e2e_ci95_scene_ms'][1]<0 for s in main)
    incremental_positive=all(s['delta_e2e_ci95_scene_ms'][1]<0 for s in increment)
    # Do not turn missing task coverage into a research-quality pass. A clear
    # failure to improve on B is separately a negative M1 contribution result.
    worse_than_b=any(s['delta_e2e_ci95_scene_ms'][0]>0 for s in increment)
    decision='NO_GO' if worse_than_b else 'INCOMPLETE'
    reason=('M1-guided C is reproducibly slower than history-only B in at least one primary workload; semantic candidates did not establish independent E2E benefit.'
        if worse_than_b else 'M1 incremental/full E2E and complete task-quality evidence do not establish the required positive result.')
    write(root/'go_no_go.json',dict(status=decision,reason=reason,exact_go=False,numerical_mode='APPROXIMATE_FP16',
        full_e2e_positive_both_workloads=e2e_positive,m1_incremental_positive_both_workloads=incremental_positive,
        independent_task_quality_coverage='PARTIAL',quality_deltas=quality_delta,
        final_policy_single_question_e2e='NOT_RUN: follow-up restricted to primary three-QA cohort after P2 single-QA controls',
        formal_configuration_modified=False,automatic_deployment=False))
    greedy=json.loads((p3/'greedy_quality_probe.json').read_text())['rows']
    gr=defaultdict(dict)
    for r in greedy:gr[r['frame_id']][r['condition']]=r
    matches=Counter();divergences=[]
    for token,g in gr.items():
        for condition in ('B','C'):
            for i,(a,b) in enumerate(zip(g['A']['tokens'],g[condition]['tokens'])):
                matches[(condition,'total')]+=1;matches[(condition,'identical')]+=a==b
                if a!=b:
                    offset=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
                    q=g['A']['questions'][i];reference=qa_pairs(token).get(q)
                    divergences.append(dict(condition=condition,frame_id=token,question=q,first_mismatch=offset,
                        native=g['A']['answers'][i],candidate=g[condition]['answers'][i],reference=reference,
                        reference_fact_check=compare_facts(facts(q,reference),facts(q,g[condition]['answers'][i])) if reference else None))
    write(root/'greedy_comparison.json',dict(status='MEASURED',counts={c:dict(total=matches[(c,'total')],identical=matches[(c,'identical')],
        verified_requests=sum(q.get('draft_status')=='target_verification' for r in greedy if r['condition']==c for q in r.get('diagnostics',{}).get('requests',[]))) for c in ('B','C')},divergences=divergences))
    numeric=json.loads((root/'numeric_replay/gpu_schedule_numeric_probe.json').read_text())
    flips=[e for e in numeric['events'] if e['serial_argmax']!=e['block_argmax']]
    (root/'numeric_stability_report.md').write_text(f'''# Numerical stability

The v1 actual-schedule witness is retained. Current v2 verifier replays exactly its earlier real frame/history tokens with an independent serial K/V reference: {len(flips)} argmax disagreement(s), details in numeric_replay/gpu_schedule_numeric_probe.json. First disagreement: {flips[0] if flips else 'not reproduced in this invocation'}. Current prefill, position/mask, accepted K/V length, pending correction/bonus, rejection crop, EOS and max length follow the same causal target computation. The v1 future-suffix invariance probe passed; updated CPU/GPU tests cover these contracts. No historical decode K/V or future/GT inputs are used.

Greedy heldout original QA identity counts: {dict((c,(matches[(c,'identical')],matches[(c,'total')])) for c in ('B','C'))}. Only C actually verified one request; B verified none. Agreement on no-draft requests is not evidence of speculative numerical equivalence. No full-model FP32 conversion or unmeasured precision mitigation was used. Batched FP16 cache arithmetic and tied/near-tied decisions remain an explicit numerical approximation.

Sampling retains installed HF processors/warpers and deterministic-draft acceptance p(d), rejection residual excluding d, and target bonus sampling. The v1 algebra, boundary tests and actual small-Llama joint-law test remain applicable; current regression re-exercises sampler logic. Real-model strict native distribution equality is NOT_CLAIMED. Sampling quality and token lengths are independently measured below the original contract (0.1/0.75, default top_k, max_new_tokens=320).

An initial v2 snapshot attempt incorrectly compared camera observation timestamps to lidar keyframe time. That interface error was fixed by matching each current CameraFrame.timestamp; an asynchronous-camera regression was added. The entire aborted attempt is preserved under initial_camera_timestamp_mismatch and excluded from these results. It was not treated as permissible numerical approximation.
''')
    runtime=json.loads((root/'runtime_before_after.json').read_text())['rows']
    old=[r['m3_wall_ms'] for r in runtime if r['implementation']=='old' and not r['source_frame']]
    new=[r['m3_wall_ms'] for r in runtime if r['implementation']=='new' and not r['source_frame']]
    write(root/'runtime_optimization.json',dict(status='MEASURED',old_m3_wall_ms=average(old),new_m3_wall_ms=average(new),
        saved_ms=average(old)-average(new),speedup=sum(old)/sum(new),paired_rounds=3,
        capture=False,prefix_planner_executed=False,metadata_cached_with_mutation_checks=True))
    (root/'implementation_summary.md').write_text(f'''# WSG-SD v2 implementation and measured decision

Decision: **{decision}**. {reason} This decision follows actual M1/E2E measurements, not a stop on FP16 token disagreement. Default formal configuration and pretrained weights remain unchanged.

The existing M3Runtime/PrefillExecutor owns one language execution path. History/world modes bypass the old prefix planner and persistent prefill K/V; expensive model/tokenizer identities are cached with real parameter/backend/config/vocabulary revision checks. Full runtime wall improved from {average(old):.3f} to {average(new):.3f} ms on the same three-question current carriers, three alternating rounds. This includes real host/runtime/cache costs, not only decode CUDA. The frozen v1 Runtime and current Runtime share the current executor in this isolation control; it is not a complete old-binary-versus-new-binary comparison. Accepted token IDs avoid one redundant GPU-to-CPU scalar read for EOS.

M1 publishes immutable committed snapshots after perception commit, with scene/frame/epoch/version/source and actual asynchronous camera support. Question dependencies cover category/existence/count/position/distance/motion/road/signals/safety/environment; ego mentions are excluded from target class extraction and pedestrian crossings are map dependencies. Deltas compare a candidate answer's committed source to the current commit, keeping UNKNOWN explicit. Qualified car identity uses the existing independently evaluated policy; map lane control points are not misrepresented as signal/crosswalk semantics. No M1 association/tracking or M2 algorithm was modified.

Answer memory retains exact matching and compatible rule-based semantic candidates, source world state, positive/negative prefix and target acceptance feedback. Positive existential proposals require evidence in both source/current snapshots; negative, count, safety, signal and missing-map dependencies are never declared stable from incomplete M1. The final policy retains B's exact-history proposal independently of unresolved semantics, and permits a four-token semantic category prefix from positive class overlap while retaining UNKNOWN identity/coverage. Candidates still receive fresh current-visual target verification. Metadata, retrieval/diff, answer management and snapshot overhead are measured separately.

Actual experiments retained here: P2 has {len(p2trials)} complete A/B/C requests (zero means NOT_RUN in a standalone final-policy reproduction). Final P3 has {len(trials)} three-QA A/B/C requests with three rotating orders. The same 24-frame/48-QA C development replay and 12 fresh scene/log-disjoint frames are used. Final-policy greedy probes use the same fresh current carriers/committed snapshots. Native-loader overview templates stay out of original-QA metrics; one known long-overview failure is retained in runtime/numeric controls. P2 single-QA results belong to the initial conservative policy; final-policy single-QA E2E was NOT_RUN and is not silently substituted.

Source bindings: execution_sources.json records the loaded code HEAD plus SHA256 of every measured source and the runner verifies unchanged bytes at completion. Original weights are rehashed by final GPU regression. Reproduce with REPRODUCE.md; raw rows, tokens, acceptance/crop logs and state provenance remain on this server.
''')
    (root/'world_delta_design.md').write_text('''# Committed question-aware state difference

freeze_committed_state reads manager.store only after matching frame ID, timestamp, committed_frame_ids, epoch and execution signature; a pending transaction is refused. Anchors are deeply immutable scalar/tuple records. Current support uses each camera exposure timestamp, not lidar keyframe time. Old retained anchors remain unobserved; current detection origin must be actual_perception. Pose transforms actual world centers into current ego coordinates. Source frame, state version, score and identity qualification remain separate claims.

Question-aware differences compare the actual historical answer provenance with current state, which may span more than one frame. The runtime retains the immediately previous committed snapshot too; when no history candidate exists it skips unnecessary diff work. Matched IDs, class/relative-position/motion changes, new observations, lost support, ambiguity, missing coverage and scene resets are represented. An observed ID is not a physical identity certificate. Car continuity may use the existing scene/log-disjoint semantic qualification. Counts/negation, full safety, environment, signals and unspecified map semantics retain UNKNOWN rather than treating no recorded objects as absence.

This result controls proposal availability/length only. It cannot return an answer, skip target verification, inherit historical decode K/V, alter a user question, drop visual tokens or change max_new_tokens. Zero-acceptance feedback is causal model output, not GT. Candidate availability, actual validation, token acceptance, local net decode gain and complete E2E benefit are separate reported quantities. Accuracy audit is offline and has independent nuScenes geometry matching plus explicit UNKNOWN denominators.
''')
    table=['| Workload | Q/frame | Comparison | E2E ratio | M3 ratio | Decode wall ratio | E2E change ms [scene CI95] |',
           '|---|---:|---|---:|---:|---:|---|']
    for s in summary:table.append(f"| {s['workload']} | {s['question_count']} | {s['comparison']} | {s['e2e_speedup']:.3f} | {s['m3_speedup']:.3f} | {s['decode_wall_speedup']:.3f} | {s['delta_e2e_ms']:.2f} {s['delta_e2e_ci95_scene_ms']} |")
    initial_table=['| P2 workload | Q/frame | Comparison | E2E ratio |','|---|---:|---|---:|']
    for s in initial_summary:initial_table.append(f"| {s['workload']} | {s['question_count']} | {s['comparison']} | {s['e2e_speedup']:.3f} |")
    (root/'e2e_performance_report.md').write_text('''# Complete M123 E2E

'''+f'''P2 retained requests: {len(p2trials)} (single/three QA; zero means not rerun). Final P3 requests: {len(trials)} (three QA). Both use three independent scene/log streams, four distinct chronological frames per scene, three rotating A/B/C orders. All original six cameras, native Object256+Ego1+Map256, current visual prefill, original questions and original sampling contract are retained. Natural QA uses original dataset questions; fixed QA repeats source-frame actual questions on changing images. Each condition has its OWN earlier model answers, never A's current answer. Each scene resets identical M1/M2/head memory; the first frame is separately marked cold.

All variants time the same outer model invocation, synchronize before/after and include M1/M2/heads, current assembly, language runtime, C snapshot/retrieval/diff and answer serialization. Data loading/collation and offline analysis are outside equally. No pre-expanded-carrier generation wall is substituted for complete M123 wall. C's post-commit snapshot cost is included in E2E and reported separately from M1 commit wall.

Ratios >1 favor the destination. Delta is destination minus source, so negative is faster. CI resamples paired scene means (three scenes only); this is limited evidence, not full nuScenes deployment coverage.

'''+ '\n'.join(table)+'\n\nP2 initial-policy controls, including single-QA (not final-policy single-QA qualification):\n\n'+'\n'.join(initial_table)+f'''

Decision **{decision}**: {reason} Cold/steady, per-scene/per-round values, module spans, output tokens, FPS/QPS and peak GPU allocation are in abc_comparison.json. Sampling output lengths vary; raw E2E and per-token/forward evidence must be read together. No answer truncation/forced token length is used.
''')
    (root/'decode_performance_report.md').write_text('''# Decode and runtime costs

See draft_acceptance_analysis.json for actual accepted/candidate tokens, verification calls, first rejection positions and module wall/CUDA spans. abc_comparison.json records A→B, B→C and A→C decode wall and complete M3, on the SAME full-request input boundary. Runtime control before/after isolates three actual current-carrier runs; runtime_optimization.json includes metadata/legacy cache/planner savings.

K8 is the only default candidate length, with a shorter prefix only for measured question-relevant changes; there is no K grid search or second executor. A verification normally includes one pending token, so eight candidates can require nine target input positions. Fresh current-frame full prefill occurs for A/B/C. Rejected suffix K/V is cropped and correction/bonus state is forwarded in the current context. Prefix reuse, same-frame visual-prefill sharing and old decode K/V are not credited as WSG gains.

Token-effective coverage (>=2 accepted tokens) is explicitly weaker than economic coverage. effective_draft_coverage.json includes paired local decode gain after scheduler/diff/management and output-token counts; actual full-request gain comes only from independently measured E2E. Candidates rejected for UNKNOWN/full-scene dependencies are recorded rather than manufactured as stable. P2 conservative gating produced no verification; exactly one P3 main-policy revision preserves B's exact-history proposal independently of unresolved world semantics and permits a four-token category prefix under positive class overlap. UNKNOWN stays explicit. No further rule/K search is used to force C gains.
''')
    (root/'quality_report.md').write_text(f'''# Independent task-quality evidence

Numerical mode: **approximate FP16**. No strict native sampling-distribution claim. Greedy observed identity counts: {dict((c,(matches[(c,'identical')],matches[(c,'total')])) for c in ('B','C'))}; divergence positions and independent reference fact checks are in greedy_comparison.json. Only one C request and no B request actually started verification in this probe, so these counts do not qualify speculative equivalence. The known long-overview numerical failure remains separately reported.

Sampling uses original settings (0.1/0.75/default top_k/320 tokens). task_quality.json grades only exact CURRENT-FRAME original QA labels using fixed binary/count/direction/motion/color/distance/road facts. Category and safety-action scores are reference fact recall, not exhaustive accuracy. Native answers are never truth. Fixed questions absent from current-frame labels remain missing; source-frame GT is never carried forward. Per-task denominators and missing references are explicit. Critical direction/motion/signal/safety findings are listed individually: {len(critical)} rule-level findings; {len(new_critical)} new failures relative to a fact-correct paired Native answer. No new labeled factual error establishes stale-history propagation here, but fixed-stream current-label gaps prevent an exhaustive audit.

Textual review of retained findings identifies omitted reference safety actions: preparing to stop for a changing light, and explicitly yielding to pedestrians. These are reference omissions, not proof that the generated advice causes a collision. All three conditions exhibit these omissions; repetitions across rounds are not independent incidents. Missing traffic-light/wet-road precautions remain visible in the full answers even where the action-set metric cannot score them. No image-based human safety certification was performed.

Paired fact-score changes and scene CIs: {quality_delta}. Candidate budget is 0.02 absolute score loss. These partially covered task-fact metrics do NOT establish that every requested driving quality category lies within two percentage points; complete driving-quality qualification remains INCOMPLETE. world_delta_accuracy.json separately audits real M1 current classes and cited identity edges against independent nuScenes annotations, with ambiguous/unmatched objects UNKNOWN rather than a scored zero or automatic success.

No full approximate deployment is enabled. A reproducible absence of M1 E2E increment can yield NO_GO independently of remaining quality coverage.
''')
    def find(workload,comparison,count=3):return next(s for s in summary if (s['workload'],s['comparison'],s['question_count'])==(workload,comparison,count))
    def dr(workload,condition,count=3):return next(s for s in drafts if (s['workload'],s['condition'],s['question_count'])==(workload,condition,count))
    answers=f'''# v2 最终问题回答

1. v1 固定分块探针与实际拒绝/裁剪顺序探针共同表明：FP16 批量 K/V 舍入可延续到单 Token 续写，在并列/近并列 logits 上翻转 argmax。v2 针对性重放发现 {len(flips)} 处分歧；因果、位置、pending 与裁剪规则保持正确。另发现并修复了新 M1 快照接口的异步相机/关键帧时间戳匹配错误，旧尝试全部排除。
2. Greedy heldout 原始 QA 一致统计见 greedy_comparison.json；Sampling 明确是近似模式，固定 GT 任务事实变化见 quality_comparison.json，未用 Native 或文本相似度充当真值。完整驾驶质量预算尚未资格化。
3. 三问题完整请求中的 K8 History B Decode wall A/B：自然 {find('natural','A->B')['decode_wall_speedup']:.3f}×，固定重复 {find('fixed','A->B')['decode_wall_speedup']:.3f}×。完整 Runtime 旧→新节省 {average(old)-average(new):.3f} ms（相同真实当前 carriers，三轮）。
4. 原来同一 48 QA 严格候选是 2/48=4.17%；v2 C 检索到 {dev['semantic_candidates']}/48 个兼容语义候选，启动验证 {dev['verification_started']}/48，至少接受两 Token 的覆盖 {dev['accepted_at_least_two_requests']}/48={dev['token_effective_coverage']:.2%}。该开发重放没有逐问题 Native 成本配对，净成本有效覆盖为 NOT_RUN；新自然流已配对的净成本有效覆盖为 0%。不把候选数或单纯接受率当净成本覆盖提升。
5. 新自然/三问 C 兼容候选 {dr('natural','C')['semantic_candidates']}/{dr('natural','C')['requests']}，启动验证 {dr('natural','C')['verification_started']}，接受 {dr('natural','C')['accepted_tokens']} Token；固定重复对应 {dr('fixed','C')['verification_started']} 次验证、{dr('fixed','C')['accepted_tokens']} Token。是否产生有效接受增量，必须与 B 和净成本一起读，不能仅看检索数量。
6. 新自然/三问 C 每问题 world diff {dr('natural','C')['world_diff_wall_ms']:.4f} ms、scheduler {dr('natural','C')['scheduler_wall_ms']:.4f} ms；当前已提交快照成本另在模块表中，已计入完整 E2E。
7. B→C 三问 E2E：自然 {find('natural','B->C')['e2e_speedup']:.3f}×，固定 {find('fixed','B->C')['e2e_speedup']:.3f}×。三轮与场景 CI 见 abc_comparison.json，M1 独立正收益（两种工作负载均成立）：{incremental_positive}。
8. A→C 完整 M123 三问 E2E：自然 {find('natural','A->C')['e2e_speedup']:.3f}×，固定 {find('fixed','A->C')['e2e_speedup']:.3f}×。没有把单独 Decode 收益外推为 E2E。
9. P2 自然/固定、单问/三问四个工作负载有完整 A/B/C 配对；一次 P3 主策略修订后重测三问自然/固定。最终策略单问 E2E 未重复，标记 NOT_RUN，不混用 P2 单问结论。源图像始终逐帧更新，概览模板未混入原始 QA。
10. 类别/数量/方位/运动/车道/信号/安全按可靠当前 QA 事实及独立标注记录。缺失样本与规则无法确定的项目保留 UNKNOWN/未评分；不能宣称全部类别都已通过 2pp 质量预算。关键错误逐项保留。
11. 最终资格：{decision}。{reason} 即使 History B 可有局部/完整收益，也不等于已经证明 M1 的独立贡献或完成近似部署质量准入。
12. 最主要瓶颈是 M1 对当前 QA 依赖的可信语义覆盖不足，语义候选数与可证据化有效草稿脱节。下一步唯一优先方向：补齐并独立校准 M1 的问题相关当前语义/观测覆盖，再评估相同调度策略；不继续 K 值搜索或制造场景稳定性。
'''
    (root/'result_answers.md').write_text(answers)
    print('REPORT',decision,'rows',len(trials),'same48',dev,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs/m3_world_guided_decode_run02'))
    report(parser.parse_args().root)
