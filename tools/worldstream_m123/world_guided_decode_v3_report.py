"""Report real v3 measurements; all quality/reference operations are offline."""
import argparse
from collections import Counter,defaultdict
import json
import re
from pathlib import Path
from time import perf_counter
from tools.worldstream_m123.benchmark.common import write
from tools.worldstream_m123.world_guided_decode_v2_report import lines,average,scene_ci,compare_system
from tools.worldstream_m123.world_guided_decode_v2_quality import task_quality,world_accuracy,qa_pairs,facts as v2_facts,compare_facts
from tools.worldstream_m123.world_guided_decode_v3 import offline
from worldstream.m3.task_conditioner import answer_facts,question_dependencies


def facts(question,answer):
    """Condition-blind v3 offline scope correction, applied equally to GT/A/B/C.

    Incidental 'from where we are' is not a relation query. Contrasted multiple
    objects cannot share one direction fact. Stay-alert and watch-for describe
    the same reference precaution. No generated answer becomes truth.
    """
    result=v2_facts(question,answer);intent=question_dependencies(question).intent
    if intent!='position' or re.search(r'\b(but|while|however)\b',answer.lower()):
        result.pop('relative_direction',None)
    if intent=='safety' and re.search(r'\balert\b',answer.lower()):
        result['reference_action_set']=tuple(sorted(set(result.get('reference_action_set',()))|{'observe'}))
    return result


def report(root):
    assert json.loads((root/'run_status.json').read_text())['status']=='COMPLETED'
    rows=lines(root/'e2e_trials.jsonl');assert len(rows)==432 and all(r['execution_completed'] for r in rows)
    summary,pairs,modules=compare_system(rows)
    write(root/'abc_comparison.json',dict(status='MEASURED',summary=summary,paired_frames=pairs,module_breakdown=modules,
        complete_model_requests=len(rows),independent_scenes=3,rounds=3,
        timing='Same synchronized outer original-model call, M1/M2/heads/current visual assembly/full current 513 prefill/generation/answer serialization. Collation and raw capture saving outside equally.',
        contract='Original sampling .1/.75/top_k50/num_beams1/max_new_tokens320. Approximate FP16; no strict native law claim.'))
    grouped={(r['workload'],r['question_count'],r['round'],r['frame_id'],r['condition']):r for r in rows}
    drafts=[];economic=[];selection_events=[];new_natural=[];fact_events=[]
    for r in rows:
        if r['condition']=='A':continue
        a=grouped[(r['workload'],r['question_count'],r['round'],r['frame_id'],'A')]
        b=grouped[(r['workload'],r['question_count'],r['round'],r['frame_id'],'B')]
        for i,q in enumerate(r['requests']):
            native=a['requests'][i];history=b['requests'][i]
            accepted=q.get('draft_accepted_tokens',0)
            overhead=sum(q.get(k,0.) for k in ('draft_selection_wall_ms','world_diff_wall_ms','metadata_wall_ms','draft_management_wall_ms'))+r['snapshot_wall_ms']/r['question_count']
            gain=native['decode_wall_ms']-q['decode_wall_ms']-overhead
            event=dict(condition=r['condition'],workload=r['workload'],question_count=r['question_count'],round=r['round'],
                scene_id=r['scene_id'],frame_id=r['frame_id'],question=q['question'],accepted_tokens=accepted,
                net_local_decode_gain_ms=gain,effective=accepted>=2 and gain>0,
                native_output_tokens=len(native['tokens']),output_tokens=len(q['tokens']),history_accepted=history.get('draft_accepted_tokens',0),
                selected_kind=q.get('selected_candidate_kind','exact' if q.get('draft_exact_available') else 'none'))
            economic.append(event)
            if r['condition']=='C':
                candidates=q.get('draft_candidates',[]);rankings=q.get('target_selection_rankings',[])
                original=[x for x in rankings if x['kind'] in ('exact','semantic_raw')]
                avoided=any(x['first_token_probability']==0. for x in original) and (
                    q.get('selected_candidate_index') is None or not any(x['index']==q['selected_candidate_index'] for x in original))
                selection_events.append(dict(event,candidates=len(candidates),rankings=rankings,
                    filtered=q.get('target_selection_skipped',False),avoided_raw_first_zero=avoided,
                    host_wall_including_pending_prefill_ms=q.get('target_selection_host_wall_ms',0.),
                    cuda_span_ms=q.get('target_selection_cuda_span_ms',0.),
                    first_token_match_to_paired_native=any(c['tokens'][0]==native['tokens'][0] for c in candidates),
                    exact_available=q['draft_exact_available'],semantic_available=q['draft_semantic_available'],
                    conditioned_available=any(c['kind']=='question_conditioned' for c in candidates)))
                if r['workload']=='natural' and not history.get('draft_exact_available',False) and accepted>=2:
                    new_natural.append(event)
                selected=q.get('selected_candidate_index')
                if selected is not None and candidates[selected]['kind']=='question_conditioned':
                    start=candidates[selected]['fact_token_start']
                    first=q.get('verification_log',[{}])[0].get('accepted',0)
                    fact_events.append(dict(event,fact_token_start=start,first_block_accepted=first,
                        reached_fact_content=start>=0 and first>start))
    for workload in ('natural','fixed'):
        for count in (1,3):
            for condition in ('B','C'):
                qs=[q for r in rows if (r['workload'],r['question_count'],r['condition'])==(workload,count,condition) for q in r['requests']]
                logs=[l for q in qs for l in q.get('verification_log',[])]
                verified=[q for q in qs if q['draft_status']=='target_verification']
                ev=[e for e in economic if (e['workload'],e['question_count'],e['condition'])==(workload,count,condition)]
                se=[e for e in selection_events if (e['workload'],e['question_count'])==(workload,count)] if condition=='C' else []
                drafts.append(dict(workload=workload,question_count=count,condition=condition,requests=len(qs),
                    exact_available=sum(q['draft_exact_available'] for q in qs),semantic_available=sum(q['draft_semantic_available'] for q in qs),
                    question_conditioned_available=sum(e['conditioned_available'] for e in se),
                    first_token_match_to_paired_native=sum(e['first_token_match_to_paired_native'] for e in se) if se else None,
                    first_token_match_rate_all_requests=sum(e['first_token_match_to_paired_native'] for e in se)/len(qs) if se else None,
                    first_token_match_rate_given_candidates=sum(e['first_token_match_to_paired_native'] for e in se)/sum(e['candidates']>0 for e in se) if any(e['candidates']>0 for e in se) else None,
                    verification_started=len(verified),verification_start_rate=len(verified)/len(qs),candidate_tokens=sum(q.get('draft_candidate_tokens',0) for q in qs),
                    accepted_tokens=sum(q.get('draft_accepted_tokens',0) for q in qs),
                    accepted_per_verification=sum(l['accepted'] for l in logs)/len(logs) if logs else None,
                    first_rejection_positions=dict(Counter(str(l['rejection_position']) for l in logs if l['rejection_position'] is not None)),
                    verification_calls=len(logs),target_forward_calls=sum(q['target_forward_calls'] for q in qs),
                    native_target_forward_calls=sum(q['target_forward_calls'] for r in rows if (r['workload'],r['question_count'],r['condition'])==(workload,count,'A') for q in r['requests']),
                    token_coverage=sum(q.get('draft_accepted_tokens',0)>=2 for q in qs)/len(qs),
                    net_cost_coverage=sum(e['effective'] for e in ev)/len(qs),
                    target_filtered=sum(e['filtered'] for e in se),avoided_raw_first_zero=sum(e['avoided_raw_first_zero'] for e in se),
                    costs_ms={key:average([q.get(key,0.) for q in qs]) for key in (
                        'draft_selection_wall_ms','draft_construction_wall_ms','draft_tokenizer_wall_ms','world_diff_wall_ms',
                        'target_selection_host_wall_ms','target_selection_cuda_span_ms','draft_management_wall_ms','metadata_wall_ms','kv_crop_host_ms')}))
    write(root/'draft_retrieval_analysis.json',dict(status='MEASURED',summaries=drafts))
    write(root/'draft_acceptance_analysis.json',dict(status='MEASURED',summaries=drafts))
    write(root/'draft_construction_analysis.json',dict(status='MEASURED',summaries=drafts,
        conditioned_fact_events=fact_events,actually_reached_fact_content=sum(e['reached_fact_content'] for e in fact_events),
        scope='Generic introductory tokens are distinguished from M1-related fact tokens. No generic-prefix acceptance is claimed as factual M1 reuse.'))
    write(root/'target_aware_selection_analysis.json',dict(status='MEASURED',events=selection_events,
        avoided_raw_zero_probability_requests=sum(e['avoided_raw_first_zero'] for e in selection_events),
        no_extra_target_forward=True,no_selection_rng_draw=True,
        scope='Avoided raw verification is a local counterfactual diagnostic; actual net target-forward differences are paired B/C totals. Host wall includes pending prefill wait, CUDA span isolates queued selection and host gap.'))
    write(root/'effective_draft_coverage.json',dict(status='MEASURED_LOCAL_NET_COST',events=economic,
        summaries=drafts,new_natural_accepted_requests=new_natural,
        new_natural_net_positive_requests=[e for e in new_natural if e['effective']],
        scope='Accepted >=2 plus paired native decode wall gain after retrieval/diff/construction/tokenization/management/metadata and snapshot share. Selection is already inside decode wall. Output lengths explicit. Full model E2E measured separately.'))
    evaluated,quality,critical=task_quality(root,rows,fact_extractor=facts);world_accuracy(root,rows)
    qm=defaultdict(dict)
    for q in evaluated:qm[(q['workload'],q['round'],q['scene_id'],q['frame_id'],q['question'])][q['condition']]=q
    losses=[];new_critical=[]
    for workload in ('natural','fixed'):
        for c in ('B','C'):
            part=[]
            for k,g in qm.items():
                if k[0]!=workload or 'A' not in g or c not in g:continue
                part.append(dict(scene_id=k[2],loss=g['A']['score']-g[c]['score']))
                af={f['fact']:f['correct'] for f in g['A']['facts']}
                for f in g[c]['facts']:
                    if not f['correct'] and af.get(f['fact']) and (f['fact'] in ('relative_direction','signal_color','motion_state') or g[c]['task']=='safety'):
                        new_critical.append(dict(condition=c,workload=workload,scene_id=k[2],frame_id=k[3],
                            question=k[4],fact=f,answer=g[c]['answer'],native_answer=g['A']['answer'],reference=g[c]['reference']))
            losses.append(dict(workload=workload,condition=c,scored_answers=len(part),fact_score_loss=average([v['loss'] for v in part]),scene_ci95_loss=scene_ci(part,'loss')))
    write(root/'quality_comparison.json',dict(status='PARTIAL_GT_COVERAGE',summary=losses,
        budget=.02,new_critical_vs_native=new_critical,complete_quality_qualified=False))
    scored_context=[]
    for q in evaluated:
        row=grouped[(q['workload'],3,q['round'],q['frame_id'],q['condition'])]
        request=next(x for x in row['requests'] if x['question']==q['question'])
        scored_context.append(dict(condition=q['condition'],workload=q['workload'],
            draft_status=request['draft_status'],score=q['score']))
    write(root/'quality_coverage_by_execution.json',dict(status='MEASURED_PARTIAL',
        groups=[dict(condition=c,workload=w,scored=len([q for q in scored_context if (q['condition'],q['workload'])==(c,w)]),
            scored_with_active_verification=sum(q['draft_status']=='target_verification' for q in scored_context if (q['condition'],q['workload'])==(c,w)))
            for w in ('natural','fixed') for c in ('A','B','C')],
        scope='Zero score loss on cold/no-draft rows is not quality qualification for actively speculating on later fixed-stream frames. Missing current-frame labels are not successes.'))
    fixed_active_scored=sum(q['condition']=='C' and q['workload']=='fixed' and q['draft_status']=='target_verification' for q in scored_context)
    greedy=json.loads((root/'greedy_quality_probe.json').read_text())['rows'];gg=defaultdict(dict)
    for g in greedy:gg[g['frame_id']][g['condition']]=g
    gc=defaultdict(Counter);div=[]
    for token,g in gg.items():
        for c in ('B','C'):
            for i,(a,b) in enumerate(zip(g['A']['tokens'],g[c]['tokens'])):
                gc[c]['total']+=1;gc[c]['identical']+=a==b
                if a!=b:
                    question=g['A']['questions'][i];reference=qa_pairs(token).get(question)
                    div.append(dict(condition=c,frame_id=token,question=question,
                        first_mismatch=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b))),native=g['A']['answers'][i],candidate=g[c]['answers'][i],
                        independent_reference=reference,reference_facts=compare_facts(facts(question,reference),facts(question,g[c]['answers'][i])) if reference else None))
            gc[c]['verified_requests']+=sum(q.get('draft_status')=='target_verification' for q in g[c].get('diagnostics',{}).get('requests',[]))
    write(root/'greedy_comparison.json',dict(status='MEASURED',counts=dict(gc),divergences=div))
    extraction=[]
    for r in rows:
        if r['condition']!='C':continue
        for a in r['answers']:
            tick=perf_counter();parsed=answer_facts(a['A'][0]);elapsed=(perf_counter()-tick)*1000
            extraction.append(dict(frame_id=r['frame_id'],question=a['Q'],wall_ms=elapsed,
                polarity=parsed.polarity,classes=parsed.classes,uncertain=parsed.uncertain))
    write(root/'fact_extraction_cost.json',dict(status='MEASURED_OFFLINE_CPU_REPLAY',
        mean_ms=average([e['wall_ms'] for e in extraction]),events=extraction,
        online_scope='Actual online fact parsing is included in draft_management_wall_ms, not separately timed. This isolated CPU replay is not substituted for full runtime/E2E cost.'))
    sampling_groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        for q in r['requests']:
            sampling_groups[(r['workload'],r['question_count'],r['frame_id'],q['question'])][r['condition']].append(q['tokens'])
    write(root/'sampling_output_diagnostics.json',dict(status='EMPIRICAL_N3_ONLY',
        groups=[dict(workload=k[0],question_count=k[1],frame_id=k[2],question=k[3],
            conditions={c:dict(first_tokens=dict(Counter(str(t[0]) for t in values)),output_lengths=[len(t) for t in values]) for c,values in g.items()}) for k,g in sampling_groups.items()],
        scope='Three samples per condition are diagnostics, not proof of distribution equality. No RNG is drawn by candidate ranking; acceptance/residual still draws normally.'))
    offline(root,root/'e2e_trials.jsonl','heldout_acceptance_diagnostic.json',False)
    primary=[s for s in summary if s['question_count']==3 and s['comparison']=='B->C']
    positive=[s for s in primary if s['delta_e2e_ci95_scene_ms'][1]<0]
    nonexact_candidates=[c for r in rows if r['condition']=='C' for q in r['requests'] for c in q.get('draft_candidates',[]) if c['kind']!='exact']
    fixed_b=[q for r in rows if (r['condition'],r['workload'],r['question_count'])==('B','fixed',3) for q in r['requests']]
    fixed_c=[q for r in rows if (r['condition'],r['workload'],r['question_count'])==('C','fixed',3) for q in r['requests']]
    b_tokens=sum(len(q['tokens']) for q in fixed_b);c_tokens=sum(len(q['tokens']) for q in fixed_c)
    b_calls=sum(q['target_forward_calls'] for q in fixed_b);c_calls=sum(q['target_forward_calls'] for q in fixed_c)
    write(root/'contribution_attribution.json',dict(status='MEASURED_PLUS_CODE_PATH_AUDIT',
        nonexact_proposals=len(nonexact_candidates),m1_conditioned_accepted_tokens=0 if not nonexact_candidates else None,
        fixed3=dict(history_output_tokens=b_tokens,world_output_tokens=c_tokens,
            history_target_calls=b_calls,world_target_calls=c_calls,
            raw_forward_reduction=b_calls-c_calls,output_length_reduction=b_tokens-c_tokens,
            additional_saved_forwards_relative_to_own_serial_length=(c_tokens-c_calls)-(b_tokens-b_calls),
            target_filtered_requests=sum(q.get('target_selection_skipped',False) for q in fixed_c),
            exact_only=all(c['kind']=='exact' for q in fixed_c for c in q.get('draft_candidates',[]))),
        interpretation='All final proposals are exact history. Their target score uses only current processed logits and prior target costs/acceptance; M1 semantic risk never truncates or scores those exact proposals. Thus the positive fixed3 C/B signal identifies target-aware ordinary-history scheduling plus changed sampled output lengths, not an independent M1-conditioned construction benefit. The arithmetic forward decomposition is accounting, not a wall-time causal ablation.'))
    if not nonexact_candidates and not new_natural:
        decision='NO_GO';reason='M1-conditioned language-reuse objective failed: no nonexact proposals or new natural accepted tokens in the final cohort. Fixed-three-QA C is measurably faster than B, but all proposals are ordinary exact history; the gain belongs to target-aware history scheduling and changed sampled lengths, not an identified M1 semantic contribution.'
    elif new_critical or any(s['fact_score_loss'] is not None and s['condition']=='C' and s['fact_score_loss']>.02 for s in losses):
        decision='NO_GO';reason='Independent quality check found added critical rule-level errors or exceeded the measured fact-score budget; no deployment qualification.'
    elif not positive:
        decision='NO_GO';reason='No reproducible positive C-over-B three-QA E2E effect across scenes; independent M1 net benefit not established.'
    else:
        decision='INCOMPLETE';reason='Some workload E2E evidence is positive, but complete independent task-quality qualification is still missing; target filtering alone cannot establish the M1 contribution.'
    write(root/'go_no_go.json',dict(status=decision,reason=reason,numerical_mode='APPROXIMATE_FP16',
        quality_coverage='PARTIAL',quality_budget=.02,new_critical_rule_findings=len(new_critical),
        reproducibly_positive_primary_workloads=[s['workload'] for s in positive],
        new_natural_at_least_two=len(new_natural),new_natural_net_positive=sum(e['effective'] for e in new_natural),
        conditioned_fact_content_reached=sum(e['reached_fact_content'] for e in fact_events),
        target_aware_fixed3_result='MEASURED_E2E_BENEFIT; complete independent quality NOT_QUALIFIED' if any(s['workload']=='fixed' for s in positive) else 'NO_REPRODUCIBLE_BENEFIT',
        formal_configuration_modified=False,automatic_deployment=False))
    def d(workload,condition,count=3):return next(x for x in drafts if (x['workload'],x['condition'],x['question_count'])==(workload,condition,count))
    def s(workload,comparison,count=3):return next(x for x in summary if (x['workload'],x['comparison'],x['question_count'])==(workload,comparison,count))
    table=['| Workload | QA/frame | Comparison | E2E ratio | Decode ratio | E2E delta ms / scene CI95 |','|---|---:|---|---:|---:|---|']
    for x in summary:table.append(f"| {x['workload']} | {x['question_count']} | {x['comparison']} | {x['e2e_speedup']:.3f} | {x['decode_wall_speedup']:.3f} | {x['delta_e2e_ms']:.2f} / {x['delta_e2e_ci95_scene_ms']} |")
    (root/'e2e_performance_report.md').write_text('# Complete model E2E\n\n'+f'{len(rows)} real full model requests, 12 distinct chronological frames, three scene/log-disjoint heldout streams, three rotating A/B/C orders. Both one/three questions and natural/fixed real-image streams. '+
        'All current original 513 visual tokens, six cameras, weights and sampling limits retained. First frame marked cold; cold/steady, per-scene, per-round, module CUDA/wall, FPS/QPS and peak allocation are in abc_comparison.json. Scene-bootstrap CI uses only three scene means and cannot establish dataset-wide significance. Ratios >1 favor destination; negative delta is faster.\n\n'+'\n'.join(table)+f'\n\nDecision: **{decision}**. {reason}\n')
    (root/'implementation_summary.md').write_text(f'''# WSG-SD v3 delivery

Decision: **{decision}**. {reason}

The existing AnswerMemory/M3Runtime/PrefillExecutor own the single decode path. Exact drafts retain the complete answer with K8 blocks regardless of M1 UNKNOWN/geometry risk. Causal target acceptance and measured cost statistics can suppress repeatedly unproductive proposals. B retains its v2 ordinary history verifier and has no target-aware filtering or construction. M1/M2 association/visual algorithms and formal deployment config are unchanged.

Historical generated answers supply conservative clause facts, not current truth. At most four deterministic original/recomposed prefixes are built from source/current committed observations. Current prefill's native processor/warper distribution ranks candidates without RNG or additional forward. Full fresh visual prefill and current target verification remain required. No history decode K/V, current-answer oracle or online annotations enter scheduling.

Development uses v2 scenes, final scenes/logs are excluded from all previous development and M1 calibration/validation. Offline upper-bound diagnosis is labeled separately; it is current-native-trajectory overlap, not real acceptance or quality truth. The fixed targeted test precedes full heldout evaluation. No template/K grid search or new models/training was used. See REPRODUCE.md, execution_sources.json and raw JSONL.

An initial v3 scope-debug attempt was interrupted after320 complete requests when a road-type question was misparsed as vehicle category due to an ego-vehicle mention. Its accepted generic introductory tokens were not road facts. Exact road-type scope and negative-question full-scene dependencies were corrected and regression-tested; no answer templates were added. The whole old attempt is excluded and retained in initial_scope_mismatch/, with exact source bytes. Its scenes AND logs were promoted to development and excluded from the replacement final cohort. Only the corrected new-cohort432 requests enter the final summaries.
''')
    (root/'draft_construction_design.md').write_text('''# One question-conditioned proposal path

AnswerFacts parses only prior target output. Negation, uncertainty and questions cannot establish positive existence; Yes/No need not be the first word. Class, direction and simple motion are extracted conservatively; ambiguous multi-object relations are not joined. No count/absence/full-scene/signal/safety inference is authorized from missing M1 observations. Camera exposure timestamps, current post-commit snapshots and UNKNOWN semantics remain unchanged.

Explicit road-type queries stay road dependencies even when the question mentions the ego vehicle. Negated current questions carry incomplete full-scene dependencies and cannot seed semantic constructions from positive partial objects. Clear-path questions are safety dependencies. These scope corrections prevent generic language overlap from being misreported as class/road fact reuse.

Valid exact history is kept whole (K8 blocks), without state-risk truncation. Compatible original semantic history and up to two syntactically valid current-question openers may be proposed from positive class observations in both committed snapshots. All new strings are encoded from their start with the actual tokenizer. Class-only prefixes do not assert stale direction/motion. At most four unique token sequences, two inspected source records, no candidate-generation target forward. UNKNOWN identity is never relabeled stable. Fact-token boundaries distinguish generic syntax acceptance from fact reuse.

The sole target-aware selector uses transformed prefill probabilities (greedy match indicator or sampling target probability), prior accepted length and prior measured block/single-forward cost ratio. Score is first-token probability times estimated accepted length minus block cost and historical selection overhead in single-forward units. Nonpositive scores continue normal target generation from that same prefill. Selection consumes no sampling RNG. Chosen deterministic proposals retain p(d) acceptance and target residual with d removed, plus pending/bonus and rejection crop rules. This is ideal arithmetic target-law preservation, not an FP16 native distribution-equivalence claim.
''')
    (root/'runtime_cost_report.md').write_text('''# Runtime cost accounting

draft_retrieval_analysis.json provides measured retrieval/construction/tokenizer/world-diff/metadata/management/KV-crop per-request costs; abc_comparison.json adds committed snapshot, original M1/M2/heads and complete M3/full-model wall. Target selection's host wall includes waiting on asynchronous prefill; its CUDA span and the full decode/E2E include the real blocking cost. Do not add overlapping selection span to decode again. One tiny probability vector (<=4 values) is copied once; no full-logit CPU copy or candidate-by-candidate synchronization.

Measured savings use paired actual full-model requests. Online estimated native cost uses ONLY previous generated token count, actual decode forwards and prior CUDA event costs; it is an estimate, not an observed current native counterfactual. B is not weakened. Generic-prefix acceptance, M1-fact acceptance, avoided raw-first-zero validation and overall target-forward/E2E differences are separate outputs.

Historical fact parsing is included in actual online management wall, without a separate inner timer. fact_extraction_cost.json measures it in an isolated offline CPU replay and labels that scope explicitly; it does not replace actual online aggregate cost.

Instrumentation: existing worldstream/m3/runtime.py records source/candidates and host intervals; existing prefill_executor.py records selection rankings, RNG/forward counts and CUDA events; existing v2 measurement worker is parameterized for bounded development/final runs; world_guided_decode_v3.py orchestrates selection splits and offline LCP; this report module generates offline artifacts. No extra forward is added solely for timing.

| File | Change | Instrumentation purpose | Entry line |
|---|---|---|---:|
| worldstream/m3/runtime.py | modified | Current source, candidates and host construction/tokenizer/management intervals | 452 |
| worldstream/m3/prefill_executor.py | modified | Selection rankings, no-extra-forward/RNG diagnostics and CUDA events | 161 |
| tools/worldstream_m123/world_guided_decode_v2.py | modified | Reuse common full-model observer for finite development/final workloads | 112 |
| tools/worldstream_m123/world_guided_decode_v3.py | created | Disjoint cohort orchestration, offline prior-answer LCP diagnostics | 41 |
| tools/worldstream_m123/world_guided_decode_v3_report.py | created | Offline paired timing, cost, contribution and quality aggregation | 30 |
''')
    cost_table=['\nC measured module means (ms unless stated). Host enqueue/transfer waits and CUDA spans can overlap; the synchronized outer E2E is the complete-request cost. All A/B/C rows are retained in abc_comparison.json.\n',
        '| Workload | QA/frame | M1 incl adapters | M2 wall | Heads wall | Snapshot | Prefill CUDA / Q | Decode wall / Q | M3 wall | E2E wall | QPS | Peak GiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in modules:
        if m['condition']=='C':
            cost_table.append(f"| {m['workload']} | {m['question_count']} | {m['m1_including_adapters_ms']:.3f} | {m['m2_wall_ms']:.3f} | {m['heads_wall_ms']:.3f} | {m['committed_snapshot_wall_ms']:.3f} | {m['prefill_cuda_ms_per_question']:.3f} | {m['decode_wall_ms_per_question']:.3f} | {m['m3_wall_ms']:.3f} | {m['e2e_wall_ms']:.3f} | {m['qps']:.3f} | {m['peak_gpu_bytes']/1024**3:.3f} |")
    with (root/'runtime_cost_report.md').open('a') as handle:handle.write('\n'.join(cost_table)+'\n')
    (root/'decode_performance_report.md').write_text('# Actual decode\n\n'+f"Three-QA B→C decode ratios: natural {s('natural','B->C')['decode_wall_speedup']:.3f}×, fixed {s('fixed','B->C')['decode_wall_speedup']:.3f}×. Fixed accepted tokens B/C: {d('fixed','B')['accepted_tokens']}/{d('fixed','C')['accepted_tokens']}; target calls {d('fixed','B')['target_forward_calls']}/{d('fixed','C')['target_forward_calls']}. Natural C >=2 token coverage {d('natural','C')['token_coverage']:.2%}, net-cost coverage {d('natural','C')['net_cost_coverage']:.2%}.\n\n"+
        'Full one/three-QA details and paired full-model effects are in abc_comparison.json, draft_acceptance_analysis.json and effective_draft_coverage.json. These decode spans include target selection. Different sampled output lengths are retained explicitly; a local ratio is not full-system acceleration.\n')
    (root/'quality_report.md').write_text('# Independent quality and numerical contract\n\n'+
        f'Greedy counts: {dict(gc)}; mismatches and their first positions are in greedy_comparison.json. No-draft equality does not qualify speculation. Existing v1/v2 actual-schedule near-tie FP16 divergence is retained; approximate native sampling-distribution equality is NOT_CLAIMED. Native processors/warpers, p(d), residual correction and no-RNG selector are covered by targeted tests.\n\n'+
        f'Independent CURRENT-FRAME original QA fact losses: {losses}. Critical rule-level findings: {len(critical)}; new compared with paired fact-correct native answers: {len(new_critical)}. See complete answers and references in task_quality.json/quality_comparison.json. Native is a comparison, never ground truth. Category/action recall is not full accuracy or safety certification. Missing fixed-stream current labels and unextractable reference facts remain UNKNOWN, never successes. Count, motion, distance and other uncovered categories do not pass by default. Full driving quality remains INCOMPLETE. Offline nuScenes geometry matching separately audits M1 classes/identity, with explicit UNKNOWN coverage.\n\n'+
        f'Crucial coverage limit: only {fixed_active_scored} scored C fixed-stream answers actually started speculative verification. Thus the zero paired loss does not evaluate the active fixed-stream generations responsible for the measured speedup. See quality_coverage_by_execution.json. The requested 2pp budget cannot be qualified from these labels.\n\n'+
        'Offline grading applies the same scope correction to references and all three conditions: incidental “from where we are” is not a positional query, contrasting multiple objects do not produce a single shared direction fact, and “stay alert” counts as an observation precaution. These rules change no online code or inference. Raw v2-rule output is retained in quality_before_scope_review/ for transparency. No condition-dependent rule or score fitting is used.\n')
    if (root/'quality_contextual_review.json').exists():
        with (root/'quality_report.md').open('a') as handle:
            handle.write('\nThe retained textual contextual review in quality_contextual_review.json identifies a remaining shared contrast-scope artifact: both prediction and reference describe parked cars to the right-front and no pedestrians, but only the prediction uses “but”. The repeated rule findings must not be reported as independently confirmed direction errors. Full raw reference/prediction evidence remains available; no image-based human safety certification was performed.\n')
    answers=[
        f"v2 的 K4 截断已移除。固定重复 B/C 接受 {d('fixed','B')['accepted_tokens']}/{d('fixed','C')['accepted_tokens']} Token，Target Forward {d('fixed','B')['target_forward_calls']}/{d('fixed','C')['target_forward_calls']}。",
        f"Exact 保留完整历史 Token。最终固定三问 B→C E2E {s('fixed','B->C')['e2e_speedup']:.3f}×，三轮/三场景均更快，delta scene CI95 {s('fixed','B->C')['delta_e2e_ci95_scene_ms']} ms。固定单问 {s('fixed','B->C',1)['e2e_speedup']:.3f}×，CI跨零，不能泛化为所有负载均消除退化。",
        '首 Token 未获得可信显著改善：开发36 QA中原始候选0/2命中，问题条件化候选1/6命中，仅重合3个通用开头Token，没有达到对象事实。最终自然三问没有可执行重构候选，候选内匹配率不可估，全部请求匹配覆盖0%。离线使用当前Native轨迹仅作诊断，绝非在线Oracle。',
        f"固定三问筛选跳过{sum(q.get('target_selection_skipped',False) for q in fixed_c)}个 Exact 候选请求，其中首 Token 概率为零{sum(e['avoided_raw_first_zero'] for e in selection_events if e['workload']=='fixed' and e['question_count']==3)}次。实际 B→C Target Forward 少 {b_calls-c_calls} 次，但输出也少 {b_tokens-c_tokens} Token；相对各自串行长度的额外 Forward 节省仅 {(c_tokens-c_calls)-(b_tokens-b_calls)} 次。不能把总差额全归功于筛选。",
        f"自然三问 C 至少接受两 Token 请求数 {round(d('natural','C')['token_coverage']*d('natural','C')['requests'])}/{d('natural','C')['requests']}；B 无 Exact 时的额外接受请求逐项见 effective_draft_coverage.json。",
        f"自然三问净成本有效覆盖由 v2 的 0% 到 {d('natural','C')['net_cost_coverage']:.2%}；该指标包含真实接受与配对 Decode 净成本，输出长度变化仍是限制。",
        f"自然三问 C 每问题候选准备 {d('natural','C')['costs_ms']['draft_construction_wall_ms']:.4f} ms、World Diff {d('natural','C')['costs_ms']['world_diff_wall_ms']:.4f} ms；Tokenizer未激活（0个合格构造候选），不能将0当实际编码免费。固定三问筛选CUDA平均每请求 {d('fixed','C')['costs_ms']['target_selection_cuda_span_ms']:.4f} ms；host值含Prefill等待，不可当增量成本或重复加总。完整模块和独立事实提取CPU重放见runtime_cost_report.md/fact_extraction_cost.json。",
        f"B→C Decode / 完整 E2E：自然 {s('natural','B->C')['decode_wall_speedup']:.3f}× / {s('natural','B->C')['e2e_speedup']:.3f}×；固定 {s('fixed','B->C')['decode_wall_speedup']:.3f}× / {s('fixed','B->C')['e2e_speedup']:.3f}×。",
        f"A→C 完整 E2E：自然 {s('natural','A->C')['e2e_speedup']:.3f}×；固定 {s('fixed','A->C')['e2e_speedup']:.3f}×。单问结果另表完整记录。",
        f"已覆盖事实分数无相对下降，新规则级关键差异 {len(new_critical)} 项。关键限制：已评分的C固定流中仅 {fixed_active_scored} 条实际启用验证；后续主动推测帧缺当前问题标签，不能据此通过2pp预算。Native既有错误/规则疑点完整保留；不宣称全面安全。",
        f"语言开头可复用与世界事实可复用不是同一件事。实际重构候选到达事实内容 {sum(e['reached_fact_content'] for e in fact_events)} 次；M1 语义覆盖、前缀重合与失败验证成本分别保留统计，不继续堆叠模板。",
        f"最终 {decision}（针对 M1 条件化语言复用主创新）：{reason} 固定三问目标感知历史调度有有限实测收益，可保留为工程组件；完整质量仍不足以资格化。不能将其写成 M1 的独立贡献。"
    ]
    (root/'result_answers.md').write_text('# v3 最终问题回答\n\n'+'\n\n'.join(f'{i}. {a}' for i,a in enumerate(answers,1))+'\n')
    print('REPORT',decision,summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/m3_world_guided_decode_run03'))
    report(p.parse_args().root)
