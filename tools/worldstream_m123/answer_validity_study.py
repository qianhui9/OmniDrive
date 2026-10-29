"""P0 active-request audit and P1 bounded causal WS-AVC feasibility study.

Never runs a model or substitutes estimated savings for measured M123 E2E.
P2/P3 are intentionally gated by independent DEV quality evidence.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
from time import perf_counter

from worldstream.m3.answer_memory import AnswerMemory
from worldstream.m3.answer_validity import QUALITY_CANDIDATES, evaluate_answer, validity_task
from worldstream.m3.world_delta import relevant
from .answer_validity_quality import load_current_annotations, annotation_grade, original_qa_grade, coverage_summary
from .benchmark.common import sha256, write
from .world_guided_decode_v3 import load_world

ROOT = Path('outputs/m3_answer_validity_cache_run01')
V3 = Path('outputs/m3_world_guided_decode_run03')
DEV = Path('outputs/m3_world_guided_decode_run02/p3')


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def jsonl(path, rows):
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def average(values):
    return sum(values) / len(values) if values else None


def active_audit(root, source, annotations):
    events = []
    for row in lines(source / 'e2e_trials.jsonl'):
        if row['question_count'] != 3:
            continue
        for index, (request, answer) in enumerate(zip(row['requests'], row['answers'])):
            execution = ('history_draft_active' if request['draft_status'] == 'target_verification'
                         else 'native_no_reuse')
            text = answer['A'][0]
            events.append(dict(condition=row['condition'], workload=row['workload'], round=row['round'],
                               scene_id=row['scene_id'], log_id=row['log_id'], frame_id=row['frame_id'],
                               frame_index=row['frame_index'], request_index=index, question=request['question'],
                               answer=text, output_tokens=len(request['tokens']), execution=execution,
                               task=validity_task(request['question']).intent,
                               draft_accepted_tokens=request.get('draft_accepted_tokens', 0),
                               prefill_cuda_ms=request['prefill_cuda_ms'], decode_cuda_ms=request['decode_cuda_ms'],
                               frame_e2e_wall_ms=row['e2e_wall_ms'],
                               qa=original_qa_grade(row['frame_id'], request['question'], text),
                               annotations=annotation_grade(text, annotations[row['frame_id']], request['question']),
                               answer_cache_hit=False))
    jsonl(root / 'active_request_quality.jsonl', events)
    groups = coverage_summary(events)
    # All requests and execution-stratified results use the same exact labels.
    all_events = [dict(e, execution='all_requests') for e in events]
    groups += coverage_summary(all_events)
    write(root / 'quality_coverage_by_execution.json', dict(status='MEASURED_PARTIAL', groups=groups))
    write(root / 'active_request_quality.json', dict(status='MEASURED_PARTIAL', groups=groups,
          tasks=dict(Counter(e['task'] for e in events)), new_GPU_generations=0,
          answer_cache=dict(requests=0, scored=0, status='NOT_RUN'),
          human_review=dict(status='NOT_RUN', records=0),
          scope='v3 A=native, B=old NON-target-aware history, C=target-aware world-guided. These are historical P0 labels, NOT new WS-AVC A/B/C. Current annotation positive witnesses give partial checks only; missing witnesses never prove absence.'))
    active = [e for e in events if e['execution'] == 'history_draft_active']
    grouped = {(e['workload'], e['round'], e['frame_id'], e['request_index'], e['condition']): e for e in events}
    critical = []
    paired = []
    for e in events:
        if e['condition'] != 'A':
            a = grouped[(e['workload'], e['round'], e['frame_id'], e['request_index'], 'A')]
            common = []
            for c in e['annotations']['checks']:
                if c['correct'] is None:
                    continue
                key = lambda f: tuple((k, tuple(v) if isinstance(v, (list, tuple)) else v)
                                      for k, v in f['claim'].items() if k not in ('clause', 'unknown'))
                ac = next((x for x in a['annotations']['checks'] if key(x) == key(c) and x['correct'] is not None), None)
                if ac is not None:
                    common.append(dict(claim=c['claim'], native_correct=ac['correct'],
                                       condition_correct=c['correct'], newly_wrong=bool(ac['correct'] and not c['correct'])))
            paired.append(dict(condition=e['condition'], workload=e['workload'], execution=e['execution'],
                               scene_id=e['scene_id'], frame_id=e['frame_id'], round=e['round'],
                               question=e['question'], common_scored_assertions=common,
                               paired_task_fact_loss=a['qa']['score'] - e['qa']['score']
                               if a['qa']['score'] is not None and e['qa']['score'] is not None else None))
        for c in e['annotations']['checks']:
            if c['correct'] is False:
                a = grouped[(e['workload'], e['round'], e['frame_id'], e['request_index'], 'A')]
                critical.append(dict(frame_id=e['frame_id'], condition=e['condition'], round=e['round'],
                                     workload=e['workload'], question=e['question'], answer=e['answer'],
                                     execution=e['execution'], finding=c,
                                     native_answer=a['answer'],
                                     native_annotation_grade=a['annotations'],
                                     severity='automated object-existence conflict; image review NOT_RUN; behaviour/signal UNKNOWN',
                                     attribution='historical_generation; no WS-AVC executed'))
    write(root / 'active_request_findings.json', dict(status='MEASURED_PARTIAL', findings=critical))
    write(root / 'active_request_paired_quality.json', dict(status='MEASURED_PARTIAL', events=paired,
          new_shared_assertion_errors=sum(c['newly_wrong'] for e in paired for c in e['common_scored_assertions']),
          scope='Paired CURRENT-frame independent labels. Only identical atomic scopes are compared; different/unscored claims remain UNKNOWN, not zero-loss evidence.'))
    report = ['# Active v3 request quality', '',
              f'Reused {len(events)} saved answer outputs, including {len(active)} actively verified requests. No new generation.', '',
              '| Condition | Workload | Execution | Requests | Current QA scored | Annotation scored requests | False assertions |',
              '|---|---|---|---:|---:|---:|---:|']
    for g in groups:
        report.append(f"| {g['condition']} | {g['workload']} | {g['execution']} | {g['requests']} | {g['exact_current_QA_scored']} | {g['annotation_scored_requests']} | {g['annotation_false_assertions']} |")
    report += ['', 'Annotation assertions and QA fact recall have separate denominators. Positive current-frame visible annotation witnesses can confirm existence or contradict explicit absence. Missing annotations, crossing intent, exact count, road semantics, signals and global safety remain UNKNOWN without exact reliable QA facts. These checks do not qualify the complete driving-quality budget.', '',
               'Repeated rounds are not independent scenes. All unscored requests remain in coverage denominators. No human validation was performed. See active_request_quality.jsonl and active_request_findings.json.']
    (root / 'active_request_quality_report.md').write_text('\n'.join(report) + '\n')
    return groups


def offline(root, source, annotations):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('ckpts/pretrain_qformer/', use_fast=False, local_files_only=True)
    raw = lines(source / 'e2e_trials.jsonl')
    # Fixed source/current states come from the corresponding real perception
    # execution; no synthetic state, repeated frame, or current answer oracle.
    worlds = {(r['workload'], r['frame_id']): load_world(r['committed_world']) for r in raw
              if r['condition'] == 'C' and r['question_count'] == 3 and r['round'] == 0}
    write(root / 'committed_worlds.json', dict(status='RETAINED_ACTUAL_M1_COMMITS',
          snapshots=[dict(workload=k[0], **asdict(v)) for k, v in worlds.items()],
          source_artifact=str(source / 'e2e_trials.jsonl')))
    events = []
    for workload in ('natural', 'fixed'):
        memory = AnswerMemory()
        binding = ('offline_original_native_generation',)
        scene = None
        seen = set()
        for row in raw:
            if (row['condition'], row['workload'], row['question_count'], row['round']) != ('A', workload, 3, 0):
                continue
            # Some prior artifacts contain duplicate timing blocks. One real
            # native trajectory per scene/frame/workload is the DEV unit.
            if row['frame_id'] in seen:
                continue
            seen.add(row['frame_id'])
            if scene != row['scene_id']:
                memory.reset()
                scene = row['scene_id']
            world = worlds[(workload, row['frame_id'])]
            ctx = dict(scene_id=scene, frame_id=world.frame_id, timestamp=world.timestamp,
                       frame_index=row['frame_index'], state_epoch=world.epoch,
                       current_state_version=world.version, state_execution_signature=world.execution_signature)
            for index, (request, answer) in enumerate(zip(row['requests'], row['answers'])):
                q = request['question']
                tick = perf_counter()
                prompt_ids = tuple(tokenizer.encode(q, add_special_tokens=False))
                records = memory.validity_candidates(q, ctx, binding)
                scored = [(record, evaluate_answer(record, q, world, ctx, binding)) for record in records]
                chosen = max(scored, key=lambda item: item[1].score) if scored else None
                check_ms = (perf_counter() - tick) * 1000
                # Decisions above are final before ANY current QA/GT/output is read.
                native_text = answer['A'][0]
                event = dict(workload=workload, scene_id=scene, log_id=row['log_id'], frame_id=world.frame_id,
                             frame_index=row['frame_index'], timestamp=world.timestamp, question=q, request_index=index,
                             question_task=asdict(validity_task(q)), candidate_count=len(records),
                             check_wall_ms=check_ms, current_native_answer=native_text,
                             current_native_output_tokens=len(request['tokens']),
                             native_qa=original_qa_grade(world.frame_id, q, native_text),
                             current_frame_e2e_wall_ms=row['e2e_wall_ms'],
                             current_prefill_cuda_ms=request['prefill_cuda_ms'],
                             current_decode_cuda_ms=request['decode_cuda_ms'],
                             actual_M3_skipped=False)
                if chosen:
                    record, decision = chosen
                    qa = original_qa_grade(world.frame_id, q, record.answer_text)
                    annotation = annotation_grade(record.answer_text, annotations[world.frame_id], q)
                    dep = record.answer_dependencies
                    task = dep.task
                    scoped = replace(task, classes=dep.classes or task.classes, directions=dep.directions,
                                     attributes=dep.attributes)
                    current_facts = [a for a in world.anchors if dep.classes and a.observed
                                     and a.anchor_type == 'object' and a.score is not None
                                     and a.score >= .35 and relevant(a, scoped)]
                    event.update(historical_question=record.question, historical_answer=record.answer_text,
                                 source_question_token_ids=record.prompt_ids, source_answer_token_ids=record.tokens,
                                 historical_output_tokens=len(record.tokens), original_generation=record.original_generation,
                                 historical_generation_cuda_cost_ms=record.generation_cost_ms,
                                 expected_net_saving_from_history_ms=record.generation_cost_ms - check_ms,
                                 source_frame=record.frame_id, source_world_version=record.world_version,
                                 source_dependencies=asdict(record.answer_dependencies),
                                 current_observed_relevant_facts=[dict(anchor_id=a.anchor_id, semantic_class=a.semantic_class,
                                     center_ego=a.center_ego, velocity_ego=a.velocity_ego, score=a.score,
                                     identity_qualified=a.identity_qualified, source_frame=a.source_frame,
                                     source_timestamp=a.source_timestamp) for a in current_facts],
                                 decision=decision.to_dict(), candidate_qa=qa, candidate_annotations=annotation,
                                 all_candidate_decisions=[dict(source_frame=r.frame_id, **d.to_dict()) for r, d in scored],
                                 reference_valid=qa['score'] == 1. if qa['score'] is not None else None,
                                 reference_scope='current-frame selected task facts only; not full-answer truth',
                                 potential_saved_prefill_decode_cuda_ms=request['prefill_cuda_ms'] + request['decode_cuda_ms'])
                events.append(event)
                tick = perf_counter()
                memory.remember(q, prompt_ids, request['tokens'], ctx, binding, world, native_text,
                                track_validity=True,
                                generation_cost_ms=request['prefill_cuda_ms'] + request['decode_cuda_ms'])
                event['memory_and_dependency_wall_ms'] = (perf_counter() - tick) * 1000
    jsonl(root / 'offline_requests.jsonl', events)
    return events


def summarize(root, events, source):
    summaries = []
    for workload in ('natural', 'fixed'):
        rows = [e for e in events if e['workload'] == workload]
        candidates = [e for e in rows if e['candidate_count']]
        eligible = [e for e in candidates if e['decision']['eligible']]
        known = [e for e in candidates if e['reference_valid'] is not None]
        summaries.append(dict(workload=workload, requests=len(rows), candidate_requests=len(candidates),
                              candidate_coverage=len(candidates) / len(rows), eligible=len(eligible),
                              independent_current_QA_labeled=len(known),
                              actually_valid_task_fact_rate=average([e['reference_valid'] for e in known]),
                              rejection_reasons=dict(Counter(e['decision']['reasons'][0] for e in candidates if not e['decision']['eligible'])),
                              unknown_candidate_ratio=sum(bool(e['decision']['unknown_dependencies']) for e in candidates) / len(candidates) if candidates else None,
                              decision_accuracy=None, false_reuse_rate=None,
                              actual_E2E_speedup=None, actual_M3_skipped=0))
    pareto = []
    for workload in ('natural', 'fixed'):
        rows = [e for e in events if e['workload'] == workload]
        native = [e for e in rows if e['native_qa']['score'] is not None]
        pareto.append(dict(workload=workload, name='no_answer_cache', budget=None, threshold=None,
                           hit_count=0, coverage=0., scored_hits=0,
                           mean_task_fact_score=average([e['native_qa']['score'] for e in native]),
                           task_fact_loss_absolute=0., observed_GPU_E2E_ms=None, status='OFFLINE_NATIVE_REFERENCE'))
        for budget, threshold in QUALITY_CANDIDATES:
            hit = [e for e in rows if e['candidate_count'] and e['decision']['eligible']
                   and e['decision']['score'] >= threshold and e['expected_net_saving_from_history_ms'] > 0.]
            labeled = [e for e in hit if e['candidate_qa']['score'] is not None and e['native_qa']['score'] is not None]
            pred_scores = [e['candidate_qa']['score'] if e in hit else e['native_qa']['score'] for e in native]
            false = [e for e in labeled if not e['reference_valid']]
            pareto.append(dict(workload=workload, name=f'budget_{int(budget * 100)}pp', budget=budget,
                               threshold=threshold, hit_count=len(hit), coverage=len(hit) / len(rows),
                               scored_hits=len(labeled), hit_label_coverage=len(labeled) / len(hit) if hit else None,
                               false_reuse_rate=len(false) / len(labeled) if labeled else None,
                               mean_task_fact_score=average(pred_scores),
                               task_fact_loss_absolute=average([e['native_qa']['score'] for e in native]) - average(pred_scores) if native else None,
                               theoretical_saved_M3_cuda_ms=sum(e['potential_saved_prefill_decode_cuda_ms'] for e in hit),
                               measured_offline_check_plus_memory_ms=sum(e['check_wall_ms'] + e['memory_and_dependency_wall_ms'] for e in rows),
                               observed_GPU_E2E_ms=None, quality_budget_qualified=False,
                               status='OFFLINE_ONLY_UNCALIBRATED'))
    write(root / 'answer_validity_analysis.json', dict(status='MEASURED_OFFLINE_CPU_REPLAY', summaries=summaries,
          score_kind='UNCALIBRATED_RULE_SCORE', events_path='offline_requests.jsonl',
          source=str(source), original_answer_source='saved actual native target generations',
          current_world_source='corresponding saved real M1 committed snapshots; not detector simulation'))
    witness_count = sum(len(e.get('current_observed_relevant_facts', [])) for e in events)
    unique_witnesses = {(e['frame_id'], a['anchor_id']) for e in events
                        for a in e.get('current_observed_relevant_facts', [])}
    write(root / 'm1_fact_coverage.json', dict(status='MEASURED_OFFLINE',
          candidate_referenced_current_object_witnesses=witness_count,
          unique_current_frame_anchor_pairs=len(unique_witnesses),
          unique_pairs=sorted(unique_witnesses), full_answers_eligible=sum(s['eligible'] for s in summaries),
          caveat='Witnesses are actual M1 predictions, not GT-confirmed facts. A relevant object does not validate appearance, negative coverage, signals, road relation or safety.'))
    write(root / 'cache_coverage_analysis.json', dict(status='MEASURED_OFFLINE', summaries=summaries,
          configured_coverages=pareto, actual_cache_hits=0, actual_M3_skips=0))
    write(root / 'quality_latency_pareto.json', dict(status='NO_QUALIFIED_PARETO_POINT', candidates=pareto,
          selection='NONE', development_source=str(source), heldout_tuning=False,
          thresholds_fixed_a_priori=True,
          caveat='Candidate epsilon labels are budgets to test, not score-to-error calibration. CUDA savings are theoretical on retained native trajectories, not measured E2E.'))
    no_hits = all(p['hit_count'] == 0 for p in pareto)
    status = 'NO_GO' if no_hits else 'INCOMPLETE'
    reason = ('No supported cache hit at any of the three predeclared DEV thresholds; current M1 observations do not cover full historical answers in this workload.'
              if no_hits else 'Some provisional DEV hits exist, but independent current-frame labels do not yet qualify a quality budget; online implementation remains gated.')
    write(root / 'go_no_go.json', dict(status=status, reason=reason,
          P0='COMPLETED_PARTIAL_QUALITY_COVERAGE', P1='COMPLETED',
          P2='NOT_RUN_OFFLINE_GATE', P3='NOT_RUN_OFFLINE_GATE',
          selected_quality_budget=.02, quality_budget_qualified=False,
          scope='Bounded retained v2 development streams; not a claim of general impossibility',
          default_deployment_modified=False, GPU_experiments=0))
    write(root / 'abc_comparison.json', dict(status='NOT_RUN_OFFLINE_GATE', A='Native OmniDrive',
          B='Target-Aware Exact History-Draft (v3 C exact path, not old v3 B)',
          C='B + M1 Answer Validity Cache', measurements=None,
          A_to_B=None, B_to_C=None, A_to_C=None, reason=reason))
    write(root / 'false_reuse_analysis.json', dict(status='NO_ONLINE_CACHE_EXECUTED',
          actual_false_reuse_rate=None, provisional_offline_false_reuse=[p for p in pareto if p['budget']],
          error_attribution='Offline candidate contradiction may be a historical model error or stale fact. No online reuse performed; current-vs-source GT required for attribution.',
          serious_driving_errors='See active_request_findings.json for historical v3 partial object-existence findings; no new WS-AVC safety certification.'))
    costs = [e['check_wall_ms'] + e['memory_and_dependency_wall_ms'] for e in events]
    cost = dict(status='MEASURED_OFFLINE_CPU_REPLAY', requests=len(events),
                mean_check_ms=average([e['check_wall_ms'] for e in events]),
                mean_dependency_memory_ms=average([e['memory_and_dependency_wall_ms'] for e in events]),
                mean_total_ms=average(costs), max_total_ms=max(costs),
                online_M123=None)
    write(root / 'runtime_cost.json', cost)
    (root / 'runtime_cost_report.md').write_text(f"# CPU replay cost\n\n{len(events)} requests; mean retrieval/scoring {cost['mean_check_ms']:.4f} ms; dependency/memory {cost['mean_dependency_memory_ms']:.4f} ms; combined {cost['mean_total_ms']:.4f} ms. These are isolated CPU replay costs, excluding GPU waiting and current M1 snapshot construction. Online M123 and GPU peak memory: NOT_RUN.\n")
    (root / 'offline_feasibility_report.md').write_text('# Bounded offline feasibility\n\n' + reason + '\n\n' +
        '\n'.join(f"- {s['workload']}: {s['requests']} requests, {s['candidate_requests']} history candidates, {s['eligible']} locally eligible answers; rejection reasons {s['rejection_reasons']}." for s in summaries) +
        '\n\nOnly existing v2 development scenes/logs were used for scoring thresholds. Natural questions were never replaced by repeated questions. Fixed questions and their actual generated answers were replayed separately on their real changing frames. No target forward was run. Quality references are read after the causal decision. Missing current-frame labels remain UNKNOWN.\n\n' +
        'No online runtime was added: the requested P1 stop condition precedes P2/P3. No finite threshold search can turn missing road/signal/safety evidence into object evidence. No quality budget or E2E benefit is claimed.\n')
    (root / 'e2e_performance_report.md').write_text('# Full M123 A/B/C\n\nNOT_RUN_OFFLINE_GATE. RTX3090 availability was checked successfully on the host. Offline evidence does not authorize a useful cache configuration. Actual skips: 0. FPS/QPS, GPU memory and paired scene speedups: NOT_RUN. Prior v3 timing is not relabeled as WS-AVC timing.\n')
    (root / 'result_answers.md').write_text('# Research answers\n\n' + '\n\n'.join([
        f"1. M1 provides {witness_count} candidate-referenced current object witnesses ({len(unique_witnesses)} unique frame/anchor pairs), with class/ego geometry and partial identity. These are M1 estimates, not GT certifications. M1 lacks signal state, visual appearance, global safety and complete negative coverage. Details: m1_fact_coverage.json and offline_requests.jsonl.",
        '2. Local positive object existence/category questions have the best supported potential; actual highest verified reuse type: NONE in this cohort. Broad scene, road, signal and safety questions lack M1 fact coverage.',
        '3. The current-frame selected-task-fact validity rates, with UNKNOWN excluded only from the scored denominator, are in answer_validity_analysis.json. They are not full-answer validity rates; that rate is UNKNOWN.',
        '4. Threshold-specific error rates are in quality_latency_pareto.json. With no scored predicted positives, precision/false-reuse is null, not perfect. Full validity accuracy is UNKNOWN.',
        f"5. For budgets 1/2/3pp, provisional DEV coverage is {[(p['workload'],p['budget'],p['coverage']) for p in pareto if p['budget']]}. No budget is empirically qualified.",
        '6. Zero requests actually skipped current M3 expansion, Prefill or Decode. P2 was gated out.',
        f"7. Isolated CPU replay combined mean is {cost['mean_total_ms']:.4f} ms. Online current snapshot/check/management cost is NOT_RUN.",
        '8. Fixed QA WS-AVC full E2E speedup: NOT_RUN.',
        '9. Original natural QA WS-AVC full E2E speedup: NOT_RUN.',
        '10. B→C independent M1 E2E benefit: NOT_ESTABLISHED. Old v3 B is not the requested target-aware baseline.',
        '11. Online cache-hit quality loss: UNKNOWN (zero online hits). Active historical v3 QA/annotation coverage is reported separately; partial annotation checks cannot qualify 2pp driving quality.',
        '12. No new online false reuse occurred. Existing v3 object-existence contradictions arise in actual model generations. Separating a bad historical answer from temporal invalidation requires reliable labels at both source and current frame; incomplete labels leave attribution UNKNOWN.',
        '13. No qualified quality–E2E Pareto configuration. Keep Native deployment. Three offline rule-score thresholds are candidates only.',
        f'14. {reason}',
        '15. Answer-level reuse is still a plausible direction for local object facts, but these retained streams do not support it as an empirically validated M3 systems-paper contribution. Broad questions require evidence M1 currently does not expose; do not add templates to force GO.'
    ]) + '\n')
    return status


def run(root, active_source, development_source, infos_path, data_root):
    root.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path('tools/worldstream_m123/answer_validity_quality.py'),
               Path('worldstream/m3/answer_validity.py'), Path('worldstream/m3/answer_memory.py')]
    bindings = {str(p): sha256(p) for p in sources}
    source_paths = [active_source / 'e2e_trials.jsonl', development_source / 'e2e_trials.jsonl']
    active_rows, dev_rows = [lines(p) for p in source_paths]
    active_scenes = {r['scene_id'] for r in active_rows}
    dev_scenes = {r['scene_id'] for r in dev_rows}
    active_logs = {r['log_id'] for r in active_rows}
    dev_logs = {r['log_id'] for r in dev_rows}
    if active_scenes & dev_scenes or active_logs & dev_logs:
        raise ValueError('P0 audit and P1 development must have disjoint scenes AND logs')
    write(root / 'experiment_plan.json', dict(status='FROZEN_BOUNDED_REPLAY',
          P0_source=str(active_source), P1_source=str(development_source),
          P0_scenes=sorted(active_scenes), P1_scenes=sorted(dev_scenes),
          P0_logs=sorted(active_logs), P1_logs=sorted(dev_logs),
          independent_scenes_and_logs=True, threshold_candidates=QUALITY_CANDIDATES,
          online_gate='Zero eligible hits => NO_GO; unqualified current-frame quality => INCOMPLETE. No P2/P3 in either case.',
          future_heldout_selection='NOT_RUN; no online cache policy qualified',
          thresholds_tuned_on_test=False))
    frame_ids = {r['frame_id'] for r in active_rows + dev_rows}
    annotations = load_current_annotations(infos_path, frame_ids, data_root)
    write(root / 'current_frame_annotation_facts.json', dict(status='OFFLINE_ONLY', frames=annotations))
    active_audit(root, active_source, annotations)
    events = offline(root, development_source, annotations)
    status = summarize(root, events, development_source)
    (root / 'answer_dependency_design.md').write_text('''# Answer dependencies and scoring

AnswerRecord retains the real target-generated question/prompt tokens, complete answer text/tokens, original scene/frame/timestamp, committed version/snapshot and model/tokenizer/generation binding. Optional answer dependencies contain classes, directions, motion, semantic attributes, related anchors and confidence. Provenance/statistics are scalar bounded tuples; no GPU tensors. Cache observations never refresh source age or create an independent target validation.

Answer-level semantic matching applies only to local object tasks with the same class/direction/proximity/selector/qualifier signature. Unsupported task types require exact question strings. This is deliberately separate from unchanged v3 token-draft parsing. Car subclasses map to car; ego mentions in road-type questions do not create vehicle-category dependencies.

The scorer reads only source/current committed actual-perception observations. Epoch, scene, version, execution binding, original source age and camera observation support are checked. It checks all extracted answer classes and claimed directions, not every object in the scene. Partial identity raises risk instead of uniformly rejecting all requests. Motion statements require real velocity evidence. Unobserved objects never establish negative existence. Signal, safety, count, appearance and road/behavior dependencies are recorded as unknown or unsupported.

Provisional risk weights are semantic .35, geometry .20, identity .10, staleness .15, unknown .20. Score = 1 - weighted risk. Local geometry movement is scaled by 10m, age by 4s, detector support floor .35; near <=30m and directly |ego_y|<=4m are explicit operational scope rules. These scores are uncalibrated. The .95/.90/.85 thresholds are only three a-priori candidates to evaluate against 1/2/3pp empirical task-quality budgets; a score is never advertised as an error probability.

Online M1/M2 tracking, asynchronous-camera snapshot construction, M3Runtime, executor and formal deployment configuration remain unchanged. P1 produced no qualified cache policy, so no online cache path was added. Existing v3 History-Draft/Target-Aware behavior remains available.

Offline labels are read by tools only after decisions. Current QA fact recall is separate from partial annotation-witness consistency. Camera projection uses each camera exposure pose; visibility>=3 and sensor-supported current annotations supply positive witnesses. Missing/occluded/unmapped objects are UNKNOWN. Annotation conflicts are automated findings for review, not human-certified severe driving errors. No Native output is truth.
''')
    (root / 'implementation_summary.md').write_text(f'''# WS-AVC bounded implementation

Research decision: **{status}**, scoped to retained development streams. See go_no_go.json.

Implemented lightweight answer dependencies, uncalibrated validity scoring, original-generation provenance/statistics, current-frame annotation audit and a reproducible P0/P1 study in the existing worldstream/tools tree. The complete saved answer is evaluated; no token-prefix agreement or current-answer oracle is required. No model, weight, training, M1/M2 algorithm or new inference framework was introduced.

P0 reuses v3 real GPU outputs and separates active target verification from requests without reuse. P1 uses existing v2 DEV scenes/logs, retaining natural and fixed QA separately. All original questions and current frames are retained; unavailable labels are explicit UNKNOWN. Source files and input artifacts are SHA-bound in execution_sources.json.

The offline gate failed. Following requirement sections 6.3 and 14, online runtime integration and new full-M123 GPU A/B/C were NOT_RUN. Default deployment and v3 generation paths are preserved. No E2E speedup, quality-budget qualification, safety certification or online cache skip is claimed. GPU availability was verified on the host RTX3090; resource availability did not cause this stop.

Tests and the final delivery commit are recorded in validation_summary.json and the final response. Raw request records are active_request_quality.jsonl and offline_requests.jsonl; related real M1 states are committed_worlds.json. Tensor captures were not duplicated.
''')
    (root / 'REPRODUCE.md').write_text('''# Reproduce P0/P1

Use OmniDrive Python and exclude faulty CPUs 4/5. All prerequisites are existing local nuScenes tables/metadata/QA, v2/v3 JSONL and the local pretrain_qformer tokenizer. No network, downloads or model forward is needed. Use a NEW output root; the study refuses to overwrite an existing result.

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_study --root outputs/m3_answer_validity_cache_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest -q tests
```

P0 source: outputs/m3_world_guided_decode_run03/e2e_trials.jsonl. Its B is the old non-target-aware verifier; its C is the existing target-aware world-guided path. Those historical conditions are never relabeled as WS-AVC A/B/C. P1 source: outputs/m3_world_guided_decode_run02/p3/e2e_trials.jsonl. Only round0 three-QA actual native outputs seed original history; corresponding C committed perception snapshots provide actual world evidence. Question tokenization uses only the existing local tokenizer.

P0 and P1 scene/log disjointness is enforced. All current answer/QA/annotation comparison happens after the causal validity decision. Predictions from M1 are not ground truth. Repeated rounds are not independent labels. QA scores are selected reference fact recall; annotation scores are partial positive-witness checks with coverage. Unknown facts cannot satisfy a quality budget.

Formal A/B/C: NOT_RUN_OFFLINE_GATE. Required future B is target-aware exact history; future C must use exactly that generator on cache misses. Do not reuse old v3 B as B. Actual online skips and new GPU model experiments are zero. Host CUDA regression tests (when available) use the existing /tmp/omnidrive_m123_latency_gpu.lock, check the GPU is idle, and never terminate other processes. CPU-only sandbox tests skip unavailable CUDA checks.
''')
    write(root / 'execution_sources.json', dict(status='COMPLETED',
          base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
          sources=bindings, artifacts={str(p): sha256(p) for p in source_paths},
          annotations=dict(infos_path=str(infos_path), infos_sha256=sha256(infos_path),
                           data_root=str(data_root), frame_ids=sorted(frame_ids),
                           tables={str(p): sha256(p) for p in (Path(data_root) / 'v1.0-trainval' / n for n in
                                    ('sample_annotation.json', 'instance.json', 'category.json'))},
                           QA_files={str(p): sha256(p) for token in sorted(frame_ids)
                                     for folder in ('conv/val', 'vqa/val')
                                     for p in [Path(data_root) / folder / (token + '.json')] if p.exists()}),
          CUDA_executed=False))
    if bindings != {str(p): sha256(p) for p in sources}:
        raise RuntimeError('Source changed during study')
    print(json.dumps(dict(status=status, requests=len(events), root=str(root))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--active-source', type=Path, default=V3)
    parser.add_argument('--development-source', type=Path, default=DEV)
    parser.add_argument('--infos', type=Path, default=Path('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl'))
    parser.add_argument('--data-root', type=Path, default=Path('data/nuscenes'))
    args = parser.parse_args()
    run(args.root, args.active_source, args.development_source, args.infos, args.data_root)
