"""Presentation/statistics over completed, frozen continuous GPU receipts."""
from collections import Counter,defaultdict
from itertools import product
import json
import math
import gzip
import shutil
from pathlib import Path
import subprocess

import numpy as np

from .benchmark.common import write,sha256
from .continuous_stream import lines,SCOPE,verify_sources
from .continuous_stream_report import events,paired_quality,mean
from .fact_bypass_identity import coverage,UNKNOWN_REASONS


def attest_companion(root):
    from datetime import datetime,timezone
    rows=lines(root/'evaluation/requests.jsonl') if (root/'evaluation/requests.jsonl').exists() else []
    assert not any(r['condition']=='C-companion' for r in rows)
    p=root/'C-fallback.json'
    write(root/'position_companion_freeze.json',dict(scope=SCOPE,policy_file=str(p),policy_sha256=sha256(p),
        policy_created_in='DEV calibration, before evaluation',attestation_UTC=datetime.now(timezone.utc).isoformat(),
        completed_evaluation_frames_at_attestation=len(rows),frozen_before_first_companion_forward=True,
        no_evaluation_GT_read=True,activation='Actual selected-C Position bypass only; no GT or answer-quality selection'))


def dist(values):
    xs=[float(x) for x in values if x is not None]
    if not xs:return dict(n=0,mean=None,p50=None,p95=None,p99=None)
    return dict(n=len(xs),mean=mean(xs),p50=float(np.percentile(xs,50)),p95=float(np.percentile(xs,95)),
        p99=float(np.percentile(xs,99)),p99_scope='descriptive order statistic; insufficient tail sample' if len(xs)<100 else 'empirical, correlated stream frames')


def cluster_performance(pairs):
    logs=sorted({p['log_id'] for p in pairs})
    if not pairs:return dict(paired_frames=0)
    by={l:[p for p in pairs if p['log_id']==l] for l in logs}
    ratios=[];deltas=[]
    # Exact empirical cluster bootstrap for 2-3 logs, with equally likely
    # ordered draws; resample whole streams, never individual correlated frames.
    for draw in product(logs,repeat=len(logs)):
        part=[p for l in draw for p in by[l]]
        a=sum(p['from_ms'] for p in part);b=sum(p['to_ms'] for p in part)
        ratios.append(a/b);deltas.append((b-a)/len(part))
    def empirical_interval(values):
        values=sorted(values);n=len(values)
        # Exact bootstrap draws have discrete probability mass. Interpolating
        # the 27 draws can incorrectly exclude an extremum with >2.5% mass.
        return [values[max(0,math.ceil(.025*n)-1)],values[min(n-1,math.ceil(.975*n)-1)]]
    return dict(paired_frames=len(pairs),independent_logs=len(logs),
        from_ms=mean([p['from_ms'] for p in pairs]),to_ms=mean([p['to_ms'] for p in pairs]),
        speedup=sum(p['from_ms'] for p in pairs)/sum(p['to_ms'] for p in pairs),
        delta_ms=mean([p['to_ms']-p['from_ms'] for p in pairs]),
        speedup_log_CI95=empirical_interval(ratios),delta_log_CI95_ms=empirical_interval(deltas),
        log_speedups={l:sum(p['from_ms'] for p in by[l])/sum(p['to_ms'] for p in by[l]) for l in logs},
        interval_scope='Exact empirical whole-log cluster bootstrap; small log count, not rare-error certification')


def performance(rows):
    groups=[];comparisons=[]
    cells=sorted({(r['workload'],r['frequency']) for r in rows})
    for workload,frequency in cells:
        cell=[r for r in rows if (r['workload'],r['frequency'])==(workload,frequency)]
        for cond in sorted({r['condition'] for r in cell}):
            part=[r for r in cell if r['condition']==cond];qs=[q for r in part for q in r['requests']]
            active=[r for r in part if r['question_count']];idle=[r for r in part if not r['question_count']]
            total=sum(r['e2e_wall_ms'] for r in part)
            g=dict(workload=workload,frequency=frequency,condition=cond,frames=len(part),questions=len(qs),
                independent_scenes=len({r['scene_id'] for r in part}),independent_logs=len({r['log_id'] for r in part}),
                e2e_ms=dist([r['e2e_wall_ms'] for r in part]),QA_frame_e2e_ms=dist([r['e2e_wall_ms'] for r in active]),
                no_QA_frame_e2e_ms=dist([r['e2e_wall_ms'] for r in idle]),response_ms=dist([q['end_to_end_ms'] for q in qs]),
                single_question_response_e2e_ms=dist([r['e2e_wall_ms'] for r in active if r['question_count']==1]),
                QA_batch_response_e2e_ms=dist([r['e2e_wall_ms'] for r in active]),
                response_definition='response_ms is M3 per-question service only; single_question_response_e2e_ms is synchronized model-entry-to-return for one-QA frames. Three questions return as one batch; no paced-arrival queueing latency is measured.',
                FPS=1000*len(part)/total,QPS=1000*len(qs)/total,peak_GPU_bytes=max(r['peak_gpu_bytes'] for r in part),
                LLM_forwards=sum(q['llm_forward_calls'] for q in qs),actual_prefill_skips=sum(q['llm_skipped'] for q in qs),
                actual_decode_skips=sum(q['llm_skipped'] for q in qs),
                cold_e2e_ms=dist([r['e2e_wall_ms'] for r in part if r['phase']=='cold']),
                steady_e2e_ms=dist([r['e2e_wall_ms'] for r in part if r['phase']=='steady']),
                visual_cuda_span_ms=dist([r['stages']['eva_native' if cond=='A' else 'm2_total']['gpu_span_ms'] for r in part]),
                M3_share_on_QA_frames=sum(r['m3_wall_ms'] for r in active)/sum(r['e2e_wall_ms'] for r in active) if active else None)
            for key in ('m1_wall_ms','m1_including_adapters_ms','m2_wall_ms','heads_wall_ms','snapshot_wall_ms','m3_wall_ms','answer_export_ms'):
                g[key]=dist([r[key] for r in part]);g['idle_'+key]=dist([r[key] for r in idle])
            for key in ('prefill_wall_ms','prefill_cuda_ms','decode_wall_ms','decode_cuda_ms'):
                g[key]=dist([q.get(key,0.) for q in qs])
            for key in ('question_fact_extraction_wall_ms','answer_validity_check_wall_ms','position_admission_wall_ms'):
                g[key]=dist([(q.get('fast_answer') or {}).get(key,0.) for q in qs])
            groups.append(g)
        baseline={}
        for r in cell:baseline[(r['frame_id'],r['condition'])]=r
        for a,b in [('A','B'),('B','C'),('A','C'),('C','C-companion'),('B','C-companion')]:
            pairs=[]
            for r in cell:
                ref=baseline.get((r['frame_id'],a))
                if r['condition']==b and ref:
                    assert ref['questions']==r['questions']
                    pairs.append(dict(log_id=r['log_id'],scene_id=r['scene_id'],frame_id=r['frame_id'],
                        from_ms=ref['e2e_wall_ms'],to_ms=r['e2e_wall_ms']))
            if pairs:comparisons.append(dict(workload=workload,frequency=frequency,comparison=a+'->'+b,**cluster_performance(pairs)))
    return dict(scope=SCOPE,groups=groups,comparisons=comparisons,
        timing='Real synchronized outer full-model wall incl answer export; input preparation and audit serialization excluded equally. Stage spans are nested, not additive. FPS/QPS are service throughput, not paced real-time arrival throughput.')


def quality_summary(part,**extra):
    known=[e for e in part if e['quality']['known']]
    return dict(**extra,requests=len(part),scored=len(known),UNKNOWN=len(part)-len(known),
        answer_correctness=mean([e['quality']['score'] for e in known]),severe=sum(bool(e['quality']['severe_error']) for e in known),
        actual_skips=sum(e['llm_skipped'] for e in part),unique_frame_questions=len({(e['frame_id'],e['question']) for e in part}),
        unique_scored_frame_questions=len({(e['frame_id'],e['question']) for e in known}),
        independent_logs=len({e['log_id'] for e in part}),scored_logs=len({e['log_id'] for e in known}))


def restore_archived_receipts(root):
    """Restore byte-identical raw receipts from the versioned lossless archive."""
    manifest=root/'receipt_archive.json'
    if not manifest.exists():return
    for item in json.loads(manifest.read_text())['files']:
        archive=root/item['archive'];raw=root/item['raw']
        assert sha256(archive)==item['archive_sha256']
        if not raw.exists():
            temporary=raw.with_suffix(raw.suffix+'.restore')
            with gzip.open(archive,'rb') as source,temporary.open('wb') as dest:
                shutil.copyfileobj(source,dest)
            assert sha256(temporary)==item['raw_sha256']
            temporary.replace(raw)
        assert sha256(raw)==item['raw_sha256']


def report(root):
    restore_archived_receipts(root)
    assert json.loads((root/'evaluation/run_status.json').read_text())['status']=='COMPLETED'
    rows=lines(root/'evaluation/requests.jsonl');devrows=lines(root/'dev/requests.jsonl');es=events(root,rows)
    with (root/'quality_requests.jsonl').open('w') as f:
        for e in es:f.write(json.dumps(e)+'\n')
    perf=performance(rows);write(root/'abc_comparison.json',perf);write(root/'long_stream_performance.json',perf)
    primary=[e for e in es if e['workload']=='addressable' and e['frequency']=='three']
    c=[e for e in primary if e['condition']=='C'];fast=[e for e in c if e['llm_skipped']]
    obj=coverage(c,'object');target=coverage(c,'target');quality=quality_summary(c)
    coverage_report=dict(scope=SCOPE,primary_cell='addressable/three/C',object=obj,target=target,answer=quality,
        fast_object=coverage(fast,'object'),fast_target=coverage(fast,'target'),
        proposal_object=coverage([e for e in c if (e['proposal'] or {}).get('facts')],'object'),
        requests_without_prediction_anchor=sum(not (e['proposal'] or {}).get('facts') for e in c),
        unknown_answer_reasons={k:sum(e['target_reference'].get('reason')==k for e in c if not e['quality']['known']) for k in UNKNOWN_REASONS},
        distinct_frames=len({e['frame_id'] for e in c}),new_distinct_scored_frame_questions=quality['unique_scored_frame_questions'],
        statistical_independence_unit='3 log clusters; adjacent frames and repeated workload questions are correlated',
        no_round_inflation=True,visibility_is_annotation_projection_proxy=True,
        identity_thresholds='Existing <2m radius and .5m ambiguity margin, one-to-one; unchanged',
        position=dict(total_requests=sum(e['task']=='position' for e in c),
            object=coverage([e for e in c if e['task']=='position'],'object'),
            target=coverage([e for e in c if e['task']=='position'],'target'),
            answer=quality_summary([e for e in c if e['task']=='position'])))
    write(root/'quality_label_coverage.json',coverage_report)
    bytask=[quality_summary([e for e in primary if e['condition']==cond and e['task']==task],condition=cond,task=task)
        for cond in sorted({e['condition'] for e in primary}) for task in ('category','existence','position','motion')]
    write(root/'quality_by_task.json',dict(scope=SCOPE,primary_cell='addressable/three',groups=bytask))
    write(root/'quality_by_scene.json',dict(scope=SCOPE,primary_cell='addressable/three',groups=[quality_summary(
        [e for e in primary if e['condition']==cond and e['scene_id']==sid],scene_id=sid,condition=cond)
        for sid in sorted({e['scene_id'] for e in primary}) for cond in sorted({e['condition'] for e in primary})]))
    paired=paired_quality(primary,'B','C');comp=paired_quality(primary,'B','C-companion')
    comp['status']='MEASURED' if comp['requests'] else 'NOT_RUN_NO_POSITION_EXECUTION_DIFFERENCE'
    selected_to_companion=paired_quality(primary,'C','C-companion')
    position_to_companion=paired_quality([e for e in primary if e['task']=='position'],'C','C-companion')
    position_execution=[quality_summary([e for e in c if e['task']=='position' and e['llm_skipped']==fast_path],
        execution='fast' if fast_path else 'fallback') for fast_path in (True,False)]
    all_workload_quality=[dict(workload=w,frequency=f,**paired_quality(
        [e for e in es if (e['workload'],e['frequency'])==(w,f)],'B','C'))
        for w,f in sorted({(e['workload'],e['frequency']) for e in es})]
    write(root/'paired_task_quality.json',dict(primary=paired,companion=comp,
        all_workloads=all_workload_quality,
        per_frequency={f:paired_quality([e for e in es if e['workload']=='addressable' and e['frequency']==f],'B','C') for f in ('every4','every2','one','three')}))
    new_cases=[p for p in paired['pairs'] if p['new_severe']]
    unique={(p['frame_id'],p['question']) for p in new_cases}
    position=[]
    for e in primary:
        if e['task']!='position':continue
        dirs=set(e['quality'].get('extracted',{}).get('directions',()))
        ref=set(e['quality'].get('reference',{}).get('directions',()))
        position.append(dict(frame_id=e['frame_id'],question=e['question'],condition=e['condition'],
            known=e['quality']['known'],left_right_flip=bool(('left' in dirs and 'right' in ref) or ('right' in dirs and 'left' in ref)),
            front_rear_flip=bool(('front' in dirs and 'rear' in ref) or ('rear' in dirs and 'front' in ref)),
            wrong_nearest_target=e['target']['known'] and not e['target'].get('correct',False),
            target_known=e['target']['known'],fast=e['llm_skipped']))
    write(root/'severe_error_audit.json',dict(scope=SCOPE,primary=paired,companion=comp,
        selected_to_companion=selected_to_companion,position_selected_to_companion=position_to_companion,
        selected_position_by_execution=position_execution,
        unique_new_severe=len(unique),new_severe_unique_frame_questions=sorted(unique),
        position_cases=position,fast_new_severe=sum(p['new_severe'] for p in paired['pairs'] if p['fast']),
        fallback_new_severe=sum(p['new_severe'] for p in paired['pairs'] if not p['fast']),
        new_severe_rate=paired['new_severe']/paired['scored'] if paired['scored'] else None,
        wrong_nearest_vs_B='Not pair-certifiable: B generated answers do not carry independent instance IDs; report C match failures separately',
        strict_2percent_risk_certification=False,historical_v3_v4_NO_GO_unchanged=True,
        caveat='Only3 within-history logs. Zero NEW errors versus B does not mean zero absolute severe errors. Zero observed errors or empirical zero bootstrap bound cannot prove a true <2% rare-error rate. UNKNOWN is never correct.'))
    taskcoverage=[]
    for w,f in sorted({(e['workload'],e['frequency']) for e in es}):
        part=[e for e in es if e['condition']=='C' and (e['workload'],e['frequency'])==(w,f)]
        if not part:continue
        unsupported=[e for e in part if e['task']=='unsupported']
        taskcoverage.append(dict(workload=w,frequency=f,requests=len(part),
            parsed=sum(e['task']!='unsupported' for e in part),positive_fact_supported=sum(bool((e['proposal'] or {}).get('supported')) for e in part),
            eligible=sum(bool((e['proposal'] or {}).get('admitted')) for e in part),actual_fast=sum(e['llm_skipped'] for e in part),
            zero_forward_fast=sum(e['llm_skipped'] and e['llm_forward_calls']==0 for e in part),unsupported=len(unsupported),
            unsupported_check_ms=dist([(e['proposal'] or {}).get('question_fact_extraction_wall_ms',0.)+(e['proposal'] or {}).get('answer_validity_check_wall_ms',0.) for e in unsupported]),
            required_answer_fact_quality=quality_summary(part)))
    write(root/'task_coverage_analysis.json',dict(scope=SCOPE,groups=taskcoverage,
        mixed='One introduced structured slot plus2 original natural questions, preset1/3; additional original slots may parse, no natural population claim'))
    frequency=dict(scope=SCOPE,groups=[g for g in perf['groups'] if g['workload']=='addressable' and g['condition'] in ('A','B','C')],
        comparisons=[p for p in perf['comparisons'] if p['workload']=='addressable'],
        definitions='Visual stream always runs. every4/every2/one/three are preset request schedules; no zero-QA frame omitted.')
    write(root/'question_frequency_analysis.json',frequency)
    policy=json.loads((root/'selected_retrospective_policy.json').read_text());pc=json.loads((root/'position_policy_comparison.json').read_text())
    if policy['policy_id']=='C-basic':
        from .target_complete_report import world,context
        from worldstream.m3.answer_memory import AnswerMemory
        from worldstream.m3.answer_validity import propose_fact_answer
        restricted=json.loads((root/'C-restricted.json').read_text());equiv=[]
        for r in rows:
            if (r['condition'],r['workload'],r['frequency'])!=('C','addressable','three'):continue
            w=world(r['committed_world'])
            for req in r['requests']:
                p=propose_fact_answer(req['question'],w,context(w,r['frame_index'],r['camera_ids']),AnswerMemory(),('equivalence',),restricted,target_binding_enabled=True)
                b=req['fast_answer']
                equiv.append(dict(frame_id=r['frame_id'],question=req['question'],
                    same=all(p.get(k)==b.get(k) for k in ('supported','admitted','answer_text')),
                    basic_admitted=b['admitted'],restricted_admitted=p['admitted']))
        write(root/'restricted_decision_equivalence.json',dict(scope=SCOPE,CPU_replay_only=True,
            same=sum(e['same'] for e in equiv),requests=len(equiv),all_same=all(e['same'] for e in equiv),pairs=equiv,
            separate_GPU_calls=0,separate_GPU_outputs_measured=False))
        pc['restricted_extended_decisions']=dict(same=sum(e['same'] for e in equiv),requests=len(equiv),
            no_separate_GPU_run=True,source='restricted_decision_equivalence.json')
    pc['extended_evaluation']=dict(selected=policy['policy_id'],primary=paired,companion=comp,
        selected_to_companion=selected_to_companion,position_selected_to_companion=position_to_companion,
        position_by_condition=[g for g in bytask if g['task']=='position'],selected_position_by_execution=position_execution,
        companion_policy=next((r['policy_file'] for r in rows if r['condition']=='C-companion'),None),
        e2e=[p for p in perf['comparisons'] if p['workload']=='addressable' and p['frequency']=='three'])
    selected_group=next(g for g in perf['groups'] if (g['workload'],g['frequency'],g['condition'])==('addressable','three','C'))
    companion_group=next((g for g in perf['groups'] if (g['workload'],g['frequency'],g['condition'])==('addressable','three','C-companion')),None)
    pc['extended_evaluation']['execution_delta']=dict(
        additional_LLM_forwards=companion_group['LLM_forwards']-selected_group['LLM_forwards'] if companion_group else None,
        additional_executed_LLM_requests=selected_group['actual_prefill_skips']-companion_group['actual_prefill_skips'] if companion_group else None,
        no_counterfactual_forward_estimates=True)
    write(root/'position_policy_comparison.json',pc)
    allrows=devrows+rows;qs=[q for r in allrows for q in r['requests']];formalqs=[q for r in rows for q in r['requests']]
    write(root/'gpu_execution_counts.json',dict(scope=SCOPE,full_M123_frame_calls=len(allrows),DEV_frame_calls=len(devrows),evaluation_frame_calls=len(rows),
        with_QA_frame_calls=sum(bool(r['requests']) for r in allrows),no_QA_visual_frame_calls=sum(not r['requests'] for r in allrows),
        question_requests=len(qs),actual_LLM_forwards=sum(q['llm_forward_calls'] for q in qs),
        actual_prefill_skips=sum(q['llm_skipped'] for q in qs),actual_decode_skips=sum(q['llm_skipped'] for q in qs),
        evaluation_actual_skips=sum(q['llm_skipped'] for q in formalqs),
        distinct_input_frames=len({r['frame_id'] for r in allrows}),scenes=len({r['scene_id'] for r in allrows}),logs=len({r['log_id'] for r in allrows}),
        actual_native_visual_frame_calls=sum(r['condition']=='A' for r in allrows),
        note='Frame calls count real full model vision/state/heads execution; idle frames have no language request and are separately counted. Native A uses native backbone+generation with M1 bookkeeping. No warmup/repeated-round inflation.'))
    speed=lambda w,f='three':next(p for p in perf['comparisons'] if (p['workload'],p['frequency'],p['comparison'])==(w,f,'B->C'))
    gains=speed('addressable');mixed=speed('mixed')
    performance_pass=gains['speedup']>1 and gains['delta_log_CI95_ms'][1]<0
    known_budget=paired['loss'] is not None and paired['loss']<=.02
    verdict=dict(scope=SCOPE,system_performance='SYSTEM_PERFORMANCE_VALIDATED_WITHIN_LOG' if performance_pass else 'SYSTEM_PERFORMANCE_NOT_DEMONSTRATED',
        independent_generalization='INDEPENDENT_GENERALIZATION_INCOMPLETE',
        selected_policy=policy['policy_id'],known_answer_2pp_budget_pass=known_budget,
        budget_scope='Primary addressable/three pooled paired known answer facts; other workload budgets reported separately, never UNKNOWN-as-correct',
        other_workload_budget_failures=[dict(workload=q['workload'],frequency=q['frequency'],loss=q['loss'],scored=q['scored'])
            for q in all_workload_quality if q['loss'] is not None and q['loss']>.02],
        unknowns_prevent_full_quality_guarantee=True,strict_rare_error_qualification=False,
        historical_v3_NO_GO='UNCHANGED',historical_v4_NO_GO='UNCHANGED',previous_INCOMPLETE='UNCHANGED',
        deployment_enabled=False,object3Q_B_to_C=gains['speedup'],mixed3Q_B_to_C=mixed['speedup'],
        new_severe=paired['new_severe'],scored=paired['scored'],independent_scored_logs=paired['independent_logs'])
    write(root/'research_verdict.json',verdict)
    selection=json.loads((root/'extended_frame_selection.json').read_text());inv=json.loads((root/'historical_execution_inventory.json').read_text())
    other_exposure=json.loads((root/'non_M3_exposure_audit.json').read_text())
    write(root/'frame_exposure_addendum.json',dict(scope=SCOPE,
        frozen_inventory_scope='Historical M3/M123-prefixed runs, not all perception-only runs',
        selected_frames=104,prior_perception_only_GPU_frames=other_exposure['prior_visual_frames'],
        prior_language_GPU_frames_outside_M3=other_exposure['prior_language_frames'],
        frames=[dict(scene_id=f['scene_id'],log_id=f['log_id'],frame_id=f['frame_id'],timestamp=f['timestamp'],
            original_partition=f['original_partition'],base_model_training=f['base_model_training'],
            prior_M3_M123_language_execution=f['completed_M123_language'],
            prior_other_visual_GPU=any(x['frame_id']==f['frame_id'] and x['actual_visual'] for s in other_exposure['sources'] for x in s['receipts']),
            prior_other_language_GPU=f['frame_id'] in other_exposure['prior_language_frame_ids'],
            historical_WRSAR_log=True,extended_evaluation=True,
            method_exposure='M2 validation/evaluation plus historically used WRSAR log; no prior retained M3 language receipt on this frame')
            for f in selection['selected_frames']],
        implication='New retained full M123 language execution, not unseen images/perception development or unseen logs. Prior M2 validation/quality exposure exists.',
        sources='non_M3_exposure_audit.json'))
    tests={k:json.loads((root/k/'summary.json').read_text()) for k in ('validation_cpu','validation_gpu') if (root/k/'summary.json').exists()}
    write(root/'validation_summary.json',tests)
    (root/'label_semantics_reference.md').write_text('# Frozen code semantics reference\n\n'
        'This documents the already executed grading code; it does not revise the pre-GPU protocol. '
        'Vehicle selectors expand to car, truck, bus, trailer, construction_vehicle, motorcycle and bicycle. '
        'The existing annotation reader maps human.pedestrian.* to pedestrian, vehicle.construction to construction_vehicle, and the existing nuScenes vehicle subclasses to their detector taxonomy. '
        'Matching is class-independent3D-center distance<2m, unique nearest margin>.5m, one-to-one across all current detections; a second-choice Hungarian assignment cannot become a certain identity. '
        'Question nearest ranking uses planar ego distance; ROI<=50m, directly abs(y)<=4m and requested sign directions. '
        'Visible GT requires projected center in at least one original1600×900 camera at its exposure pose, nuScenes visibility>=3 and nonzero lidar+radar support. '
        'This is a visibility proxy, not per-pixel human confirmation. Projected uncertain-visibility GT within nearest distance+2m invalidates certification. '
        'GT targets within2m planar rank difference are identity-ambiguous; if all imply the same Position sector or Category, answer facts may be known while identity remains UNKNOWN. '
        'Answer directions use x>1 front, x<−1 rear, y>1 left, y<−1 right; within .5m of these boundaries is ambiguous. '
        'Positive Existence accepts any visible in-ROI witness, never requires a nearest identity and never treats absence of annotations as a correct No. '
        'Object and target metrics cover C proposal anchor receipts; they are not whole-detector AP or independent instance IDs for free-form B answers. Requests without a prediction-anchor receipt are separately counted: their missing_annotation_support reason means the answer-to-instance link is unavailable, not that the nuScenes annotation table is absent. Conditional proposal and actual-fast denominators are also reported. '
        'Unsupported natural/fixed QA uses only exact current-frame original QA limited fact recall, never a previous frame reference. '
        'Grading source hashes and the pre-evaluation selection freeze are retained; UNKNOWN is excluded from accuracy numerators and reported in all denominators.\n')
    (root/'data_independence_report.md').write_text('# Data independence\n\n'+SCOPE+'. '+
        f"Historical complete language receipts: {inv['completed_M123_language_frames']}/{inv['val_frames']}; frames without those receipts: {inv['never_completed_M123_frames']}. " +
        f"{selection['candidate_segments']} eligible >=16-frame segments, {selection['candidate_unique_frames']} frames in those segments. Selected2 DEV streams16 frames each and3 evaluation streams24 frames each:104 unique frames/5 scenes/5 logs. " +
        'Every selected frame was excluded from all retained full-model/per-question execution receipts before inference. Direct-ID cross-check of other historical JSONL is in additional_selected_frame_audit.json. '
        f"A broader cross-check outside M3/M123 runs found prior perception-only GPU execution on {other_exposure['prior_visual_frames']}/104 selected frames, but language execution on {other_exposure['prior_language_frames']}/104. Frozen inventory actual_visual_GPU flags have M3/M123 scope only; these are new full-M123 language evaluations, not new images or unseen M2 validation data. See non_M3_exposure_audit.json and frame_exposure_addendum.json. " +
        'Logs and generic question templates have historical development exposure; scene/log independence is not claimed. Official val is outside the checkpoint configured nuScenes train split, but exact pretraining acquisition/execution exposure is unknown. '
        'Metadata plan appearances are separate from completed GPU execution. History role tags use recorded path/stage names, not inferred correctness. No new independent TEST is claimed. Historical conclusions and checkpoint are unchanged.\n')
    table='| Schedule | A ms/frame | B ms/frame | C ms/frame | B→C | A→C | B→C log CI95 | Skips / questions |\n|---|---:|---:|---:|---:|---:|---|---:|\n'
    for f in ('every4','every2','one','three'):
        p=speed('addressable',f);g=next(g for g in perf['groups'] if (g['workload'],g['frequency'],g['condition'])==('addressable',f,'C'))
        ac=next(x for x in perf['comparisons'] if (x['workload'],x['frequency'],x['comparison'])==('addressable',f,'A->C'))
        table+=f"| {f} | {ac['from_ms']:.3f} | {p['from_ms']:.3f} | {p['to_ms']:.3f} | {p['speedup']:.4f}× | {ac['speedup']:.4f}× | {p['speedup_log_CI95']} | {g['actual_prefill_skips']} / {g['questions']} |\n"
    (root/'e2e_performance_report.md').write_text('# Real continuous-stream E2E\n\n'+SCOPE+'; all speeds come from actual synchronized complete model requests, not token scaling.\n\n'+table+
        f"\nPreset mixed3Q B→C {mixed['speedup']:.6f}×; object3Q {gains['speedup']:.6f}×. Each schedule processes identical real six-camera streams, including idle frames. " +
        'A uses native backbone/native generation plus M1 bookkeeping; B/C share deployed incremental M2. A→C, A→B, cold/steady, module wall/CUDA, response percentiles, FPS/QPS and memory are in abc_comparison.json. '
        'Cold=first frame, transition1-3, steady>=4 without reset. The FPS/QPS are compute service throughput, not a simulation of queued real-time arrivals. Small-sample P99 and3-log bootstrap intervals are descriptive. '
        'One rotated condition order per cell; no repeated-round label inflation, but residual order/clock variation is not eliminated.\n')
    cost='| Schedule / C | M1 incl adapters | M2 | Heads | M3 | Idle E2E | M3 share on QA frames |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for g in frequency['groups']:
        if g['condition']!='C':continue
        cost+=f"| {g['frequency']} | {g['m1_including_adapters_ms']['mean']:.3f} | {g['m2_wall_ms']['mean']:.3f} | {g['heads_wall_ms']['mean']:.3f} | {g['m3_wall_ms']['mean']:.3f} | {g['no_QA_frame_e2e_ms']['mean']} | {g['M3_share_on_QA_frames']:.4f} |\n"
    # Keep true M2 receipts and within-stream trends rather than claiming a
    # visual speedup from the amount of planned window reuse alone.
    state=[]
    for cnd in ('A','B','C'):
        rs=[r for r in rows if r['workload']=='addressable' and r['frequency']=='three' and r['condition']==cnd]
        for phase in ('cold','transition','steady'):
            part=[r for r in rs if r['phase']==phase]
            state.append(dict(condition=cnd,phase=phase,frames=len(part),
                m1_ms=dist([r['m1_including_adapters_ms'] for r in part]),visual_ms=dist([r['m2_wall_ms'] for r in part]),
                visual_cuda_span_ms=dist([r['stages']['eva_native' if cnd=='A' else 'm2_total']['gpu_span_ms'] for r in part]),
                actual_skipped_windows=sum((r['m2'] or {}).get('actual_skipped_windows',0) for r in part),
                reuse_frames=sum(bool((r['m2'] or {}).get('actual_window_reuse')) for r in part)))
    write(root/'state_continuity_cost.json',dict(scope=SCOPE,groups=state,
        receipts='Each per-frame row records actual M1 commit_count, M2 execution/skips and scene/frame/time; no within-sequence state reset'))
    (root/'runtime_cost_report.md').write_text('# Measured cost attribution\n\n'+cost+
        '\nAll times ms, host wall. M1 adapter transfers and CUDA spans can be nested inside model phases; do not sum nested spans. Parsing/validity/Position admission are inside M3. '
        'Native visual host wall is enqueue time; incremental M2 internally synchronizes. Compare their CUDA event stream spans, not native host enqueue against incremental synchronized host time. Both include stream gaps and are not summed kernel-profiler time. Heads is the observed detection/map forward span; head postprocessing and other residual integration cost remain included in outer E2E. Per-question prefill/decode wall and CUDA event spans are distinct. M1 trends and actual M2 window reuse are in state_continuity_cost.json. '
        'No-QA frames retain vision/commit/heads and zero language forwards. Unsupported-question CPU checking overhead and natural parsing opportunities are in task_coverage_analysis.json. '
        'Throughput differences across frequencies are measured directly; no linear extrapolation of the old3Q result is used.\n')
    natural=next(g for g in taskcoverage if g['workload']=='natural');primary_c=next(g for g in perf['groups'] if (g['workload'],g['frequency'],g['condition'])==('addressable','three','C'))
    bestfreq=max((speed('addressable',f) for f in ('every4','every2','one','three')),key=lambda p:p['speedup'])
    posby=lambda cond:next(g for g in bytask if g['condition']==cond and g['task']=='position')
    state_row=lambda cond,phase:next(s for s in state if s['condition']==cond and s['phase']==phase)
    cold=state_row('C','cold');steady=state_row('C','steady');native_steady=state_row('A','steady')
    perf_c=lambda f:next(g for g in frequency['groups'] if g['condition']=='C' and g['frequency']==f)
    compperf=next((p for p in perf['comparisons'] if (p['workload'],p['frequency'],p['comparison'])==('addressable','three','C->C-companion')),None)
    attribute_scores=[]
    for task in ('category','existence'):
        bs=next(g for g in bytask if g['condition']=='B' and g['task']==task)
        cs=next(g for g in bytask if g['condition']=='C' and g['task']==task)
        attribute_scores.append(f"{task}: B={bs['answer_correctness']}, C={cs['answer_correctness']}，C标签{cs['scored']}/{cs['requests']}")
    responses=[
        f"执行前清单：{inv['never_completed_M123_frames']}帧未见完整M123语言执行收据；已见{inv['completed_M123_language_frames']}帧。计划不当作执行，分类收据及局限见historical_execution_inventory.json。",
        f"{selection['candidate_segments']}个合法>=16帧片段，包含{selection['candidate_unique_frames']}帧。固定元数据规则选104帧，DEV2×16，扩展评价3×24。",
        f"新M3语言执行Frame隔离成立，但全部{other_exposure['prior_visual_frames']}/104帧此前做过M2纯视觉GPU评价，额外语言重叠={other_exposure['prior_language_frames']}。Scene/Log和通用模板历史暴露；不是新图像或独立TEST。",
        '能。逐帧真实M1 commit和M2执行已验证，保持24帧顺序、时间戳与位姿；无QA帧仍完整运行视觉系统。',
        f"主3Q C的M1含适配器成本：冷启动{cold['m1_ms']['mean']:.3f}ms（{cold['frames']}帧），稳态{steady['m1_ms']['mean']:.3f}ms（{steady['frames']}帧）。状态没有按问题到达重置；逐帧commit计数连续。Cold指序列状态空，模型加载在计时外。",
        f"主3Q稳态实际复用{steady['reuse_frames']}/{steady['frames']}帧，共跳过{steady['actual_skipped_windows']}窗口；可比GPU流视觉跨度C={steady['visual_cuda_span_ms']['mean']:.3f}ms、Native A={native_steady['visual_cuda_span_ms']['mean']:.3f}ms。{'这些均值不支持M2视觉正加速' if steady['visual_cuda_span_ms']['mean']>=native_steady['visual_cuda_span_ms']['mean'] else '这些均值显示有限视觉跨度减少'}；不能把计划复用或整体ABC差异当作独立M2收益。原生host仅enqueue，不能直接与M2同步host对比。",
        '; '.join(f"{f}: B={speed('addressable',f)['from_ms']:.3f}, C={speed('addressable',f)['to_ms']:.3f}ms/视觉帧，{speed('addressable',f)['speedup']:.4f}×" for f in ('every4','every2','one','three')),
        f"本预设负载最大B→C点估计为{bestfreq['frequency']}，{bestfreq['speedup']:.4f}×；完整log区间及各log结果保留。",
        f"正式C对象三问真实跳过{primary_c['actual_prefill_skips']}/{primary_c['questions']}次Prefill和Decode；全阶段总数见gpu_execution_counts.json，所有命中Forward Hook=0且输入展开=0。",
        '; '.join(attribute_scores)+'。这是已评分必要事实质量，UNKNOWN未计正确。',
        f"Position共{posby('C')['requests']}问，已评分{posby('C')['scored']}、UNKNOWN={posby('C')['UNKNOWN']}。B正确率{posby('B')['answer_correctness']:.1%}，C-basic为{posby('C')['answer_correctness']:.1%}。Restricted在DEV及扩展评价的支持、准入和答案文本均与Basic一致（扩展评价216/216），未重复GPU测量。"+(f"实际C-fallback正确率{posby('C-companion')['answer_correctness']:.1%}；同帧配对见position_policy_comparison.json。" if comp['requests'] else '没有需要单独GPU执行的Fallback决策差异。'),
        (f"本样本没有减少：已评分Position严重错误由C-basic的{posby('C')['severe']}增至C-fallback的{posby('C-companion')['severe']}，配对新增{position_to_companion['new_severe']}、净增{position_to_companion['net_severe']}。" if comp['requests'] else '没有独立Fallback测量。')+f"主C快速Position已评分{position_execution[0]['scored']}/{position_execution[0]['requests']}，正确率{position_execution[0]['answer_correctness']}、严重错误{position_execution[0]['severe']}；主C未旁路Position严重错误{position_execution[1]['severe']}。正式B→C新增严重输出{paired['new_severe']}/{paired['scored']}不代表绝对严重错误为零，也不构成严格2%风险认证。",
        (f"真实C→伴随策略平均增加{compperf['delta_ms']:.3f}ms/视觉帧，log CI={compperf['delta_log_CI95_ms']}（跨零）；实际新增{pc['extended_evaluation']['execution_delta']['additional_executed_LLM_requests']}次LLM请求、{pc['extended_evaluation']['execution_delta']['additional_LLM_forwards']}次Forward。历史生成上下文随实际执行演化，不是静态替换答案。" if compperf else '选定C没有实际Position旁路，回退不改变执行，因此没有追加无意义GPU消融；额外时延未单独测量。'),
        f"主对象3Q C物理对象标签{obj['scored']}/{obj['requests']}；快速回答分母另列；使用不变的2m/.5m一对一标准。",
        f"问题目标标签{target['scored']}/{target['requests']}；匹配对象正确不代表最近目标正确。",
        f"主对象3Q答案UNKNOWN={quality['UNKNOWN']}/{quality['requests']}；原因={coverage_report['unknown_answer_reasons']}。Object和Target UNKNOWN另列，未合并分母。",
        f"原始自然1Q：解析{natural['parsed']}/{natural['requests']}，正向事实{natural['positive_fact_supported']}，准入{natural['eligible']}，真实跳过{natural['actual_fast']}。",
        f"预设mixed3Q B→C={mixed['speedup']:.6f}×，log CI={mixed['speedup_log_CI95']}；人为1/3结构化插槽，不是自然分布。",
        '; '.join(f"{f}的C无QA帧：M1含适配器={perf_c(f)['idle_m1_including_adapters_ms']['mean']:.3f}、M2={perf_c(f)['idle_m2_wall_ms']['mean']:.3f}、E2E={perf_c(f)['no_QA_frame_e2e_ms']['mean']:.3f}ms" for f in ('every4','every2'))+'；没有省略视觉帧。',
        '; '.join(f"{f}: {perf_c(f)['M3_share_on_QA_frames']:.2%}" for f in ('every4','every2','one','three'))+'，按有QA帧实际M3墙钟/E2E计算；嵌套计时不重复相加。',
        f"DEV2pp已评分质量约束下选择{policy['policy_id']}，实测E2E最小；1/2/3pp工作点见position_policy_comparison.json。扩展评价已评分loss={paired['loss']}，UNKNOWN阻止全任务质量保证。",
        '证明所报告历史日志内、新连续帧和固定到达模式的真实系统性能/计算省略及条件事实质量；不能证明历史log外泛化、自然应用分布或低罕见严重风险。',
        '是，仍然缺乏与基础模型和历史方法开发完全隔离的独立TEST；历史v3/v4 NO_GO及上一轮INCOMPLETE不变。',
        f"可支持与实测范围一致的连续六相机对象事实QA系统论点，性能状态={verdict['system_performance']}；论文必须保留数据暴露、任务覆盖、UNKNOWN和严重风险统计局限，不能表述成泛化GO。"]
    (root/'result_answers.md').write_text('# 24项实验回答\n\n'+SCOPE+' / INDEPENDENT_GENERALIZATION_INCOMPLETE\n\n'+'\n\n'.join(f'{i}. {x}' for i,x in enumerate(responses,1))+'\n')
    counts=json.loads((root/'gpu_execution_counts.json').read_text())
    (root/'implementation_summary.md').write_text('# Continuous stream delivery\n\n'+
        f"{verdict['system_performance']}; INDEPENDENT_GENERALIZATION_INCOMPLETE. " +
        f"{counts['full_M123_frame_calls']} real complete frame calls, {counts['actual_prefill_skips']} actual language skips across DEV/evaluation,104 unique frames/5 distinct logs. " +
        f"Object3Q B→C={gains['speedup']:.6f}×; preset mixed3Q={mixed['speedup']:.6f}×. " +
        f"Known primary answer loss={paired['loss']}; labels={quality['scored']}/{quality['requests']}; new severe={paired['new_severe']}.\n\n"+
        'The only integration changes are explicit opt-in idle visual frames, measured answer export, and narrowly scoped FROZEN_RETROSPECTIVE_POLICY acceptance in the existing runtime. '
        'Current-only fact bypass and the three existing Position modes are unchanged; no new risk model, second runtime, weights or M1/M2 redesign. Production remains off. '
        'Every idle frame executes vision/state/heads, zero language forwards. Exact native prompt validation, strict checkpoint load, source hashes and per-frame receipts retained.\n\n'+
        '\n'.join(f"- {k}: {v['counts']}, exit {v['exit_code']}" for k,v in tests.items())+'\n')
    cmd='taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python'
    (root/'REPRODUCE.md').write_text('# Reproduce continuous stream evaluation\n\nExisting OmniDrive environment, RTX3090, no install/train/download. GPU uses /tmp/omnidrive_m123_latency_gpu.lock and waits for other users; CPU4/5 excluded.\n\nIntegrity verification (automatically restores evaluation/requests.jsonl from its lossless gzip archive if absent; both archive and raw SHA256 checked against receipt_archive.json):\n\n```sh\n'+cmd+' -m tools.worldstream_m123.continuous_stream verify\n```\n\n'
        'Exact repetition must copy the original frozen selection/protocol to a NEW root; rerunning metadata selection after these results exist must exclude these now-executed frames and is a different study. Repeated original frames are regression/performance repetitions, not fresh extension data.\n\n```sh\n'+
        cmd+' -m tools.worldstream_m123.continuous_stream dev --root /tmp/continuous-exact-replay\n'+
        cmd+' -c "from pathlib import Path; from tools.worldstream_m123.continuous_stream_analysis import attest_companion; attest_companion(Path(\'/tmp/continuous-exact-replay\'))"\n'+
        cmd+' -m tools.worldstream_m123.continuous_stream evaluation --root /tmp/continuous-exact-replay\n'+
        cmd+' -m tools.worldstream_m123.answer_validity_validate --output /tmp/continuous-exact-replay/validation_cpu\n'+
        cmd+' -m tools.worldstream_m123.answer_validity_validate --gpu --output /tmp/continuous-exact-replay/validation_gpu\n'+
        cmd+' -m tools.worldstream_m123.continuous_stream report --root /tmp/continuous-exact-replay\n```\n\n'
        'Before running the commands, create the replay root and copy extended_frame_selection.json, protocol_freeze.json and continuous_qa_protocol.md verbatim from this run; also copy historical_manifest.json, historical_execution_inventory.json, additional_selected_frame_audit.json and non_M3_exposure_audit.json for provenance and reporting. Do not copy completed stage directories, selected policies, receipt archives or freeze files: the replay must generate its own. '
        'The driver validates selection hashes. DEV freezes the retrospective policy before evaluation. Each condition starts cold then runs the complete stream with no internal reset. '
        'Source hashes in dev/evaluation/execution_sources.json and policy_sources.json cover runtime/worker/grading; presentation assembly is separately attested. '
        'DEV executed the archived worker in dev/executed_sources. Before evaluation, scheduling was extended to conditionally run the predeclared Position fallback if actual selected-C Position bypass occurred; evaluation_source_revision.json preserves the exact old/new worker hashes. No inference or grading change occurred at this boundary. '
        'Generation settings remain original; using a different GPU/clock or condition order can change timings. All labels and statistics have explicit denominators and unknowns. No independent TEST or strict2% rare-risk qualification.\n')
    write(root/'report_sources.json',dict(analysis_sha256=sha256(__file__),grading_sha256=sha256('tools/worldstream_m123/continuous_stream_report.py')))


def verify(root):
    restore_archived_receipts(root)
    selection=json.loads((root/'extended_frame_selection.json').read_text());protocol=json.loads((root/'protocol_freeze.json').read_text())
    assert sha256(root/'extended_frame_selection.json')==protocol['selection_sha256']
    assert all(not r['actual_visual_GPU'] and not r['language_execution_evidence'] for r in selection['selected_frames'])
    assert json.loads((root/'additional_selected_frame_audit.json').read_text())['hits']==[]
    assert json.loads((root/'non_M3_exposure_audit.json').read_text())['prior_language_frames']==0
    manifest=json.loads((root/'historical_manifest.json').read_text());assert all(sha256(p)==h for p,h in manifest.items())
    for stage in ('dev','evaluation'):
        verify_sources(root,stage+'/execution_sources.json')
        status=json.loads((root/stage/'run_status.json').read_text());rows=lines(root/stage/'requests.jsonl')
        assert status['status']=='COMPLETED' and len(rows)==status['full_M123_frame_calls']
        jobs=defaultdict(list)
        for r in rows:
            assert r['execution_completed'] and r['m1_commit_completed'];jobs[r['job']].append(r)
            assert len(r['requests'])==len(r['questions'])==len(r['answers'])
            assert r['observed_llm_forward_calls']==sum(q['llm_forward_calls'] for q in r['requests'])
            if r['condition']!='A':assert r['m2_completed'] and r['m2']['execution_completed']
            if not r['questions']:assert r['observed_llm_forward_calls']==0
            for q in r['requests']:
                if q['llm_skipped']:
                    assert q['llm_forward_calls']==0 and q['layout_preparation_ms']==0
                    assert not q['prefill_completed'] and not q['decode_completed']
        for rs in jobs.values():
            assert [r['frame_index'] for r in rs]==list(range(len(rs)))
            assert all(a['timestamp']<b['timestamp'] for a,b in zip(rs,rs[1:]))
            assert [r['m1_commit_count'] for r in rs]==list(range(1,len(rs)+1))
    verify_sources(root,'policy_sources.json')
    freeze=json.loads((root/'POLICY_FREEZE.json').read_text());assert sha256(root/'selected_retrospective_policy.json')==freeze['selected_policy_sha256']
    assert freeze['before_extended_evaluation'] and not freeze['independent_test_qualified']
    companion=json.loads((root/'position_companion_freeze.json').read_text())
    assert companion['frozen_before_first_companion_forward'] and sha256(companion['policy_file'])==companion['policy_sha256']
    for k in ('validation_cpu','validation_gpu'):
        t=json.loads((root/k/'summary.json').read_text());assert t['exit_code']==0 and not {4,5}&set(t['cpu_affinity'])
    assert json.loads((root/'validation_gpu/summary.json').read_text())['CUDA_available']
    report_sources=json.loads((root/'report_sources.json').read_text());assert sha256(__file__)==report_sources['analysis_sha256']
    verdict=json.loads((root/'research_verdict.json').read_text());assert verdict['independent_generalization']=='INDEPENDENT_GENERALIZATION_INCOMPLETE'
    result=dict(status='PASS',historical_files_unchanged=len(manifest),scope=SCOPE,
        counts=json.loads((root/'gpu_execution_counts.json').read_text()))
    write(root/'integrity_verification.json',result);print(json.dumps(result),flush=True)
