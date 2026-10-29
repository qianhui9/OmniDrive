"""Independent current-frame task-fact labels and DEV-only WRSAR calibration."""
from collections import Counter,defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import re

from worldstream.m3.task_conditioner import question_facts
from worldstream.m3.answer_validity import fact_region,fact_directions
from worldstream.m3.world_delta import VEHICLES
from .answer_validity_quality import load_current_annotations,original_qa_grade
from .benchmark.common import write,sha256
from .selective_answer_reuse import CONFIDENCE_CANDIDATES


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def mean(values):return sum(values)/len(values) if values else None


def label(question,objects):
    """Positive visible witnesses only; no physical/visual-absence equivalence."""
    profile=question_facts(question)
    if profile.intent=='unsupported':return dict(known=False,reason='outside_structured_domain')
    if profile.classes==('object',):return dict(known=False,reason='generic_visual_object_taxonomy_incomplete')
    classes=set(profile.classes)
    if 'vehicle' in classes:classes=(classes-{'vehicle'})|VEHICLES
    if 'object' in classes:classes=set()
    eligible=[g for g in objects if g['visible'] and (not classes or g['semantic_class'] in classes)
              and fact_region(g['center_ego'],profile)]
    if not eligible:return dict(known=False,reason='no_positive_visible_witness_absence_UNKNOWN')
    eligible.sort(key=lambda g:math.hypot(*g['center_ego'][:2]))
    first=eligible[0]
    if profile.selector=='nearest':
        near=[g for g in eligible[1:] if math.hypot(*g['center_ego'][:2])-math.hypot(*first['center_ego'][:2])<2.]
        if (profile.intent=='category' and any(g['semantic_class']!=first['semantic_class'] for g in near)):
            return dict(known=False,reason='annotated_nearest_class_ambiguous')
        if profile.intent=='position' and (near or any(abs(abs(v)-1.)<.5 for v in first['center_ego'][:2])):
            return dict(known=False,reason='annotated_nearest_or_sector_ambiguous')
    motion=None
    if profile.intent=='motion':
        attrs=set(first.get('attributes',[]))
        if 'vehicle.moving' in attrs:motion='moving'
        elif attrs&{'vehicle.parked','vehicle.stopped'}:motion='stationary'
        else:return dict(known=False,reason='independent_current_motion_label_unavailable')
    return dict(known=True,task=profile.intent,semantic_class=first['semantic_class'],positive=True,
                motion=motion,available_classes=sorted({g['semantic_class'] for g in eligible}),
                directions=fact_directions(first['center_ego']),annotation_ids=[g['annotation_id'] for g in eligible],
                selected_annotation=first['annotation_id'],source_frame=first['source_frame'],
                center_ego=first['center_ego'],required_facts=profile.required,
                metric='required_current_visible_object_facts',profile=asdict(profile))


def extracted_classes(text):
    text=str(text).lower()
    return {c for c,p in [('car',r'cars?|sedans?|suvs?|vans?|taxis?'),('truck',r'trucks?'),
        ('bus',r'bus|buses'),('pedestrian',r'pedestrians?|people|person'),('bicycle',r'bicycles?|cyclists?'),
        ('motorcycle',r'motorcycles?'),('trailer',r'trailers?'),('construction_vehicle',r'construction vehicles?')]
        if re.search(r'\b(?:'+p+r')\b',text)}


def grade(question,answer,token,objects):
    ref=label(question,objects)
    if question_facts(question).intent=='unsupported':
        qa=original_qa_grade(token,question,answer)
        return dict(known=qa['score'] is not None,score=qa['score'],reference=qa,
                    metric='limited_exact_current_QA_fact_recall',severe_error=False)
    if not ref['known']:return dict(known=False,score=None,reference=ref,severe_error=None)
    text=' '.join(str(answer).lower().split())
    negative=bool(re.match(r'no\b',text) or re.search(r'\b(?:there (?:is|are)|i (?:can |do )?see) (?:no|not any|not a)\b',text)
                  or re.search(r'\b(?:cannot|can\x27t|do not) (?:see|determine|identify)\b',text))
    uncertain=bool(re.search(r'\b(uncertain|unsure|cannot determine|could be|might be|not visible|not clear)\b',text))
    classes=extracted_classes(text)
    dirs=tuple(d for d,p in [('front','front|ahead'),('rear','rear|behind'),('left','left'),('right','right')]
               if re.search(r'\b(?:'+p+r')\b',text))
    task=ref['task'];correct=False
    if task=='category':correct=classes=={ref['semantic_class']} and not negative and not uncertain
    elif task=='existence':
        positive=bool(re.match(r'yes\b',text) or re.search(r'\b(?:there (?:is|are)|i (?:can )?see|visible|present)\b',text))
        correct=positive and not negative and not uncertain
        if classes and not classes&set(ref['available_classes']):correct=False
        expected_dirs=set(question_facts(question).directions)
        if dirs and expected_dirs and not expected_dirs<=set(dirs):correct=False
    elif task=='position':correct=set(dirs)==set(ref['directions']) and not negative and not uncertain
    elif task=='motion':
        moving=bool(re.search(r'\b(moving|driving|travelling|traveling)\b',re.sub(r'\bnot moving\b','',text)))
        stationary=bool(re.search(r'\b(stationary|stopped|parked)\b',text))
        correct=(moving!=stationary and ('moving' if moving else 'stationary')==ref['motion'] and not uncertain)
    severe=bool((task=='existence' and negative) or (task=='position' and any(
        a in dirs and b in ref['directions'] for a,b in [('front','rear'),('rear','front'),('left','right'),('right','left')])))
    return dict(known=True,score=float(correct),reference=ref,metric='required_current_visible_object_facts',
                extracted=dict(classes=sorted(classes),directions=dirs,negative=negative,uncertain=uncertain),severe_error=severe)


def load_labels(rows):
    tokens={r['frame_id'] for r in rows}
    objects=load_current_annotations('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl',tokens,'data/nuscenes')
    from tools.worldstream_m1.io_utils import iter_json_array
    table=Path('data/nuscenes/v1.0-trainval')
    attrs={a['token']:a['name'] for a in iter_json_array(table/'attribute.json')}
    by_id={g['annotation_id']:g for parts in objects.values() for g in parts}
    for ann in iter_json_array(table/'sample_annotation.json'):
        if ann['token'] in by_id:by_id[ann['token']]['attributes']=[attrs[a] for a in ann.get('attribute_tokens',[])]
    return objects


def scored_requests(rows,objects):
    events=[]
    for row in rows:
        for i,(request,answer) in enumerate(zip(row['requests'],row['answers'])):
            q=request['question'];text=answer['A'][0];proposal=request.get('fast_answer')
            g=grade(q,text,row['frame_id'],objects[row['frame_id']])
            e=dict(stage=row['stage'],condition=row['condition'],workload=row['workload'],question_count=row['question_count'],
                round=row['round'],scene_id=row['scene_id'],frame_id=row['frame_id'],request_index=i,
                question=q,answer=text,quality=g,llm_skipped=request['llm_skipped'],
                answer_source=request['answer_source'],llm_forward_calls=request['llm_forward_calls'],
                request_wall_ms=request.get('end_to_end_ms'),frame_e2e_wall_ms=row['e2e_wall_ms'],proposal=proposal,
                output_tokens=len(request['tokens']),task=question_facts(q).intent,
                draft_active=request.get('draft_status')=='target_verification')
            if proposal and proposal['supported']:
                e['candidate_quality']=grade(q,proposal['answer_text'],row['frame_id'],objects[row['frame_id']])
            events.append(e)
    return events


def wilson(errors,n):
    if not n:return None
    z=1.96;p=errors/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    spread=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-spread),min(1.,center+spread)]


def calibrate(root):
    rows=lines(root/'shadow/requests.jsonl');objects=load_labels(rows)
    events=scored_requests(rows,objects)
    with (root/'shadow_quality_requests.jsonl').open('w') as f:
        for e in events:f.write(json.dumps(e)+'\n')
    primary=[e for e in events if e['workload']=='addressable']
    known=[e for e in primary if e['quality']['known']]
    candidates=[e for e in primary if e.get('candidate_quality')]
    labeled=[e for e in candidates if e['candidate_quality']['known']]
    bins=[]
    for lo,hi in ((.35,.55),(.55,.75),(.75,1.01)):
        part=[e for e in labeled if lo<=e['proposal']['support_score']<hi]
        errors=sum(e['candidate_quality']['score']<1 for e in part)
        bins.append(dict(confidence_interval=[lo,hi],scored=len(part),errors=errors,
                         observed_error_rate=errors/len(part) if part else None,binomial_wilson95=wilson(errors,len(part)),
                         independent_frames=len({e['frame_id'] for e in part}),
                         independent_scenes=len({e['scene_id'] for e in part})))
    configs=[]
    for threshold in CONFIDENCE_CANDIDATES:
        hits=[e for e in candidates if e['proposal']['support_score']>=threshold]
        scored=[e for e in hits if e['candidate_quality']['known'] and e['quality']['known']]
        loss=sum(e['quality']['score']-e['candidate_quality']['score'] for e in scored)/len(known) if known else None
        configs.append(dict(min_confidence=threshold,requests=len(primary),baseline_scored=len(known),hits=len(hits),
            scored_hits=len(scored),scored_correct=sum(e['candidate_quality']['score']==1 for e in scored),
            hit_label_coverage=len(scored)/len(hits) if hits else None,coverage=len(hits)/len(primary),
            absolute_task_quality_loss=loss,
            false_fast_rate=mean([1-e['candidate_quality']['score'] for e in scored]),
            new_severe_errors=sum(e['candidate_quality']['severe_error'] and not e['quality']['severe_error'] for e in scored),
            distinct_correct_frames=len({e['frame_id'] for e in scored if e['candidate_quality']['score']==1}),
            estimated_saved_m3_ms=sum(e['request_wall_ms'] for e in hits),
            UNKNOWN_hits=len(hits)-len(scored),
            empirical_metric_scope='current positive witness task facts only; no unknown-as-correct'))
    expected=mean([e['request_wall_ms'] for e in primary])
    artifact=dict(status='EMPIRICAL_DEV_CALIBRATION',configs=configs,bins=bins,
        source=str(root/'shadow/requests.jsonl'),source_sha256=sha256(root/'shadow/requests.jsonl'),
        label_source='Current nuScenes sample_annotation reverse links, camera poses, visibility and sensor support',
        score_kind='Detector support is not answer-correctness probability',
        sample_independence='Three scenes, twelve current frames; requests within a scene correlated. Wilson intervals are descriptive, not scene-independent guarantees.',
        baseline='Target-aware exact History-Draft with shadow bookkeeping, still full target generation',
        development_only=True,normal_m3_ms_estimate=expected,
        candidate_requests=len(candidates),candidate_labeled=len(labeled),candidate_correct=sum(e['candidate_quality']['score']==1 for e in labeled),
        full_reuse_opportunities=sum(e['proposal']['answer_source']=='full_answer_reuse' for e in candidates),
        fact_grounded_opportunities=sum(e['proposal']['answer_source']=='fact_grounded_response' for e in candidates))
    write(root/'answer_validity_calibration.json',artifact)
    for config in configs:
        if config['distinct_correct_frames']>=3 and config['absolute_task_quality_loss']<=.03 and config['new_severe_errors']==0:
            write(root/f"policy_conf{int(config['min_confidence']*100)}.json",dict(
                status='FROZEN_RESEARCH_POLICY',policy_id=f"confidence_{int(config['min_confidence']*100)}",
                budget=.02,min_confidence=config['min_confidence'],expected_normal_m3_ms=expected,
                calibration_sha256=sha256(root/'answer_validity_calibration.json'),calibration_metrics=config,
                automatic_deployment=False,quality_qualification='DEV_PAIRED_POSITIVE_FACT_METRIC_WITH_COVERAGE'))
    policies=[]
    for budget in (.01,.02,.03):
        usable=[c for c in configs if c['absolute_task_quality_loss'] is not None and c['absolute_task_quality_loss']<=budget
                and c['distinct_correct_frames']>=3 and c['scored_correct']>=3 and c['hit_label_coverage']>=.5
                and c['new_severe_errors']==0]
        if not usable:
            policies.append(dict(budget=budget,status='NO_QUALIFIED_DEV_CANDIDATE'));continue
        best=max(usable,key=lambda c:(c['estimated_saved_m3_ms'],-c['min_confidence']))
        policy=dict(status='FROZEN_RESEARCH_POLICY',policy_id=f'wrsar_dev_{int(budget*100)}pp',
            budget=budget,min_confidence=best['min_confidence'],expected_normal_m3_ms=expected,
            calibration_sha256=sha256(root/'answer_validity_calibration.json'),calibration_metrics=best,
            quality_qualification='PARTIAL_FACT_LABELS; empirically within budget on scored requests, not a calibrated error probability',
            automatic_deployment=False)
        write(root/f'policy_{int(budget*100)}pp.json',policy);policies.append(policy)
    write(root/'policy_selection.json',dict(status='DEV_ONLY_SELECTION',policies=policies,test_tuning=False))
    selected=next((p for p in policies if p.get('budget')==.02 and p['status']=='FROZEN_RESEARCH_POLICY'),None)
    if selected is None:selected=next((p for p in policies if p['status']=='FROZEN_RESEARCH_POLICY'),None)
    if selected:write(root/'selected_policy.json',selected)
    write(root/'shadow_candidate_quality.json',artifact)
    (root/'shadow_evaluation_report.md').write_text(f'''# Real RTX3090 Shadow

{len(rows)} complete M123 model calls / {len(events)} language requests, including 36 addressable and 24 separate diagnostics. The target-aware LLM executed for every request; actual skips: 0.

Addressable candidates: {len(candidates)}; independently scored: {len(labeled)}; task-fact correct: {artifact['candidate_correct']}. Full-history opportunities: {artifact['full_reuse_opportunities']}; fact-driven opportunities: {artifact['fact_grounded_opportunities']}. See raw decisions and all UNKNOWN cases in shadow_quality_requests.jsonl. No human review was performed.

Three predeclared confidence floors were compared against the SAME paired B outputs using current-frame labels. The calibration uses empirical task loss with coverage and uncertainty, not the previous uncalibrated weighted validity score. Selected online research policy: {selected['policy_id'] if selected else 'NONE'}. Missing labels prevent full driving-quality certification. Shadow costs are observed, but savings are theoretical because the LLM was not skipped.
''')
    print('CALIBRATION',json.dumps(dict(candidate_correct=artifact['candidate_correct'],configs=configs,selected=selected)),flush=True)


def report(root):
    rows=lines(root/'run/requests.jsonl')
    assert json.loads((root/'run/run_status.json').read_text())['status']=='COMPLETED'
    objects=load_labels(rows);events=scored_requests(rows,objects)
    with (root/'quality_requests.jsonl').open('w') as f:
        for e in events:f.write(json.dumps(e)+'\n')
    write(root/'current_frame_facts.json',dict(source='offline_only_current_nuScenes_annotations',frames=objects))
    plan=json.loads((root/'experiment_plan.json').read_text())
    policy=json.loads((root/'selected_policy.json').read_text())
    summaries=[];pairs=[];modules=[]
    from .world_guided_decode_v2_report import scene_ci
    for workload in plan['workloads']:
        for count in (1,3):
            rs=[r for r in rows if r['workload']==workload and r['question_count']==count]
            grouped=defaultdict(dict)
            for r in rs:grouped[(r['round'],r['scene_id'],r['frame_id'])][r['condition']]=r
            assert all(set(g)=={'A','B','C'} for g in grouped.values())
            for a,b in [('A','B'),('B','C'),('A','C')]:
                part=[]
                for (repeat,scene,token),g in grouped.items():
                    x,y=g[a],g[b]
                    part.append(dict(workload=workload,question_count=count,comparison=a+'->'+b,
                        round=repeat,scene_id=scene,frame_id=token,from_ms=x['e2e_wall_ms'],to_ms=y['e2e_wall_ms'],
                        delta_ms=y['e2e_wall_ms']-x['e2e_wall_ms'],cold_scene_frame=x['frame_index']==0))
                pairs.extend(part)
                scenes={s:sum(p['from_ms'] for p in part if p['scene_id']==s)/sum(p['to_ms'] for p in part if p['scene_id']==s)
                        for s in sorted({p['scene_id'] for p in part})}
                summaries.append(dict(workload=workload,question_count=count,comparison=a+'->'+b,pairs=len(part),
                    e2e_speedup=sum(p['from_ms'] for p in part)/sum(p['to_ms'] for p in part),
                    mean_delta_ms=mean([p['delta_ms'] for p in part]),delta_scene_CI95_ms=scene_ci(part,'delta_ms'),
                    scene_speedups=scenes,round_speedups=[sum(p['from_ms'] for p in part if p['round']==i)/
                        sum(p['to_ms'] for p in part if p['round']==i) for i in range(3)],
                    steady_speedup=sum(p['from_ms'] for p in part if not p['cold_scene_frame'])/
                        sum(p['to_ms'] for p in part if not p['cold_scene_frame'])))
            for condition in ('A','B','C'):
                part=[r for r in rs if r['condition']==condition];qs=[q for r in part for q in r['requests']]
                fp=[q['fast_answer'] for q in qs if q.get('fast_answer')]
                modules.append(dict(workload=workload,question_count=count,condition=condition,frames=len(part),
                    requests=len(qs),e2e_ms=mean([r['e2e_wall_ms'] for r in part]),
                    m1_wall_ms=mean([r['m1_wall_ms'] for r in part]),
                    m1_including_adapters_ms=mean([r['m1_including_adapters_ms'] for r in part]),
                    m2_wall_ms=mean([r['m2_wall_ms'] for r in part]),heads_wall_ms=mean([r['heads_wall_ms'] for r in part]),
                    snapshot_wall_ms=mean([r['snapshot_wall_ms'] for r in part]),m3_wall_ms=mean([r['m3_wall_ms'] for r in part]),
                    prefill_cuda_ms_per_question=mean([q['prefill_cuda_ms'] for q in qs]),
                    prefill_host_enqueue_wall_ms_per_question=mean([q['prefill_wall_ms'] for q in qs]),
                    decode_cuda_ms_per_question=mean([q['decode_cuda_ms'] for q in qs]),
                    decode_stream_wall_ms_per_question=mean([q['decode_wall_ms'] for q in qs]),
                    question_wall_ms=mean([q['end_to_end_ms'] for q in qs]),
                    fact_extraction_ms=mean([p['question_fact_extraction_wall_ms'] for p in fp]),
                    validity_check_ms=mean([p['answer_validity_check_wall_ms'] for p in fp]),
                    answer_construction_ms=mean([p['fast_answer_construction_wall_ms'] for p in fp]),
                    fast_source_counts=dict(Counter(q['answer_source'] for q in qs)),
                    actual_m3_skips=sum(q['llm_skipped'] for q in qs),
                    actual_llm_forward_calls=sum(q['llm_forward_calls'] for q in qs),
                    fps=len(part)*1000/sum(r['e2e_wall_ms'] for r in part),qps=len(qs)*1000/sum(r['e2e_wall_ms'] for r in part),
                    peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in part),
                    mean_output_tokens=mean([len(q['tokens']) for q in qs])))
    write(root/'abc_comparison.json',dict(status='MEASURED_FULL_M123_GPU',summaries=summaries,
        paired_frames=pairs,module_breakdown=modules,
        A='Native complete M123',B='Target-aware exact K8 History-Draft',C='B + WRSAR',
        timing='CUDA-synchronized same original outer model call, including M1/M2/heads/commit/snapshot/M3 and answer output. Data collation and raw JSONL audit serialization outside equally.',
        phase_timing='Prefill host enqueue wall and CUDA event span reported separately; decode stream wall includes HF host gaps. Phase means are not summed to manufacture E2E.'))
    quality=[]
    emap={(e['workload'],e['question_count'],e['round'],e['frame_id'],e['request_index'],e['condition']):e for e in events}
    differences=[];failures=[]
    for e in events:
        if e['condition']=='A':continue
        key=(e['workload'],e['question_count'],e['round'],e['frame_id'],e['request_index'])
        for baseline in (['A'] if e['condition']=='B' else ['A','B']):
            ref=emap[key+(baseline,)]
            known=e['quality']['known'] and ref['quality']['known']
            loss=ref['quality']['score']-e['quality']['score'] if known else None
            item=dict(workload=e['workload'],question_count=e['question_count'],round=e['round'],
                scene_id=e['scene_id'],frame_id=e['frame_id'],request_index=e['request_index'],
                comparison=baseline+'->'+e['condition'],loss=loss,llm_skipped=e['llm_skipped'],
                answer_source=e['answer_source'],known=known)
            differences.append(item)
            if known and e['quality']['score']<1:
                failures.append(dict(item,question=e['question'],baseline_answer=ref['answer'],answer=e['answer'],
                    baseline_quality=ref['quality'],quality=e['quality'],
                    new_wrong=ref['quality']['score']==1,
                    new_severe_error=bool(e['quality']['severe_error'] and not ref['quality']['severe_error']),
                    causal_attribution='current_M1_fact_response' if e['answer_source']=='fact_grounded_response' else
                        'original_target_answer_reused' if e['answer_source']=='full_answer_reuse' else 'normal_target_generation'))
    for workload in plan['workloads']:
        for count in (1,3):
            for condition in ('A','B','C'):
                es=[e for e in events if (e['workload'],e['question_count'],e['condition'])==(workload,count,condition)]
                filters=[('all',lambda e:True),('fast_hit',lambda e:e['llm_skipped']),
                         ('miss',lambda e:not e['llm_skipped']),('full_answer_reuse',lambda e:e['answer_source']=='full_answer_reuse'),
                         ('fact_grounded_response',lambda e:e['answer_source']=='fact_grounded_response'),
                         ('history_draft_active',lambda e:e['draft_active'])]
                for name,keep in filters:
                    part=[e for e in es if keep(e)];scored=[e for e in part if e['quality']['known']]
                    quality.append(dict(workload=workload,question_count=count,condition=condition,execution=name,
                        requests=len(part),scored=len(scored),UNKNOWN=len(part)-len(scored),
                        label_coverage=len(scored)/len(part) if part else None,
                        mean_task_fact_score=mean([e['quality']['score'] for e in scored]),
                        independent_frame_questions=len({(e['frame_id'],e['question']) for e in scored}),
                        independent_scenes=len({e['scene_id'] for e in scored}),
                        observed_errors=sum(e['quality']['score']<1 for e in scored),
                        false_response_rate=mean([1-e['quality']['score'] for e in scored]),
                        severe_errors=sum(bool(e['quality']['severe_error']) for e in scored)))
    losses=[]
    for workload in plan['workloads']:
        for count in (1,3):
            for comparison in ('A->B','B->C','A->C'):
                part=[e for e in differences if (e['workload'],e['question_count'],e['comparison'])==(workload,count,comparison) and e['known']]
                losses.append(dict(workload=workload,question_count=count,comparison=comparison,scored_pairs=len(part),
                    absolute_task_quality_loss=mean([e['loss'] for e in part]),
                    scene_CI95=scene_ci(part,'loss') if part else None))
    write(root/'task_quality.json',dict(status='MEASURED_PARTIAL_INDEPENDENT_LABELS',groups=quality,paired_losses=losses,
        events_path='quality_requests.jsonl',errors=failures,
        caveat='Positive visible annotation facts only for structured queries. Original QA uses limited exact-current reference fact recall. Unknown visual absence/safety and missing labels are never correct by default.'))
    write(root/'false_reuse_analysis.json',dict(status='MEASURED_PARTIAL',errors=failures,
        new_severe_vs_B=[e for e in failures if e['comparison']=='B->C' and e['new_severe_error']],
        scope='Actual task facts, not lexical/token differences. Same-scene rounds are correlated.'))
    fast=[e for e in events if e['llm_skipped']]
    write(root/'fast_answer_analysis.json',dict(status='MEASURED_GPU_ACTUAL_SKIPS',total_skips=len(fast),
        source_counts=dict(Counter(e['answer_source'] for e in fast)),
        skipped_forward_calls_verified_zero=all(e['llm_forward_calls']==0 for e in fast),
        groups=[g for g in quality if g['execution'] in ('fast_hit','full_answer_reuse','fact_grounded_response')]))
    write(root/'cache_coverage_analysis.json',dict(status='MEASURED',groups=modules,
        full_history_answer_reuse=sum(e['answer_source']=='full_answer_reuse' for e in fast),
        current_fact_bypass=sum(e['answer_source']=='fact_grounded_response' for e in fast)))
    addressable=[g for g in quality if g['workload']=='addressable']
    write(root/'m1_addressable_qa_results.json',dict(status='MEASURED_PARTIAL_LABELS',quality=addressable,
        paired_losses=[g for g in losses if g['workload']=='addressable'],
        performance=[s for s in summaries if s['workload']=='addressable']))
    shadow=lines(root/'shadow_quality_requests.jsonl')
    old=lines(Path('outputs/m3_answer_validity_cache_run01/offline_requests.jsonl'))
    rejected=[e for e in old if e.get('decision',{}).get('reasons')==['M1_does_not_cover_task']]
    write(root/'answer_dependency_analysis.json',dict(status='MEASURED',
        prior_unsupported_candidates=len(rejected),prior_unsupported_fixed_candidates=sum(e['workload']=='fixed' for e in rejected),
        prior_unsupported_task_types=dict(Counter(e['question_task']['intent'] for e in rejected)),
        explanation='Old complete answers require road/map semantics, appearance or global evidence absent from M1. New F(Q) answers only required class/positive presence/direction; extra color/subclass/scene claims are not copied.',
        full_reuse_grammar='All substantive assertions must match currently supported closed object-fact grammar; richer original text cannot be promoted by partial fact coverage.'))
    write(root/'question_fact_coverage.json',dict(status='MEASURED_CURRENT_M1_SUPPORT',
        workloads=[dict(workload=w,requests=len([e for e in shadow if e['workload']==w]),
            supported=sum(bool(e.get('proposal',{}).get('supported')) for e in shadow if e['workload']==w),
            reasons=dict(Counter(e['proposal']['reason'] for e in shadow if e['workload']==w))) for w in ('addressable','diagnostic')],
        question_types=[dict(task=t,requests=len([e for e in shadow if e['task']==t]),
            supported=sum(e['proposal']['supported'] for e in shadow if e['task']==t),
            known_labels=sum(e['quality']['known'] for e in shadow if e['task']==t)) for t in ('category','existence','position','motion')],
        map_semantics_audit=dict(source='worldstream/m1/omnidrive_adapter.py: outputs_from_model',
            actual_semantic_class='lane_control_points',signal_color_available=False,full_road_semantics_available=False)))
    temporal=[]
    for stage in ('shadow','run'):
        source_rows=lines(root/stage/'requests.jsonl')
        # One actual B addressable stream per scene for independent change audit.
        streams=defaultdict(list)
        for r in source_rows:
            if (r['condition'],r['workload'],r['question_count'],r['round'])==('B','addressable',3,0):streams[r['scene_id']].append(r)
        for scene,stream in streams.items():
            previous=None
            for r in stream:
                s=r['committed_world'];now={a['anchor_id']:a for a in s['anchors'] if a['observed'] and a['anchor_type']=='object'}
                if previous:
                    old=previous[1];common=set(old)&set(now)
                    temporal.append(dict(stage=stage,scene_id=scene,frame_id=r['frame_id'],
                        entered=len(set(now)-set(old)),lost_observation=len(set(old)-set(now)),
                        lost_observation_semantics='departure_OR_occlusion_OR_detector_miss_UNKNOWN',
                        common=len(common),sector_changes=sum(fact_directions(old[a]['center_ego'])!=fact_directions(now[a]['center_ego']) for a in common),
                        ego_translation_m=math.dist([previous[0]['ego_to_world'][i] for i in (3,7,11)],
                                                   [s['ego_to_world'][i] for i in (3,7,11)]),
                        current_uncertain_identity=sum(not a['identity_qualified'] for a in now.values())))
                previous=(s,now)
    write(root/'temporal_change_audit.json',dict(status='OBSERVED_NOT_SCENE_SELECTION',events=temporal,
        occlusion_or_identity_error_GT='UNKNOWN without independent image/identity review; no human review performed'))
    curves=pareto_report(root)
    positive=next(s for s in summaries if (s['workload'],s['question_count'],s['comparison'])==('addressable',3,'B->C'))
    qloss=next(s for s in losses if (s['workload'],s['question_count'],s['comparison'])==('addressable',3,'B->C'))
    hit=next(g for g in quality if (g['workload'],g['question_count'],g['condition'],g['execution'])==('addressable',3,'C','fast_hit'))
    severe=[e for e in failures if e['comparison']=='B->C' and e['workload']=='addressable' and e['new_severe_error']]
    partial=(positive['e2e_speedup']>1 and positive['delta_scene_CI95_ms'][1]<0
             and all(s>1 for s in positive['scene_speedups'].values())
             and qloss['absolute_task_quality_loss'] is not None and qloss['absolute_task_quality_loss']<=policy['budget']
             and hit['scored']>0 and hit['label_coverage']>=.8 and not severe)
    status='PARTIAL_GO' if partial else 'INCOMPLETE' if hit['label_coverage'] is None or hit['label_coverage']<.8 else 'NO_GO'
    reason=('Measured positive-annotation M1-addressable task-fact scope only: actual language bypass and scene-paired E2E gain, with empirical paired loss inside the DEV-selected budget. General natural QA/safety quality is not qualified.' if partial else
            'Measured results do not meet all declared scoped quality/coverage/scene-E2E criteria; inspect errors, UNKNOWN coverage and paired scene intervals. No test threshold tuning.')
    write(root/'go_no_go.json',dict(status=status,reason=reason,selected_policy=policy,
        actual_fast_answers=len(fast),addressable_three_QA_B_to_C=positive,addressable_quality_loss=qloss,
        label_coverage_fast_hits=hit['label_coverage'],new_severe_errors=len(severe),
        whole_driving_quality_budget='INCOMPLETE',formal_deployment_modified=False,
        scientific_attribution='World-Referenced Selective Language Bypass; full-cache contribution claimed only if separately measured.'))
    write(root/'active_request_quality.json',dict(status='MEASURED_PARTIAL',groups=quality,paired_losses=losses))
    (root/'active_request_quality_report.md').write_text('# Actual active-request quality\n\n'+
        '\n'.join(f"- {g['workload']} {g['question_count']}Q {g['condition']} {g['execution']}: {g['scored']}/{g['requests']} scored; score {g['mean_task_fact_score']}; errors {g['observed_errors']}." for g in quality if g['execution'] in ('all','fast_hit','history_draft_active'))+
        '\n\nEvery score uses current independent labels and retains UNKNOWN in coverage. Full-history and fact-driven answers have separate raw provenance. No human review or global safety certification was performed. Detailed required facts, omissions and new severe findings are in task_quality.json and quality_requests.jsonl.\n')
    (root/'e2e_performance_report.md').write_text('# Real full M123 GPU performance\n\n'+
        '\n'.join(f"- {s['workload']} {s['question_count']}Q {s['comparison']}: {s['e2e_speedup']:.4f}x, scene speedups {s['scene_speedups']}, delta CI95 {s['delta_scene_CI95_ms']} ms." for s in summaries)+
        '\n\nSame synchronized outer model call includes all M1/M2/heads/current commit/snapshot/fact checking/construction/current M3/answer output. No theoretical token savings replace E2E. Raw per-module costs, output lengths, FPS/QPS and GPU peak memory are in abc_comparison.json. Prefill host enqueue and CUDA spans are distinct; phase timings are not added together as a substitute for outer wall.\n')
    shadow_count=len(lines(root/'shadow/requests.jsonl'));pareto_count=len(lines(root/'pareto/requests.jsonl'))
    excluded=root/'initial_prompt_context'
    excluded_calls=sum(len(lines(excluded/s/'requests.jsonl')) for s in ('shadow','pareto','run')
                       if (excluded/s/'requests.jsonl').exists())
    write(root/'run_counts.json',dict(full_M123_GPU_model_calls=len(rows)+shadow_count+pareto_count,
        excluded_initial_attempt_model_calls=excluded_calls,
        actual_model_calls_including_excluded=len(rows)+shadow_count+pareto_count+excluded_calls,
        shadow_model_calls=shadow_count,pareto_model_calls=pareto_count,formal_ABC_model_calls=len(rows),
        formal_ABC_question_requests=len(events),actual_M3_skips=len(fast),
        independent_test_scenes=len(plan['test_scenes']),independent_test_frames=sum(len(s['tokens']) for s in plan['test_scenes'])))
    (root/'implementation_summary.md').write_text(f'''# WRSAR delivery

Decision: **{status}**. {reason}

Implemented question-referenced fact coverage in task_conditioner/answer_validity, source-preserving AnswerMemory reuse, an opt-in shadow/online decision before M3 expansion in the existing Runtime, and the existing exact K8 target-aware verifier on every miss. No new model, weight update, training, M1 tracker or M2 visual algorithm. Defaults remain unchanged and selective_answer_mode is off.

Actual full M123 GPU calls: {len(rows)+shadow_count+pareto_count}; formal A/B/C language requests: {len(events)}; actual M3 skips: {len(fast)}. Source counts: {dict(Counter(e['answer_source'] for e in fast))}. Runtime diagnostics and an independent language-forward hook agree: fast requests execute zero LLM calls and zero expansion. Shadow executes every target generation and is never counted as skip speedup.

DEV Shadow selected a confidence floor {policy['min_confidence']} for the default {policy['budget']*100:g}pp relative quality budget. Empirical confidence bins are not error probabilities. All budgets/thresholds were fixed before TEST; repeated rounds are not independent scenes. Original natural, fixed and introduced addressable workloads remain separate. Quality is limited to independently labeled current required facts; unavailable absence/road/safety labels remain UNKNOWN.

Default deployment remains prefill-mandatory. Explicit opt-in WRSAR responses complete the same M123 execution contract only with a current committed perception receipt, admitted policy, zero expansion/LLM calls and declared nonexecution of Prefill/Decode. Final B/C measurements enable that contract; A uses the existing native-full benchmark override. An independent forward hook checks the actual calls. This is research qualification, not deployment activation.
''')
    make_answers(root,summaries,losses,quality,events,curves,status,reason)
    (root/'REPRODUCE.md').write_text('''# WRSAR reproduction

Use existing local data, checkpoints and OmniDrive Python. No downloads. Every command excludes faulty CPUs4/5. GPU stages acquire /tmp/omnidrive_m123_latency_gpu.lock and wait for idle GPU memory, never stop another process. Use a NEW output root for each reproduction; stages refuse to overwrite their directories. The fixed native generation parameters remain max_new_tokens320, sampling .1/.75/num_beams1/top_k50 and full six-camera inputs.

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.selective_answer_reuse --root outputs/wrsar_reproduce01 --stage prepare
```

Repeat the same command prefix/root with stages `shadow`, `calibrate`, `pareto`, `run`, `report` in that order. Shadow and pareto/run need host CUDA access. Only a shadow-qualified DEV policy enables online stages. Final TEST never tunes a threshold. The three predeclared detector floors provide the measured DEV Pareto curve; the budgets can select the same policy, which must be reported as the same operating point.

Per-stage execution_sources.json and executed_sources/ bind exact GPU source bytes. The initial attempt is retained in initial_prompt_context/ and excluded: the fixed first-frame native-token assertion caught a city-vs-site context mismatch before any fixed/addressable TEST call. Corrected encoding matches the native loader on Boston and Singapore tests; all Shadow/calibration/Pareto/TEST stages were rerun. No initial TEST labels were evaluated. The original metadata-selected DEV-disjoint TEST cohort was retained unchanged because additional image-complete independent logs were unavailable; no TEST answers/GT informed threshold selection. Calibration source and selected-policy hashes bind the freeze before the replacement test. Raw logs contain answers/tokens, current M1 evidence, forward counts and synchronized frame timings.

For tests, use the spawn-safe existing `tools.worldstream_m123.answer_validity_validate --gpu --output <NEW validation directory>` with the same taskset/Python prefix. CPU testing omits --gpu. Regression CUDA checks are reported separately from full model GPU experiments. Raw duplicate model answer exports remain local; no large tensor captures are duplicated.\n''')
    print('REPORT',status,positive,qloss,hit,flush=True)


def pareto_report(root):
    rows=lines(root/'pareto/requests.jsonl');objects=load_labels(rows)
    events=scored_requests(rows,objects);curves=[]
    baseline={(e['round'],e['frame_id'],e['request_index']):e for e in events if e['condition']=='B'}
    for condition in sorted({r['condition'] for r in rows}):
        rs=[r for r in rows if r['condition']==condition];es=[e for e in events if e['condition']==condition]
        paired=[(e,baseline[(e['round'],e['frame_id'],e['request_index'])]) for e in es]
        known=[(e,b) for e,b in paired if e['quality']['known'] and b['quality']['known']]
        confidence=None if condition=='B' else int(condition[1:])/100
        curves.append(dict(condition=condition,min_confidence=confidence,full_model_frames=len(rs),
            e2e_wall_ms=mean([r['e2e_wall_ms'] for r in rs]),
            task_score=mean([e['quality']['score'] for e,b in known]),
            task_loss_vs_B=mean([b['quality']['score']-e['quality']['score'] for e,b in known]),
            fast_hits=sum(e['llm_skipped'] for e in es),requests=len(es),scored_pairs=len(known),
            fast_scored=sum(e['llm_skipped'] and e['quality']['known'] for e in es),
            new_severe_errors=sum(e['quality']['severe_error'] and not b['quality']['severe_error'] for e,b in known)))
    policies=json.loads((root/'policy_selection.json').read_text())['policies']
    write(root/'quality_latency_pareto.json',dict(status='MEASURED_DEV_FULL_M123; FINAL_TEST_SELECTED_POLICY_ONLY',
        measured_operating_points=curves,budget_selected_policies=policies,
        test_tuning=False,duplicate_budget_policies_are_same_operating_point=True,
        quality_metric='Current positive-annotation required facts; missing labels UNKNOWN'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4))
    for c in curves:
        if c['task_loss_vs_B'] is None:continue
        ax.scatter(c['task_loss_vs_B']*100,c['e2e_wall_ms'],s=55)
        ax.annotate(c['condition'],(c['task_loss_vs_B']*100,c['e2e_wall_ms']),xytext=(5,5),textcoords='offset points')
    ax.set_xlabel('Required-fact quality loss vs B (percentage points)')
    ax.set_ylabel('Full M123 frame latency (ms)');ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(root/'quality_latency_pareto.png',dpi=180);fig.savefig(root/'quality_latency_pareto.pdf');plt.close(fig)
    return curves


def make_answers(root,summaries,losses,quality,events,curves,status,reason):
    cal=json.loads((root/'answer_validity_calibration.json').read_text())
    sel=json.loads((root/'selected_policy.json').read_text())
    skips=[e for e in events if e['llm_skipped']]
    part=[s for s in summaries if s['comparison']=='B->C']
    qpart=[q for q in losses if q['comparison']=='B->C']
    costs=[e['proposal'] for e in events if e['proposal']]
    content=[
        '1. Yes. The old 24 fixed candidates fail because complete original answers depend on road/appearance/global semantics M1 does not supply. answer_dependency_analysis.json gives task counts. Core supported facts can now answer the limited structured question without copying unsupported white/sedan/scene claims.',
        '2. Current M1 supports observed object class, positive presence and ego-relative geometry. Reliable simple motion is implemented but actual velocity support is audited, never assumed. Map publishes lane_control_points, not traffic-light colors or complete road semantics.',
        f"3. DEV addressable support: {cal['candidate_requests']}/36 before policy; labeled proposals {cal['candidate_labeled']}. Question-type coverage and diagnostic motion abstentions: question_fact_coverage.json. TEST coverage stays workload/condition/count specific in m1_addressable_qa_results.json.",
        f"4. DEV full-history opportunities {cal['full_reuse_opportunities']}; current-fact opportunities {cal['fact_grounded_opportunities']}. Actual TEST skip sources: {dict(Counter(e['answer_source'] for e in skips))}. Full-cache and fact bypass are not combined as cache efficacy.",
        f"5. Empirical DEV calibration was performed for only three detector floors. Bins, errors and descriptive Wilson intervals are in answer_validity_calibration.json. Scores are not probabilities. Selected confidence floor: {sel['min_confidence']}. Sample size and scene dependence limit reliability.",
        f"6. Empirical DEV threshold errors: {[(c['min_confidence'],c['false_fast_rate'],c['scored_hits']) for c in cal['configs']]}. Measured TEST errors, UNKNOWN and new severe cases are in task_quality.json/false_reuse_analysis.json; no text-similarity score is used.",
        f"7. DEV budget choices: {[(p.get('budget'),p.get('min_confidence'),p.get('calibration_metrics',{}).get('coverage')) for p in json.loads((root/'policy_selection.json').read_text())['policies']]}. Different budgets choosing one policy are one operating point, not three independent improvements.",
        f"8. Real Shadow found {cal['candidate_correct']} correct scored primary proposals among {cal['candidate_labeled']} scored proposals. Shadow kept all target LLM generation, so it demonstrates opportunity rather than real bypass speedup.",
        f"9. Actual formal TEST skips: {len(skips)}; all have zero independently observed LLM forward calls and zero input expansion. Full GPU request counts including DEV/Shadow are in run_counts.json.",
        f"10. TEST mean exclusive CPU fact extraction/check/construction costs (ms): {mean([c['question_fact_extraction_wall_ms'] for c in costs])}, {mean([c['answer_validity_check_wall_ms'] for c in costs])}, {mean([c['fast_answer_construction_wall_ms'] for c in costs])}. Current snapshot and model metadata are separately included in full E2E/module costs.",
        f"11. Full scene-paired B→C results: {part}. No estimated token savings replace these synchronized original-model measurements.",
        '12. Original natural, fixed-repeat and introduced M1-addressable one/three-question results are separate in abc_comparison.json and e2e_performance_report.md. Addressable gains must not be relabeled as natural-QA gains.',
        f"13. Paired TEST task losses vs B: {qpart}. Cache/fact hits and misses have separate task scores/coverage and severe findings in task_quality.json. Missing visual-negative/safety labels preclude a general driving-quality budget claim.",
        '14. A→B is the existing target-aware exact K8 generator. B→C is selective M1 fast response plus its management cost. Source-specific counts attribute each skip to full original answer reuse or current fact grounding. If full-history count is zero, the contribution is Selective Language Bypass, not demonstrated Answer Cache.',
        f"15. DEV selected {sel['policy_id']}, confidence {sel['min_confidence']}. Actual DEV measured points: {curves}. TEST evaluated only this pre-frozen operating policy, without retuning.",
        f'16. {status}. {reason} Only the explicitly scored fact/task scope can support a contribution; global natural driving QA and safety remain unqualified. Default deployment was not changed.'
    ]
    (root/'result_answers.md').write_text('# WRSAR research answers\n\n'+'\n\n'.join(content)+'\n')
