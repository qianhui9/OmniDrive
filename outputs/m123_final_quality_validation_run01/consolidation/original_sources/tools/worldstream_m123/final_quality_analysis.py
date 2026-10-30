"""Offline final-quality audit. No GT is passed into inference or admission."""
import argparse
from collections import Counter, defaultdict
import copy
import gzip
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .benchmark.common import sha256, write
from .critical_path_analysis import job, stage_rows, diagnostic_performance
from .continuous_stream_report import events, paired_quality
from .continuous_stream_analysis import quality_summary, mean, dist
from .fact_bypass_identity import match_instances, question_target

ROOT = Path('outputs/m123_final_quality_validation_run01')
OLD = Path('outputs/m123_critical_path_and_joint_tuning_run01')
ERROR_FRAME = 'a7753324e1e44e4598ed2b9a9c26c4c1'


def read(path):
    return json.loads(Path(path).read_text())


def stream(path):
    path = Path(path)
    opener = path.open if path.exists() else lambda mode: gzip.open(str(path) + '.gz', mode)
    with opener('rt') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prepare(root):
    from worldstream.m1.config import M1Config
    from worldstream.m2.config import M2Config
    root.mkdir(exist_ok=False)
    for name in ('extended_frame_selection.json', 'current_gt_annotations.json', 'reference_policy.json'):
        shutil.copy2(OLD / name, root / name)
    shutil.copytree(OLD / 'baseline_sources/worldstream/m1', root / 'baseline_sources/worldstream/m1')
    selection = read(root / 'extended_frame_selection.json')
    protocol = dict(base_commit='40c8cdce74ef2e5cbdb8c2bb6eaa5ef42e2ad071',
        selection_sha256=sha256(root/'extended_frame_selection.json'),
        scope='RETROSPECTIVE_CONFIGURATION_SELECTION', independent_TEST=False,
        causal_case_role='POST_HOC_CAUSAL_DIAGNOSIS_ONLY',
        DEV_selection='Only first two fixed extended_DEV streams, 16 frames each. Historical evaluation/error stream forbidden for selection.',
        original_M1_threshold=M1Config().min_detection_score, original_M2_budget=M2Config().ordinary_budget,
        parameter_source=['worldstream/m1/config.py', 'worldstream/m2/config.py', 'GPU effective_runtime_jobs.jsonl'],
        candidate_limit=['R0: original/original', 'R1: .4/1', 'R2: original/1', 'R3: .4/original'],
        M3='Unchanged current_only C-basic .45/.75, temporal risk/completeness disabled',
        common_candidate_objective='Measured three-question DEV E2E on identical 32 frames; no invented missing five-workload means for R2/R3.',
        decision_rule='Known paired B-C loss <= budget; separately zero new severe on DEV. Prefer candidates without >2pp loss in visible committed class-correct object recall or question-relevant eligible target recall versus R0. Then minimize common-cell E2E. If none qualifies retain fastest risk-failed research candidate. No further search.',
        budgets=[.01,.02,.03], default_budget=.02,
        detector_protocol='Class-independent unique <2m / .5m ambiguity; bottom-to-gravity box conversion; raw score>=.3 fixed. All decoded boxes also retained for error trace.',
        recall_scope='All mapped GT and separately visibility-qualified GT in already exposed subset; unmatched predictions UNKNOWN, never automatic FP.',
        FP_policy='No certified exhaustive per-prediction negative adjudication: FP and unconditional precision NOT_CERTIFIED. Report conditional matched precision and conservative bounds.',
        feature_observer='Actual returned M2 NCHW; per-camera/4x4 tiles/token quantiles, after model timer',
        historical_performance='Prior frozen five-workload ABC results stay separate from new diagnostic means',
        production_default=False, no_training=True, sampling_unchanged=True)
    write(root/'protocol_freeze.json', protocol)
    write(root/'data_roles.json', dict(previous=read(OLD/'data_roles.json'),
        new_independent_TEST=False, DEV_used_for_selection_only=True,
        new_causal_frames_are_old_evaluation=True))
    configs={'R0':({},{}), 'R1':({'min_detection_score':.4},{'ordinary_budget':1}),
             'R2':({}, {'ordinary_budget':1}), 'R3':({'min_detection_score':.4},{})}
    scene=copy.deepcopy(next(s for s in selection['selected'] if ERROR_FRAME in s['tokens']))
    scene['tokens']=scene['tokens'][:scene['tokens'].index(ERROR_FRAME)+1]
    causal=[]
    for label in ('R0','R3','R2','R1'):
        m1,m2=configs[label]
        causal.append(job(root,scene,label,m1=m1,m2=m2,profile=False,audit_stage='causal'))
    causal.append(job(root,scene,'Full',profile=False,m2_full=True,audit_stage='causal'))
    write(root/'causal_jobs.json',causal)
    dev=[]
    for s in selection['selected']:
        if s['role']!='extended_DEV':continue
        for label in ('Full','R0','R2','R2_B'):
            j=job(root,s,label,condition='B' if label=='R2_B' else 'C',
                m2={'ordinary_budget':1} if label.startswith('R2') else {},
                m2_full=label=='Full',profile=False,audit_stage='dev_quality',
                feature_reference_key=s['scene_id'],audit_save_features=label=='Full',
                audit_compare_features=label in ('R0','R2'))
            dev.append(j)
    write(root/'dev_quality_jobs.json',dev)
    files=subprocess.check_output(['git','ls-files','outputs','worldstream','projects'],text=True).splitlines()
    write(root/'historical_integrity_manifest.json',{p:sha256(p) for p in files if Path(p).is_file()})
    write(root/'pre_GPU_freeze.json',dict(protocol_sha256=sha256(root/'protocol_freeze.json'),
        job_hashes={stage:sha256(root/(stage+'_jobs.json')) for stage in ('causal','dev_quality')},
        historical_unchanged=True, planned_new_frame_calls=sum(len(j['scene']['tokens']) for j in causal+dev)))
    print('PREPARED',root)


def raw_anchors(row, floor=.3):
    from worldstream.m1.omnidrive_adapter import CLASS_NAMES
    raw=row['visual_quality']
    result=[]
    for i,(box,cls,score) in enumerate(zip(raw['boxes'],raw['labels'],raw['scores'])):
        if score<floor:continue
        center=list(box[:3]);center[2]+=box[5]/2
        result.append(dict(anchor_id=str(i),anchor_type='object',observed=True,
            source_frame=row['frame_id'],source_timestamp=row['timestamp'],
            evidence_origin='actual_perception',center_ego=center,
            semantic_class=CLASS_NAMES[cls],score=score,cameras=[]))
    return result


def current_anchors(row):
    return [a for a in (row.get('committed_world') or {}).get('anchors',[])
        if a['anchor_type']=='object' and a['observed'] and a['source_frame']==row['frame_id']
        and a['source_timestamp']==row['timestamp'] and a['evidence_origin']=='actual_perception']


def eligible(anchors,question):
    from worldstream.m3.task_conditioner import question_facts
    from worldstream.m3.answer_validity import fact_region
    from worldstream.m3.world_delta import VEHICLES
    p=question_facts(question);classes=set(p.classes)
    if 'vehicle' in classes:classes=(classes-{'vehicle'})|VEHICLES
    if 'object' in classes:classes=set()
    return sorted([a for a in anchors if a['cameras'] and a['score'] is not None
        and math.isfinite(a['score']) and a['score']>=.35
        and (not classes or a['semantic_class'] in classes)
        and all(math.isfinite(v) for v in a['center_ego']) and fact_region(a['center_ego'],p)],
        key=lambda a:(math.hypot(*a['center_ego'][:2]),-a['score'],a['anchor_id']))


def detection_frame(row, anchors, objects):
    """Explicit GT denominators, ambiguity-aware recall and precision bounds."""
    matches=match_instances(anchors,objects,row['frame_id'],row['timestamp'])
    correct={m['annotation_id'] for m in matches.values() if m['known'] and m['class_correct']}
    by_annotation={m['annotation_id']:m for m in matches.values() if m['known']}
    sets={
        'all_mapped_GT':objects,
        'visible_GT':[g for g in objects if g['visible']],
        'visible_within50m':[g for g in objects if g['visible'] and math.hypot(*g['center_ego'][:2])<=50.]}
    for cls in sorted({g['semantic_class'] for g in objects}):
        sets['class/'+cls]=[g for g in objects if g['visible'] and g['semantic_class']==cls]
    for low,high in ((0,10),(10,20),(20,30),(30,50),(50,math.inf)):
        sets[f'distance/{low}-{high}m']=[g for g in objects if g['visible'] and low<=math.hypot(*g['center_ego'][:2])<high]
    groups={}
    for name,gs in sets.items():
        tp=sum(g['annotation_id'] in correct for g in gs)
        fn=unknown=0
        for g in gs:
            if g['annotation_id'] in correct:continue
            potential=any(a['semantic_class']==g['semantic_class'] and math.dist(a['center_ego'],g['center_ego'])<2. for a in anchors)
            if potential:unknown+=1
            else:fn+=1
        groups[name]=dict(GT=len(gs),TP=tp,FN=fn,UNKNOWN_GT=unknown,
            localization_sum_m=sum(by_annotation[g['annotation_id']]['center_distance_m'] for g in gs if g['annotation_id'] in correct))
    known=[m for m in matches.values() if m['known']]
    return dict(groups=groups,predictions=len(anchors),matched=len(known),class_correct=sum(m['class_correct'] for m in known),
        confirmed_class_FP=sum(not m['class_correct'] for m in known),UNKNOWN_predictions=len(anchors)-len(known),matches=matches)


def aggregate_detection(frames):
    groups={}
    for name in sorted({k for r in frames for k in r['groups']}):
        c=Counter()
        for r in frames:c.update(r['groups'].get(name,{}))
        n=c['GT'];tp=c['TP'];u=c['UNKNOWN_GT']
        groups[name]=dict(c,proven_class_recall=tp/n if n else None,
            class_recall_bounds=[tp/n,(tp+u)/n] if n else None,
            mean_proven_localization_m=c['localization_sum_m']/tp if tp else None)
    totals={k:sum(r[k] for r in frames) for k in ('predictions','matched','class_correct','confirmed_class_FP','UNKNOWN_predictions')}
    n=totals['predictions'];t=totals['class_correct'];u=totals['UNKNOWN_predictions']
    return dict(groups=groups,**totals,FP=None,
        FP_status='NOT_CERTIFIED: unmatched/ambiguous detections are UNKNOWN, not assumed false positives',
        conditional_matched_class_precision=t/totals['matched'] if totals['matched'] else None,
        unconditional_class_precision_bounds=[t/n,(t+u)/n] if n else None)


def association(rows, objects):
    counts=Counter();lifetimes=[];recoveries=[];streams=defaultdict(list)
    for r in rows:streams[r['scene_id']].append(r)
    for scene,part in streams.items():
        previous={};previous_instances={};tracks=defaultdict(list);last_seen={};seen_at={}
        for r in sorted(part,key=lambda r:r['frame_index']):
            anchors=current_anchors(r)
            matches=match_instances(anchors,objects[r['frame_id']],r['frame_id'],r['timestamp'])
            current={aid:m['instance_id'] for aid,m in matches.items() if m['known']}
            visible={g['instance_id'] for g in objects[r['frame_id']] if g['visible']}
            for a,_ in r['visual_quality']['association_pairs']:
                if a not in matches:continue
                counts['accepted_links']+=1
                if a in previous and a in current:
                    counts['scored_links']+=1
                    counts['correct_links' if previous[a]==current[a] else 'wrong_links']+=1
                    if previous[a]==current[a] and current[a] in visible and current[a] in seen_at:
                        counts['correct_visible_links']+=1
                else:counts['UNKNOWN_links']+=1
            instances={v:k for k,v in current.items()}
            common=set(instances)&set(previous_instances)
            counts['consecutive_scored_GT_instances']+=len(common)
            counts['GT_instance_ID_changes']+=sum(instances[i]!=previous_instances[i] for i in common)
            if seen_at:
                counts['consecutive_visible_GT_opportunities']+=len(visible&set(seen_at))
            for inst,aid in instances.items():
                tracks[aid].append(r['timestamp'])
                if inst in last_seen and r['frame_index']>last_seen[inst][0]+1:
                    recoveries.append(dict(scene=scene,instance=inst,gap_frames=r['frame_index']-last_seen[inst][0]-1,
                        same_anchor=aid==last_seen[inst][1],frame_id=r['frame_id']))
                last_seen[inst]=(r['frame_index'],aid)
            previous=current;previous_instances=instances;seen_at={i:True for i in visible}
        lifetimes.extend(dict(scene_id=scene,anchor_id=aid,scored_observed_frames=len(ts),
            observed_span_s=max(ts)-min(ts)) for aid,ts in tracks.items())
    return dict(counts=dict(counts),conditional_link_precision=counts['correct_links']/counts['scored_links'] if counts['scored_links'] else None,
        effective_association_coverage=counts['correct_visible_links']/counts['consecutive_visible_GT_opportunities'] if counts['consecutive_visible_GT_opportunities'] else None,
        ID_change_rate=counts['GT_instance_ID_changes']/counts['consecutive_scored_GT_instances'] if counts['consecutive_scored_GT_instances'] else None,
        observed_retention=lifetimes,loss_recovery_events=recoveries,
        scope='Partial unique-GT matching. Observed spans are censored by 16-frame streams and UNKNOWN gaps, not true lifetime; not MOTA/IDF1/HOTA.')


def audit_rows(root, rows, label, objects):
    raw=[];committed=[];targets=[];frame_metrics=[]
    for r in rows:
        obs=objects[r['frame_id']];ra=raw_anchors(r);wa=current_anchors(r)
        raw_d=detection_frame(r,ra,obs);world_d=detection_frame(r,wa,obs)
        raw.append(raw_d);committed.append(world_d)
        for q,req in zip(r['questions'],r['requests']):
            target=question_target(q,obs);allowed=set(target.get('acceptable_annotation_ids',[]))
            ea=eligible(wa,q);eids={a['anchor_id'] for a in ea}
            def retained(matches,ids=None):
                return any(m['known'] and m['class_correct'] and m['visible'] and m['annotation_id'] in allowed
                    for aid,m in matches.items() if ids is None or aid in ids)
            record=dict(label=label,scene_id=r['scene_id'],frame_id=r['frame_id'],question=q,
                target=target,known=target['known'],raw_recalled=retained(raw_d['matches']) if target['known'] else None,
                committed_recalled=retained(world_d['matches']) if target['known'] else None,
                eligible_recalled=retained(world_d['matches'],eids) if target['known'] else None,
                eligible_count=len(ea),actual_skip=req['llm_skipped'],llm_forwards=req['llm_forward_calls'])
            for stage,aa in (('raw',ra),('committed',wa),('eligible',ea)):
                potential=any(a['semantic_class']==g['semantic_class'] and math.dist(a['center_ego'],g['center_ego'])<2.
                    for a in aa for g in obs if g['annotation_id'] in allowed and g['visible'])
                record[stage+'_coverage_status']=('GT_UNKNOWN' if not target['known'] else
                    'CONFIRMED_RECALLED' if record[stage+'_recalled'] else
                    'PREDICTION_MATCHING_UNKNOWN' if potential else 'NO_CORRECT_CLASS_CANDIDATE_WITHIN_2M')
            targets.append(record)
        frame_metrics.append(dict(label=label,scene_id=r['scene_id'],log_id=r['log_id'],frame_id=r['frame_id'],
            phase=r['phase'],global_suffix_reuse=bool(r['m2']['global_suffix_skipped_blocks']),
            feature_error=r['visual_quality'].get('full_feature_difference'),
            raw=raw_d['groups']['visible_GT'],committed=world_d['groups']['visible_GT'],
            raw_detection_predictions=raw_d['predictions'],committed_predictions=world_d['predictions'],
            e2e_ms=r['e2e_wall_ms'],m1_ms=r['m1_including_adapters_ms'],m2_host_ms=r['m2_wall_ms'],
            m2_cuda_ms=r['stages']['m2_total']['gpu_span_ms']))
    known=[t for t in targets if t['known']]
    target_summary=dict(requests=len(targets),scored=len(known),UNKNOWN=len(targets)-len(known),
        **{k:sum(t[k] for t in known)/len(known) if known else None for k in ('raw_recalled','committed_recalled','eligible_recalled')},
        successes={k:sum(t[k] for t in known) for k in ('raw_recalled','committed_recalled','eligible_recalled')},
        failure_breakdown={stage:dict(Counter(t[stage+'_coverage_status'] for t in known)) for stage in ('raw','committed','eligible')},
        UNKNOWN_reasons=dict(Counter(t['target']['reason'] for t in targets if not t['known'])))
    return dict(label=label,frames=len(rows),independent_logs=len({r['log_id'] for r in rows}),
        raw=aggregate_detection(raw),committed=aggregate_detection(committed),target_summary=target_summary,
        targets=targets,frame_metrics=frame_metrics,association=association(rows,objects))


def causal_audit(root,rows,objects,stem='severe_error_counterfactual'):
    records=[]
    for row in rows:
        if row['frame_id']!=ERROR_FRAME:continue
        question='Where is the nearest pedestrian relative to the vehicle?'
        target=question_target(question,objects[ERROR_FRAME]);center=target['center_ego']
        raw=raw_anchors(row,0.);world=current_anchors(row);legal=eligible(world,question)
        match=match_instances(world,objects[ERROR_FRAME],ERROR_FRAME,row['timestamp'])
        candidates=[]
        for a in raw:
            if a['semantic_class']!='pedestrian':continue
            nearest=min(objects[ERROR_FRAME],key=lambda g:math.dist(a['center_ego'],g['center_ego']))
            equivalent=[w for w in world if w['semantic_class']=='pedestrian' and math.dist(w['center_ego'],a['center_ego'])<1e-4]
            candidates.append(dict(a,GT_target_distance_m=math.dist(a['center_ego'],center),
                nearest_physical_GT_annotation=nearest['annotation_id'],nearest_physical_GT_class=nearest['semantic_class'],
                nearest_physical_GT_distance_m=math.dist(a['center_ego'],nearest['center_ego']),
                passes_original_M1=a['score']>=.3,passes_fast_M1=a['score']>=.4,
                committed_anchor_ids=[w['anchor_id'] for w in equivalent],
                eligible_anchor_ids=[w['anchor_id'] for w in equivalent if w in legal]))
        req=next(q for q in row['requests'] if q['question']==question)
        event=next(e for e in events(root,[row]) if e['question']==question)
        records.append(dict(label=row['job_spec']['label'],diagnostic_id={'R0':'D00','R3':'D10','R2':'D01','R1':'D11','Full':'Full'}[row['job_spec']['label']],
            frame_id=ERROR_FRAME,history_frames=row['frame_index']+1,history_start=row['job_spec']['scene']['tokens'][0],
            M1=row['job_spec']['m1'],M2=row['job_spec']['m2'],full_control=row['job_spec'].get('m2_full',False),
            target=target,raw_pedestrians=candidates,
            nearest_target_decoded_candidates=[a for a in candidates if a['GT_target_distance_m']<2.],
            committed_pedestrians=[dict(a,GT_match=match.get(a['anchor_id'])) for a in world if a['semantic_class']=='pedestrian'],
            M3_legal_candidates=[dict(a,distance_xy_m=math.hypot(*a['center_ego'][:2]),GT_match=match.get(a['anchor_id'])) for a in legal],
            M2_state={k:row['m2'].get(k) for k in ('global_anchor_frame_id','global_anchor_age','global_update_mode',
                'global_suffix_skipped_blocks','global_suffix_executed_blocks','executed_windows','reused_windows','global_feature_drift')},
            M1_association_pairs=row['visual_quality']['association_pairs'],M1_commit_count=row['m1_commit_count'],
            proposal=req['fast_answer'],answer=event['answer'],quality=event['quality'],
            actual_LLM_forwards=req['llm_forward_calls'],actual_skip=req['llm_skipped']))
    assert len(records)==5
    target_instance=records[0]['target']['selected_instance']
    evolution=[]
    for row in rows:
        gt=next((g for g in objects[row['frame_id']] if g['instance_id']==target_instance),None)
        nearby=[a for a in raw_anchors(row,0.) if gt and a['semantic_class']=='pedestrian' and math.dist(a['center_ego'],gt['center_ego'])<2.]
        nearby_world=[a for a in current_anchors(row) if gt and a['semantic_class']=='pedestrian' and math.dist(a['center_ego'],gt['center_ego'])<2.]
        evolution.append(dict(label=row['job_spec']['label'],frame_id=row['frame_id'],frame_index=row['frame_index'],
            historical_target_instance_center=gt['center_ego'] if gt else None,
            best_target_nearby_score=max((a['score'] for a in nearby),default=None),
            target_nearby_committed_candidates=[dict(anchor_id=a['anchor_id'],score=a['score'],center_ego=a['center_ego']) for a in nearby_world],
            committed_objects=len(current_anchors(row)),
            suffix_reused=bool(row['m2']['global_suffix_skipped_blocks']),
            executed_windows=row['m2']['executed_windows'],reused_windows=row['m2']['reused_windows'],
            feature_digest=row['visual_quality']['feature'],
            actual_frame_QA=[dict(question=q['question'],admitted=q['llm_skipped'],proposal=q.get('fast_answer')) for q in row['requests']]))
    first_differences=[]
    for a,b in (('R0','R3'),('R2','R1'),('R0','R2'),('R3','R1')):
        left={r['frame_index']:r for r in rows if r['job_spec']['label']==a}
        right={r['frame_index']:r for r in rows if r['job_spec']['label']==b}
        differences=[i for i in left if any(left[i]['visual_quality'][k]!=right[i]['visual_quality'][k] for k in ('boxes','labels','scores'))]
        feature_diff=[i for i in left if left[i]['visual_quality']['feature']!=right[i]['visual_quality']['feature']]
        first_differences.append(dict(comparison=a+'->'+b,first_decoded_detection_difference_frame_index=min(differences) if differences else None,
            first_returned_feature_difference_frame_index=min(feature_diff) if feature_diff else None,
            differing_frames=differences,scope='Entire closed-loop replay; does not isolate a single prior object or window as sole cause'))
    write(root/(stem+'.json'),dict(scope='POST_HOC_CAUSAL_DIAGNOSIS_ONLY',records=records,
        history_evolution=evolution,first_differences=first_differences,
        protocol=f"Each arm resets then replays the identical complete {records[0]['history_frames']}-frame prefix from {records[0]['history_start']}; stops at case, no isolated-frame parameter edit.",
        independent_TEST=False,selection_allowed=False))
    return records


def load_candidates(root):
    sets={};sources=[]
    for label,stage,oldlabel in [('R0','tune_m1','tune_m1_0_C'),('R3','tune_m1','tune_m1_1_C'),('R1','tune_m2','tune_m2_1_C')]:
        part=[r for r in stream(OLD/stage/'requests.jsonl') if r['job_spec']['label']==oldlabel]
        sets[label]=part
        sources.append(dict(label=label,condition='C',path=str(OLD/stage/'requests.jsonl.gz'),
            sha256=sha256(OLD/stage/'requests.jsonl.gz'),filter_label=oldlabel,frames=len(part),new_GPU_calls=0))
        sets[label+'_B']=[r for r in stream(OLD/stage/'requests.jsonl') if r['job_spec']['label']==oldlabel[:-1]+'B']
    fresh=list(stream(root/'dev_quality/requests.jsonl'))
    sets['R2']=[r for r in fresh if r['job_spec']['label']=='R2']
    sets['R2_B']=[r for r in fresh if r['job_spec']['label']=='R2_B']
    write(root/'historical_receipt_reuse.json',dict(sources=sources,counts_exclude_reuse=True))
    return sets,fresh


def performance_group(rows):
    requests=[q for r in rows for q in r['requests']]
    return dict(frames=len(rows),independent_logs=len({r['log_id'] for r in rows}),
        E2E_ms=dist([r['e2e_wall_ms'] for r in rows]),M1_wall_ms=dist([r['m1_including_adapters_ms'] for r in rows]),
        M2_host_ms=dist([r['m2_wall_ms'] for r in rows]),M2_CUDA_ms=dist([r['stages']['m2_total']['gpu_span_ms'] for r in rows]),
        suffix_refresh_frames=sum(not bool(r['m2']['global_suffix_skipped_blocks']) for r in rows),
        suffix_reuse_frames=sum(bool(r['m2']['global_suffix_skipped_blocks']) for r in rows),
        question_requests=len(requests),LLM_forwards=sum(q['llm_forward_calls'] for q in requests),
        actual_skips=sum(q['llm_skipped'] for q in requests),
        prefill_including_cache_CUDA_ms=dist([q['prefill_cuda_ms'] for q in requests]),
        decode_forward_CUDA_ms=dist([q['decode_cuda_ms'] for q in requests]),
        M3_runtime_wall_ms=dist([r['m3_wall_ms'] for r in rows]))


def select(root):
    if (root/'final_research_policy.json').exists():
        raise ValueError('Research parameters already frozen; use report for reanalysis or prepare a NEW replay root.')
    sets,fresh=load_candidates(root);objects=read(root/'current_gt_annotations.json')
    audits={label:audit_rows(root,sets[label],label,objects) for label in ('R0','R1','R2','R3')}
    es=events(root,[dict(r,condition=label) for label,rows in sets.items() for r in rows])
    points=[]
    base=audits['R0']
    for label in ('R0','R1','R2','R3'):
        a=audits[label];q=paired_quality(es,label+'_B',label)
        rows=sets[label]
        points.append(dict(label=label,M1=rows[0]['job_spec']['m1'],M2=rows[0]['job_spec']['m2'],
            performance=performance_group(rows),quality=q,
            fact_quality=quality_summary([e for e in es if e['condition']==label]),
            visible_object_recall=a['committed']['groups']['visible_GT']['proven_class_recall'],
            question_eligible_target_recall=a['target_summary']['eligible_recalled'],
            coverage_gate=(base['committed']['groups']['visible_GT']['proven_class_recall']-
                a['committed']['groups']['visible_GT']['proven_class_recall']<=.02 and
                base['target_summary']['eligible_recalled']-a['target_summary']['eligible_recalled']<=.02),
            measurement_source='Historical actual GPU receipts' if label!='R2' else 'New actual GPU dev_quality receipts'))
    bybudget={}
    for budget in (.01,.02,.03):
        qualified=[p for p in points if p['coverage_gate'] and p['quality']['loss'] is not None
            and p['quality']['loss']<=budget and p['quality']['new_severe']==0]
        if not qualified:
            qualified=points
        bybudget[str(budget)]=min(qualified,key=lambda p:p['performance']['E2E_ms']['mean'])['label']
    chosen=next(p for p in points if p['label']==bybudget['0.02'])
    fastest=min(points,key=lambda p:p['performance']['E2E_ms']['mean'])
    write(root/'limited_configuration_comparison.json',dict(scope='RETROSPECTIVE_CONFIGURATION_SELECTION',
        common_cell='addressable/three',frames_per_candidate=32,independent_logs=2,
        points=points,by_budget=bybudget,selected=chosen['label'],fastest=fastest['label'],
        selected_penalty_ms=chosen['performance']['E2E_ms']['mean']-fastest['performance']['E2E_ms']['mean'],
        protocol_sha256=sha256(root/'protocol_freeze.json'),causal_case_used_for_selection=False,
        five_workload_means_not_available_for_all_four=True))
    front=[p['label'] for p in points if not any(
        o['performance']['E2E_ms']['mean']<=p['performance']['E2E_ms']['mean'] and
        o['visible_object_recall']>=p['visible_object_recall'] and
        o['question_eligible_target_recall']>=p['question_eligible_target_recall'] and
        o['fact_quality']['answer_correctness']>=p['fact_quality']['answer_correctness'] and
        (o['performance']['E2E_ms']['mean']<p['performance']['E2E_ms']['mean'] or
         o['visible_object_recall']>p['visible_object_recall'] or
         o['question_eligible_target_recall']>p['question_eligible_target_recall'] or
         o['fact_quality']['answer_correctness']>p['fact_quality']['answer_correctness']) for o in points)]
    write(root/'quality_latency_pareto.json',dict(points=points,pareto_labels=front,by_budget=bybudget,
        selection='Predeclared known-answer and separate coverage gates, then common-cell E2E; no claim of global optimum',
        statistical_limit='Two exposed DEV logs; differences descriptive. Four candidate observations span separate execution sessions.'))
    write(root/'m1_detection_coverage.json',dict(protocol=read(root/'protocol_freeze.json')['detector_protocol'],
        groups=[dict(label=k,raw=v['raw'],committed=v['committed']) for k,v in audits.items()],
        scope='32 paired DEV frames / 2 exposed logs. GT object observations, not unique physical instances.',
        FN_definition='No predicted correct-class center within 2m. Potential but non-unique matches are UNKNOWN_GT; recall is proven TP / complete mapped GT denominator.',
        precision_limitation='Unmatched predictions are UNKNOWN. Certified FP and unconditional precision unavailable; matched class errors and precision bounds supplied.'))
    write(root/'m1_target_recall.json',dict(groups={k:v['target_summary'] for k,v in audits.items()},
        requests=[t for v in audits.values() for t in v['targets']],
        denominator='Unique/positive-witness-set question_target identity-known requests, separate from answer-known denominator. UNKNOWN never counted correct.',
        M3_eligibility='Exact C-basic current object/camera/.35/class/region predicates; eligibility is before .45 confidence/.75 binding admission.'))
    write(root/'m1_association_quality.json',dict(groups={k:v['association'] for k,v in audits.items()},standard_MOT=False))
    with (root/'dev_quality_requests.jsonl').open('w') as f:
        for e in es:f.write(json.dumps(e)+'\n')
    write(root/'task_quality.json',dict(scope='DEV_ONLY',groups={k:quality_summary([e for e in es if e['condition']==k]) for k in sets},
        paired={p['label']:p['quality'] for p in points}, historical_final_R1=read(OLD/'task_quality.json'),
        UNKNOWN_never_correct=True))
    write(root/'quality_by_task.json',dict(groups=[dict(label=k,task=t,**quality_summary([e for e in es if e['condition']==k and e['task']==t]))
        for k in sets for t in ('category','existence','position')]))
    propagation=[]
    for label in ('R0','R1','R2','R3'):
        part=[e for e in es if e['condition']==label];targets=audits[label]['targets'];scored=[e for e in part if e['target']['known']]
        propagation.append(dict(label=label,committed_objects=audits[label]['committed']['predictions'],
            requests=len(part),parseable=sum(e['task']!='unsupported' for e in part),
            target_coverage=audits[label]['target_summary'],eligible_candidates=sum(t['eligible_count'] for t in targets),
            actual_admitted_and_skips=sum(e['llm_skipped'] for e in part),
            actual_LLM_forwards=sum(e['llm_forward_calls'] for e in part),
            fast_quality=quality_summary([e for e in part if e['llm_skipped']]),
            fallback_quality=quality_summary([e for e in part if not e['llm_skipped']]),
            wrong_target=sum(not e['target']['correct'] for e in scored),target_identity_scored=len(scored),
            target_identity_UNKNOWN=len(part)-len(scored),performance=performance_group(sets[label])))
    switches=[]
    ref={(e['frame_id'],e['question']):e for e in es if e['condition']=='R0'}
    for label in ('R1','R2','R3'):
        for e in [e for e in es if e['condition']==label]:
            old=ref[(e['frame_id'],e['question'])]
            if e['llm_skipped']!=old['llm_skipped']:
                switches.append(dict(label=label,frame_id=e['frame_id'],question=e['question'],
                    R0_skip=old['llm_skipped'],candidate_skip=e['llm_skipped'],
                    LLM_forward_delta=e['llm_forward_calls']-old['llm_forward_calls'],
                    request_wall_delta_ms=e['request_wall_ms']-old['request_wall_ms'],
                    R0_quality=old['quality'],candidate_quality=e['quality']))
    write(root/'m123_fact_propagation.json',dict(groups=propagation,route_switches=switches,
        no_M3_algorithm_change=True,selection_GT_offline_only=True))
    actual_cfg=list(stream(root/'dev_quality/effective_runtime_jobs.jsonl'))
    effective=next(c for c in actual_cfg if c['label']=='R0')
    m1=dict(effective['M1'],**chosen['M1']);m2=dict(effective['M2'],**chosen['M2'])
    policy=read(root/'reference_policy.json')
    m3={k:policy[k] for k in ('main_path','position_mode','min_confidence','max_binding_risk','temporal_risk_enabled','target_completeness_enabled')}
    m3.update(question_rng_enabled=True,target_binding_enabled=True,reuse_strategy='exact_prefix',decode_strategy='world_guided',experimental_speculative_sampling=True)
    final=dict(status='FROZEN_RESEARCH_CANDIDATE',selected=chosen['label'],selection_scope='RETROSPECTIVE_CONFIGURATION_SELECTION',
        M1=m1,M2=m2,M3=m3,parameter_overrides=dict(M1=chosen['M1'],M2=chosen['M2']),
        selection_basis='DEV known-answer loss and separate visible-object/question-target coverage gates before minimum E2E. Historical severe case excluded.',
        quality_budget=.02,budget_sensitivity=bybudget,known_answer_denominator=chosen['quality']['scored'],
        UNKNOWN=chosen['quality']['UNKNOWN'],DEV_new_severe=chosen['quality']['new_severe'],
        current_fastest_historical_risk='FAILED: R1 has one new front/rear severe error; immutable historical verdict',
        severe_error_status='INDEPENDENT_RISK_NOT_CERTIFIED; zero DEV new severe is not rare-error assurance',
        E2E_evidence_sources=['limited_configuration_comparison.json','selected_confirmation_performance.json'],
        independent_TEST=False,research_performance_accepted=True,
        strict_risk_accepted=False,production_deployment_allowed=False,production_default=False,
        auto_apply=False,base_commit=read(root/'protocol_freeze.json')['base_commit'])
    write(root/'final_research_policy.json',final)
    # One fixed 16-frame DEV stream, all five workloads. Existing selected 3Q B/C
    # receipts are reused; other 13 cells need actual timing, not interpolation.
    scene=sets[chosen['label']][0]['job_spec']['scene'];jobs=[]
    for workload,frequency in [('addressable','every4'),('addressable','every2'),('addressable','one'),('addressable','three'),('mixed','three')]:
        for condition in ('A','B','C'):
            if (workload,frequency)==('addressable','three') and condition!='A':continue
            jobs.append(job(root,scene,'selected_'+condition,condition=condition,workload=workload,frequency=frequency,
                m1=chosen['M1'],m2=chosen['M2'],profile=False,audit_stage='selected_confirm',
                audit_map_capture=True,native_without_m1=condition=='A'))
    write(root/'selected_confirm_jobs.json',jobs)
    write(root/'pre_confirmation_freeze.json',dict(selected=chosen['label'],
        policy_sha256=sha256(root/'final_research_policy.json'),jobs_sha256=sha256(root/'selected_confirm_jobs.json'),
        additional_frames=sum(len(j['scene']['tokens']) for j in jobs),
        purpose='Confirm frozen candidate frequencies and true Native comparisons on one exposed DEV log; not new selection or independent TEST.'))
    print(json.dumps(dict(selected=chosen['label'],fastest=fastest['label'],bybudget=bybudget,
        candidate_points=[(p['label'],p['performance']['E2E_ms']['mean'],p['visible_object_recall'],p['question_eligible_target_recall']) for p in points])))


def report(root):
    sets,fresh=load_candidates(root);objects=read(root/'current_gt_annotations.json')
    records=causal_audit(root,list(stream(root/'causal/requests.jsonl')),objects)
    return assemble_report(root,sets,fresh,objects,records)


def prepare_scene_start(root):
    from tools.worldstream_m1.io_utils import iter_json_array
    from .answer_validity_quality import load_current_annotations
    sample_path=Path('data/nuscenes/v1.0-trainval/sample.json')
    samples={s['token']:s for s in iter_json_array(sample_path)}
    tokens=[ERROR_FRAME]
    while samples[tokens[0]]['prev']:tokens.insert(0,samples[tokens[0]]['prev'])
    scene=copy.deepcopy(read(root/'causal_jobs.json')[0]['scene']);scene['tokens']=tokens
    actual=next(s for s in read('data/nuscenes/v1.0-trainval/scene.json') if s['token']==scene['scene_id'])
    assert tokens[0]==actual['first_sample_token']
    jobs=[]
    for j in read(root/'causal_jobs.json'):
        j=copy.deepcopy(j);j['scene']=scene;j['audit_stage']='scene_start_causal';jobs.append(j)
    write(root/'scene_start_causal_jobs.json',jobs)
    objects=read(root/'current_gt_annotations.json');needed=set(tokens)-set(objects)
    sources=[]
    for path in sorted(Path('outputs').glob('*/current_gt_annotations.json')):
        if not needed:break
        data=read(path);hits=needed&set(data)
        if hits:
            objects.update({t:data[t] for t in hits});needed-=hits
            sources.append(dict(path=str(path),sha256=sha256(path),frames=sorted(hits)))
    if needed:
        objects.update(load_current_annotations('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl',needed,'data/nuscenes'))
    write(root/'scene_start_gt_annotations.json',objects)
    write(root/'scene_start_protocol_addendum.json',dict(scope='POST_HOC_CAUSAL_DIAGNOSIS_ONLY',
        true_scene_first_sample=actual['first_sample_token'],history_frames=len(tokens),
        historical_segment_start_index=tokens.index(read(root/'causal_jobs.json')[0]['scene']['tokens'][0]),
        reason='Historical final used a legal cold-start segment, not official scene first sample. Preserve that exact-error replay AND add official full-scene-origin replay; do not conflate initial conditions.',
        no_configuration_reselection=True,frozen_policy_sha256=sha256(root/'final_research_policy.json'),
        jobs_sha256=sha256(root/'scene_start_causal_jobs.json'),GT_sha256=sha256(root/'scene_start_gt_annotations.json'),
        reused_GT_sources=sources,frames_not_previously_in_this_run=len(tokens)-8,
        all_extra_frames_role='CAUSAL_HISTORY_ONLY, never DEV or TEST qualification',
        same_request_RNG='Frame identity, question index, experiment seed/repeat are identical across arms. Question RNG uses actual frame_id, not segment-local index.'))
    print('SCENE_START_DIAGNOSIS_PREPARED',len(tokens)*len(jobs))


def window_chain_audit(root,rows,objects):
    """Trace the previous M1 witness into actual M2 window execution."""
    by={(r['job_spec']['label'],r['frame_index']):r for r in rows}
    index=max(r['frame_index'] for r in rows)
    gt_target=question_target('Where is the nearest pedestrian relative to the vehicle?',objects[ERROR_FRAME])
    chains=[]
    for low,high in (('R0','R3'),('R2','R1')):
        before=by[(low,index-1)];other_before=by[(high,index-1)]
        current=by[(low,index)];other=by[(high,index)]
        previous_gt=next(g for g in objects[before['frame_id']] if g['instance_id']==gt_target['selected_instance'])
        candidates=[a for a in current_anchors(before) if a['semantic_class']=='pedestrian'
            and math.dist(a['center_ego'],previous_gt['center_ego'])<2. and .3<=a['score']<.4]
        matches=match_instances(current_anchors(before),objects[before['frame_id']],before['frame_id'],before['timestamp'])
        witnesses=[]
        for a in candidates:
            aid=a['anchor_id'];uncertainty=[r for r in current['m2']['region_refresh']['uncertainty_regions'] if r.get('source_track_id')==aid]
            witnesses.append(dict(previous_committed_anchor=a,independent_GT_match=matches.get(aid),
                high_threshold_previous_same_geometry_present=any(math.dist(b['center_ego'],a['center_ego'])<1e-5 for b in current_anchors(other_before)),
                current_M1_precheck=current['m1_precheck']['anchors'].get(aid),M2_uncertainty_regions=uncertainty))
        changes=[dict(block=p['block'],executed_only_low_threshold=sorted(set(p['executed_window_ids'])-set(q['executed_window_ids'])),
            executed_only_high_threshold=sorted(set(q['executed_window_ids'])-set(p['executed_window_ids'])))
            for p,q in zip(current['m2']['per_block'],other['m2']['per_block'])]
        chains.append(dict(comparison=low+'->'+high,history_start=current['job_spec']['scene']['tokens'][0],
            previous_frame=before['frame_id'],error_frame=ERROR_FRAME,witnesses=witnesses,
            all_previous_decoded_detections_exactly_equal=all(by[(low,i)]['visual_quality']['boxes']==by[(high,i)]['visual_quality']['boxes']
                and by[(low,i)]['visual_quality']['scores']==by[(high,i)]['visual_quality']['scores'] for i in range(index)),
            all_previous_executed_window_ids_equal=all([b['executed_window_ids'] for b in by[(low,i)]['m2']['per_block']]==
                [b['executed_window_ids'] for b in by[(high,i)]['m2']['per_block']] for i in range(index)),
            error_frame_prefix_window_changes=changes,
            low_threshold_M2_per_camera=current['m2']['per_camera'],high_threshold_M2_per_camera=other['m2']['per_camera'],
            ordinary_window_ids_equal=current['m2']['region_refresh']['ordinary_window_ids']==other['m2']['region_refresh']['ordinary_window_ids'],
            suffix_executed_blocks_low=current['m2']['global_suffix_executed_blocks'],suffix_executed_blocks_high=other['m2']['global_suffix_executed_blocks'],
            suffix_reuse_at_error_low=bool(current['m2']['global_suffix_skipped_blocks']),suffix_reuse_at_error_high=bool(other['m2']['global_suffix_skipped_blocks'])))
    write(root/'causal_window_chain.json',dict(scope='POST_HOC_OFFICIAL_SCENE_ORIGIN',chains=chains,
        interpretation='Previous-frame threshold filtering removes a real target witness and its uncertainty-triggered mandatory M2 refresh. In each fixed-budget contrast the current prefix loses windows19/22 in CAM_FRONT_LEFT for blocks0/1; both arms execute global suffix22 blocks. Evidence identifies the changed window pair, not an isolated single-window intervention.',
        no_online_GT=True,no_runtime_change=True))
    return chains


def feature_report(root,fresh,objects):
    groups={label:[r for r in fresh if r['job_spec']['label']==label] for label in ('Full','R0','R2')}
    audits={k:audit_rows(root,v,k,objects) for k,v in groups.items()}
    es=events(root,[dict(r,condition=k) for k,part in groups.items() for r in part])
    pairs=[];summaries=[];strata=[]
    full={r['frame_id']:r for r in groups['Full']}
    by_event=defaultdict(list)
    for e in es:by_event[(e['condition'],e['frame_id'])].append(e)
    for label in ('R0','R2'):
        a=audits[label]
        for fm in a['frame_metrics']:
            row=next(r for r in groups[label] if r['frame_id']==fm['frame_id'])
            ref=full[fm['frame_id']]
            qs=[e for e in by_event[(label,fm['frame_id'])] if e['quality']['known']]
            ts=[t for t in a['targets'] if t['frame_id']==fm['frame_id'] and t['known']]
            pairs.append(dict(fm,full_e2e_ms=ref['e2e_wall_ms'],full_m2_cuda_ms=ref['stages']['m2_total']['gpu_span_ms'],
                actual_skips=sum(q['llm_skipped'] for q in row['requests']),
                answer_scored=len(qs),answer_correct=sum(e['quality']['score'] for e in qs),
                answer_severe=sum(e['quality']['severe_error'] for e in qs),
                question_target_scored=len(ts),question_target_raw_recalled=sum(t['raw_recalled'] for t in ts),
                question_target_eligible_recalled=sum(t['eligible_recalled'] for t in ts)))
        labelpairs=[p for p in pairs if p['label']==label]
        latency=diagnostic_performance([dict(log_id=p['log_id'],from_ms=p['full_m2_cuda_ms'],to_ms=p['m2_cuda_ms']) for p in labelpairs])
        summaries.append(dict(label=label,feature_relative_l2=dist([p['feature_error']['relative_l2'] for p in labelpairs]),
            feature_RMSE=dist([p['feature_error']['rmse'] for p in labelpairs]),
            maximum_absolute_error=max(p['feature_error']['max_abs'] for p in labelpairs),
            M2_CUDA_Full_to_incremental=latency,quality=paired_quality(es,'Full',label),
            detection=audits[label]['raw'],question_targets=audits[label]['target_summary'],
            performance=performance_group(groups[label])))
        for reuse in (False,True):
            for phase in ('cold','transition','steady'):
                part=[p for p in labelpairs if p['global_suffix_reuse']==reuse and p['phase']==phase]
                if not part:continue
                strata.append(dict(label=label,suffix_reuse=reuse,phase=phase,frames=len(part),
                    relative_l2=dist([p['feature_error']['relative_l2'] for p in part]),
                    raw_visible_GT=sum(p['raw']['GT'] for p in part),raw_TP=sum(p['raw']['TP'] for p in part),
                    E2E_ms=mean([p['e2e_ms'] for p in part]),M2_CUDA_ms=mean([p['m2_cuda_ms'] for p in part])))
    spatial=[]
    for label in ('R0','R2'):
        lp=[p for p in pairs if p['label']==label]
        camera_ids=groups[label][0]['camera_ids']
        for camera in range(6):
            vals=[p['feature_error']['per_camera'][camera]['relative_l2'] for p in lp]
            spatial.append(dict(label=label,camera_index=camera,camera_id=camera_ids[camera],relative_l2=dist(vals)))
    correlations=[]
    for label in ('R0','R2'):
        part=[p for p in pairs if p['label']==label and p['raw']['GT']]
        x=np.array([p['feature_error']['relative_l2'] for p in part]);y=np.array([1-p['raw']['TP']/p['raw']['GT'] for p in part])
        correlations.append(dict(label=label,frames=len(part),feature_vs_visible_detection_miss_Pearson=float(np.corrcoef(x,y)[0,1])
            if len(x)>1 and x.std()>0 and y.std()>0 else None,causal_interpretation=False))
    top_regions=sorted([dict(label=p['label'],frame_id=p['frame_id'],suffix_reuse=p['global_suffix_reuse'],**tile)
        for p in pairs for tile in p['feature_error']['tiles_4x4']],key=lambda r:r['relative_l2'],reverse=True)[:12]
    stable=[]
    for p in pairs:
        known=[e for e in by_event[(p['label'],p['frame_id'])] if e['quality']['known']]
        reference={e['request_index']:e for e in by_event[('Full',p['frame_id'])]}
        if known and p['feature_error']['relative_l2']>.8 and all(e['quality']['score']==reference[e['request_index']]['quality']['score'] for e in known):
            stable.append(dict(label=p['label'],frame_id=p['frame_id'],relative_l2=p['feature_error']['relative_l2'],
                suffix_reuse=p['global_suffix_reuse'],known_answers=len(known),
                identical_known_answer_scores_to_Full=True,all_other_tasks_or_UNKNOWN_certified=False))
    write(root/'m2_feature_quality.json',dict(observer='actual_forward_from_m1_return_NCHW',legacy_callback_evidence_used=False,
        measurement_outside_model_timer=True,groups=summaries,strata=strata,spatial=spatial,
        frame_pairs=pairs,correlations=correlations,top_spatial_errors=top_regions,
        large_error_stable_known_answer_examples=stable,feature_error_is_not_accuracy=True))
    write(root/'m2_detection_quality.json',dict(scope='32 paired DEV frames / 2 logs, same M1=.3',
        groups={k:dict(raw=v['raw'],committed=v['committed'],targets=v['target_summary'],association=v['association']) for k,v in audits.items()},
        official_nuScenes_mAP_NDS=False,reason='Diagnostic historical subset; no official complete evaluation execution',
        map_quality='Budget3 lane subset independently evaluated in map_quality_audit.json; full-data and budget1 MAP_QUALITY_NOT_CERTIFIED',
        map_reason='This paired feature/detection capture has no lane outputs. Subsequent frozen-budget3 confirmation retains actual predictions for independent available lane labels.'))
    write(root/'m2_full_incremental_quality.json',dict(groups=summaries,Full_performance=performance_group(groups['Full']),
        Full_detection=audits['Full']['raw'],M1_fixed=.3,independent_legal_state_evolution=True,
        interpretation='Intervention compares whole closed-loop sequences through same M2 executor; not an isolated-frame feature replacement. No second runtime.'))
    write(root/'m2_quality_latency_tradeoff.json',dict(groups=summaries,strata=strata,correlations=correlations,
        conclusion='Use paired detection and target/answer labels with actual latency, not feature distance alone. Only two exposed DEV logs.'))
    return summaries,pairs


def selected_confirmation(root,sets):
    from .continuous_stream_analysis import performance
    from .critical_path_analysis import diagnostic_performance
    selected=read(root/'final_research_policy.json')['selected']
    rows=list(stream(root/'selected_confirm/requests.jsonl'))
    scene=rows[0]['scene_id']
    reuse=[]
    for condition in ('B','C'):
        key=selected+('_B' if condition=='B' else '')
        for r in sets[key]:
            if r['scene_id']==scene:
                rows.append(dict(r,condition=condition,reused_receipt=True));reuse.append(dict(condition=condition,frame_id=r['frame_id'],origin_stage=r['stage']))
    assert len(rows)==240
    perf=performance(rows)
    pairs=[]
    for workload,frequency in [('addressable','every4'),('addressable','every2'),('addressable','one'),('addressable','three'),('mixed','three')]:
        cell=[r for r in rows if (r['workload'],r['frequency'])==(workload,frequency)]
        by={(r['condition'],r['frame_id']):r for r in cell}
        for a,b in (('B','C'),('A','C')):
            result=diagnostic_performance([dict(log_id=r['log_id'],from_ms=by[(a,r['frame_id'])]['e2e_wall_ms'],to_ms=r['e2e_wall_ms']) for r in cell if r['condition']==b])
            pairs.append(dict(workload=workload,frequency=frequency,comparison=a+'->'+b,**result))
    # The generic legacy performance helper returns degenerate one-log intervals.
    # Suppress them: one log does not estimate population/log uncertainty.
    for c in perf['comparisons']:
        for k in list(c):
            if 'CI' in k:c[k]=None
        c['uncertainty']='Not estimable from one independent log'
    es=events(root,rows)
    cellq=[dict(workload=w,frequency=f,**paired_quality([e for e in es if (e['workload'],e['frequency'])==(w,f)],'B','C'))
        for w,f in sorted({(e['workload'],e['frequency']) for e in es})]
    for q in cellq:
        q['loss_log_CI95']=q['new_severe_log_CI95']=None
    write(root/'selected_confirmation_performance.json',dict(selected=selected,scope='ONE_EXPOSED_DEV_LOG_DIAGNOSTIC',
        frames_per_cell=16,new_model_calls=208,reused_model_receipts=32,
        native_A='True native vision+generation, M1/M2/M3 disabled',groups=perf['groups'],comparisons=pairs,
        quality=cellq,reused=reuse,all_intervals_population_unestimable=True,
        timing='CUDA-synchronized model service wall. Input preparation and offline audit/export serialization excluded equally; answer export included. No arrival/queue simulation.',
        language_timing='B/C prefill_cuda_ms includes cache GPU span; decode CUDA is sum of decode forwards. Legacy prefill_wall alias is CPU enqueue, decode_wall alias is CUDA loop span, not synchronized pure phase host wall.',
        memory='Process peak can retain inactive allocations from other conditions; not standalone minimal footprint.'))
    with (root/'confirmation_quality_requests.jsonl').open('w') as f:
        for e in es:f.write(json.dumps(e)+'\n')
    return pairs,cellq


def archive_receipts(root):
    existing={r['raw']:r for r in read(root/'receipt_archive.json')['files']} if (root/'receipt_archive.json').exists() else {}
    archives=[r for r in existing.values() if not (root/r['raw']).exists()]
    for raw in sorted(root.glob('*/requests.jsonl')):
        dest=raw.with_suffix('.jsonl.gz')
        previous=existing.get(str(raw.relative_to(root)))
        if previous and dest.exists() and sha256(raw)==previous['raw_sha256'] and sha256(dest)==previous['archive_sha256']:
            archives.append(previous);continue
        with raw.open('rb') as src,dest.open('wb') as dst:
            with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as gz:shutil.copyfileobj(src,gz)
        archives.append(dict(raw=str(raw.relative_to(root)),archive=str(dest.relative_to(root)),
            raw_sha256=sha256(raw),archive_sha256=sha256(dest)))
    write(root/'receipt_archive.json',dict(files=sorted(archives,key=lambda r:r['raw']),lossless=True,no_intermediate_tensors=True))


def parameter_freeze_path(root):
    archive=root/'policy_archive/pre_confirmation_policy.json'
    return archive if archive.exists() else root/'final_research_policy.json'


def qualify_final_policy(root,scene_events,scene_rows):
    """Update risk evidence only; preserve exact pre-confirmation selection."""
    frozen_path=parameter_freeze_path(root)
    original=read(frozen_path)
    assert sha256(frozen_path)==read(root/'pre_confirmation_freeze.json')['policy_sha256']
    selected=original['selected']
    part=[e for e in scene_events if e['condition']==selected]
    known=[e for e in part if e['quality']['known']]
    severe=[e for e in known if e['quality']['severe_error']]
    full={(e['frame_id'],e['request_index']):e for e in scene_events if e['condition']=='Full'}
    cases=[]
    for e in severe:
        r=next(r for r in scene_rows if r['job_spec']['label']==selected and r['frame_id']==e['frame_id'])
        gt=e['target_reference'].get('center_ego')
        near=[dict(a,GT_target_distance_m=math.dist(a['center_ego'],gt)) for a in raw_anchors(r,0.)
            if gt and a['semantic_class']==e['target_reference'].get('semantic_class') and math.dist(a['center_ego'],gt)<2.]
        cases.append(dict(event=e,Full_event=full[(e['frame_id'],e['request_index'])],
            raw_target_vicinity_candidates=near,M3_eligible_candidates=eligible(current_anchors(r),e['question']),
            interpretation='Persists under Full when Full_event is also severe. Do not attribute solely to incremental budget. Current positive witness set does not certify true nearest-visible semantics.'))
    write(root/'residual_severe_cases.json',dict(scope='POST_HOC_SCENE_HISTORY_ONLY',selected=selected,cases=cases,
        no_matched_normal_B=True,no_new_vs_B_claim=True,no_parameter_reselection=True))
    archive=root/'policy_archive/pre_confirmation_policy.json'
    if not archive.exists():
        archive.parent.mkdir(exist_ok=True);shutil.copy2(root/'final_research_policy.json',archive)
    final=copy.deepcopy(original)
    final.update(severe_error_status='FAILED_OBSERVED_RESIDUAL_SEVERE_ON_POSTHOC_SCENE; INDEPENDENT_RISK_NOT_CERTIFIED',
        posthoc_scene_quality=dict(requests=len(part),scored=len(known),UNKNOWN=len(part)-len(known),
            absolute_severe=len(severe),fast_severe=sum(e['llm_skipped'] for e in severe),
            matched_normal_B_unavailable=True,new_relative_to_B=None,
            scope='POST_HOC_CAUSAL_HISTORY_ONLY_NOT_DEV_OR_TEST_SELECTION'),
        qualification_evidence=['quality_qualification_addendum.json','residual_severe_cases.json','scene_start_task_quality.json','map_quality_audit.json'],
        parameter_freeze_archive=str(archive.relative_to(root)),
        qualification_metadata_updated_after_diagnosis=True)
    for key in ('M1','M2','M3','selected','parameter_overrides','selection_basis','quality_budget','budget_sensitivity'):
        assert final[key]==original[key]
    write(root/'final_research_policy.json',final)
    write(root/'policy_qualification_revision.json',dict(scope='RISK_METADATA_ONLY_PARAMETERS_UNCHANGED',
        frozen_policy_archive=str(archive.relative_to(root)),frozen_policy_sha256=sha256(archive),
        qualified_final_policy_sha256=sha256(root/'final_research_policy.json'),
        reason='After fixed-parameter confirmation, full-scene causal grading reveals a residual severe error. Final policy must disclose it. Exact pre-confirmation file retained; no threshold, budget, sampling, M3 rule or selection changed.'))
    return final


def map_quality(root):
    """Existing independent OpenLane evaluator, explicitly a 16-frame subset."""
    import pickle
    import traceback
    from .benchmark.metrics import finite_metrics
    from tools.worldstream_m2.map_eval import evaluate_map
    annotation_path=Path('data/nuscenes/data_dict_subset_B_val.pkl')
    info_path=Path('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    rows=[r for r in stream(root/'selected_confirm/requests.jsonl')
        if (r['workload'],r['frequency'])==('addressable','every4')]
    tokens={r['frame_id'] for r in rows}
    infos={i['token']:i for i in pickle.load(info_path.open('rb'))['infos'] if i['token'] in tokens}
    full_gt=pickle.load(annotation_path.open('rb'))
    subset={infos[t]['lane_info']:full_gt[infos[t]['lane_info']] for t in sorted(tokens)}
    def jsonable(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [jsonable(x) for x in v]
        return v
    write(root/'independent_lane_GT_subset.json',dict(frames=[dict(frame_id=t,lane_info=infos[t]['lane_info'],
        annotation=jsonable(subset[infos[t]['lane_info']]['annotation'])) for t in sorted(tokens)],
        source=str(annotation_path),source_sha256=sha256(annotation_path)))
    out=dict(status='SUBSET_EVALUATED',scope='ONE_EXPOSED_DEV_LOG_16_FRAMES',
        selected=read(root/'final_research_policy.json')['selected'],M1=.3,M2_budget=3,
        GT_frames=len(subset),GT_lane_observations=sum(len(v['annotation']['lane_centerline']) for v in subset.values()),
        official_full_dataset_certification=False,
        budget1_quality='MAP_QUALITY_NOT_CERTIFIED: budget1 lane predictions were not retained by the earlier worker; this new confirmation uses the frozen budget3 policy',
        other_unscored_geometry='Only available lane/centerline evaluation, not all map attributes or safety semantics',
        evaluator='Existing tools.worldstream_m2.map_eval.evaluate_map: unchanged official per-frame functions, workers4 CPU, original subset aggregation',
        evaluator_sha256=sha256('tools/worldstream_m2/map_eval.py'),groups={})
    out['reported_primary_metrics']=['OpenLane-V2 DET_l','3D Lane F-Score']
    out['not_interpreted_as_model_quality']=['DET_t','TOP_ll','TOP_lt','OpenLane composite']
    out['uninterpreted_reason']='Existing helper supplies empty traffic elements and zero topology placeholders. Raw evaluator outputs retained for reproducibility, but only actual lane predictions support a lane-quality claim.'
    predictions_seen=[]
    for cond in ('A','B','C'):
        part=[r for r in rows if r['condition']==cond]
        assert len(part)==16
        pred=dict(method='OmniDrive',authors=[],**{'e-mail':'','institution / company':'','country / region':'China'},results={})
        for r in part:
            lane=r['visual_quality']['lane_predictions'];n=len(lane['scores'])
            pred['results'][infos[r['frame_id']]['lane_info']]=dict(predictions=dict(
                lane_centerline=[dict(id=i,points=np.asarray(points,dtype=np.float32),confidence=float(np.asarray(score).item()))
                    for i,(points,score) in enumerate(zip(lane['points'],lane['scores']))],
                traffic_element=[],topology_lclc=np.zeros((n,n)),topology_lcte=np.zeros((n,0))))
        fingerprint={r['frame_id']:r['visual_quality']['lane_predictions'] for r in part}
        same=next((name for name,data in predictions_seen if data==fingerprint),None)
        if same:
            out['groups'][cond]=dict(out['groups'][same],exact_same_predictions_as=same,extra_evaluator_execution=False)
            continue
        try:
            with (root/('map_'+cond+'_evaluation.log')).open('w') as log:
                from contextlib import redirect_stdout,redirect_stderr
                with redirect_stdout(log),redirect_stderr(log):
                    result=evaluate_map(subset,pred,workers=4)
            value,unavailable=finite_metrics(result)
            out['groups'][cond]=dict(metrics=value,nonfinite_metrics_unavailable=unavailable,
                GT_frames=len(subset),predicted_lane_observations=sum(len(r['visual_quality']['lane_predictions']['scores']) for r in part))
        except Exception:
            out['groups'][cond]=dict(status='MAP_QUALITY_NOT_CERTIFIED',error=traceback.format_exc())
            out['status']='PARTIAL_OR_UNAVAILABLE'
        predictions_seen.append((cond,fingerprint))
        write(root/'map_quality_audit.json',out)
    write(root/'map_quality_audit.json',out)
    print(json.dumps(out))


def assemble_report(root,sets,fresh,objects,records):
    audits={label:audit_rows(root,sets[label],label,objects) for label in ('R0','R1','R2','R3')}
    write(root/'m1_detection_coverage.json',dict(protocol=read(root/'protocol_freeze.json')['detector_protocol'],
        groups=[dict(label=k,raw=v['raw'],committed=v['committed']) for k,v in audits.items()],
        scope='32 paired DEV frames / 2 exposed logs. GT object observations, not unique physical instances.',
        FN_definition='No correct-class prediction center within2m. Ambiguous potentially matched GT is UNKNOWN_GT, not proven FN.',
        precision_limitation='FP is not fully certified; unmatched detections are UNKNOWN. Precision conditional on unique match plus conservative bounds.'))
    write(root/'m1_target_recall.json',dict(groups={k:v['target_summary'] for k,v in audits.items()},
        requests=[t for v in audits.values() for t in v['targets']],
        denominator='47 identity-known GT target requests per candidate, 49/96 GT UNKNOWN; separate prediction matching UNKNOWN failure breakdown.',
        M3_eligibility='Exact current-object, camera, .35 score, class and region predicates, before answer admission .45/.75.'))
    write(root/'m1_association_quality.json',dict(groups={k:v['association'] for k,v in audits.items()},standard_MOT=False,
        effective_coverage='Numerator counts only correct links for instances visible in BOTH adjacent frames; denominator all such visible GT opportunities.'))
    propagation=read(root/'m123_fact_propagation.json')
    for g in propagation['groups']:g['target_coverage']=audits[g['label']]['target_summary']
    write(root/'m123_fact_propagation.json',propagation)
    features,fpairs=feature_report(root,fresh,objects)
    confirm,cq=selected_confirmation(root,sets)
    policy=read(root/'final_research_policy.json');comparison=read(root/'limited_configuration_comparison.json')
    assert sha256(parameter_freeze_path(root))==read(root/'pre_confirmation_freeze.json')['policy_sha256']
    stage_counts={};actual=[]
    for stage in ('causal','dev_quality','selected_confirm','scene_start_causal'):
        rows=list(stream(root/stage/'requests.jsonl'));status=read(root/stage/'run_status.json');assert status['status']=='COMPLETED'
        actual.extend(rows)
        stage_counts[stage]=dict(full_model_calls=len(rows),stateful_M123_calls=sum(r['condition']!='A' for r in rows),
            native_calls=sum(r['condition']=='A' for r in rows),question_requests=sum(len(r['requests']) for r in rows),
            QA_frames=sum(bool(r['requests']) for r in rows),idle_frames=sum(not r['requests'] for r in rows),
            actual_LLM_forwards=sum(r['observed_llm_forward_calls'] for r in rows),actual_skips=sum(q['llm_skipped'] for r in rows for q in r['requests']))
    totals={k:sum(s[k] for s in stage_counts.values()) for k in next(iter(stage_counts.values()))}
    write(root/'gpu_execution_counts.json',dict(new_only=totals,by_stage=stage_counts,
        distinct_input_frames=len({r['frame_id'] for r in actual}),distinct_logs=len({r['log_id'] for r in actual}),
        historical_reused_receipts_do_not_count_as_new=True,
        actual_prefill_skips=totals['actual_skips'],actual_decode_skips=totals['actual_skips'],
        regressions_are_zero_full_model_experiments=True,
        all_frames_used_only_as_DEV_or_posthoc_diagnosis=True,new_independent_TEST_frames=0))
    points={p['label']:p for p in comparison['points']};selected=points[policy['selected']]
    old_severe=read(OLD/'severe_error_audit.json')
    scene_records=causal_audit(root,list(stream(root/'scene_start_causal/requests.jsonl')),
        read(root/'scene_start_gt_annotations.json'),stem='scene_start_severe_error_counterfactual')
    import tempfile
    scene_rows=list(stream(root/'scene_start_causal/requests.jsonl'))
    window_chain_audit(root,scene_rows,read(root/'scene_start_gt_annotations.json'))
    with tempfile.TemporaryDirectory(prefix='omnidrive-scene-offline-',dir='/tmp') as temporary:
        offline=Path(temporary)
        write(offline/'current_gt_annotations.json',read(root/'scene_start_gt_annotations.json'))
        scene_events=events(offline,[dict(r,condition=r['job_spec']['label']) for r in scene_rows])
    with (root/'scene_start_quality_requests.jsonl').open('w') as f:
        for e in scene_events:f.write(json.dumps(e)+'\n')
    scene_quality=dict(scope='POST_HOC_CAUSAL_DIAGNOSIS_ONLY_NOT_SELECTION',
        groups={k:quality_summary([e for e in scene_events if e['condition']==k]) for k in ('R0','R1','R2','R3','Full')},
        paired_vs_R0={k:paired_quality(scene_events,'R0',k) for k in ('R1','R2','R3','Full')},
        paired_vs_Full={k:paired_quality(scene_events,'Full',k) for k in ('R0','R1','R2','R3')},
        note='New severe here is relative to another C-basic causal arm, not matched normal-LLM B. One scene, not independent risk certification.')
    for pair_groups in ('paired_vs_R0','paired_vs_Full'):
        for q in scene_quality[pair_groups].values():q['loss_log_CI95']=q['new_severe_log_CI95']=None
    write(root/'scene_start_task_quality.json',scene_quality)
    policy=qualify_final_policy(root,scene_events,scene_rows)
    lane=read(root/'map_quality_audit.json')
    lane.update(reported_primary_metrics=['OpenLane-V2 DET_l','3D Lane F-Score'],
        not_interpreted_as_model_quality=['DET_t','TOP_ll','TOP_lt','OpenLane composite'],
        uninterpreted_reason='Empty traffic elements and zero topology are placeholders from existing helper; only actual lane predictions support a lane-quality claim.')
    write(root/'map_quality_audit.json',lane)
    write(root/'quality_qualification_addendum.json',dict(qualified_policy_sha256=sha256(root/'final_research_policy.json'),
        original_parameter_freeze_sha256=sha256(parameter_freeze_path(root)),
        parameter_selection_unchanged=True,qualification_scope='Known B-C budget is conditional on paired B using the SAME M1/M2; independent coverage gates are separate.',
        M2_Full_to_selected=next(f['quality'] for f in features if f['label']==policy['selected']),
        M2_Full_to_selected_2pp_known_budget_pass=next(f['quality']['loss']<=.02 for f in features if f['label']==policy['selected']),
        note='Selected budget3 loses one of47 known answers versus Full (2.13pp), despite better target coverage than budget1. This is retained as residual approximation cost, not hidden by the B-C gain. No whole-system/full-task quality certification.',
        selected_residual_absolute_severe=policy['posthoc_scene_quality'],
        map_evidence='map_quality_audit.json',strict_risk_pass=False,production_deployment_allowed=False))
    write(root/'severe_error_audit.json',dict(DEV={k:p['quality'] for k,p in points.items()},
        selected_confirmation=cq,historical_R1=old_severe,
        causal_answers={r['label']:dict(answer=r['answer'],severe=r['quality']['severe_error'],actual_skip=r['actual_skip']) for r in records},
        true_scene_start_causal_answers={r['label']:dict(answer=r['answer'],severe=r['quality']['severe_error'],actual_skip=r['actual_skip']) for r in scene_records},
        selected_residual_absolute_severe=policy['posthoc_scene_quality'],residual_case_file='residual_severe_cases.json',
        no_independent_2percent_rare_risk_certification=True,
        no_historical_NO_GO_rewritten=True))
    verdict=dict(statuses=['QUALITY_LATENCY_CANDIDATE_SELECTED','SELECTED_CANDIDATE_RESIDUAL_RISK_FAILED','INDEPENDENT_VALIDATION_INCOMPLETE'],
        selected=policy['selected'],fastest= comparison['fastest'],
        fastest_historical_risk_status='FASTEST_CANDIDATE_RISK_FAILED',
        selection_scope='RETROSPECTIVE_CONFIGURATION_SELECTION',
        research_performance_accepted=True,strict_risk_accepted=False,production_deployment_allowed=False,
        no_algorithm_change=True,M1_hash_optimization_retained=True,M2_sole_executor_retained=True,M3_C_basic_retained=True,
        historical_v3_v4_NO_GO_preserved=True,previous_fastest_failure_preserved=True,
        causal_finding='Historical 8-frame cold start shows M1/M2 interaction (.4/1 fails; .4/3 correct; .3/1 fallback). Official 16-frame scene-origin replay shows .4 fails with BOTH M2 budgets, while .3 retains front-left with both. In every failed final frame correct-target vicinity decoded scores are already below .3, so the effect is history-mediated, not direct final-frame .4 filtering. C-basic cannot certify nearest-GT completeness from observed witnesses.',
        causal_window_chain_evidence='causal_window_chain.json: previous .3543 witness removed by .4, mandatory CAM_FRONT_LEFT windows19/22 absent in both prefix blocks; current global suffix fully executes in all arms',
        M2_Full_to_selected_known_answer_loss=next(f['quality']['loss'] for f in features if f['label']==policy['selected']),
        full_task_quality_certified=False,
        map_quality_evidence='map_quality_audit.json; budget3 lane subset only, budget1 unscored',
        official_scene_start_case={r['label']:dict(answer=r['answer'],fast=r['actual_skip'],severe=r['quality']['severe_error']) for r in scene_records},
        selected_residual_absolute_severe=policy['posthoc_scene_quality'],
        constraints='Independent TEST unavailable. Two DEV logs and one confirmation log do not certify a rare serious-error rate. Full-data/budget1 map/lane quality not certified.',
        next_action='Freeze the sole method and this candidate; proceed to paper experiments with predeclared new independent data. No further tuning to the exposed case.')
    write(root/'research_verdict.json',verdict)
    write_narrative(root,policy,comparison,features,fpairs,records,confirm,cq,totals)
    archive_receipts(root)
    write(root/'integrity_verification.json',verify(root))
    print(json.dumps(dict(status='FINAL_AUDIT_ASSEMBLED',selected=policy['selected'],new_GPU=totals)))


def write_narrative(root,policy,comparison,features,fpairs,records,confirm,cq,totals):
    points={p['label']:p for p in comparison['points']}
    detection=read(root/'m1_detection_coverage.json')['groups']
    det={g['label']:g for g in detection};targets=read(root/'m1_target_recall.json')['groups']
    assoc=read(root/'m1_association_quality.json')['groups']
    selected=policy['selected'];r0=det['R0']['committed'];r3=det['R3']['committed'];r1=det['R1']['committed']
    feat={g['label']:g for g in features};case={r['label']:r for r in records}
    def pct(x):return '不可评分' if x is None else f'{100*x:.2f}%'
    def rec(d):return pct(d['groups']['visible_GT']['proven_class_recall'])
    table='|配置|M1阈值|M2预算|DEV E2E ms/帧|可见对象确认Recall|问题目标Eligible Recall|已评分答案正确率|新增严重错误 vs B|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for k,p in points.items():
        table+=f"|{k}|{p['M1'].get('min_detection_score',.3)}|{p['M2'].get('ordinary_budget',3)}|{p['performance']['E2E_ms']['mean']:.3f}|{pct(p['visible_object_recall'])}|{pct(p['question_eligible_target_recall'])}|{pct(p['fact_quality']['answer_correctness'])}|{p['quality']['new_severe']}|\n"
    causal_text='''# 严重错误因果审计

只用于 Post-hoc Causal Diagnosis，不参与最终参数选择。四组从历史收据使用的同一合法连续流起点重放全部8帧，至错误帧结束；不是只在错误帧改参数。另有同执行器 Full 控制。完整原始检测、M1提交、M2缓存/刷新、M3候选和请求Forward保存在压缩收据及 severe_error_counterfactual.json。

GT最近可见Pedestrian是前左方，中心约(10.0937,8.1488,1.3329)m；其唯一GT身份通过原有可见性/最近目标协议，未改变GT定义。但多重检测框的独立匹配可能UNKNOWN，不能把附近框都认证为同一GT目标。

|组|参数|实际回答路径及答案|
|---|---|---|
'''
    for k,identifier in [('R0','D00'),('R3','D10'),('R2','D01'),('R1','D11'),('Full','Full')]:
        r=case[k];causal_text+=f"|{identifier}/{k}|{r['M1']} / {r['M2']}|{'FAST' if r['actual_skip'] else 'LLM'}: {r['answer']}（严重={r['quality']['severe_error']}）|\n"
    causal_text+='''
以下针对**历史8帧重置起点**：连续状态中存在M1×M2交互，以及C-basic对可见最近目标覆盖的原有限制。官方Scene起点结论见下节，不能将该8帧条件的预算归因外推到所有起点。

1. **当前帧直接M1过滤**：D11的Front-Left附近解码候选存在，但最高约0.1984，已经低于原始0.3。不是0.4在当前帧直接过滤一个0.3原本会接受的正确目标。
2. **M2 Approximation Effect**：固定M1=.4，只恢复M2预算3，Front-Left附近高分框约0.7638被保留，FAST回答正确方位；预算1对应约0.1984且目标不进入世界状态。该对照是整个合法闭环序列的因果效应，不证明仅错误帧某个特征区域是唯一原因。
3. **M1历史介导效应**：固定M2预算1，只恢复M1=.3，当前检测又有约0.7942的Front-Left框及更近约0.3949框；M3选择更近低分框，未通过.45准入，转LLM。M1阈值会改变前序世界状态及后续M2刷新，不能把M1视为只影响当前帧后处理的独立开关。
4. **M3 Target Selection Effect**：D11正确目标没有进入合法当前候选集合；M3按现有集合距离排序选择后左方真实Pedestrian，其3D匹配约0.386m。没有证据显示距离排序实现把一个已存在、更近、合格的Front-Left目标排在后面。真正局限是把不完整检测集合的最近对象当成问题所要求的最近可见对象。检测置信度/绑定风险不等于该语义正确概率。D01的低分最近框会使高分次近框不能直接回答，这是原选择/准入逻辑，不在本轮修改。
5. **Base Perception Limitation**：不能把本例完全归为Full基础视觉漏检；Full和D00均有高分前左候选并回答Front-Left。模型的一般检测与重复框问题仍存在。
6. **Evaluation Ambiguity**：GT目标唯一、方位远离边界，本例无GT方位歧义；预测框身份匹配的UNKNOWN另行保留。D00/Full答对方位也不等于每个重复框都有已认证身份。

历史R1错误仍是新增严重前后反转。恢复单一参数后本例不再出现反转，只支持事后机制诊断，不能用本例给新配置授予独立TEST通过。历史v3/v4 NO_GO和上一轮R1风险失败均不改写。
'''
    scene=read(root/'scene_start_severe_error_counterfactual.json')
    causal_text+='\n## 官方Scene首帧的完整历史补充\n\n历史起点位于实际Scene第8个sample之后。额外从官方first_sample_token重放16帧，五个控制共80次模型调用；配置冻结不变。以下结果与上面的历史8帧起点不合并。\n\n'
    for r in scene['records']:
        causal_text+=f"- {r['label']}: {'FAST' if r['actual_skip'] else 'LLM'}，{r['answer']}；severe={r['quality']['severe_error']}。\n"
    causal_text+='\n完整结果及所有前序目标分数、M1状态数、M2刷新/复用见scene_start_severe_error_counterfactual.json；原始GT缓存保留不动，额外历史GT在独立文件。起点敏感性限制因果结论的外推，不能拿其中一个有利起点覆盖另一个实际错误。\n'
    causal_text+='\n真实Scene首帧结果表明：.4阈值在预算3和预算1下均出现Rear-Left；.3在两种预算下均回答Front-Left。失败臂当前GT附近最高分仍仅约.1903/.1980，已低于.3；因此仍是前序M1状态改变经M2及检测头连续记忆传播的效应，不是最终帧0.4直接剔除一个0.3本会保留的候选。预算1不是跨起点的必要原因；M1历史参数影响更稳健，但也不能定位单个先前窗口为唯一原因。Full有高分前左候选，同时较近.3901候选使M3转LLM，说明全量视觉并不自动保证完整QA正确。\n'
    causal_text+='\n## 已定位的M1→M2窗口链路\n\n在真实Scene起点固定预算3的对照中，前15帧的原始检测和M2前缀执行窗口ID均逐帧完全一致；错误帧前一帧，目标附近Pedestrian分数0.3543037474被M1=.3提交、被.4排除。下一帧，该保留锚点在CAM_FRONT_LEFT的观测不确定（uncertainty_exceeds_search），因此合法地要求强制刷新窗口19、22；.4缺少此锚点，两个窗口在M2块0、1均被复用，共少4个实际窗口计算。预算1的固定对照同样缺少这一窗口对。两臂的普通窗口选择相同，全局后缀都完整执行22块，**本帧不是全局后缀复用导致的错误**。真实返回特征及检测仅从本帧开始分岔，高分前左目标下降至约.19，随后M3在残缺候选中选择后左对象。causal_window_chain.json保存前帧独立GT匹配、M1预检查、强制刷新区域和实际执行ID。该证据定位到改变的窗口对，没有声称分别干预单个窗口。\n'
    residual=read(root/'residual_severe_cases.json')
    causal_text+='\n## 冻结候选残余严重错误\n\nR0在真实Scene前序仍有1/39已评分绝对严重错误（48总请求、9 UNKNOWN），为FAST左右方位错误；同一错误在Full中也存在。它不是相对正常LLM B的“新增”计数，因为该补充没有B臂。完整帧、答案、GT、候选及Full对应事件见residual_severe_cases.json。不能因为重点前后错误修复就宣称R0通过严格风险验收。风险元数据在最终策略文件中补充，原始参数冻结文件逐字节归档，所有参数及选择完全不改。\n'
    (root/'severe_error_causal_audit.md').write_text(causal_text)
    (root/'implementation_summary.md').write_text('# 最终核验与配置冻结\n\n'+
        f'最终研究候选：{selected}，M1阈值{policy["M1"]["min_detection_score"]}、M2预算{policy["M2"]["ordinary_budget"]}、M3 current_only C-basic .45/.75。保留M1哈希优化、唯一M2执行器和全部M3原算法；生产默认关闭。\n\n'+
        table+'\n配置选择严格只用两条既有DEV；错误帧四组对照仅事后归因。新增外部观测脚本final_quality_capture.py捕获真实M2返回值、6相机及4×4区域特征误差，所有离线质量代码在final_quality_analysis.py。未训练、下载、安装依赖、修改基础权重或新增正式runtime。\n\n'+
        f'新增{totals["full_model_calls"]}次完整模型调用，含{totals["stateful_M123_calls"]}次M123和{totals["native_calls"]}次纯Native；复用历史收据不计为新增。新运行、历史正式R1结果、单日志冻结配置确认分别报告，不合并总体均值。\n\n'+
        '完整GT分母、可见性分母和匹配歧义分开；无法认证的FP为null，未匹配预测为UNKNOWN，提供条件精确率与上下界，不冒充官方nuScenes mAP/NDS或标准MOT。Map/Lane资格以map_quality_audit.json为准。\n')
    validation_paths={k:root/('release_'+k+'_validation/summary.json') for k in ('cpu','cuda')}
    if all(p.exists() for p in validation_paths.values()):
        summaries={k:read(p) for k,p in validation_paths.items()}
        assert all(s['exit_code']==0 for s in summaries.values())
        write(root/'regression_summary.json',dict(final=summaries,source_paths={k:str(p.relative_to(root)) for k,p in validation_paths.items()},
            full_model_experiment_calls=0))
        with (root/'implementation_summary.md').open('a') as f:
            f.write(f"\n最终回归：CPU {summaries['cpu']['counts']}；RTX3090 CUDA {summaries['cuda']['counts']}。修复并验证了仅gzip检出时重新分析仍保留原始收据SHA清单。所有服务端算法与配置源文件不变。\n")
    answers=[
        '是M1阈值与M2近似在连续状态演化中的交互：D11的前左高分目标丢失，M3用剩余真实后左对象回答“nearest”。不是当前帧单纯从0.3升到0.4造成的直接过滤。详见severe_error_causal_audit.md。',
        '解码300框中有GT中心2m内的Pedestrian候选；D11最高约0.1984，低于原始保留阈值。严格唯一匹配不足时仍标UNKNOWN，不凭近邻几何认证每个重复框。',
        '在错误当前帧不成立：相关候选已低于0.3。但提高阈值改变前序M1状态和后续M2刷新，有闭环介导效应；一般DEV中确实存在额外覆盖损失。',
        '会。在M1=.4的成对全历史干预中，预算3保留约.7638的前左高分框，预算1仅剩约.1984；在M1=.3时变化不同，说明存在交互。',
        '最终选中对象不符合真实最近可见目标语义，但D11正确目标不在M3合法当前集合；没有观察到对已存在合格最近目标的argmin排序实现错误。C-basic缺少完整场景覆盖保证。',
        'D00本例FAST回答Front-Left，无严重前后反转。该结果仅为事后对照；不表示原参数在其他帧没有严重错误。',
        f'DEV 32帧M1输出{r0["predictions"]}→{r3["predictions"]}；确认几何匹配{r0["matched"]}→{r3["matched"]}；确认类别正确{r0["class_correct"]}→{r3["class_correct"]}。可见GT确认TP {r0["groups"]["visible_GT"]["TP"]}→{r3["groups"]["visible_GT"]["TP"]}，分母165。计数为逐帧对象观察，非唯一实例。',
        f'M1=.3→.4、M2预算3时，可见对象确认Recall {rec(r0)}→{rec(r3)}；条件匹配类别Precision {pct(r0["conditional_matched_class_precision"])}→{pct(r3["conditional_matched_class_precision"])}。完整Precision/FP无法认证，见上下界；不能将条件Precision当总体Precision。',
        f'47个身份可评分请求：Committed {targets["R0"]["successes"]["committed_recalled"]}/47→{targets["R3"]["successes"]["committed_recalled"]}/47；Eligible {targets["R0"]["successes"]["eligible_recalled"]}/47→{targets["R3"]["successes"]["eligible_recalled"]}/47。R1 Eligible={targets["R1"]["successes"]["eligible_recalled"]}/47；R2={targets["R2"]["successes"]["eligible_recalled"]}/47。49/96 UNKNOWN保留。',
        f'部分匹配的关联计数、ID变化、对象观察跨度及丢失恢复均在m1_association_quality.json。R0/R3 ID变化率={pct(assoc["R0"]["ID_change_rate"])}/{pct(assoc["R3"]["ID_change_rate"])}。匹配覆盖和对象集合变化使条件精确率不能单独代表连续性；仅两条DEV，不能宣称统计显著或标准MOT改善。',
        f'M1固定.3、32帧真实返回Tensor：预算3 Relative L2均值{feat["R0"]["feature_relative_l2"]["mean"]:.6f}，预算1均值{feat["R2"]["feature_relative_l2"]["mean"]:.6f}；逐帧RMSE/max_abs见m2_feature_quality.json，不采用旧Block回调。',
        '按cold/transition/steady、全局后缀复用/刷新、6相机、4×4区域和token误差分位数逐帧报告；不能把特征误差数值直接读作感知精度。最大区域与对应帧可直接由frame_pairs查证。',
        f'M1固定.3的预算3/1原始可见目标确认Recall分别{rec(feat["R0"]["detection"])}/{rec(feat["R2"]["detection"])}；Full参照、定位误差、各类别/距离、FN/UNKNOWN见m2_detection_quality.json。是子集诊断，不是官方全量mAP/NDS。',
        f'原始Question Target Recall预算3={pct(targets["R0"]["raw_recalled"])}，预算1={pct(targets["R2"]["raw_recalled"])}；Eligible分别{pct(targets["R0"]["eligible_recalled"])}/{pct(targets["R2"]["eligible_recalled"])}。根据配对DEV结果判断，不能只用跳过窗口或特征相关性推断漏检因果。',
        '见map_quality_audit.json：若有可用独立车道标注与合法评测接口，报告子集指标及分母；其范围不外推为全数据集认证，也不使用自一致性替代GT。',
        '真实M2 CUDA与完整E2E见m2_quality_latency_tradeoff.json，刷新/复用按逐帧记录分层。Full→Incremental是同执行器复用消融；不是新的代码优化，也不等于理论窗口数加速。',
        '详见m123_fact_propagation.json：保留对象、可解析问题、目标覆盖、合法候选、准入/实际跳过、FAST/fallback质量及路由切换增加的Forward/实测请求时延全部分开。较少对象与较多FAST不能自动证明系统更优。',
        '有。R1相对R0在DEV的可见对象与问题目标覆盖下降；历史评价还出现一个新增严重前后反转。速度优势和平均答案正确率不能抹去这些代价。',
        f'按预先冻结的质量与覆盖门槛，选择{selected}。比较分母为相同32帧/2条DEV的对象三问；只在R0/R1已具备的五负载均值与其他候选缺测时，不伪造四候选统一五负载总体。',
        f'最终选择{selected}，观测最快为{comparison["fastest"]}；最佳质量并非对所有指标绝对占优的全局最优，只是限定候选与预设门槛所支持的选择。',
        f'在共同DEV三问分母上，冻结候选比观测最快平均多{comparison["selected_penalty_ms"]:.3f}ms/帧。其他负载的新确认单独报告，不与此数字混为同一总体。',
        '本例的参数相关因素是M1/M2闭环交互造成可靠前左候选缺失；M3把局部候选的nearest当真实nearest属于原方法覆盖局限。没有给所有其他严重错误强行归因；完整DEV及历史严重错误保留在severe_error_audit.json。',
        f'冻结候选的共同DEV三问实际跳过{points[selected]["performance"]["actual_skips"]}/{points[selected]["performance"]["question_requests"]}；新五负载确认分别提供真实Forward/Prefill/Decode跳过计数。历史R1的127/216只属于R1，不移植给冻结候选。',
        f'1%、2%、3%已评分答案损失预算分别选择{comparison["by_budget"]}，同时执行独立于答案均值的覆盖和严重错误门槛。UNKNOWN的保守全请求损失界未必落在这些预算内。',
        '不足。暴露数据、少量日志、UNKNOWN和历史新增严重错误不支持严格2%罕见严重风险认证；不能把DEV零新增严重或小于2%的点估计当统计保证。',
        f'{selected}：M1 min_detection_score={policy["M1"]["min_detection_score"]}；M2 ordinary_budget={policy["M2"]["ordinary_budget"]}；M3 current_only/basic/.45/.75。唯一推荐文件final_research_policy.json，其他候选仅离线审计，不自动改变生产配置。',
        '配置DEV仅2日志，最终新性能确认仅1日志。额外8个真实Scene前序帧只为因果历史，不进入DEV/TEST。历史评价、两种起点因果重放、新确认分开。没有新的合格独立TEST，泛化、全量检测/地图质量和严格风险仍未认证。',
        '可以冻结当前主方法与单一研究候选，进入预注册主实验/论文评估准备；不继续添加风险算法或搜索。下一步需要真正独立数据和完整任务评价，现有证据不能写成生产GO或独立质量验收通过。'
    ]
    feature_detail=read(root/'m2_feature_quality.json')
    biggest=feature_detail['top_spatial_errors'][0]
    stable=feature_detail['large_error_stable_known_answer_examples']
    answers[11]+=f" 最大区域例：{biggest['label']}，Frame {biggest['frame_id']}，camera index={biggest['camera']}、4×4 tile=({biggest['row']},{biggest['col']})，Relative L2={biggest['relative_l2']:.6f}，suffix reuse={biggest['suffix_reuse']}。"
    if stable:
        e=stable[0];answers[11]+=f" 高误差但有限已知答案评分稳定的例：{e['label']}/{e['frame_id']}，L2={e['relative_l2']:.6f}，{e['known_answers']}个已评分答案与Full同分；不覆盖UNKNOWN或其他感知任务。"
    answers[13]+=' proven Recall下降包含明确缺少候选和预测身份匹配UNKNOWN两部分，m1_target_recall.json已分别给出，不能全部称为视觉漏检。'
    lane=read(root/'map_quality_audit.json')
    la=lane['groups']['A']['metrics'];lc=lane['groups']['C']['metrics']
    answers[14]=f"已完成独立车道子集评价：16帧、{lane['GT_lane_observations']}个GT车道观察。Native A→冻结配置C的DET_l {la['OpenLane-V2 Score']['DET_l']:.8f}→{lc['OpenLane-V2 Score']['DET_l']:.8f}，3D Lane F-score {la['F-Score for 3D Lane']['score']:.8f}→{lc['F-Score for 3D Lane']['score']:.8f}。仅预算3/单日志子集；预算1及全量Map质量仍MAP_QUALITY_NOT_CERTIFIED。空traffic/零topology导致的其他原始复合指标不作为质量结论。"
    answers[15]+=f" 本轮32帧M1=.3同执行器Full CUDA均值{features[0]['M2_CUDA_Full_to_incremental']['from_ms']:.3f}ms；预算3为{feat['R0']['performance']['M2_CUDA_ms']['mean']:.3f}ms，预算1为{feat['R2']['performance']['M2_CUDA_ms']['mean']:.3f}ms。"
    scene_answers={r['label']:r['answer'] for r in scene['records']}
    answers[0]='准确来源依赖连续历史起点：历史8帧起点是M1/M2交互（仅.4/1反转）；官方Scene首帧16帧起点是.4在两种预算下均反转、.3均前左。共同点是M1历史状态经M2/检测头连续记忆传播后，正确目标附近候选分数已降到.3以下；不是最终帧0.4直接过滤一个0.3本会保留的高分目标。M3又把不完整候选集合的nearest当真实nearest，形成显式错误。不能定位一个先前窗口为唯一原因。详见两个起点的完整收据与因果审计。'
    answers[0]+=' 已定位关键链：前帧.3543目标被.4过滤→下一帧CAM_FRONT_LEFT窗口19/22失去不确定性强制刷新（块0/1共少4次）→特征与高分检测在错误帧首次分岔→M3错误最近目标。双方此帧均执行全局后缀22块，不能归咎于此帧的全局后缀复用。'
    answers[3]+=' 此结论只在历史8帧起点成立；真实Scene16帧起点的.4/3也失败，因此预算1不是该错误跨起点的必要条件。'
    answers[5]+=' 官方Scene首帧D00结果为：'+scene_answers['R0']+'。'
    answers[21]+=' 真实Scene起点的对照进一步表明M1阈值历史效应存在于两种M2预算；不能把本例一概归咎于预算1。'
    answers[21]+=' R0另有FAST左右错误在Full下同样存在，属于未被参数恢复消除的方法/基础感知与目标覆盖局限，不能归因于预算1单独造成。'
    answers[24]+=' R0在额外Scene历史中仍有1/39已评分绝对严重错误（9/48 UNKNOWN），同帧Full也错误；没有正常B对照，不能叫新增vs B，但足以阻止把冻结配置标成无严重错误。'
    perf=read(root/'selected_confirmation_performance.json')
    c_three=next(g for g in perf['groups'] if (g['condition'],g['workload'],g['frequency'])==('C','addressable','three'))
    answers[22]+=f" 冻结配置单日志三问确认（复用该日志的历史实测收据）为{c_three['actual_prefill_skips']}/{c_three['questions']}={100*c_three['actual_prefill_skips']/c_three['questions']:.2f}%；Prefill和Decode同时跳过的是同一批请求，不能重复计数。"
    (root/'result_answers.md').write_text('# 28项最终回答\n\n'+table+'\n'+
        '\n\n'.join(f'{i}. {answer}' for i,answer in enumerate(answers,1))+'\n')
    perf_table='|负载|A ms/帧|B ms/帧|C ms/帧|B→C|A→C|\n|---|---:|---:|---:|---:|---:|\n'
    for workload,frequency in [('addressable','every4'),('addressable','every2'),('addressable','one'),('addressable','three'),('mixed','three')]:
        b=next(p for p in confirm if (p['workload'],p['frequency'],p['comparison'])==(workload,frequency,'B->C'))
        a=next(p for p in confirm if (p['workload'],p['frequency'],p['comparison'])==(workload,frequency,'A->C'))
        perf_table+=f"|{workload}/{frequency}|{a['from_ms']:.3f}|{b['from_ms']:.3f}|{a['to_ms']:.3f}|{b['speedup']:.4f}×|{a['speedup']:.4f}×|\n"
    (root/'e2e_performance_report.md').write_text('# 冻结配置性能确认\n\n'+
        f'{selected}，一条暴露DEV流、每格16帧；三问B/C的32个收据复用先前实际执行，其余208次新完整模型调用。无独立TEST意义；一个日志无法估计日志总体不确定性，因此不提供退化bootstrap置信区间。\n\n'+
        perf_table+'\n低问答频率C慢于纯Native，M1/M2管理代价仍存在。FPS/QPS为计算服务吞吐，不是传感器到达或排队时延。M3阶段字段定义见JSON，不把CPU enqueue或CUDA loop span当同步纯阶段Host Wall。\n\n'+
        '全部三日志历史R1结果引用 ../m123_critical_path_and_joint_tuning_run01/abc_comparison.json，仅归属于R1；不与本轮R0单日志值拼成总体，也不跨日志把差值当参数因果收益。\n')
    py='taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python'
    reproduce=f'''# 复现

基准提交40c8cdce74ef2e5cbdb8c2bb6eaa5ef42e2ad071；当前审计提交包含本目录与两个外部脚本。环境warm reuse，无安装/训练/下载；权重ckpts/iter_10548.pth，SHA256 6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d。GPU RTX3090。所有命令必须使用上述taskset；GPU被占用只等待，锁/tmp/omnidrive_m123_latency_gpu.lock。

以下命令须在 /data/qh/phdCode/work4/OmniDrive 项目根目录执行。select仅允许第一次选择，已有final_research_policy.json时拒绝覆盖；之后只能report重新核验，或使用新的复现目录。

最终参数只有R0一套。policy_archive/pre_confirmation_policy.json保留确认前的原始冻结文件及SHA；final_research_policy.json随后只补充已观察到的残余严重错误风险元数据。policy_qualification_revision.json逐项证明M1/M2/M3、质量预算、候选选择均未修改。档案不构成第二套推荐运行路径。

无GPU历史完整性校验：

```sh
{py} -m tools.worldstream_m123.final_quality_analysis verify
```

本次三个真实GPU阶段的执行状态、源文件哈希、严格checkpoint验证、seeded CUDA witness与原始请求均在对应子目录。完整请求JSONL已无损gzip，receipt_archive.json包含原文与压缩SHA256；读取器可直接读取.gz，不需要大型Tensor。

在现有收据上重新组装离线报告（不会训练或运行推理，保留冻结策略）：

```sh
{py} -m tools.worldstream_m123.final_quality_analysis report
```

重新推理必须使用全新输出目录；复用相同帧只是回归/性能重放，不是新的独立TEST。顺序如下：

```sh
{py} -m tools.worldstream_m123.final_quality_analysis prepare --root /tmp/omnidrive-final-quality-replay
{py} -m tools.worldstream_m123.final_quality_capture causal --root /tmp/omnidrive-final-quality-replay
{py} -m tools.worldstream_m123.final_quality_capture dev_quality --root /tmp/omnidrive-final-quality-replay
{py} -m tools.worldstream_m123.final_quality_analysis select --root /tmp/omnidrive-final-quality-replay
{py} -m tools.worldstream_m123.final_quality_capture selected_confirm --root /tmp/omnidrive-final-quality-replay
{py} -m tools.worldstream_m123.final_quality_analysis prepare_scene_start --root /tmp/omnidrive-final-quality-replay
{py} -m tools.worldstream_m123.final_quality_capture scene_start_causal --root /tmp/omnidrive-final-quality-replay
{py} -m tools.worldstream_m123.final_quality_analysis map_quality --root /tmp/omnidrive-final-quality-replay
{py} -m tools.worldstream_m123.final_quality_analysis report --root /tmp/omnidrive-final-quality-replay
```

Full必须先于对应Incremental，以在/tmp/omnidrive-final-quality-features生成同序列当前返回NCHW参照。不能同时运行两个重复实验写同一临时特征命名空间；GPU锁串行整个阶段。临时Tensor不提交Git，不参与在线准入。

新阶段统计从实际请求逐行计算，历史复用不计新增；causal40帧、DEV128帧、冻结后五负载确认208帧、官方Scene首帧起点causal80帧。确认中的32个三问B/C收据来自先前DEV，其余208实际执行。既有正式R1五负载实验保持独立，绝不混入本轮均值。

CPU/CUDA回归：

```sh
{py} -m tools.worldstream_m123.answer_validity_validate --output /tmp/omnidrive-final-quality-cpu
{py} -m tools.worldstream_m123.answer_validity_validate --gpu --output /tmp/omnidrive-final-quality-cuda
```

GPU命令需要真实主机CUDA访问；沙箱可能隐藏CUDA/NVML，应在已授权主机执行，不能把沙箱不可见误记为GPU实验失败。生产默认关闭；零DEV新严重错误并非严格风险认证。独立TEST不足与历史NO_GO均保留。
'''
    (root/'REPRODUCE.md').write_text(reproduce)


def verify(root):
    original=read(root/'historical_integrity_manifest.json')
    failures=[p for p,h in original.items() if not Path(p).is_file() or sha256(p)!=h]
    assert not failures, failures
    freezes=read(root/'pre_GPU_freeze.json')
    assert freezes['protocol_sha256']==sha256(root/'protocol_freeze.json')
    for stage,h in freezes['job_hashes'].items():assert sha256(root/(stage+'_jobs.json'))==h
    stages={}
    for stage in ('causal','dev_quality','selected_confirm','scene_start_causal'):
        p=root/stage/'run_status.json'
        if not p.exists():continue
        status=read(p);assert status['status']=='COMPLETED'
        hashes=read(root/stage/'execution_sources.json')
        for name,h in hashes.items():
            if sha256(name)==h:continue
            alternate=root/'capture_v1_sources/final_quality_capture.py'
            assert stage in ('causal','dev_quality') and name=='tools/worldstream_m123/final_quality_capture.py' and sha256(alternate)==h,(stage,name)
        rows=list(stream(root/stage/'requests.jsonl'))
        assert len(rows)==status['full_M123_frame_calls']
        assert all(r['observed_llm_forward_calls']==sum(q['llm_forward_calls'] for q in r['requests']) for r in rows)
        assert all(not q['llm_skipped'] or q['llm_forward_calls']==0 for r in rows for q in r['requests'])
        assert all(r['feature_receipt_version']=='returned_m2_nchw_v2' for r in rows)
        stages[stage]=dict(status='PASS',full_model_calls=len(rows),pinned_source_files=len(hashes))
    archive_path=root/'receipt_archive.json'
    if archive_path.exists():
        import hashlib
        for r in read(archive_path)['files']:
            assert sha256(root/r['archive'])==r['archive_sha256']
            digest=hashlib.sha256()
            with gzip.open(root/r['archive'],'rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
            assert digest.hexdigest()==r['raw_sha256']
    if (root/'pre_confirmation_freeze.json').exists():
        freeze=read(root/'pre_confirmation_freeze.json')
        assert sha256(parameter_freeze_path(root))==freeze['policy_sha256']
        assert sha256(root/'selected_confirm_jobs.json')==freeze['jobs_sha256']
    if (root/'scene_start_protocol_addendum.json').exists():
        addendum=read(root/'scene_start_protocol_addendum.json')
        assert sha256(root/'scene_start_causal_jobs.json')==addendum['jobs_sha256']
        assert sha256(root/'scene_start_gt_annotations.json')==addendum['GT_sha256']
        assert sha256(parameter_freeze_path(root))==addendum['frozen_policy_sha256']
    if (root/'policy_qualification_revision.json').exists():
        revision=read(root/'policy_qualification_revision.json')
        assert sha256(root/'final_research_policy.json')==revision['qualified_final_policy_sha256']
        before=read(parameter_freeze_path(root));after=read(root/'final_research_policy.json')
        for k in ('M1','M2','M3','selected','parameter_overrides','selection_basis','quality_budget','budget_sensitivity'):assert before[k]==after[k]
    assert sha256(root/'current_gt_annotations.json')==sha256(OLD/'current_gt_annotations.json')
    for stage in ('causal','scene_start_causal'):
        causal=list(stream(root/stage/'requests.jsonl')) if (root/stage/'run_status.json').exists() else []
        ref={r['frame_index']:r for r in causal if r['job_spec']['label']=='R0'}
        for r in causal:
            base=ref[r['frame_index']]
            assert r['frame_id']==base['frame_id'] and r['timestamp']==base['timestamp']
            assert r['camera_ids']==base['camera_ids'] and len(r['camera_ids'])==6
            assert r['questions']==base['questions']
            assert r['job_spec']['scene']['tokens']==base['job_spec']['scene']['tokens']
        if stage=='scene_start_causal' and causal:
            assert ref[0]['frame_id']==read(root/'scene_start_protocol_addendum.json')['true_scene_first_sample']
    result=dict(status='PASS',historical_files_unchanged=len(original),stages=stages,
        job_protocol_parameter_freezes_unchanged=True,policy_qualification_metadata_revision_attested=True,GT_cache_unchanged=True,
        causal_inputs_and_full_prefix_identical=True,source_revisions_attested=True,
        no_GT_online=True,strict_risk_or_generalization_certified=False)
    print(json.dumps(result))
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=['prepare','select','prepare_scene_start','map_quality','report','verify'])
    p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args()
    globals()[a.stage](a.root)


if __name__=='__main__':main()
