"""Post-inference provenance/integrity audit. Never imported by M3 decisions."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

from .benchmark.common import write,sha256


def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def match_annotation(center,objects):
    distances=sorted((math.dist(center,g['center_ego']),g['annotation_id'],g) for g in objects if g['visible'])
    if not distances or distances[0][0]>=2. or (len(distances)>1 and distances[1][0]-distances[0][0]<=.5):
        return None
    return distances[0][2]


def audit(root,verify_checkpoint=False):
    assert json.loads((root/'run/run_status.json').read_text())['status']=='COMPLETED'
    freeze=json.loads((root/'POLICY_FREEZE.json').read_text())
    for section in ('files','quality_sources'):
        assert all(sha256(p)==h for p,h in freeze[section].items()),'Frozen policy/quality source changed'
    bound={}
    for stage in ('shadow','pareto','run'):
        binding=json.loads((root/stage/'execution_sources.json').read_text())
        for source,digest in binding['sources'].items():
            p=Path(source);relative=p.relative_to(Path.cwd()) if p.is_absolute() else p
            assert sha256(root/stage/'executed_sources'/relative)==digest
        bound[stage]=dict(status='EXACT_EXECUTED_SOURCES_VERIFIED',sources=len(binding['sources']))
    rows=lines(root/'run/requests.jsonl');objects=json.loads((root/'current_frame_facts.json').read_text())['frames']
    mapped={};class_rows=[];edges=[];previous={}
    for r in rows:
        if (r['condition'],r['workload'],r['question_count'],r['round'])!=('B','addressable',3,0):continue
        scene=r['scene_id'];world=r['committed_world'];now={}
        for a in world['anchors']:
            if not a['observed'] or a['anchor_type']!='object':continue
            gt=match_annotation(a['center_ego'],objects[r['frame_id']])
            now[a['anchor_id']]=gt
            class_rows.append(dict(frame_id=r['frame_id'],anchor_id=a['anchor_id'],predicted_class=a['semantic_class'],
                known=gt is not None,GT_class=gt['semantic_class'] if gt else None,
                correct=a['semantic_class']==gt['semantic_class'] if gt else None))
        if scene in previous:
            old=previous[scene]
            for aid in set(now)&set(old):
                a,b=old[aid],now[aid]
                edges.append(dict(scene_id=scene,current_frame=r['frame_id'],anchor_id=aid,
                    known=a is not None and b is not None,
                    correct=a['instance_id']==b['instance_id'] if a and b else None,
                    previous_instance=a['instance_id'] if a else None,current_instance=b['instance_id'] if b else None))
        previous[scene]=now;mapped[r['frame_id']]=now
    quality=lines(root/'quality_requests.jsonl');by_key={
        (e['workload'],e['question_count'],e['round'],e['frame_id'],e['request_index'],e['condition']):e for e in quality}
    fast_pairs=[]
    for workload in ('natural','fixed','addressable'):
        for count in (1,3):
            es=[e for e in quality if (e['workload'],e['question_count'],e['condition'])==(workload,count,'C') and e['llm_skipped']]
            paired=[]
            for e in es:
                a=by_key[(workload,count,e['round'],e['frame_id'],e['request_index'],'A')]
                b=by_key[(workload,count,e['round'],e['frame_id'],e['request_index'],'B')]
                if e['quality']['known'] and a['quality']['known'] and b['quality']['known']:paired.append((a,b,e))
            avg=lambda k:sum(t[k]['quality']['score'] for t in paired)/len(paired) if paired else None
            fast_pairs.append(dict(workload=workload,question_count=count,actual_fast_hits=len(es),scored_hits=len(paired),
                A_score_same_hit_mask=avg(0),B_score_same_hit_mask=avg(1),C_score=avg(2),
                loss_A_to_C=(avg(0)-avg(2)) if paired else None,loss_B_to_C=(avg(1)-avg(2)) if paired else None,
                scope='Same CURRENT independent labels on the actual C hit mask; no Native-answer truth.'))
    write(root/'fast_hit_paired_quality.json',dict(status='MEASURED_PARTIAL',groups=fast_pairs))
    from worldstream.m3.task_conditioner import question_facts
    history={};coverage=[]
    for row in rows:
        stream=(row['workload'],row['question_count'],row['round'],row['condition'],row['scene_id'])
        records=history.setdefault(stream,{})
        for q in row['requests']:
            profile=question_facts(q['question']);matches=[]
            for old_q,(index,timestamp,old_profile) in records.items():
                if (0<row['frame_index']-index<=8 and 0<row['timestamp']-timestamp<=4.
                        and (old_q==q['question'] or (profile.intent!='unsupported' and profile.signature==old_profile.signature))):
                    matches.append(old_q)
            p=q.get('fast_answer')
            coverage.append(dict(workload=row['workload'],question_count=row['question_count'],condition=row['condition'],
                frame_id=row['frame_id'],question=q['question'],history_available=bool(matches),
                M1_supported=bool(p and p['supported']) if p else None,actual_skip=q['llm_skipped']))
            if q['answer_source']=='current_target_model_generation':
                records[q['question']]=(row['frame_index'],row['timestamp'],profile)
    write(root/'history_and_fact_coverage.json',dict(status='OFFLINE_CAUSAL_HISTORY_REPLAY_AND_MEASURED_SUPPORT',
        groups=[dict(workload=w,question_count=n,condition=c,
            requests=len([e for e in coverage if (e['workload'],e['question_count'],e['condition'])==(w,n,c)]),
            history_available=sum(e['history_available'] for e in coverage if (e['workload'],e['question_count'],e['condition'])==(w,n,c)),
            M1_supported=sum(bool(e['M1_supported']) for e in coverage if (e['workload'],e['question_count'],e['condition'])==(w,n,c)) if c=='C' else None,
            actual_skips=sum(e['actual_skip'] for e in coverage if (e['workload'],e['question_count'],e['condition'])==(w,n,c)))
            for w in ('natural','fixed','addressable') for n in (1,3) for c in ('A','B','C')],
        events=coverage,scope='Only prior actual target generations enter replay. Fact/cache outputs do not refresh source age or add independent model witnesses. History availability is separate from current fact support, correctness and measured E2E gain.'))
    task_groups=[]
    for workload in ('natural','fixed','addressable'):
        for condition in ('A','B','C'):
            for task in ('category','existence','position','motion','unsupported'):
                es=[e for e in quality if (e['workload'],e['condition'],e['task'])==(workload,condition,task)]
                scored=[e for e in es if e['quality']['known']]
                fast=[e for e in es if e['llm_skipped']]
                fast_scored=[e for e in fast if e['quality']['known']]
                task_groups.append(dict(workload=workload,condition=condition,task=task,
                    requests=len(es),scored=len(scored),UNKNOWN=len(es)-len(scored),
                    errors=sum(e['quality']['score']<1 for e in scored),
                    error_rate=sum(1-e['quality']['score'] for e in scored)/len(scored) if scored else None,
                    fast_requests=len(fast),fast_scored=len(fast_scored),
                    fast_errors=sum(e['quality']['score']<1 for e in fast_scored),
                    fast_error_rate=sum(1-e['quality']['score'] for e in fast_scored)/len(fast_scored) if fast_scored else None,
                    unique_frame_questions=len({(e['frame_id'],e['question']) for e in scored}),
                    status='MEASURED_PARTIAL' if es else 'NOT_PRESENT'))
    write(root/'quality_by_task.json',dict(status='MEASURED_PARTIAL',formal_test=task_groups,
        motion_scope='Reliable-motion diagnostics are separate DEV Shadow requests; no unsupported motion response is silently scored correct.'))
    bounds=[]
    for workload in ('natural','fixed','addressable'):
        for count in (1,3):
            es=[e for e in quality if (e['workload'],e['question_count'],e['condition'])==(workload,count,'C')]
            known_loss=0.;unknown_changed=0;unknown_identical=0;known=0
            for e in es:
                b=by_key[(workload,count,e['round'],e['frame_id'],e['request_index'],'B')]
                if e['quality']['known'] and b['quality']['known']:
                    known+=1;known_loss+=b['quality']['score']-e['quality']['score']
                elif e['answer']==b['answer']:unknown_identical+=1
                else:unknown_changed+=1
            bounds.append(dict(workload=workload,question_count=count,requests=len(es),known_pairs=known,
                unknown_changed=unknown_changed,unknown_identical=unknown_identical,
                empirical_known_loss=known_loss/known if known else None,
                worst_case_all_request_loss_lower=(known_loss-unknown_changed)/len(es),
                worst_case_all_request_loss_upper=(known_loss+unknown_changed)/len(es),
                rule='Unknown changed answers have loss in [-1,+1]. Identical answers imply equal score, not correctness. No unknown is scored correct.',
                safety_qualification='INCOMPLETE; bounds do not certify absence or human driving safety'))
    write(root/'quality_budget_bounds.json',dict(status='CONSERVATIVE_DETERMINISTIC_BOUNDS_NOT_CONFIDENCE_INTERVALS',groups=bounds,
        policy_unchanged=True,scope='Supplementary bound for the already defined bounded required-fact metric; no threshold or GO decision is tuned from this bound.'))
    errors=[];severe=[]
    for e in quality:
        if e['condition']!='C':continue
        b=by_key[(e['workload'],e['question_count'],e['round'],e['frame_id'],e['request_index'],'B')]
        if e['quality']['known'] and b['quality']['known'] and e['quality']['severe_error'] and not b['quality']['severe_error']:
            severe.append(dict(frame_id=e['frame_id'],question=e['question'],answer=e['answer'],
                baseline_answer=b['answer'],workload=e['workload'],question_count=e['question_count'],
                answer_source=e['answer_source'],quality=e['quality']))
        if not e['llm_skipped'] or not e['quality']['known'] or e['quality']['score']==1:continue
        p=e['proposal'];fact=p['facts'][0];g=match_annotation(fact['center_ego'],objects[e['frame_id']])
        cause=('M1_class_estimate' if g and g['semantic_class']!=fact['semantic_class'] else
               'question_target_binding_or_geometry' if g else 'UNKNOWN_no_unique_geometric_GT_match')
        errors.append(dict(frame_id=e['frame_id'],question=e['question'],answer=e['answer'],
            round=e['round'],question_count=e['question_count'],source=e['answer_source'],fact=fact,
            expected=e['quality']['reference'],matched_annotation=g,cause=cause,
            historical_answer_source=p.get('source_generation')))
    known_class=[e for e in class_rows if e['known']];known_edges=[e for e in edges if e['known']]
    write(root/'perception_grounding_audit.json',dict(status='OFFLINE_PARTIAL_ANNOTATION_AUDIT',
        frame_coverage=[dict(frame_id=t,physical_annotations=len(gs),visible_sensor_supported=len([g for g in gs if g['visible']])) for t,gs in objects.items()],
        class_scored=len(known_class),class_unknown=len(class_rows)-len(known_class),
        conditional_class_accuracy=sum(e['correct'] for e in known_class)/len(known_class) if known_class else None,
        identity_scored=len(known_edges),identity_unknown=len(edges)-len(known_edges),
        identity_wrong=sum(not e['correct'] for e in known_edges),class_rows=class_rows,identity_edges=edges,
        actual_fast_error_causes=dict(Counter(e['cause'] for e in errors)),fast_errors=errors,
        matching='Class-blind unique nearest visible GT center within 2m, nearest gap >0.5m; unmatched UNKNOWN. Not complete detection/identity/safety certification.'))
    write(root/'severe_error_audit.json',dict(status='MEASURED_PARTIAL',new_severe_vs_B=severe,
        unique_frame_questions=len({(e['frame_id'],e['question']) for e in severe}),
        criterion='C opposite-direction/existence contradiction and B not severe, including B incomplete answers.'))
    checkpoint=json.loads((root/'checkpoint_verification.json').read_text()) if (root/'checkpoint_verification.json').exists() else dict(status='NOT_REHASHED',scope='strict checkpoint key/shape load validated in each stage')
    if verify_checkpoint:
        digest=sha256('ckpts/iter_10548.pth')
        expected=json.loads((root/'run/run_status.json').read_text())['checkpoint_sha256']
        assert digest==expected,'Original checkpoint digest differs'
        checkpoint=dict(status='ORIGINAL_CHECKPOINT_SHA256_VERIFIED',sha256=digest)
        write(root/'checkpoint_verification.json',checkpoint)
    write(root/'integrity_audit.json',dict(status='PASS',policy_and_quality_freeze_verified=True,
        measured_sources=bound,checkpoint=checkpoint,
        actual_fast_forward_calls=sum(e['llm_forward_calls'] for e in quality if e['llm_skipped']),
        GT_online=False,future_information_online=False,human_review='NOT_RUN',
        safety_scope='See severe_error_audit.json; annotations and rule predicates do not replace image-based human safety review.'))
    print('AUDIT PASS',len(errors),'fast errors;',len(severe),'new severe paired findings',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=Path('outputs/m3_selective_answer_reuse_run01'))
    p.add_argument('--verify-checkpoint',action='store_true');a=p.parse_args();audit(a.root,a.verify_checkpoint)
