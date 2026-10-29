"""Frozen DEV calibration/grading. Final presentation is a separate module."""
from collections import Counter
import json
import math
from pathlib import Path

from .benchmark.common import write,sha256
from .continuous_stream import lines,source_hashes,SCOPE
from .target_complete_report import world,context,grade_current
from .fact_bypass_identity import question_target,evaluate,match_instances,request_identity
from .answer_validity_quality import load_current_annotations
from .world_guided_decode_v2_report import scene_ci


def mean(xs):return sum(xs)/len(xs) if xs else None


def labels(root,rows):
    path=root/'current_gt_annotations.json'
    if path.exists():objects=json.loads(path.read_text())
    else:objects={}
    tokens={r['frame_id'] for r in rows}-set(objects)
    if tokens:
        objects.update(load_current_annotations('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl',tokens,'data/nuscenes'))
        write(path,objects)
    return objects


def grade(q,text,token,objects):
    from worldstream.m3.task_conditioner import question_facts
    if question_facts(q).intent=='unsupported':return grade_current(q,text,token,objects,'val')
    return evaluate(q,text,objects[token],question_target(q,objects[token]),token)


def events(root,rows):
    from worldstream.m3.task_conditioner import question_facts
    objects=labels(root,rows);es=[]
    for r in rows:
        w=r.get('committed_world');matched=match_instances(w['anchors'],objects[r['frame_id']],r['frame_id'],r['timestamp']) if w else {}
        for qi,(req,answer) in enumerate(zip(r['requests'],r['answers'])):
            q=req['question'];target=question_target(q,objects[r['frame_id']]);identity=request_identity(req.get('fast_answer'),target,matched)
            es.append(dict(stage=r['stage'],condition=r['condition'],workload=r['workload'],frequency=r['frequency'],
                scene_id=r['scene_id'],log_id=r['log_id'],frame_id=r['frame_id'],frame_index=r['frame_index'],request_index=qi,
                question=q,task=question_facts(q).intent,answer=answer['A'][0],
                quality=grade(q,answer['A'][0],r['frame_id'],objects),**identity,target_reference=target,
                llm_skipped=req['llm_skipped'],llm_forward_calls=req['llm_forward_calls'],proposal=req.get('fast_answer'),
                request_wall_ms=req['end_to_end_ms']))
    return es


def paired_quality(es,a,b):
    key=lambda e:(e['workload'],e['frequency'],e['frame_id'],e['request_index'])
    bases={key(e):e for e in es if e['condition']==a};pairs=[];all_pairs=0
    for e in es:
        if e['condition']!=b:continue
        ref=bases.get(key(e))
        if not ref:continue
        assert ref['question']==e['question'];all_pairs+=1
        if ref['quality']['known'] and e['quality']['known']:
            pairs.append(dict(scene_id=e['log_id'],frame_id=e['frame_id'],question=e['question'],
                loss=ref['quality']['score']-e['quality']['score'],new_severe=int(bool(e['quality']['severe_error']) and not ref['quality']['severe_error']),
                severe_delta=int(bool(e['quality']['severe_error']))-int(bool(ref['quality']['severe_error'])),
                B=ref['quality']['score'],C=e['quality']['score'],fast=e['llm_skipped']))
    unknown=all_pairs-len(pairs);loss=sum(p['loss'] for p in pairs)
    return dict(comparison=a+'->'+b,requests=all_pairs,scored=len(pairs),UNKNOWN=unknown,
        B_score=mean([p['B'] for p in pairs]),C_score=mean([p['C'] for p in pairs]),loss=mean([p['loss'] for p in pairs]),
        loss_log_CI95=scene_ci(pairs,'loss') if pairs else None,
        conservative_all_request_loss_bounds=[(loss-unknown)/all_pairs,(loss+unknown)/all_pairs] if all_pairs else None,
        new_severe=sum(p['new_severe'] for p in pairs),net_severe=sum(p['severe_delta'] for p in pairs),
        new_severe_log_CI95=scene_ci(pairs,'new_severe') if pairs else None,
        independent_logs=len({p['scene_id'] for p in pairs}),pairs=pairs)


def calibrate_dev(root):
    from worldstream.m3.answer_memory import AnswerMemory
    from worldstream.m3.answer_validity import propose_fact_answer
    rows=[r for r in lines(root/'dev/requests.jsonl') if r['condition']=='B-shadow'];objects=labels(root,rows)
    es=events(root,rows);base=json.loads((root/'shadow_policy.json').read_text())
    wrong=[];position_labeled=0
    for e in es:
        p=e['proposal']
        if e['task']=='position' and p and p['admitted']:
            g=grade(e['question'],p['answer_text'],e['frame_id'],objects)
            if g['known']:
                position_labeled+=1
                if g['score']<1:wrong.append(p['support_score'])
    floor=math.nextafter(max(wrong),1.) if wrong else base['min_confidence']
    candidates=[dict(base,policy_id='C-basic',position_mode='basic'),
        dict(base,policy_id='C-restricted',position_mode='restricted',position_min_confidence=floor),
        dict(base,policy_id='C-fallback',position_mode='fallback')]
    replays=[];decisions=[]
    for p in candidates:
        ds=[]
        for r in rows:
            w=world(r['committed_world'])
            for req in r['requests']:
                proposal=propose_fact_answer(req['question'],w,context(w,r['frame_index'],r['camera_ids']),AnswerMemory(),('extended_DEV',),p,target_binding_enabled=True)
                ds.append(proposal['admitted']);replays.append(dict(policy_id=p['policy_id'],frame_id=r['frame_id'],question=req['question'],proposal=proposal))
        decisions.append(ds)
    differing=sum(a!=b for a,b in zip(decisions[0],decisions[1]));fallback_diff=sum(a!=b for a,b in zip(decisions[0],decisions[2]))
    calibration=dict(scope=SCOPE,source_sha256=sha256(root/'dev/requests.jsonl'),questions=len(es),
        admitted_Position_labels=position_labeled,incorrect_admitted_Position=len(wrong),restricted_floor=floor,
        restricted_changed_decisions=differing,fallback_changed_decisions=fallback_diff,
        no_new_algorithm=True,restricted_fit='Maximum wrong admitted Position support + next float, extended DEV only; .45 unchanged if none')
    write(root/'dev_calibration.json',calibration)
    with (root/'dev_policy_replay.jsonl').open('w') as f:
        for r in replays:f.write(json.dumps(r)+'\n')
    active=[]
    for i,p in enumerate(candidates):
        p.update(calibration_sha256=sha256(root/'dev_calibration.json'),expected_normal_m3_ms=mean([e['request_wall_ms'] for e in es]),
            budget=.02,min_net_gain_ms=0.,temporal_risk_enabled=False,target_completeness_enabled=False)
        path=root/(p['policy_id']+'.json');write(path,p)
        if i==0 or (i==1 and differing) or (i==2 and fallback_diff):active.append(path)
    write(root/'dev_active_policies.json',dict(active=list(map(str,active)),skipped_restricted_as_identical=differing==0,
        skipped_fallback_as_identical=fallback_diff==0))
    return active


def freeze_dev(root):
    rows=lines(root/'dev/requests.jsonl');es=events(root,rows);active=json.loads((root/'dev_active_policies.json').read_text())['active']
    points=[]
    for path in active:
        pid=json.loads(Path(path).read_text())['policy_id'];part=[r for r in rows if r['condition']==pid]
        q=paired_quality(es,'B-shadow',pid)
        points.append(dict(policy_id=pid,policy_file=path,mean_e2e_ms=mean([r['e2e_wall_ms'] for r in part]),
            actual_skips=sum(e['llm_skipped'] for e in es if e['condition']==pid),quality=q))
    usable=[p for p in points if p['quality']['loss'] is not None and p['quality']['loss']<=.02]
    assert usable,'No retrospective DEV policy satisfies known paired2pp; cannot silently freeze one'
    best=min(usable,key=lambda p:p['mean_e2e_ms']);policy=json.loads(Path(best['policy_file']).read_text())
    policy.update(status='FROZEN_RETROSPECTIVE_POLICY',selection_scope='Retrospective DEV only, not independent TEST qualification',
        development_metrics=best,production_default=False)
    write(root/'selected_retrospective_policy.json',policy)
    companion=next((p['policy_file'] for p in points if p['policy_id']==('C-basic' if best['policy_id']=='C-fallback' else 'C-fallback')),None)
    if not companion:companion=next((p['policy_file'] for p in points if p['policy_id']!=best['policy_id']),None)
    write(root/'position_policy_comparison.json',dict(scope=SCOPE,stage='extended_DEV_REAL_GPU',points=points,
        cpu_equivalence=json.loads((root/'dev_calibration.json').read_text()),
        budgets=[dict(budget=b,best=min([p for p in points if p['quality']['loss'] is not None and p['quality']['loss']<=b],key=lambda p:p['mean_e2e_ms'])['policy_id']
            if any(p['quality']['loss'] is not None and p['quality']['loss']<=b for p in points) else None) for b in (.01,.02,.03)],
        no_strict_rare_error_certification=True))
    write(root/'policy_sources.json',source_hashes())
    write(root/'POLICY_FREEZE.json',dict(scope=SCOPE,before_extended_evaluation=True,selected_policy_sha256=sha256(root/'selected_retrospective_policy.json'),
        selected_policy=best['policy_id'],companion_policy=companion,protocol_sha256=sha256(root/'protocol_freeze.json'),
        grading_sha256=sha256(__file__),independent_test_qualified=False))
    print('FROZEN',best['policy_id'],best['mean_e2e_ms'],flush=True)


def report(root):
    from .continuous_stream_analysis import report as assemble
    assemble(root)


def verify(root):
    from .continuous_stream_analysis import verify as check
    check(root)
