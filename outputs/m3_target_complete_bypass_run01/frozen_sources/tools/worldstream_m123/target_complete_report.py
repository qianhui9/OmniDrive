"""Independent offline grading, finite DEV selection and v4 research reports."""
from collections import Counter,defaultdict
from dataclasses import replace
from functools import lru_cache
import json
import math
from pathlib import Path
import shutil
from .benchmark.common import write,sha256
from .selective_answer_reuse_report import lines,mean,grade,label,wilson
from .temporal_selective_report import performance,quality
from .world_guided_decode_v2_report import scene_ci


def world(data):
    from worldstream.m3.world_state import CommittedWorldState,CommittedAnchor,CurrentCompetitionEvidence
    return CommittedWorldState(**dict(data,anchors=tuple(CommittedAnchor(**a) for a in data['anchors']),
        competition_evidence=tuple(CurrentCompetitionEvidence(**a) for a in data.get('competition_evidence',()))))


def labels(root,rows):
    from .answer_validity_quality import load_current_annotations
    from tools.worldstream_m1.io_utils import iter_json_array
    plan=json.loads((root/'experiment_plan.json').read_text());tokens={r['frame_id'] for r in rows}
    objects=load_current_annotations(plan['dataset_infos'],tokens,'data/nuscenes')
    table=Path('data/nuscenes/v1.0-trainval');attrs={a['token']:a['name'] for a in iter_json_array(table/'attribute.json')}
    by={g['annotation_id']:g for gs in objects.values() for g in gs}
    for a in iter_json_array(table/'sample_annotation.json'):
        if a['token'] in by:by[a['token']]['attributes']=[attrs[t] for t in a.get('attribute_tokens',[])]
    return objects


@lru_cache(maxsize=512)
def qa_references(token,split):
    refs={}
    for folder in ('conv','vqa'):
        p=Path('data/nuscenes')/folder/split/(token+'.json')
        if p.exists():refs.update({q['question']:q.get('answer',q.get('answers','')) for q in json.loads(p.read_text())})
    return refs


def grade_current(q,answer,token,objects,split='train'):
    from worldstream.m3.task_conditioner import question_facts
    if question_facts(q).intent!='unsupported':return grade(q,answer,token,objects)
    from .world_guided_decode_v3_report import facts
    from .world_guided_decode_v2_quality import compare_facts
    ref=qa_references(token,split).get(q)
    expected=facts(q,ref) if isinstance(ref,str) and ref.strip() else []
    checks=compare_facts(expected,facts(q,answer)) if expected else []
    return dict(known=bool(checks),score=mean([float(c['correct']) for c in checks]),severe_error=False,
        metric='limited_exact_current_QA_fact_recall',reference=dict(source='exact_current_frame_original_QA',
        reference=ref,checks=checks,reason=None if checks else 'no_reliable_reference_facts'))


def events(rows,objects):
    from worldstream.m3.task_conditioner import question_facts
    es=[]
    for row in rows:
        for qi,(request,answer) in enumerate(zip(row['requests'],row['answers'])):
            q=request['question'];text=answer['A'][0];p=request.get('fast_answer')
            es.append(dict(stage=row['stage'],condition=row['condition'],workload=row['workload'],question_count=row['question_count'],
                round=row['round'],scene_id=row['scene_id'],frame_id=row['frame_id'],request_index=qi,
                question=q,answer=text,quality=grade_current(q,text,row['frame_id'],objects[row['frame_id']]),
                llm_skipped=request['llm_skipped'],answer_source=request['answer_source'],llm_forward_calls=request['llm_forward_calls'],
                proposal=p,task=question_facts(q).intent,request_wall_ms=request['end_to_end_ms']))
    return es


def metrics(es,comparisons):
    out=quality(es,comparisons)
    emap={(e['workload'],e['question_count'],e['round'],e['frame_id'],e['request_index'],e['condition']):e for e in es}
    for group in out['groups']:
        part=[e for e in es if (e['workload'],e['question_count'],e['condition'])==(group['workload'],group['question_count'],group['condition'])
            and (group['execution']=='all' or e['llm_skipped']==(group['execution']=='fast_hit'))]
        group['independent_scenes']=len({e['scene_id'] for e in part})
        group['independent_scored_scenes']=len({e['scene_id'] for e in part if e['quality']['known']})
    for loss in out['paired_losses']:
        a,b=loss['comparison'].split('->');paired=[]
        for e in es:
            if (e['workload'],e['question_count'],e['condition'])!=(loss['workload'],loss['question_count'],b):continue
            ref=emap.get((e['workload'],e['question_count'],e['round'],e['frame_id'],e['request_index'],a))
            if ref and e['quality']['known'] and ref['quality']['known']:
                paired.append(dict(scene_id=e['scene_id'],new_severe=float(e['quality']['severe_error'] and not ref['quality']['severe_error']),
                    severe_delta=float(e['quality']['severe_error'])-float(ref['quality']['severe_error'])))
        loss['new_severe_rate']=mean([p['new_severe'] for p in paired])
        loss['new_severe_scene_CI95']=scene_ci(paired,'new_severe')
        loss['severe_rate_delta']=mean([p['severe_delta'] for p in paired])
        loss['severe_delta_scene_CI95']=scene_ci(paired,'severe_delta')
        loss['independent_scored_scenes']=len({p['scene_id'] for p in paired})
    out['scope']+=' Base-model training partition exposure possible; independent experimental logs are not proof of unseen-model generalization.'
    return out


def context(w,fi,camera_ids):
    return dict(scene_id=w.scene_id,frame_id=w.frame_id,timestamp=w.timestamp,state_epoch=w.epoch,
        state_execution_signature=w.execution_signature,current_state_version=w.version,frame_index=fi,camera_ids=camera_ids)


def calibrate(root):
    from .target_complete_bypass import CONFIGS
    from worldstream.m3.answer_validity import propose_fact_answer
    from worldstream.m3.answer_memory import AnswerMemory
    rows=lines(root/'shadow/requests.jsonl');objects=labels(root,rows);es=events(rows,objects);counter=[];previous=None
    by={(e['frame_id'],e['request_index']):e for e in es}
    for row in rows:
        w=world(row['committed_world'])
        if previous is not None and previous.scene_id!=w.scene_id:previous=None
        for qi,request in enumerate(row['requests']):
            e=by[(row['frame_id'],qi)]
            for i,c in enumerate(CONFIGS):
                p=propose_fact_answer(request['question'],w,context(w,row['frame_index'],row['camera_ids']),AnswerMemory(),('offline-dev',),
                    dict(expected_normal_m3_ms=1000.,**c),previous,True,False,True)
                counter.append(dict(index=i,scene_id=w.scene_id,frame_id=w.frame_id,question=request['question'],proposal=p,
                    baseline_quality=e['quality'],candidate_quality=grade_current(request['question'],p['answer_text'],w.frame_id,objects[w.frame_id])
                    if p['supported'] else None,estimated_saved_m3_ms=request['end_to_end_ms'] if p['admitted'] else 0.))
        previous=w
    summaries=[]
    for i,c in enumerate(CONFIGS):
        xs=[e for e in counter if e['index']==i];hits=[e for e in xs if e['proposal']['admitted']]
        known=[e for e in hits if e['candidate_quality']['known'] and e['baseline_quality']['known']]
        summaries.append(dict(index=i,**c,requests=len(xs),estimated_hits=len(hits),scored_hits=len(known),UNKNOWN=len(hits)-len(known),
            candidate_score=mean([e['candidate_quality']['score'] for e in known]),
            new_severe=sum(e['candidate_quality']['severe_error'] and not e['baseline_quality']['severe_error'] for e in known),
            unique_scored_frames=len({e['frame_id'] for e in known}),independent_scenes=len({e['scene_id'] for e in xs}),
            estimated_saved_m3_ms=sum(e['estimated_saved_m3_ms'] for e in xs)))
    normal=mean([e['request_wall_ms'] for e in es])
    write(root/'shadow_calibration.json',dict(stage='DEV_REAL_GPU_SHADOW',configs=summaries,normal_m3_ms=normal,
        actual_skips=0,scope='Offline same-current-output proposals are estimates, not measured skips or E2E. Only Pareto stage measures full online cost.'))
    for i,c in enumerate(CONFIGS):
        write(root/f'candidate_{i}.json',dict(status='FROZEN_RESEARCH_POLICY',policy_id=f'target_dev_{i}',budget=.02,
            expected_normal_m3_ms=normal,min_net_gain_ms=0.,**c,temporal_risk_enabled=False,
            calibration_sha256=sha256(root/'shadow_calibration.json'),automatic_deployment=False))
    with (root/'shadow_quality_requests.jsonl').open('w') as f:
        for e in es:f.write(json.dumps(e)+'\n')
    with (root/'shadow_counterfactuals.jsonl').open('w') as f:
        for e in counter:f.write(json.dumps(e)+'\n')
    from worldstream.m3.task_conditioner import question_facts
    calibration=[]
    for e in counter:
        p=e['proposal']
        if e['index']!=1 or not p.get('facts'):continue
        fact=p['facts'][0];physical=match_physical(fact['center_ego'],objects[e['frame_id']])
        calibration.append(dict(scene_id=e['scene_id'],frame_id=e['frame_id'],question=e['question'],
            task=question_facts(e['question']).intent,observation_kind=fact['observation_kind'],confidence=fact['confidence'],
            camera_count=len(fact['cameras']),target_risk=p['target_completeness']['risk'],
            competitor_count=len(p['target_completeness']['competitors']),candidate_quality=e['candidate_quality'],
            physical_class_correct=physical['semantic_class']==fact['semantic_class'] if physical else None))
    bins=[];groups=defaultdict(list)
    for e in calibration:
        conf=next((f'{lo}-{hi}' for lo,hi in ((.35,.55),(.55,.75),(.75,1.01)) if lo<=e['confidence']<hi),'other')
        risk=next((f'{lo}-{hi}' for lo,hi in ((0.,.35),(.35,.5),(.5,.65),(.65,1.01)) if lo<=e['target_risk']<hi),'other')
        groups[(e['task'],e['observation_kind'],conf,risk,bool(e['competitor_count']),e['camera_count'])].append(e)
    for key,part in groups.items():
        known=[e for e in part if e['candidate_quality']['known']]
        bins.append(dict(task=key[0],observation_kind=key[1],confidence_bin=key[2],target_risk_bin=key[3],
            has_competitor=key[4],camera_count=key[5],requests=len(part),known=len(known),UNKNOWN=len(part)-len(known),
            fact_score=mean([e['candidate_quality']['score'] for e in known]),
            physical_class_score=mean([float(e['physical_class_correct']) for e in part if e['physical_class_correct'] is not None]),
            independent_frames=len({e['frame_id'] for e in part}),independent_scenes=len({e['scene_id'] for e in part})))
    write(root/'dev_evidence_calibration.json',dict(rows=calibration,bins=bins,
        scope='Same real DEV Shadow snapshot; one fixed feature extractor per question, no policy-repetition inflation. Object physical match and question-fact score differ. Bins are descriptive small-sample observations, not calibrated probabilities.'))
    (root/'shadow_evaluation_report.md').write_text('# DEV Shadow\n\n'+f'{len(rows)} real full M123 calls, {len(es)} questions, zero actual skips. '+
        'All questions still execute the original K8 target model. Counterfactual policy candidates and estimated saving remain offline estimates.\n\n'+
        '\n'.join(json.dumps(c) for c in summaries)+'\n')
    print('CALIBRATION',json.dumps(summaries),flush=True)


def freeze(root):
    from .target_complete_bypass import CONFIGS
    rows=lines(root/'pareto/requests.jsonl');objects=labels(root,rows);es=events(rows,objects)
    points=[]
    for i,c in enumerate(CONFIGS):
        subset=[e for e in es if e['condition'] in ('B',f'C{i}')]
        q=metrics(subset,[('B',f'C{i}')]);loss=q['paired_losses'][0]
        hits=[e for e in subset if e['condition']==f'C{i}' and e['llm_skipped']]
        points.append(dict(index=i,**c,mean_e2e_ms=mean([r['e2e_wall_ms'] for r in rows if r['condition']==f'C{i}']),
            hits=len(hits),unique_scored_fast_frames=len({e['frame_id'] for e in hits if e['quality']['known']}),
            quality=q,**loss))
    policies=[]
    for budget in (.01,.02,.03):
        usable=[p for p in points if p['loss']<=budget and p['scene_CI95'][1]<=budget
            and p['conservative_all_request_loss_bounds'][1]<=budget and p['new_severe_rate']<=.02
            and p['new_severe_scene_CI95'][1]<=.02 and p['unique_scored_fast_frames']>=3]
        if not usable:policies.append(dict(budget=budget,status='NO_QUALIFIED_DEV_POLICY'));continue
        best=min(usable,key=lambda p:p['mean_e2e_ms'])
        policy=json.loads((root/f"candidate_{best['index']}.json").read_text());policy.update(budget=budget,
            online_dev_metrics={k:v for k,v in best.items() if k!='quality'})
        write(root/f'policy_{round(budget*100)}pp.json',policy);policies.append(dict(budget=budget,index=best['index']))
    write(root/'quality_latency_pareto.json',dict(stage='DEV_REAL_FULL_M123_GPU',points=points,policies=policies,
        preset_candidates=4,no_test_tuning=True))
    selected=root/'policy_2pp.json'
    if not selected.exists():
        write(root/'go_no_go.json',dict(status='INCOMPLETE',reason='No DEV policy meets predeclared quality/severe gates; TEST not run',deployment_enabled=False))
        raise RuntimeError('No qualified2pp DEV policy; do not run TEST')
    shutil.copyfile(selected,root/'selected_policy.json')
    best=json.loads(selected.read_text());base=json.loads((root/'candidate_0.json').read_text())
    base.update(expected_normal_m3_ms=best['expected_normal_m3_ms'])
    write(root/'basic_ablation_policy.json',base)
    targets=[p for p in points if p['target_completeness_enabled'] and p['loss']<=.02
        and p['new_severe_rate']<=.02 and p['conservative_all_request_loss_bounds'][1]<=.02]
    target=min(targets,key=lambda p:p['mean_e2e_ms'])['index'] if targets else 1
    tp=json.loads((root/f'candidate_{target}.json').read_text());write(root/'target_ablation_policy.json',tp)
    # When the selected method is a target policy, its ablation changes only
    # the completeness decision; the shared .45/.75 base gate stays identical.
    if best['target_completeness_enabled']:write(root/'target_ablation_policy.json',best)
    sources=[*map(str,__import__('tools.worldstream_m123.selective_answer_reuse',fromlist=['source_files']).source_files()),
        'tools/worldstream_m123/target_complete_bypass.py','tools/worldstream_m123/target_complete_report.py',
        'tools/worldstream_m123/selective_answer_reuse_report.py','tools/worldstream_m123/answer_validity_quality.py',
        'tools/worldstream_m123/temporal_selective_report.py','tools/worldstream_m123/world_guided_decode_v2_quality.py',
        'tools/worldstream_m123/world_guided_decode_v3_report.py','worldstream/m1/omnidrive_adapter.py','worldstream/m1/state_manager.py']
    sources=list(dict.fromkeys(sources))
    for p in sources:
        path=Path(p);rel=path.relative_to(Path.cwd()) if path.is_absolute() else path
        dst=root/'frozen_sources'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
    write(root/'POLICY_FREEZE.json',dict(before_test=True,selected_policy_sha256=sha256(root/'selected_policy.json'),
        protocol_sha256=sha256(root/'protocol_freeze.json'),sources={p:sha256(p) for p in sources},
        ablation_policies={n:sha256(root/n) for n in ('basic_ablation_policy.json','target_ablation_policy.json')}))
    print('FROZEN',json.dumps(dict(policy=best['policy_id'],target_enabled=best['target_completeness_enabled'],
        metrics=best['online_dev_metrics'])),flush=True)


def report(root):
    rows=lines(root/'run/requests.jsonl');abrows=lines(root/'ablation/requests.jsonl')
    objects=labels(root,rows);es=events(rows,objects);ae=events(abrows,objects)
    write(root/'current_frame_facts.json',dict(source='Independent current annotations, offline only',frames=objects))
    with (root/'quality_requests.jsonl').open('w') as f:
        for e in es+ae:f.write(json.dumps(e)+'\n')
    perf=performance(rows,[('A','B'),('B','C'),('A','C')]);q=metrics(es,[('A','B'),('B','C'),('A','C')])
    for m in perf['module_breakdown']:
        part=[r for r in rows if (r['workload'],r['question_count'],r['condition'])==(m['workload'],m['question_count'],m['condition'])]
        requests=[x for r in part for x in r['requests']]
        for key in ('current_fact_extraction_wall_ms','target_completeness_wall_ms'):
            m[key+'_per_question']=sum((x.get('fast_answer') or {}).get(key,0.) for x in requests)/len(requests)
        for name,key in [('answer_memory_selection_wall_ms','draft_selection_wall_ms'),('answer_memory_management_wall_ms','draft_management_wall_ms')]:
            m[name]=mean([x.get(key,0.) for x in requests])
        m['rng_management_wall_ms']=mean([(x.get('question_rng') or {}).get('management_wall_ms',0.) for x in requests]) if m['condition']!='A' else None
    write(root/'abc_comparison.json',perf);write(root/'task_quality.json',q)
    policy=json.loads((root/'selected_policy.json').read_text());selected='C-target' if policy['target_completeness_enabled'] else 'C-basic'
    crows=[dict(r,condition=selected) for r in rows if r['condition']=='C' and r['workload'] in ('addressable','mixed')]
    ce=[dict(e,condition=selected) for e in es if e['condition']=='C' and e['workload'] in ('addressable','mixed')]
    ap=performance(crows+abrows,[('C-basic','C-target')]);aq=metrics(ce+ae,[('C-basic','C-target')])
    def evidence(row):
        w=row['committed_world'];keys=('semantic_class','center_ego','velocity_ego','score','cameras','source_frame','source_timestamp','observed','observation_kind')
        return dict(pose=w['ego_to_world'],timestamp=w['timestamp'],anchors=sorted(
            ({k:a[k] for k in keys} for a in w['anchors'] if a['anchor_type']=='object'),key=lambda a:json.dumps(a,sort_keys=True)))
    counterparts={(r['workload'],r['question_count'],r['round'],r['frame_id']):r for r in crows}
    equality=[dict(workload=r['workload'],question_count=r['question_count'],frame_id=r['frame_id'],
        equal=evidence(r)==evidence(counterparts[(r['workload'],r['question_count'],r['round'],r['frame_id'])])) for r in abrows]
    amap={(e['workload'],e['question_count'],e['round'],e['frame_id'],e['request_index'],e['condition']):e for e in ce+ae}
    veto=[]
    for e in ce+ae:
        if e['condition']!='C-target':continue
        b=amap[(e['workload'],e['question_count'],e['round'],e['frame_id'],e['request_index'],'C-basic')]
        if not b['llm_skipped'] or e['llm_skipped']:continue
        bg,tg=b['quality'],e['quality']
        category='beneficial_veto' if bg['known'] and tg['known'] and bg['score']<tg['score'] else 'unnecessary_veto' if bg['known'] and bg['score']==1 else 'unknown_veto'
        veto.append(dict(scene_id=e['scene_id'],frame_id=e['frame_id'],question=e['question'],workload=e['workload'],
            question_count=e['question_count'],veto_class=category,basic_quality=bg,target_quality=tg,
            avoided_severe=bool(bg['known'] and tg['known'] and bg['severe_error'] and not tg['severe_error']),
            target_proposal=e['proposal'],basic_answer=b['answer'],target_answer=e['answer']))
    independent=any((l['loss']<0 and l['scene_CI95'][1]<=0) or
        (l['severe_rate_delta']<0 and l['severe_delta_scene_CI95'][1]<=0) for l in aq['paired_losses'])
    ablation=dict(status='REAL_FULL_M123_ABLATION',performance=ap,quality=aq,current_evidence_equality=equality,
        current_evidence_equal=all(e['equal'] for e in equality),vetoes=veto,
        veto_counts=dict(Counter(e['veto_class'] for e in veto)),independent_quality_or_severe_value=independent,
        scope='Same current M1 committed detections/poses, images, weights, RNG, .45/.75 base gate. C-target alone pays for current decoded competition readout/projection/analysis; C-basic retains v3 path. Companion timing is a separate phase.')
    write(root/'current_vs_target_complete_ablation.json',ablation)
    new=[e for e in q['errors'] if e['comparison']=='B->C' and e['new_severe_error']]
    fast=[e for e in es if e['llm_skipped']]
    write(root/'severe_error_audit.json',dict(new_severe_vs_B=new,paths=dict(Counter(e['path'] for e in new)),
        unique_new_severe=len({(e['frame_id'],e['question']) for e in new}),all_errors=q['errors'],
        all_severe_outputs=[e for e in es if e['quality']['known'] and e['quality']['severe_error']],
        historical_v3='NO_GO_UNCHANGED',research_safety_distinct=True))
    tasks=[];scenes=[];source=defaultdict(list);risk_events=[];nearest=[]
    from .selective_answer_reuse_audit import match_annotation
    for e in es+ae:
        p=e.get('proposal') or {}
        if p.get('target_completeness'):
            risk_events.append(dict(stage=e['stage'],condition=e['condition'],scene_id=e['scene_id'],frame_id=e['frame_id'],question=e['question'],
                task=e['task'],admitted=e['llm_skipped'],final_quality=e['quality'],
                quality=grade_current(e['question'],p['answer_text'],e['frame_id'],objects[e['frame_id']])
                    if p['supported'] else dict(known=False,score=None),**p['target_completeness']))
        if e['llm_skipped']:
            fact=p['facts'][0];source[(e['stage'],e['condition'],e['task'],fact['observation_kind'])].append(e)
            ref=e['quality']['reference'];physical=match_physical(fact['center_ego'],objects[e['frame_id']])
            visible=match_annotation(fact['center_ego'],objects[e['frame_id']])
            nearest.append(dict(stage=e['stage'],condition=e['condition'],scene_id=e['scene_id'],frame_id=e['frame_id'],question=e['question'],task=e['task'],
                actual_fact_quality=e['quality'],selected_fact=fact,physical_match=physical,
                physical_class_correct=physical['semantic_class']==fact['semantic_class'] if physical else None,
                target_identity_correct=(visible['annotation_id']==ref['selected_annotation']) if visible and ref.get('known') and ref.get('selected_annotation') else None))
    for c in ('A','B','C'):
        for task in ('category','existence','position','motion','unsupported'):
            part=[e for e in es if e['condition']==c and e['task']==task];tasks.append(summary(part,condition=c,task=task))
    for w,n,c,sid in sorted({(e['workload'],e['question_count'],e['condition'],e['scene_id']) for e in es}):
        part=[e for e in es if (e['workload'],e['question_count'],e['condition'],e['scene_id'])==(w,n,c,sid)]
        scenes.append(summary(part,workload=w,question_count=n,condition=c,scene_id=sid))
    write(root/'quality_by_task.json',tasks);write(root/'quality_by_scene.json',dict(groups=scenes,
        scope='Scene-CI bootstrap resamples independent logs with equal scene weights. Overall scores weight known requests.'))
    write(root/'fast_answer_quality.json',dict(actual_skips=len(fast),zero_llm_forward=all(e['llm_forward_calls']==0 for e in fast),
        groups=[g for g in q['groups'] if g['execution']=='fast_hit'],nearest_identity=nearest,
        no_identity_match='UNKNOWN; never assume unmatched detection is false or correct',
        quality_scope='Current positive visible annotation facts; no complete scene, absence, appearance or safety certificate'))
    write(root/'observation_source_quality.json',dict(groups=[summary(v,stage=k[0],condition=k[1],task=k[2],observation_kind=k[3]) for k,v in source.items()],
        physical_object_and_question_target_quality_are_separate=True))
    bins=[]
    for lo,hi in ((0.,.35),(.35,.5),(.5,.65),(.65,1.01)):
        part=[e for e in risk_events if lo<=e['risk']<hi];known=[e for e in part if e['quality']['known']]
        bins.append(dict(interval=[lo,hi],requests=len(part),known=len(known),UNKNOWN=len(part)-len(known),
            empirical_error_rate=mean([1.-e['quality']['score'] for e in known]),
            independent_frames=len({e['frame_id'] for e in part}),independent_scenes=len({e['scene_id'] for e in part})))
    write(root/'question_conditioned_risk.json',dict(score_kind='RELATIVE_SUPPORT_RISK_NOT_PROBABILITY',events=risk_events,bins=bins))
    write(root/'low_confidence_competitor_analysis.json',dict(events=[e for e in risk_events if e['competitors']],
        actual_vetoes=veto,unique_veto_cases=len({(e['frame_id'],e['question']) for e in veto}),
        candidates_are_positive_witnesses=False,source='Already decoded current detector outputs, no extra network'))
    def s(w,n=3):return next(e for e in perf['summaries'] if (e['workload'],e['question_count'],e['comparison'])==(w,n,'B->C'))
    addr=s('addressable');loss=next(e for e in q['paired_losses'] if (e['workload'],e['question_count'],e['comparison'])==('addressable',3,'B->C'))
    hit=next(g for g in q['groups'] if (g['workload'],g['question_count'],g['condition'],g['execution'])==('addressable',3,'C','fast_hit'))
    gain=addr['e2e_speedup']>1 and addr['delta_scene_CI95_ms'][1]<0 and all(x>1 for x in addr['scene_speedups'].values())
    point_quality=loss['loss']<=policy['budget'];quality_ok=point_quality and loss['scene_CI95'][1]<=policy['budget'] and loss['conservative_all_request_loss_bounds'][1]<=policy['budget']
    severe_ok=loss['new_severe_rate']<=.02 and loss['new_severe_scene_CI95'][1]<=.02
    status='PARTIAL_GO' if gain and quality_ok and severe_ok else 'NO_GO'
    if not ablation['current_evidence_equal'] or hit['independent_frame_questions']<3:status='INCOMPLETE'
    if gain and point_quality and severe_ok and not quality_ok:status='INCOMPLETE'
    decision=dict(status=status,selected_policy_id=policy['policy_id'],target_completeness_enabled=policy['target_completeness_enabled'],
        addressable_three_QA=addr,paired_quality=loss,credible_speed_gain=gain,quality_budget_pass=quality_ok,
        research_severe_gate_pass=severe_ok,independent_target_completeness_value=independent,
        new_severe_outputs=len(new),unique_new_severe=len({(e['frame_id'],e['question']) for e in new}),
        independent_test_logs=4,independent_test_frames=32,base_model_train_exposure_possible=True,
        broad_natural_quality='INCOMPLETE',deployment_safety='NOT_QUALIFIED',deployment_enabled=False,
        historical_v3_verdict='NO_GO_UNCHANGED',test_retuning=False)
    write(root/'go_no_go.json',decision)
    counts={k:len(lines(root/k/'requests.jsonl')) for k in ('diagnostic','shadow','pareto','run','ablation')}
    write(root/'run_counts.json',dict(stages=counts,full_M123_calls=sum(counts.values()),formal_question_requests=len(es),
        actual_formal_skips=len(fast),ablation_question_requests=len(ae)))
    (root/'e2e_performance_report.md').write_text('# Actual RTX3090 full M123\n\n'+
        '\n'.join(f"- {e['workload']}{e['question_count']}Q {e['comparison']}: {e['e2e_speedup']:.6f}x, delta sceneCI={e['delta_scene_CI95_ms']}ms." for e in perf['summaries'])+
        '\n\nFour new logs, eight consecutive frames, one round. Condition order rotates by scene/workload; actual event/outer timings, no theoretical token acceleration. Base-model training exposure is possible.\n')
    (root/'runtime_cost_report.md').write_text('# Measured execution costs\n\n'+
        'C-target alone reads already decoded competition scalars inside M1 output adapter and projects centers in snapshot. It does not alter M1 filtering/association, M2 reuse or any model forward. Basic/B do not pay the additional readout cost. Unsupported requests exit before target checks; normal K8 memory and RNG work remain.\n\n'+
        '\n'.join(f"- {m['workload']}{m['question_count']}Q {m['condition']}: E2E={m['e2e_wall_ms']:.4f}ms/frame, snapshot={m['snapshot_wall_ms']:.4f}, parsing={m['question_fact_extraction_wall_ms_per_question']:.6f}ms/Q, extraction={m['current_fact_extraction_wall_ms_per_question']:.6f}, target check={m['target_completeness_wall_ms_per_question']:.6f}, miss decision={m['miss_decision_wall_ms']}ms." for m in perf['module_breakdown'])+
        '\n\nExtraction and completeness are nested inside validity; do not sum them again. M1/M2/Heads and enqueue/CUDA/outer wall spans differ; complete E2E variation cannot be attributed solely to microsecond parsing. FPS/QPS, memory, forwards and phase timings are in abc_comparison.json.\n')
    mixed=[e for e in es if e['condition']=='C' and e['workload']=='mixed']
    (root/'mixed_workload_report.md').write_text('# Preset mixed workload\n\n'+f"One introduced structured question and two original questions per frame (preset1/3 slots). Actual skips={sum(e['llm_skipped'] for e in mixed)}/{len(mixed)}, B→C={s('mixed')['e2e_speedup']:.6f}x, delta sceneCI={s('mixed')['delta_scene_CI95_ms']}. Not a natural population estimate. Known/UNKNOWN quality and all-request bounds in task_quality.json.\n")
    (root/'implementation_summary.md').write_text('# WRSAR v4\n\n'+f"Research verdict **{status}**; best DEV policy {policy['policy_id']}, target completeness enabled={policy['target_completeness_enabled']}. {sum(counts.values())} actual full M123 calls; {len(fast)} formal actual skips.\n\n"+
        'P0: correct low-score pedestrian candidates are filtered at M1 .3 or M3 .35, while the category case retains its motorcycle but selects insufficiently supported near Car. Detection truth and nearest-target truth differ. The opt-in readout reuses existing CPU decoder scalars before the unchanged M1 filter; immutable competition evidence never becomes an Answer Witness. Current detector-center evidence remains distinct from M2 reuse qualification.\n\n'+
        f"New TEST four disjoint experimental logs/32 frames, historical scene/log metadata excluded. Source partition is physical train; possible base-model training exposure is disclosed. Addressable3Q {addr['e2e_speedup']:.6f}x; mixed3Q {s('mixed')['e2e_speedup']:.6f}x. Paired quality B−C={loss['loss']}, sceneCI={loss['scene_CI95']}, conservative all-request bounds={loss['conservative_all_request_loss_bounds']}; new severe={len(new)} outputs/{decision['unique_new_severe']} unique cases. Independent target-completeness value={independent}.\n\n"+
        'All thresholds, grading sources, average/severe research criteria were frozen before TEST. One runtime/generator, unchanged weights/sampling/RNG. Production options remain off; deployment safety is NOT_QUALIFIED. Historical v3 NO_GO is unchanged.\n')
    answers(root,decision,perf,q,ablation,policy,fast)
    print(json.dumps(dict(verdict=status,counts=counts,actual_formal_skips=len(fast),addressable=addr,
        quality=loss,severe_gate=severe_ok,target_enabled=policy['target_completeness_enabled'],independent_value=independent)),flush=True)


def match_physical(center,objects):
    candidates=sorted((math.dist(center,g['center_ego']),g['annotation_id'],g) for g in objects)
    return candidates[0][2] if candidates and candidates[0][0]<2. and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>.5) else None


def summary(part,**extra):
    known=[e for e in part if e['quality']['known']]
    return dict(**extra,requests=len(part),scored=len(known),UNKNOWN=len(part)-len(known),
        score=mean([e['quality']['score'] for e in known]),severe=sum(bool(e['quality']['severe_error']) for e in known),
        actual_skips=sum(e['llm_skipped'] for e in part),
        unique_frame_questions=len({(e['frame_id'],e['question']) for e in part}),
        unique_scored_frame_questions=len({(e['frame_id'],e['question']) for e in known}),
        independent_scenes=len({e['scene_id'] for e in part}))


def answers(root,d,perf,q,ab,policy,fast):
    def speed(w,n=3):return next(s for s in perf['summaries'] if (s['workload'],s['question_count'],s['comparison'])==(w,n,'B->C'))
    basic=next(g for g in ab['quality']['groups'] if (g['workload'],g['question_count'],g['condition'],g['execution'])==('addressable',3,'C-basic','fast_hit'))
    target=next(g for g in ab['quality']['groups'] if (g['workload'],g['question_count'],g['condition'],g['execution'])==('addressable',3,'C-target','fast_hit'))
    tasks=json.loads((root/'quality_by_task.json').read_text());pareto=json.loads((root/'quality_latency_pareto.json').read_text())
    texts=[
        '1. 严重Pedestrian案例中，当前正确Front-Left候选原始得分0.286888，被M1 0.3过滤。另选Front-Right有当前观测支持，却不能证明其为真正最近可见目标。全链路和物理/可见GT分别保留在nearest_target_failure_audit.json。',
        '2. 三个旧可评分快速错误：两个目标过滤（分别M1 .3和M3 .35），一个目标完整性不足。不是三个原始检测缺失，也没有证实排序代码按距离排序出错。',
        '3. 最小合法Anchor仅是检测集合中的最近者；集合可缺失低分目标，也可能含不满足问句可见目标要求的候选。相机描述子不证明最近排序完整。',
        '4. 已有解码输出中的更近/接近且会改变类别或方位的低分对象、当前目标分数、相机数及证据来源可提示歧义。未来、GT、历史预测均未用于当前准入。',
        '5. 同一Runtime：问题解析→当前正向Witness→只对相关选择任务计算竞争分数比/距离支配与任务来源条件→准入或原K8生成。竞争候选只提升不确定性。风险不是概率。',
        '6. 实测解析、当前事实提取、完整性检查、快照、M1输出适配器和E2E见runtime_cost_report.md及abc_comparison.json；嵌套时间不重复相加。',
        f"7. 新TEST同证据核心消融拒绝={ab['veto_counts']}；有益/无益/UNKNOWN按独立标签分开，不能把低分候选本身当作真实肯定事实。",
        '8. 正向存在性可由合法当前单个Witness支持；最近类别/方位按DEV校准的置信度、竞争风险和中心投影相机数准入。中心投影不证明无遮挡/身份/排序，也不是M2复用证书。',
        '9. v3当前检测证据修复保留，未恢复90%全框复用资格过滤。旧Car实际输出回归见prior_regression.json；回归不是新TEST证据。',
        f"10. 新增严重输出相对B={d['new_severe_outputs']}，唯一Frame/Question={d['unique_new_severe']}；C-basic与C-target比较见核心消融，所有严重左右/前后案例在severe_error_audit.json。",
        '11. Category/Existence/Position/Motion的A/B/C分母、UNKNOWN、分数和严重错误分别见quality_by_task.json；不存在的可靠Motion标签不计正确。',
        f"12. 正式请求真实跳过{len(fast)}次，零输入展开/Prefill/Decode/验证Forward。分负载实际分母见fast_answer_quality.json；DEV及消融另计。",
        f"13. 同一新TEST对象三问C-basic命中{basic['requests']}/96，C-target命中{target['requests']}/96；不把不同数据流直接当成相对v3的因果提升。",
        '14. 未命中进入同一K8生成器；正常B和C-basic不承担新增竞争读取，C-target承担读取/快照/检查成本。不支持问题不扫描竞争候选。配对E2E与模块时延共同解释，不能归因于单个微秒级解析。',
        f"15. 对象三问B→C={speed('addressable')['e2e_speedup']:.6f}x，场景差CI={speed('addressable')['delta_scene_CI95_ms']}ms。",
        f"16. 预设1/3结构化混合三问B→C={speed('mixed')['e2e_speedup']:.6f}x，差CI={speed('mixed')['delta_scene_CI95_ms']}ms；不代表自然分布。",
        f"17. 原始自然1Q={speed('natural',1)['e2e_speedup']:.6f}x，3Q={speed('natural')['e2e_speedup']:.6f}x；机会、跳过数和场景CI分别报告，不能从对象负载外推。",
        '18. 1/2/3pp DEV最优候选='+json.dumps(pareto['policies'],ensure_ascii=False)+f"；本轮2pp选择{policy['policy_id']}，target enabled={policy['target_completeness_enabled']}。研究严重风险边界预先冻结为已评分请求新严重率及场景CI上界≤2pp；部署安全始终不合格。",
        f"19. 最终{d['status']}：可信速度收益={d['credible_speed_gain']}，质量预算={d['quality_budget_pass']}，研究严重风险={d['research_severe_gate_pass']}。UNKNOWN、32帧/4日志和基础模型训练分区暴露限制仍保留。",
        f"20. 新完整性模块独立质量/严重风险改善证据={d['independent_target_completeness_value']}。仅在相同新数据/生成器下的核心消融能支持这一判断；最终可选更简单Current-Only，不能为了新模块名字牺牲实测最优。历史v3 NO_GO不变。"]
    (root/'result_answers.md').write_text('# 20项最终回答\n\n'+'\n\n'.join(texts)+'\n')
