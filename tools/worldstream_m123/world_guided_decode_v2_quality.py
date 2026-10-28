"""Offline task-fact grading against original QA / independent nuScenes labels.

No native-answer truth, language similarity score, or online decision hooks.
Conservative extraction abstains on ambiguous reference facts. These limited
task-fact scores are reported with coverage, not as a complete driving benchmark.
"""
from collections import defaultdict,Counter
from pathlib import Path
import json
import math
import re
import numpy as np
from worldstream.m3.task_conditioner import question_dependencies
from tools.worldstream_m123.benchmark.common import write

NUMBER={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}


def qa_pairs(token):
    result={}
    for folder in ('conv/val','vqa/val'):
        p=Path('data/nuscenes')/folder/(token+'.json')
        if p.exists():
            for row in json.loads(p.read_text()):result[row['question']]=row['answer']
    return result


def polarity(text):
    text=text.strip().lower()
    if re.match(r'yes\b',text):return 'yes'
    if re.match(r'no\b',text) or re.match(r'there (?:is|are) (?:no|not any)\b',text):return 'no'
    return None


def facts(question,answer):
    q=question.lower();a=answer.lower();dep=question_dependencies(q)
    result={}
    if re.match(r'(?:is|are|do|does|can|should)\b',q):
        p=polarity(a)
        if p is not None:result['binary_answer']=p
    if dep.intent=='signal':
        colors=set(re.findall(r'\b(red|green|yellow|amber)\b',a));colors={'yellow' if c=='amber' else c for c in colors}
        if len(colors)==1:result['signal_color']=next(iter(colors))
    if dep.intent=='category':
        found=[]
        for name,pattern in (('car',r'cars?|sedans?|suvs?|taxi|taxis|vans?'),('truck',r'trucks?'),
            ('bus',r'bus|buses'),('pedestrian',r'pedestrians?|people'),('bicycle',r'bicycles?|cyclists?'),('motorcycle',r'motorcycles?')):
            if (re.search(r'\b(?:'+pattern+r')\b',a)
                    and not re.search(r'\b(?:no|not|without)\s+(?:\w+\s+){0,2}(?:'+pattern+r')\b',a)):
                found.append(name)
        if found:result['reference_class_set']=tuple(sorted(found))
    if dep.intent=='position' or re.search(r'\bwhere\b',q):
        text=re.sub(r'\bahead\b','front',a);text=re.sub(r'\bbehind\b','rear',text)
        found=set(re.findall(r'\b(front|rear|left|right)\b',text))
        if found and not ({'left','right'}<=found or {'front','rear'}<=found):result['relative_direction']=tuple(sorted(found))
    if dep.intent=='motion':
        stationary=bool(re.search(r'\b(stationary|parked|stopped)\b',a))
        moving=bool(re.search(r'\b(moving|travelling|traveling|driving)\b',a))
        if stationary!=moving:result['motion_state']='stationary' if stationary else 'moving'
    if re.search(r'how many|number of|count',q):
        values=re.findall(r'\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b',a)
        if 'lane' in q:
            lane=re.findall(r'\b(\d+|one|two|three|four|five|six)\s+(?:traffic )?lanes?\b',a)
            if lane:values=lane
        unique={NUMBER.get(v,int(v) if v.isdigit() else None) for v in values}
        if len(unique)==1:result['object_count']=next(iter(unique))
    if dep.intent=='distance':
        values=[float(v) for v in re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:meters?|metres?|m)\b',a)]
        if len(values)==1:result['distance_m']=values[0]
    if dep.intent=='safety':
        actions=[]
        for name,pattern in (('slow',r'slow|reduce (?:the )?speed'),('yield',r'yield|give way'),
            ('stop',r'stop|halt|wait'),('observe',r'cautio\w*|aware|watch|look|check|vigilant|observe')):
            if re.search(r'\b(?:'+pattern+r')\b',a):actions.append(name)
        if actions:result['reference_action_set']=tuple(sorted(actions))
    if dep.intent=='road':
        types=set(re.findall(r'\b(t-junction|four-way|three-way|roundabout|crossroads)\b',a))
        if len(types)==1:result['road_subtype']=next(iter(types))
    return result


def compare_facts(expected,observed):
    rows=[]
    for name,value in expected.items():
        actual=observed.get(name)
        if name in ('reference_class_set','reference_action_set'):correct=actual is not None and set(value)<=set(actual)
        elif name=='distance_m':correct=actual is not None and abs(value-actual)<=max(2.,.1*value)
        else:correct=actual==value
        rows.append(dict(fact=name,expected=value,observed=actual,correct=bool(correct)))
    return rows


def task_quality(root,trials):
    evaluated=[];missing=Counter();critical=[]
    for row in trials:
        # Three-QA cohort is primary, avoiding double-counting Q1 from the
        # separately timed single-question workload. Fixed workload GT gaps
        # remain missing; source-frame answers are never target-frame labels.
        if row['question_count']!=3:continue
        refs=qa_pairs(row['frame_id'])
        for request,output in zip(row['requests'],row['answers']):
            question=request['question'];reference=refs.get(question)
            base=dict(condition=row['condition'],round=row['round'],workload=row['workload'],
                scene_id=row['scene_id'],frame_id=row['frame_id'],question=question,
                task=question_dependencies(question).intent,answer=output['A'][0])
            if reference is None:
                missing[(row['condition'],row['workload'],'question_not_in_current_frame_GT')]+=1;continue
            if not isinstance(reference,str) or not reference.strip():
                missing[(row['condition'],row['workload'],'nontext_or_empty_current_frame_GT')]+=1;continue
            expected=facts(question,reference)
            if not expected:
                missing[(row['condition'],row['workload'],'reference_facts_not_unambiguously_extractable')]+=1;continue
            scores=compare_facts(expected,facts(question,output['A'][0]))
            evaluated.append(dict(base,reference=reference,facts=scores,score=sum(s['correct'] for s in scores)/len(scores)))
            for score in scores:
                if not score['correct'] and (score['fact'] in ('signal_color','relative_direction','motion_state') or base['task']=='safety'):
                    critical.append(dict(base,reference=reference,failure=score))
    summaries=[]
    for workload in ('natural','fixed'):
        for condition in ('A','B','C'):
            rows=[r for r in evaluated if r['workload']==workload and r['condition']==condition]
            tasks={}
            for task in sorted({r['task'] for r in rows}):
                rs=[r for r in rows if r['task']==task]
                tasks[task]=dict(scored_answers=len(rs),mean_answer_fact_score=float(np.mean([r['score'] for r in rs])))
            summaries.append(dict(workload=workload,condition=condition,scored_answers=len(rows),
                mean_answer_fact_score=float(np.mean([r['score'] for r in rows])) if rows else None,tasks=tasks))
    write(root/'task_quality.json',dict(status='MEASURED_PARTIAL_COVERAGE',summaries=summaries,evaluated=evaluated,
        missing=[dict(condition=k[0],workload=k[1],reason=k[2],count=v) for k,v in missing.items()],critical_errors=critical,
        scope='Exact current-frame original QA labels only. No fixed-stream semantic pseudo-GT. Rule-extracted binary/count/direction/motion/color/distance/road facts; category/action scores are reference fact recall, not precision or exhaustive safety correctness. Unscored reference text is not a measured success.'))
    return evaluated,summaries,critical


def world_accuracy(root,trials,development_root=None):
    """Independent annotation audit, strictly after all inference decisions."""
    import pickle
    from mmcv import Config
    from tools.worldstream_m1.dataset import load_oracle
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    with open(cfg.data.test.ann_file,'rb') as handle:infos=pickle.load(handle)['infos']
    snapshots={}
    for r in trials:
        if r['condition']=='C':snapshots[(r['frame_id'],r['committed_world']['epoch'])]=r['committed_world']
    dev=[json.loads(s) for s in ((development_root or root)/'development_coverage.jsonl').read_text().splitlines()]
    for r in dev:snapshots[(r['frame_id'],r['committed_world']['epoch'])]=r['committed_world']
    selected=[i for i in infos if i['token'] in {key[0] for key in snapshots}]
    oracle=load_oracle(selected,'data/nuscenes')
    mapped={};class_rows=[]
    class_frames=set()
    for (token,epoch),state in snapshots.items():
        transform=np.array(state['ego_to_world']).reshape(4,4)
        gt=[g for g in oracle.get(token,[]) if g['has_sensor_support']]
        ids={}
        for a in state['anchors']:
            if not a['observed'] or a['anchor_type']!='object':continue
            assert a['source_frame']==token and a['source_timestamp']==state['timestamp']
            center=(transform@np.r_[a['center_ego'],1.])[:3]
            distances=sorted((float(np.linalg.norm(g['geometry'].center-center)),j) for j,g in enumerate(gt))
            known=bool(distances and distances[0][0]<2. and (len(distances)<2 or distances[1][0]-distances[0][0]>.5))
            g=gt[distances[0][1]] if known else None
            ids[a['anchor_id']]=g['instance_id'] if g else None
            if token not in class_frames:
                class_rows.append(dict(frame_id=token,anchor_id=a['anchor_id'],known=known,
                    predicted_class=a['semantic_class'],gt_class=g['class'] if g else None,
                    class_correct=a['semantic_class']==g['class'] if g else None))
        class_frames.add(token)
        mapped[(token,epoch)]=ids
    # Evaluate only same-scene/epoch source/current edges actually cited by C.
    edges=[];seen=set()
    for row in trials:
        if row['condition']!='C':continue
        for q in row['requests']:
            delta=q.get('world_delta') or {}
            for aid in delta.get('matched_anchor_ids',[]):
                source,current=delta['previous_frame'],delta['current_frame']
                versions=delta['evidence_versions']
                source_epoch,current_epoch=versions[0][0],versions[1][0]
                key=(source,current,aid)
                if key in seen:continue
                seen.add(key)
                a=mapped.get((source,source_epoch),{}).get(aid);b=mapped.get((current,current_epoch),{}).get(aid)
                edges.append(dict(source_frame=source,current_frame=current,anchor_id=aid,
                    known=a is not None and b is not None,correct=a==b if a is not None and b is not None else None))
    known=[r for r in class_rows if r['known']];scored=[r for r in edges if r['known']]
    write(root/'world_delta_accuracy.json',dict(status='MEASURED_PARTIAL_GT',
        current_observed_objects=len(class_rows),class_scored=len(known),class_unknown=len(class_rows)-len(known),
        matched_class_accuracy=sum(r['class_correct'] for r in known)/len(known) if known else None,
        cited_identity_edges=len(edges),identity_scored=len(scored),identity_unknown=len(edges)-len(scored),
        identity_correct=sum(r['correct'] for r in scored),identity_wrong=sum(not r['correct'] for r in scored),
        identity_accuracy=sum(r['correct'] for r in scored)/len(scored) if scored else None,
        class_rows=class_rows,identity_edges=edges,complete_scene_coverage_claimed=False,
        scope='Offline nuScenes annotation geometry, unique nearest any-class center within 2m and >0.5m nearest-gap; no class-conditioned matching tautology. Unmatched/ambiguous observations are UNKNOWN. Cited identity IDs are scoped to their snapshot epoch. Not a complete scene/geometry correctness certificate.'))
