"""Single M1 association method only; cached detections, original partial-oracle scorer.

Scoring functions copied from dqtrack_policy_eval.py, with all external tracker
and method dispatch code removed. GT labels are used only after commit.
"""
import argparse
import copy
from dataclasses import replace, asdict
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter
import numpy as np
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m1.association import legal_assignment
from worldstream.m1.state_types import json_value

def strict_labels(predictions,annotations,gate=1.):
    """Offline partial oracle: unique class/size/1m edges in BOTH directions."""
    costs=np.full((len(predictions),len(annotations)),np.inf)
    for i,p in enumerate(predictions):
        for j,a in enumerate(annotations):
            if p.semantic_class!=a['class'] or not a['has_sensor_support']:continue
            if p.geometry_world.size is None:continue
            ratios=np.maximum(p.geometry_world.size/a['geometry'].size,a['geometry'].size/p.geometry_world.size)
            if np.any(ratios>2):continue
            distance=np.linalg.norm(p.geometry_world.center-a['geometry'].center)
            if distance<=gate:costs[i,j]=distance
    finite=np.isfinite(costs)
    return {i:annotations[j]['instance_id'] for i,j in legal_assignment(costs)
            if finite[i].sum()==1 and finite[:,j].sum()==1}

def percentiles(values):
    a=np.asarray(values,float)
    return dict(n=len(a),p50_ms=float(np.percentile(a,50)) if len(a) else None,
        p95_ms=float(np.percentile(a,95)) if len(a) else None,
        throughput_frames_per_s=1000*len(a)/a.sum() if len(a) and a.sum()>0 else None)

def rates(c):
    def ratio(a,b):return c.get(a,0)/c.get(b,0) if c.get(b,0) else None
    return dict(c,wrong_association_rate=ratio('wrong','scored'),
        effective_recall=ratio('correct','eligible'),association_coverage=ratio('accepted','candidates'),
        rejection_rate=1-ratio('accepted','candidates') if c.get('candidates',0) else None,
        alignment_precision=ratio('alignment_correct','alignment_scored'),
        alignment_recall=ratio('alignment_correct','alignment_possible'))

def evaluate_main_scene(batches,annotations,config,saved=None):
    from .identity_metrics import ContinuityMetrics, possible_instances
    manager=WorldStateManager(config)
    continuity=ContinuityMetrics();fast_diagnosis=[];recoveries=[];recovery_opportunities=[]
    actual_source_labels={};instance_history={};visibility_history={};wrong_relations=[]
    counters={k:0 for k in ('accepted','candidates','correct','wrong','scored','unknown','eligible',
        'identity_switches','fragments','continuity_breaks','tracks','aligned','alignment_correct',
        'alignment_scored','alignment_unknown','alignment_possible','ambiguous','rejected','proposals')}
    previous={};last_instance={};previous_gt_geometry={};groups={};traces=[];times=[];bridge_times=[]
    total_gt_visible=0;occluded=0;fast=0;near_opposing=0
    for ordinal,(frame,outputs) in enumerate(batches):
        tick=perf_counter();r=manager.precheck(frame)
        manager.commit(frame,copy.deepcopy(outputs),r)
        times.append((perf_counter()-tick)*1000);d=manager.last_commit_diagnostics
        # Explicit post-commit boundary: no GT labels are attached to Frame/outputs.
        anns=annotations.get(frame.frame_id,[]);ann_by={a['instance_id']:a for a in anns}
        labels=strict_labels(outputs.anchors,anns)
        selected=d['selected_prediction_indices'];committed=d['committed_anchor_ids']
        current={aid:labels[index] for aid,index in zip(committed,selected) if index in labels}
        prev_instances=set(previous.values());now_instances=set(current.values())
        eligible=prev_instances&now_instances
        density='sparse' if len(anns)<10 else 'moderate' if len(anns)<30 else 'dense'
        motions={}
        for identity,a in ann_by.items():
            if identity in previous_gt_geometry:
                point,ts=previous_gt_geometry[identity]
                speed=np.linalg.norm(a['geometry'].center-point)/max(frame.timestamp-ts,1e-9)
                motions[identity]='fast' if speed>=8 else 'moving' if speed>=.5 else 'static'
            else:motions[identity]='unknown'
        def strata(identity,semantic):
            a=ann_by.get(identity,{})
            return ['class:'+semantic,'density:'+density,'visibility:'+str(a.get('visibility','unknown')),
                    'motion:'+motions.get(identity,'unknown')]
        def bump(keys,name,n=1):
            for key in keys:groups.setdefault(key,{});groups[key][name]=groups[key].get(name,0)+n
        if ordinal:
            counters['candidates']+=len(selected);counters['eligible']+=len(eligible)
            for identity in eligible:bump(strata(identity,ann_by[identity]['class']),'eligible')
        correct_ids=set();edge_rows=[]
        for aid,index in d['association_pairs']:
            old=previous.get(aid);new=labels.get(index);known=old is not None and new is not None
            counters['accepted']+=1
            status='correct' if known and old==new else 'wrong' if known else 'unknown'
            counters[status]+=1;counters['scored']+=int(known)
            if status=='wrong':wrong_relations.append([frame.frame_id,old,new])
            keys=strata(new,outputs.anchors[index].semantic_class)
            bump(keys,'accepted');bump(keys,status);bump(keys,'scored',int(known))
            if status=='correct':correct_ids.add(new)
            edge_rows.append(dict(anchor_id=aid,prediction_index=index,previous_instance=old,current_instance=new,status=status))
        counters['continuity_breaks']+=len(eligible-correct_ids)
        for aid,identity in current.items():
            old=last_instance.get(identity)
            if old:
                counters['identity_switches']+=int(old[0]!=aid)
                counters['fragments']+=int(old[1]<ordinal-1)
            last_instance[identity]=(aid,ordinal)
        possible=possible_instances(outputs.anchors,anns)
        continuity.update(anns,{identity:aid for aid,identity in current.items()},possible)
        saved_metrics={name:reader.update(frame.frame_id,selected,labels,anns,possible)
                       for name,reader in (saved or {}).items()}
        td=d.get('association_tracks') or {}
        index_by_gt={identity:index for index,identity in labels.items() if index in selected}
        for identity in sorted(eligible):
            if motions.get(identity)=='fast':
                fast_diagnosis.append(dict(frame_id=frame.frame_id,instance_id=identity,
                    prediction_index=index_by_gt[identity],correct=identity in correct_ids))
        for aid in td.get('recovered',[]):
            old=actual_source_labels.get(aid);new=current.get(aid)
            known=old is not None and new is not None
            recoveries.append(dict(frame_id=frame.frame_id,anchor_id=aid,previous_instance=old,current_instance=new,
                status='correct' if known and old==new else 'wrong' if known else 'unknown'))
        for identity,index in index_by_gt.items():
            last=instance_history.get(identity)
            if last and last['ordinal']<ordinal-1 and frame.timestamp-last['timestamp']<=config.association_track_ttl_s:
                intervening=visibility_history.get(identity,[])[last['ordinal']+1:ordinal]
                kind='sensor_support_gap' if any(v=='unsupported' for v in intervening) else 'gt_absent_gap' if any(v=='absent' for v in intervening) else 'partial_oracle_detection_gap'
                recovery_opportunities.append(dict(frame_id=frame.frame_id,instance_id=identity,kind=kind,
                    gap_frames=ordinal-last['ordinal']-1,identity_preserved=last['anchor_id'] in current and current[last['anchor_id']]==identity,
                    dormant_applied=last['anchor_id'] in td.get('recovered',[])))
            aid=next(aid for aid,value in current.items() if value==identity)
            instance_history[identity]=dict(ordinal=ordinal,timestamp=frame.timestamp,anchor_id=aid)
        for identity in set(visibility_history)|set(ann_by):
            if identity not in visibility_history:visibility_history[identity]=['absent']*ordinal
            ann=ann_by.get(identity)
            visibility_history[identity].append('absent' if ann is None else 'supported' if ann['has_sensor_support'] else 'unsupported')
        # A newer unscored observation invalidates the older GT label for this
        # supplemental source/target relation; never fill unknown with history.
        actual_source_labels.update({aid:current.get(aid) for aid in committed})
        bd=None
        # Difficulty labels are offline summaries, never online evidence.
        total_gt_visible+=len(anns);occluded+=sum(str(a['visibility']) in ('1','2','3') for a in anns)
        fast+=sum(v=='fast' for v in motions.values())
        moving=[]
        for identity,a in ann_by.items():
            if identity in previous_gt_geometry:
                point,ts=previous_gt_geometry[identity];moving.append((a['geometry'].center,(a['geometry'].center-point)/max(frame.timestamp-ts,1e-9)))
        near_opposing+=sum(np.linalg.norm(p[:2]-q[:2])<4 and np.dot(v[:2],w[:2])<0
            for i,(p,v) in enumerate(moving) for q,w in moving[i+1:])
        previous_gt_geometry={identity:(a['geometry'].center,frame.timestamp) for identity,a in ann_by.items()}
        previous=current
        traces.append(dict(frame_id=frame.frame_id,scene_id=frame.scene_id,edges=edge_rows,
            certified=list(manager.get_current_certifications().certified_anchor_ids),
            pairs=d['association_pairs'],dqtrack=bd,m1_ms=times[-1],
            association_tracks=d.get('association_tracks'),
            committed_anchor_ids=committed,selected_prediction_indices=selected,scored_identity_by_anchor=current))
    metrics=rates(counters)
    metrics.update(continuity.counts,legacy_gt_gap_count=counters['fragments'],
        correct_associations=counters['correct'],wrong_associations=counters['wrong'],unknown_associations=counters['unknown'])
    return dict(metrics=metrics,fast_diagnosis=fast_diagnosis,recoveries=recoveries,
        recovery_opportunities=recovery_opportunities,saved_continuity_metrics=saved_metrics if batches else {},
        wrong_relations=wrong_relations,strata={k:rates(v) for k,v in groups.items()},
        m1=percentiles(times),bridge=percentiles(bridge_times),traces=traces,
        difficulty=dict(gt_observations=total_gt_visible,occluded_observations=occluded,
            fast_observations=fast,near_opposing_motion_pairs=near_opposing,
            crossing_semantics='4m opposing-motion proxy; not visually verified crossings'),
        GT_scope='unique 1m partial matching; unknown excluded; not official nuScenes tracking metrics')
