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

def evaluate_main_scene(batches,annotations,config):
    manager=WorldStateManager(config)
    counters={k:0 for k in ('accepted','candidates','correct','wrong','scored','unknown','eligible',
        'identity_switches','fragments','continuity_breaks','tracks','aligned','alignment_correct',
        'alignment_scored','alignment_unknown','alignment_possible','ambiguous','rejected','proposals')}
    previous={};last_instance={};previous_gt_geometry={};groups={};traces=[];times=[];bridge_times=[]
    total_gt_visible=0;occluded=0;fast=0;near_opposing=0
    for ordinal,(frame,outputs) in enumerate(batches):
        tick=perf_counter();r=manager.precheck(frame);manager.commit(frame,copy.deepcopy(outputs),r)
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
            association_observation=d.get('association_observation'),association_tracks=d.get('association_tracks')))
    return dict(metrics=rates(counters),strata={k:rates(v) for k,v in groups.items()},
        m1=percentiles(times),bridge=percentiles(bridge_times),traces=traces,
        difficulty=dict(gt_observations=total_gt_visible,occluded_observations=occluded,
            fast_observations=fast,near_opposing_motion_pairs=near_opposing,
            crossing_semantics='4m opposing-motion proxy; not visually verified crossings'),
        GT_scope='unique 1m partial matching; unknown excluded; not official nuScenes tracking metrics')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--protocol',default='outputs/m1_dqtrack_policy_20261021/protocol.json')
    p.add_argument('--candidates',default='outputs/m1_dqtrack_policy_20261021/omnidrive/model_observations.pkl')
    p.add_argument('--reference',default='outputs/m1_dqtrack_policy_20261021/analysis_summary_owned.json')
    p.add_argument('--infos',default='data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    p.add_argument('--data-root',default='data/nuscenes')
    p.add_argument('--config',default='configs/worldstream_m1_world_identity.yaml')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    config=replace(M1Config.load(a.config),association_track_prior=True,logging=False,
                   dqtrack_mode='off',association_observation_guard=False)
    protocol=json.loads(Path(a.protocol).read_text())
    sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for root in ('worldstream/m1','tools/worldstream_m1')
             for f in Path(root).glob('*.py')}
    inputs={n:dict(path=getattr(a,n),sha256=hashlib.sha256(Path(getattr(a,n)).read_bytes()).hexdigest())
            for n in ('protocol','candidates','infos','config','reference')}
    definition=dict(method='world_referenced_association',runs=1,external_tracker_called=False,
        new_perception_inference=False,config=asdict(config),inputs=inputs,sources=sources,
        scoring='Original strict 1m partial oracle and previous-visible denominators, unknown excluded from scored',
        scorer_origin='tools/worldstream_m1/dqtrack_policy_eval.py (static copy; not imported or executed)')
    (out/'run_definition.json').write_text(json.dumps(definition,indent=2))
    with open(a.infos,'rb') as f:by_token={i['token']:i for i in pickle.load(f)['infos']}
    with open(a.candidates,'rb') as f:observations=pickle.load(f)
    from .dataset import frame_from_info,load_oracle
    wanted=[by_token[t] for spec in protocol['development'] for t in spec['samples']]
    annotations=load_oracle(wanted,a.data_root)
    results={}
    for spec in protocol['development']:
        batches=[(frame_from_info(by_token[t],a.data_root),observations[t]) for t in spec['samples']]
        result=evaluate_main_scene(batches,annotations,config)
        (out/(spec['scene_id']+'.json')).write_text(json.dumps(json_value(result),allow_nan=False))
        results[spec['scene_id']]={k:v for k,v in result.items() if k!='traces'}
        print(spec['scene_id'],result['metrics'],flush=True)
        del batches
    counts={k:sum(s['metrics'][k] for s in results.values())
            for k,v in next(iter(results.values()))['metrics'].items() if isinstance(v,int)}
    metrics=rates(counts)
    reference=json.loads(Path(a.reference).read_text())['methods']['geometry']['metrics']
    comparable=metrics['candidates']==reference['candidates'] and metrics['eligible']==reference['eligible']
    gates=dict(wrong=metrics['wrong']<=reference['wrong'],recall=metrics['effective_recall']>=reference['effective_recall'],
        coverage=metrics['association_coverage']>=reference['association_coverage'],
        switches=metrics['identity_switches']<reference['identity_switches'],
        fragments=metrics['fragments']<reference['fragments'],continuity=metrics['continuity_breaks']<reference['continuity_breaks'])
    summary=dict(metrics=metrics,saved_reference=reference,denominators_equal=comparable,gates=gates,
        decision='GO' if comparable and all(gates.values()) else 'NO_GO',scenes=results,
        fragments_definition='GT scored-visible ordinal gaps; independent of assigned identity; preserved unchanged')
    (out/'summary.json').write_text(json.dumps(json_value(summary),indent=2,allow_nan=False))
    print('FINAL',json.dumps(json_value({k:v for k,v in summary.items() if k!='scenes'})),flush=True)


if __name__=='__main__':main()
