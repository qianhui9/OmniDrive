"""Development-only paired policy diagnosis; never creates online qualification.

GT labels are constructed and consumed after each manager.commit. ResearchManager
only injects a frozen numeric profile, never annotations. Independent qualification
cannot be inferred from exposed nuScenes validation scenes.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1.association import legal_assignment
from worldstream.m1.state_types import WorldAnchor,json_value
from worldstream.m1.geometry import box_geometry


class ResearchManager(WorldStateManager):
    def _dqtrack_policy_profile(self):
        return dict(self.research_profile),dict(status='OFFLINE_RESEARCH_ONLY_NOT_QUALIFIED')


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


def evaluate_scene(batches,annotations,config,path,method,profile):
    mode='off' if method in ('geometry','guard_geometry') else 'shadow' if method=='shadow' else 'hint'
    cls=ResearchManager if method=='research_tiebreak' else WorldStateManager
    cfg=replace(config,dqtrack_mode=mode,dqtrack_results_path=str(path),logging=False,
                dqtrack_association_policy='calibrated_tiebreak' if cls is ResearchManager else 'conservative',
                association_observation_guard=method in ('guard_geometry','guard_hint'))
    manager=cls(cfg)
    if cls is ResearchManager:manager.research_profile=profile
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
        bd=d.get('dqtrack')
        if bd:
            counters['tracks']+=bd['available_tracks'];counters['aligned']+=bd['aligned']
            counters['ambiguous']+=bd['ambiguous'];counters['rejected']+=len(bd['rejected'])
            counters['proposals']+=len(bd.get('policy',{}).get('proposals',[]))
            bridge_times.append(bd['elapsed_ms'])
            data=json.loads(Path(path).read_text())
            sample_token=data['m1_manifest']['frames'][frame.frame_id]['sample_token']
            raw=data['results'].get(sample_token,[])
            tracks=[];track_indices=[]
            from pyquaternion import Quaternion
            for i,row in enumerate(raw):
                try:
                    matrix=np.eye(4);matrix[:3,:3]=Quaternion(row['rotation']).rotation_matrix;matrix[:3,3]=row['translation']
                    tracks.append(WorldAnchor(str(i),frame.scene_id,'object',row['tracking_name'],
                        box_geometry(np.zeros(3),np.array(row['size'])[[1,0,2]],to_world=matrix),frame.timestamp))
                    track_indices.append(i)
                except (ValueError,KeyError):continue
            track_gt=strict_labels(tracks,anns)
            gt_by_id={raw[track_indices[i]]['tracking_id']:identity for i,identity in track_gt.items()}
            counters['alignment_possible']+=len(set(gt_by_id.values())&set(labels.values()))
            for j,h in bd.get('alignments',{}).items():
                identity=gt_by_id.get(h['tracking_id']);candidate=labels.get(selected[int(j)])
                known=identity is not None and candidate is not None
                counters['alignment_scored']+=int(known);counters['alignment_unknown']+=int(not known)
                counters['alignment_correct']+=int(known and identity==candidate)
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
            association_observation=d.get('association_observation')))
    return dict(metrics=rates(counters),strata={k:rates(v) for k,v in groups.items()},
        m1=percentiles(times),bridge=percentiles(bridge_times),traces=traces,
        difficulty=dict(gt_observations=total_gt_visible,occluded_observations=occluded,
            fast_observations=fast,near_opposing_motion_pairs=near_opposing,
            crossing_semantics='4m opposing-motion proxy; not visually verified crossings'),
        GT_scope='unique 1m partial matching; unknown excluded; not official nuScenes tracking metrics')


def scene_bootstrap(scenes,metric='wrong_association_rate',seed=20261021):
    rng=np.random.RandomState(seed);values=[]
    for _ in range(2000):
        sampled=[scenes[i]['metrics'] for i in rng.randint(0,len(scenes),len(scenes))]
        denominator=sum(s['scored'] for s in sampled)
        if denominator:values.append(sum(s['wrong'] for s in sampled)/denominator)
    return dict(unit='scene',independent_validation=False,n_scenes=len(scenes),draws=2000,
        descriptive_interval=np.percentile(values,[2.5,97.5]).tolist() if values else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','candidates','dqtrack-root','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--infos',default='data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    p.add_argument('--data-root',default='data/nuscenes')
    p.add_argument('--config',default='configs/worldstream_m1_balanced.yaml')
    p.add_argument('--min-score',type=float,default=.5)
    p.add_argument('--methods',nargs='+',choices=['geometry','shadow','conservative','research_tiebreak',
        'guard_geometry','guard_hint'],default=['geometry','shadow','conservative','research_tiebreak'])
    a=p.parse_args();output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(Path(a.protocol).read_text())
    # Profile is frozen before this program loads any GT evaluation labels.
    profile=dict(min_score=a.min_score,min_observations=3,max_hint_age_s=1.)
    (output/'research_profile.json').write_text(json.dumps(profile,indent=2))
    with open(a.infos,'rb') as f:infos=pickle.load(f)['infos']
    by_token={i['token']:i for i in infos}
    with open(a.candidates,'rb') as f:candidates=pickle.load(f)
    from .dataset import frame_from_info,load_oracle
    config=M1Config.load(a.config);results={}
    wanted=[by_token[t] for spec in protocol['development'] for t in spec['samples']]
    annotations=load_oracle(wanted,a.data_root)
    for spec in protocol['development']:
        scene=spec['scene_id'];folder=output/scene;folder.mkdir()
        batches=[(frame_from_info(by_token[t],a.data_root),candidates[t]) for t in spec['samples']]
        results[scene]={}
        for method in a.methods:
            result=evaluate_scene(batches,annotations,config,Path(a.dqtrack_root)/scene/'normalized_tracking.json',method,profile)
            with (folder/(method+'.json')).open('x') as f:json.dump(json_value(result),f,allow_nan=False)
            results[scene][method]={k:v for k,v in result.items() if k!='traces'}
            print(scene,method,result['metrics'],flush=True)
    summary=dict(protocol_sha256=hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest(),profile=profile,
        deployment_decision='NOT_QUALIFIED_NO_INDEPENDENT_SCENES',scenes=results,methods={})
    for method in a.methods:
        rows=[r[method] for r in results.values()];counts={k:sum(r['metrics'].get(k,0) for r in rows)
            for k,v in rows[0]['metrics'].items() if isinstance(v,int)}
        summary['methods'][method]=dict(metrics=rates(counts),wrong_rate_scene_bootstrap=scene_bootstrap(rows))
    (output/'summary.json').write_text(json.dumps(json_value(summary),indent=2,allow_nan=False))


if __name__=='__main__':main()
