"""Independent annotation scoring AFTER annotation-free fact extraction.

Frozen, deliberately conservative diagnostic of available physical evidence.
No source GT enters TaskFact construction, M1, M3, or scheduling.
"""
from collections import Counter,defaultdict
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np

from worldstream.m1.task_facts import extract_task_facts
from worldstream.m1.state_types import json_value
from .audit import write
from .streaming_replay import distribution
from .v3_physical import upper95

ROOT=Path('outputs/system_optimization_v4')


def freeze():
    path=ROOT/'task_fact_protocol.json'
    if path.exists():raise FileExistsError('Fact protocol immutable')
    write(path,dict(frozen_before_v4_fact_oracle=True,online_consumption_allowed=False,
        physical_risk=dict(far_upper95_max=.05,min_support_scenes=3,min_accepted_scorable=59),
        metric_distance_error_m_max=.5,motion_vector_error_mps_max=.5,
        direction='exact longitudinal/lateral signs in current ego; unknown zero or boundary-spanning interval never certifies',
        persistence='same independent instance with current sensor support and visibility=4',
        offline_identity='same-class source nearest within 2m; reject if multiple eligible same-class GT or competing predicted source within 2m',
        splits='same frozen v4 scene split; all old 29 exploratory scenes excluded',
        extractor='missing independent identity, foreground, metric and motion bounds => Unknown; never detector score as localization bound',
        source_gate='two manually reviewed original metric questions; Planning remains No-Go',
        thresholds='new Fact score tolerances frozen independently; old M1 physical protocol remains unchanged',
        gt_scope='offline scoring only AFTER all annotation-free extractions persisted; not online inputs, certificates or scheduling'))


def main():
    protocol=json.loads((ROOT/'task_fact_protocol.json').read_text())
    if json.loads((ROOT/'gpu_status.json').read_text())['status']!='completed':raise RuntimeError('Wait for all real captures')
    raw=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']
    infos={i['token']:i for i in raw};paths=sorted(ROOT.glob('*/oracle_frames/*.pkl.gz'))
    if not paths:raise ValueError('No real snapshots')
    online_path=ROOT/'traces/task_fact_extractions.jsonl';frames=[];times=[];selected=set();source_hashes={}
    # Complete extraction before importing/loading any Oracle annotations.
    with online_path.open('w') as out:
        for path in paths:
            source_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            data=pickle.load(gzip.open(path,'rb'));frame=data['frame'];facts=[]
            tick=perf_counter()
            for aid,anchor in data['history'].items():
                if anchor.anchor_type!='object':continue
                facts.extend(extract_task_facts(anchor,frame,data['result'].diagnostics.get('anchors',{}).get(aid,{})))
            wall=(perf_counter()-tick)*1000;times.append(wall)
            payload=dict(frame=frame.frame_id,scene=frame.scene_id,split=data['split'],
                verification_wall_ms=wall,facts=[asdict(f) for f in facts])
            out.write(json.dumps(json_value(payload),allow_nan=False)+'\n')
            frames.append((path,payload));selected.add(frame.frame_id)
            selected.update(a.source_frame_id for a in data['history'].values() if a.source_frame_id)
    from tools.worldstream_m1.dataset import load_oracle
    from worldstream.m1.geometry import predict_geometry
    oracle=load_oracle([infos[t] for t in selected],'data/nuscenes')
    records=[]
    for path,payload in frames:
        data=pickle.load(gzip.open(path,'rb'));frame=data['frame'];history=data['history'];pose=np.linalg.inv(frame.ego_to_world)
        current={a['instance_id']:a for a in oracle[frame.frame_id]}
        for fact in payload['facts']:
            anchor=history[fact['anchor_id']];labels=oracle.get(anchor.source_frame_id,[])
            eligible=[a for a in labels if a['class']==anchor.semantic_class and a['has_sensor_support']
                and np.linalg.norm(anchor.geometry_world.center-a['geometry'].center)<=2.]
            identity=None
            if len(eligible)==1:
                source=eligible[0]
                competitors=[a for a in history.values() if a.anchor_type=='object' and a.source_frame_id==anchor.source_frame_id
                    and a.semantic_class==anchor.semantic_class and np.linalg.norm(a.geometry_world.center-source['geometry'].center)<=2.]
                if len(competitors)==1:identity=source['instance_id']
            ann=current.get(identity);supported=bool(ann and ann['has_sensor_support'] and str(ann['visibility'])=='4')
            correct=None;spatial_error=None;motion_error=None
            if identity is not None:
                correct=False
                if supported:
                    point=(pose@np.r_[ann['geometry'].center,1.])[:3]
                    predicted=predict_geometry(anchor,frame.timestamp).center
                    spatial_error=float(np.linalg.norm(predicted-ann['geometry'].center))
                    kind=fact['fact_type'];value=fact['fact_value']
                    if kind=='object_persistence':correct=value is True
                    elif kind=='relative_direction' and value is not None:
                        correct=bool(point[0]!=0 and point[1]!=0 and value==dict(longitudinal=int(np.sign(point[0])),lateral=int(np.sign(point[1]))))
                    elif kind=='metric_distance' and value is not None:
                        correct=abs(value-float(np.linalg.norm(point)))<=protocol['metric_distance_error_m_max']
                    elif kind=='motion_relation' and value is not None:
                        age=frame.timestamp-anchor.source_timestamp
                        if age>0:
                            average=(ann['geometry'].center-eligible[0]['geometry'].center)/age
                            motion_error=float(np.linalg.norm(np.asarray(value)-pose[:3,:3]@average))
                            correct=motion_error<=protocol['motion_vector_error_mps_max']
            records.append(dict(frame=frame.frame_id,scene=frame.scene_id,split=payload['split'],fact_id=fact['fact_id'],
                fact_type=fact['fact_type'],accepted=fact['certified'],oracle_valid=correct,
                rejection_reason=fact['rejection_reason'],spatial_error_m=spatial_error,motion_error_mps=motion_error,
                offline_identity_known=identity is not None))
    def summarize(rows):
        accepted=[r for r in rows if r['accepted']];scorable=[r for r in rows if r['oracle_valid'] is not None]
        scored_accept=[r for r in accepted if r['oracle_valid'] is not None];errors=sum(not r['oracle_valid'] for r in scored_accept)
        valid=sum(bool(r['oracle_valid']) for r in scorable);recovered=sum(bool(r['oracle_valid']) for r in scored_accept)
        scenes=len({r['scene'] for r in scored_accept});upper=upper95(errors,len(scored_accept))
        return dict(trials=len(rows),accepted=len(accepted),accepted_scorable=len(scored_accept),scorable=len(scorable),
            fact_acceptance=len(accepted)/len(rows) if rows else None,fact_accuracy=recovered/len(scored_accept) if scored_accept else None,
            valid_fact_recovery=recovered/valid if valid else None,oracle_valid_candidate_values=valid,
            physical_false_acceptance=errors/len(scored_accept) if scored_accept else None,physical_false_acceptance_upper95=upper,
            independent_accepted_scene_support=scenes,
            spatial_error_m_all_scorable=distribution([r['spatial_error_m'] for r in scorable if r['spatial_error_m'] is not None]),
            motion_error_mps_all_scorable=distribution([r['motion_error_mps'] for r in scorable if r['motion_error_mps'] is not None]),
            physical_quality_pass=bool(len(scored_accept)>=59 and scenes>=3 and upper is not None and upper<.05),
            rejection_reasons=dict(Counter(r['rejection_reason'] for r in rows)))
    results=dict(status='completed_independent_offline_diagnostic',online_gt_used=False,
        frames=len(frames),scenes=len({p['scene'] for _,p in frames}),source_snapshots_sha256=source_hashes,
        annotation_free_extractions_sha256=hashlib.sha256(online_path.read_bytes()).hexdigest(),
        overall=summarize(records),by_fact_type={k:summarize([r for r in records if r['fact_type']==k]) for k in sorted({r['fact_type'] for r in records})},
        by_split={s:summarize([r for r in records if r['split']==s]) for s in ('calibration','heldout','final_test')},
        verification_wall_ms_per_frame=distribution(times),
        task_fact_coverage=0.,coverage_scope='no certified fact; no mapping from arbitrary partial anchors to original query referents',
        decision='No-Go: available evidence cannot independently certify nonzero task fact coverage',
        limitations=['Conservative unique 2m source assignment is an offline association, not guaranteed identity truth',
            'Correlated frames/objects: binomial bounds descriptive; scene support additionally required',
            'Spatial errors report candidate values, not accepted-fact accuracy',
            'Motion oracle uses source-current average velocity, not instantaneous velocity',
            'Extractor explicitly exposes missing evidence; this is a feasibility diagnostic, not a new geometry estimator'])
    write(ROOT/'task_fact_results.json',results)
    write(ROOT/'task_fact_risk_coverage.json',dict(status=results['status'],protocol=protocol,
        overall=results['overall'],by_split=results['by_split'],by_fact_type=results['by_fact_type'],
        selected_candidate=None,independent_quality_pass=False,decision=results['decision']))
    with (ROOT/'traces/task_fact_oracle.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    print(json.dumps(dict(frames=results['frames'],overall=results['overall'],decision=results['decision']),indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    freeze() if args.freeze else main()
