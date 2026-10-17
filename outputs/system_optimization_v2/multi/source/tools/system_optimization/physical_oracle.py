"""Independent annotation oracle, applied OFFLINE to captured actual M1 inputs.

No annotation object reaches a model, worker, precheck, cache or online query.
Reuses the pre-existing frozen 2m source / 1m current physical protocol.
"""
import argparse,gzip,hashlib,json,pickle
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from worldstream.m1 import M1Config,WorldAnchor
from worldstream.m1.geometry import project_anchor,region_slices
from tools.worldstream_m1.dataset import load_oracle
from tools.worldstream_m1.evaluate import oracle_assignment
from tools.worldstream_m1.refinement_common import physical_details,summarize_rows,interval
from .identity import token_index,validate_sequence
from .audit import write


def main(root='outputs/system_optimization_v2'):
    root=Path(root)
    paths=list(root.glob('*/oracle_frames/*.pkl.gz'))
    if not paths:raise ValueError('No actual preprocessed M1 snapshots; refusing fabricated oracle output')
    raw=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']
    infos=token_index(raw);selected={p.name[:-7] for p in paths}
    oracle=load_oracle([infos[t] for t in selected],'data/nuscenes')
    config=M1Config.load('configs/worldstream_m1_coverage.yaml')
    records=[];events=[];funnel=Counter();reasons=Counter();seen=set();previous={}
    for path in sorted(paths,key=lambda p:(infos[p.name[:-7]]['scene_token'],infos[p.name[:-7]]['timestamp'],str(p))):
        data=pickle.load(gzip.open(path,'rb'));frame=data['frame'];result=data['result']
        key=(frame.scene_id,frame.frame_id)
        if key in seen:continue
        seen.add(key)
        info=infos[frame.frame_id]
        if info['scene_token']!=frame.scene_id or abs(info['timestamp']/1e6-frame.timestamp)>1e-5:
            raise ValueError('Oracle snapshot identity mismatch')
        current=oracle[frame.frame_id]
        groups=defaultdict(list)
        for aid,anchor in data['history'].items():groups[anchor.source_frame_id].append(anchor)
        identities={};source_anns={}
        for source,anchors in groups.items():
            source_labels=oracle.get(source,[])
            matched=oracle_assignment(anchors,source_labels)
            by_id={a['instance_id']:a for a in source_labels}
            for i,anchor in enumerate(anchors):
                identities[anchor.anchor_id]=matched.get(i)
                source_anns[anchor.anchor_id]=by_id.get(matched.get(i))
        for aid,check in result.diagnostics.get('anchors',{}).items():
            anchor=data['history'].get(aid)
            if anchor is None:raise ValueError('Precheck anchor missing from actual historical snapshot')
            camera=list(check.get('cameras',{}).values())
            funnel['historical_anchor_trials']+=1;reasons.update(check.get('reasons',[]))
            eligible=[c for c in camera if c.get('geometry_valid')]
            geometry=bool(eligible)
            temporal=geometry and bool(check.get('within_horizon') and check.get('motion_known')) and 'calibration_or_camera_timing_unknown' not in check.get('reasons',[])
            # Match the actual manager's all-eligible-view rule. One good
            # camera cannot override contradictory evidence in another view.
            observation=temporal and all(c.get('observation_consistent') for c in eligible)
            visibility=observation and all(c.get('visibility_and_coverage_valid') for c in eligible)
            source=visibility and not set(check.get('reasons',[])).intersection(('source_score_insufficient','history_insufficient','history_innovation_unverified'))
            funnel['geometrically_decidable']+=geometry;funnel['temporal_motion_calibration_supported']+=temporal
            funnel['observation_supported']+=observation;funnel['visibility_supported']+=visibility
            funnel['source_history_quality_supported']+=source;funnel['finally_verified']+=bool(check.get('valid'))
            records.append(dict(scene=frame.scene_id,frame=frame.frame_id,anchor=aid,accepted=bool(check.get('valid')),
                split=data['split'],reasons=check.get('reasons',[]),
                **physical_details(anchor,current,identities.get(aid),source_anns.get(aid),frame.timestamp)))
        old=previous.get(frame.scene_id)
        if old is not None:
            old_frame,old_annotations=old
            for cid,cam in frame.cameras.items():
                old_camera=old_frame.cameras[cid];old_ids={}
                for ann in old_annotations:
                    anchor=WorldAnchor('oracle',frame.scene_id,'object',ann['class'],ann['geometry'],old_frame.timestamp)
                    if ann['has_sensor_support'] and str(ann['visibility'])=='4' and project_anchor(anchor,old_camera,config).valid:
                        old_ids[ann['instance_id']]=ann
                for ann in current:
                    if not ann['has_sensor_support']:continue
                    anchor=WorldAnchor('oracle',frame.scene_id,'object',ann['class'],ann['geometry'],frame.timestamp)
                    projected=project_anchor(anchor,cam,config)
                    if not projected.valid:continue
                    prior=next((a for a in old_annotations if a['instance_id']==ann['instance_id']),None)
                    kinds=[]
                    if ann['instance_id'] not in old_ids and str(ann['visibility'])=='4':kinds.append('newly_visible')
                    if prior and np.linalg.norm(prior['geometry'].center-ann['geometry'].center)>.25:kinds.append('moving_object')
                    if prior and int(ann['visibility'])<int(prior['visibility']):kinds.append('visibility_drop')
                    if not kinds:continue
                    coverage=float(data['masks'][cid][region_slices(projected.region,cam.image.shape)].mean())
                    events.append(dict(scene=frame.scene_id,frame=frame.frame_id,camera=cid,instance=ann['instance_id'],
                        kinds=kinds,invalidation_coverage=coverage,recalled=coverage>=.9))
        previous[frame.scene_id]=(frame,current)
    summary=summarize_rows(records)
    by_event={}
    for kind in ('newly_visible','moving_object','visibility_drop'):
        chosen=[e for e in events if kind in e['kinds']];hits=sum(e['recalled'] for e in chosen)
        by_event[kind]=dict(events=len(chosen),recalled=hits,recall=hits/len(chosen) if chosen else None,
            descriptive_binomial_95ci=interval(hits,len(chosen)))
    result=dict(evidence='new offline evaluation of actual preprocessed M1 snapshots and independent annotation tables',
        unique_frames=len(seen),scenes=len({s for s,t in seen}),physical=summary,critical_region_events=by_event,
        critical_change_recall=sum(e['recalled'] for e in events)/len(events) if events else None,
        task_consumable_state_recall=None,
        task_recall_reason='Annotation object identities do not certify completeness for original global Planning/count questions; no synthetic correct answers',
        motion_occlusion_failures=dict(
            accepted_dynamic_position_errors=sum(r['accepted'] and r['center_error_m'] is not None and r['center_error_m']>1 and r['gt_speed_mps'] is not None and r['gt_speed_mps']>=.5 for r in records),
            accepted_insufficient_visibility=sum(r['accepted'] and r['identity_known'] and r['visibility'] is not None and r['visibility']!='4' for r in records),
            scope='Independent current position/visibility failures on accepted histories; association with motion is not proof of its causal origin'),
        physical_records=records,critical_records=events,
        limitations=['2m predicted-source assignment can be ambiguous; unmatched identities remain unscorable',
            'Map anchors have no independent map identity/physical oracle in this annotation protocol and remain unscorable',
            'Projected annotation regions are not per-pixel occlusion truth or closed-loop safety',
            'Binomial intervals are descriptive; repeated objects/views are correlated. Scene summaries are provided'],
        by_scene={s:summarize_rows([r for r in records if r['scene']==s]) for s in sorted({s for s,t in seen})})
    write(root/'physical_oracle_results.json',result)
    write(root/'state_yield_results.json',dict(funnel=dict(funnel),rejection_reasons=dict(reasons),
        exact_m1_rules_unchanged=True,task_query_hit_rate='see actual M3 request diagnostics in formal frames',
        online_ground_truth_used=False))
    print('Physical oracle:',len(seen),'frames,',len(records),'anchor trials,',len(events),'critical region events')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/system_optimization_v2');main(p.parse_args().root)
