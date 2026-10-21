"""Offline conditional orientation, motion and timestamp error-source diagnostics."""
from collections import defaultdict
import json
from pathlib import Path
import pickle
import numpy as np
from .dataset import load_oracle,prediction_outputs
from .io_utils import read_prediction_subset,write_json
from .independent_refinement import distribution
from worldstream.m1 import M1Config


def axial_heading(geometry):
    size=np.asarray(geometry.size)
    if size.shape!=(3,) or abs(size[0]-size[1])<.01:return None
    edges=geometry.points[1:]-geometry.points[0]
    lengths=np.linalg.norm(edges,axis=1)
    candidates=edges[np.abs(lengths-size[0])<.005]
    if not len(candidates):return None
    unit=candidates/np.linalg.norm(candidates,axis=1,keepdims=True)
    if any(abs(np.dot(unit[0],v))<.999 for v in unit[1:]):return None
    xy=unit[0,:2]
    return xy/np.linalg.norm(xy) if np.linalg.norm(xy)>.1 else None


def main():
    root=Path('outputs/module1_evidence_optimization');lookup={};outputs={}
    needed=set();records={};assoc={}
    for stage in ('development','calibration','heldout','final_test'):
        records[stage]=sum((json.loads(p.read_text()) for p in (root/'traces'/stage).glob('*/population.json')),[])
        assoc[stage]=sum((json.loads(p.read_text()) for p in (root/'traces'/stage).glob('*/association.json')),[])
        needed.update(r['frame'] for r in records[stage])
    for file in ['data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl']:
        data=pickle.load(open(file,'rb'))['infos'];lookup.update({i['token']:i for i in data if i['token'] in needed});del data
    oracle=load_oracle(list(lookup.values()),'data/nuscenes')
    with (root/'model_observations.pkl').open('rb') as handle:outputs.update(pickle.load(handle))
    old_needed={r['frame'] for r in records['development']}
    old=read_prediction_subset('test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json',old_needed)
    outputs.update({token:prediction_outputs(pred,M1Config()) for token,pred in old.items()})
    result={};samples=[]
    for stage,rows in records.items():
        yaw=[];yaw_unknown=0;size=[];motion=[];extrap=[];static_motion=[];switch=[]
        pop={(r['frame'],r['anchor']):r for r in rows}
        for r in rows:
            if r['center_error_m'] is None:continue
            gt=next((a for a in oracle[r['frame']] if a['instance_id']==r['nearest_reference_only']),None)
            if gt is None:continue
            g=outputs[r['frame']].anchors[r['prediction']].geometry_world
            a,b=axial_heading(g),axial_heading(gt['geometry'])
            if a is None or b is None:yaw_unknown+=1
            else:yaw.append(float(np.degrees(np.arccos(np.clip(abs(np.dot(a,b)),-1,1)))))
            size.append(float(np.linalg.norm(g.size-gt['geometry'].size)))
        for r in assoc[stage]:
            if not r['source_instance']:continue
            source=next((a for a in oracle[r['source_frame']] if a['instance_id']==r['source_instance']),None)
            current=next((a for a in oracle[r['frame']] if a['instance_id']==r['source_instance']),None)
            if source is None or current is None:continue
            dt=(lookup[r['frame']]['timestamp']-lookup[r['source_frame']]['timestamp'])/1e6
            displacement=float(np.linalg.norm(current['geometry'].center-source['geometry'].center))
            error=float(np.linalg.norm(np.asarray(r['source_center'])-current['geometry'].center))
            motion.append(displacement/dt if dt>0 else np.nan);extrap.append(error)
            oldpop=pop.get((r['source_frame'],r['anchor']))
            if oldpop and oldpop['semantic_class'] in ('traffic_cone','barrier'):static_motion.append(displacement)
            if r['correct'] is False:switch.append(dict(scene=r['scene'],frame=r['frame'],source_frame=r['source_frame'],
                source_instance=r['source_instance'],wrong_current_instance=r['current_instance'],
                actual_source_instance_displacement_m=displacement,historical_current_centre_error_m=error,
                local_margin_m=r['local_margin_m'],global_cost_margin_m=r['global_cost_margin_m'],cardinality_lost=r['cardinality_lost']))
        result[stage]=dict(axial_box_orientation_error_deg=distribution(yaw),orientation_symmetry_or_axis_unscorable=yaw_unknown,
            size_error_norm_m=distribution(size),source_instance_motion_mps=distribution(motion),
            historical_static_prediction_current_centre_error_m=distribution(extrap),
            static_class_annotation_displacement_m=distribution(static_motion),
            static_prior_displacement_above_10cm=sum(v>.1 for v in static_motion),
            condition='Unique offline 2m source identities only; box orientation modulo 180deg, symmetric boxes unscorable',
            switch_examples=switch)
    skews=[abs(c['timestamp']/1e6-i['timestamp']/1e6) for i in lookup.values() for c in i['cams'].values()]
    write_json(root/'geometry_error_sources.json',dict(splits=result,camera_timestamp_absolute_skew_s=distribution(skews),
        source_pose_true_error='Not identifiable: input pose has no independent physical reference',
        calibration_true_error='Not identifiable: shared nuScenes calibration is a contract, not an error bound',
        uncensored_nearest_distance_is_not_true_identity_error=True,
        online_use=False,posthoc_diagnostic_only_no_threshold_or_method_changes=True))
    print('conditional geometry/motion error-source diagnostics complete',flush=True)


if __name__=='__main__':main()
