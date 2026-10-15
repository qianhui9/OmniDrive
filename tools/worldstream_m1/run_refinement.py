"""Frozen multi-scene protocol: historical detection replay, offline labels only after precheck."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import pickle
from time import perf_counter
import cv2
import numpy as np
from worldstream.m1 import M1Config, WorldStateManager, WorldAnchor
from worldstream.m1.geometry import project_anchor, region_slices
from .dataset import frame_from_info, load_oracle, prediction_outputs
from .evaluate import oracle_assignment
from .io_utils import read_prediction_subset, write_json
from .refinement_common import summarize_rows, physical_details, ratio

METHODS={
 'old_strict': dict(temporal_visibility=False,local_support=False),
 'old_appearance_ablation':dict(temporal_visibility=False,local_support=False,require_visibility_evidence=False),
 'without_temporal_visibility':dict(temporal_visibility=False,local_support=True,require_visibility_evidence=False),
 'without_local_support':dict(temporal_visibility=True,local_support=False),
 'new_full':dict(),
 'new_full_tight_patch':dict(support_residual_max=.02),
 'new_full_loose_patch':dict(support_residual_max=.06),
}


def run_scene(root, frames, predictions, oracle, config, method, split):
    path=root/method
    if (path/'summary.json').exists(): return json.loads((path/'summary.json').read_text())
    path.mkdir(parents=True,exist_ok=True)
    manager=WorldStateManager(config); identities={}; source_anns={}; previous_visible={}
    rows=[]; pixels=[]; timings=[]; new_regions=[]; association=Counter(); previous_pose=None
    camera_diagnostics=[]
    for index, frame in enumerate(frames):
        result=manager.precheck(frame)  # Never current outputs or oracle arguments.
        anns=oracle[frame.frame_id]; by_id={a['instance_id']:a for a in anns}
        if previous_pose is None: ego_translation=ego_rotation=0.
        else:
            delta=np.linalg.inv(previous_pose)@frame.ego_to_world
            ego_translation=float(np.linalg.norm(delta[:3,3])); ego_rotation=float(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))
        for aid,check in result.diagnostics['anchors'].items():
            anchor=manager.store.anchors[aid]; eligible=[v for v in check['cameras'].values() if v['geometry_valid']]
            fixed=[manager.validator.residual(obs.descriptor,manager.validator.descriptor(frame.cameras[cid].image,obs.region))
                   for cid,obs in anchor.observation_support.items() if cid in frame.cameras]
            detail=physical_details(anchor,anns,identities.get(aid),source_anns.get(aid),frame.timestamp)
            rows.append(dict(scene_id=frame.scene_id,frame_id=frame.frame_id,anchor_id=aid,identity=identities.get(aid),
                method=method,split=split,age_s=check['age_s'],semantic_class=anchor.semantic_class,accepted=check['valid'],
                geometry_only=bool(check['within_horizon'] and check['motion_known'] and eligible),
                appearance_only=bool(fixed and max(fixed)<=config.residual_threshold),
                appearance_residual=max(fixed,default=float('inf')),
                residual=max((v['residual'] for v in eligible),default=float('inf')),
                joint_without_residual=bool(check['within_horizon'] and check['motion_known'] and eligible
                    and 'calibration_or_camera_timing_unknown' not in check['reasons']
                    and all(v['visibility_and_coverage_valid'] for v in eligible)),
                observation_consistent=bool(eligible and all(v['observation_consistent'] for v in eligible)),
                reasons=check['reasons'],ego_translation_m=ego_translation,ego_rotation_rad=ego_rotation,**detail))
        for cid,cam in frame.cameras.items():
            regions=result.invalidation_regions[cid]
            pixels.append(dict(frame_id=frame.frame_id,camera=cid,pixels=cam.image.shape[0]*cam.image.shape[1],
                **{k:regions[k] for k in ('verified_stable_area_ratio','changed_area_ratio','unknown_area_ratio','invalidation_area_ratio','refresh_override')}))
            if cid in manager.last_evidence:
                camera_diagnostics.append(dict(frame_id=frame.frame_id,camera=cid,**manager.last_evidence[cid].diagnostics))
            visible=set()
            for ann in anns:
                a=WorldAnchor('oracle',frame.scene_id,'object',ann['class'],ann['geometry'],frame.timestamp)
                p=project_anchor(a,cam,config)
                if not p.valid: continue
                visible.add(ann['instance_id'])
                if index and ann['instance_id'] not in previous_visible.get(cid,set()):
                    coverage=float(manager.last_masks[cid][region_slices(p.region,cam.image.shape)].mean())
                    new_regions.append(dict(frame_id=frame.frame_id,camera=cid,instance=ann['instance_id'],coverage=coverage,recalled=coverage>=.9))
            previous_visible[cid]=visible
        outputs=prediction_outputs(predictions[frame.frame_id],config)
        current=[WorldAnchor(str(j),frame.scene_id,p.anchor_type,p.semantic_class,p.geometry_world,frame.timestamp) for j,p in enumerate(outputs.anchors)]
        assignments=oracle_assignment(current,anns)
        manager.commit(frame,outputs,result)
        for aid,j in manager.last_commit_diagnostics['association_pairs']:
            association['proposed']+=1
            if identities.get(aid) is not None and assignments.get(j) is not None:
                association['scorable']+=1; association['correct']+=int(identities[aid]==assignments[j])
        for j,aid in enumerate(manager.last_commit_diagnostics['committed_anchor_ids']):
            identities[aid]=assignments.get(j); source_anns[aid]=by_id.get(assignments.get(j))
        timings.append(dict(result.diagnostics['timings_ms'],**manager.last_commit_diagnostics,
            m1_overall_wall_ms=result.diagnostics['timings_ms']['precheck_total']+manager.last_commit_diagnostics['commit_total_ms']))
        # Preserve audit records; large exact masks only for full method and original modes.
        from worldstream.m1.state_types import json_value
        with (path/'state.jsonl').open('a') as handle:
            handle.write(json.dumps(json_value(dict(precheck=result.to_dict(),commit=manager.last_commit_diagnostics)),allow_nan=False)+'\n')
        if method=='new_full':
            masks={cid+'_invalid':mask for cid,mask in manager.last_masks.items()}
            for cid,e in manager.last_evidence.items(): masks.update({cid+'_stable':e.stable_mask,cid+'_changed':e.changed_mask})
            (path/'masks').mkdir(exist_ok=True)
            np.savez_compressed(path/'masks'/f'{index:03d}.npz',**masks)
        previous_pose=frame.ego_to_world
    write_json(path/'oracle_records.json',rows); write_json(path/'pixels.json',pixels)
    write_json(path/'new_regions.json',new_regions); write_json(path/'timings.json',timings)
    write_json(path/'visibility.json',camera_diagnostics)
    summary=summarize_rows(rows)
    summary.update(scene_id=frames[0].scene_id,method=method,split=split,frames=len(frames),
        invalidation_area_ratio=float(np.mean([p['invalidation_area_ratio'] for p in pixels])),
        stable_area_ratio=float(np.mean([p['verified_stable_area_ratio'] for p in pixels])),
        changed_area_ratio=float(np.mean([p['changed_area_ratio'] for p in pixels])),
        unknown_area_ratio=float(np.mean([p['unknown_area_ratio'] for p in pixels])),
        new_regions=len(new_regions),new_region_recall=ratio(sum(r['recalled'] for r in new_regions),len(new_regions)),
        new_region_coverage=float(np.mean([r['coverage'] for r in new_regions])) if new_regions else None,
        association=dict(association),semantic_change_fpr=None,stable_false_invalidation_rate=None,
        negative_label_limitation='No reliable pixel-level static/changed ground truth; unlabelled background is not negative.',
        m1_wall_p50_ms=float(np.median([t['m1_overall_wall_ms'] for t in timings])))
    write_json(path/'summary.json',summary); print(root.name,method,json.dumps(summary),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/module1_refinement')
    p.add_argument('--frames',type=int,default=12)
    p.add_argument('--scene-indices',default='0,1,4,5,6,7,8,9,10,11')
    p.add_argument('--methods',default=','.join(METHODS))
    p.add_argument('--predictions',default='test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json')
    args=p.parse_args(); cv2.setNumThreads(1)
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    infos=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']
    scenes=sorted({i['scene_token'] for i in infos})
    specs=[]
    for idx in map(int,args.scene_indices.split(',')):
        chosen=sorted([i for i in infos if i['scene_token']==scenes[idx]],key=lambda i:i['timestamp'])[:args.frames]
        specs.append((f'scene_{idx:02d}',chosen,'calibration' if idx>=8 else 'evaluation'))
    if 0 in list(map(int,args.scene_indices.split(','))):
        specs.append(('scene_00_interval_1s',sorted([i for i in infos if i['scene_token']==scenes[0]],key=lambda i:i['timestamp'])[::2][:args.frames],'evaluation'))
    all_infos=[i for _,v,_ in specs for i in v]
    # Frozen before observing round-two test outcomes. No test-selected thresholds.
    protocol=dict(args=vars(args),methods=METHODS,config=M1Config.load('configs/worldstream_m1_refinement.yaml').to_dict(),
        scene_splits={name:dict(scene_id=v[0]['scene_token'],split=s,samples=[i['token'] for i in v],timestamps=[i['timestamp']/1e6 for i in v]) for name,v,s in specs},
        calibration_note='Reserved val prediction scenes 8..11, disjoint from evaluated scenes; not official held-out nuScenes test.',
        source='Historical Full-Model Prediction Replay',new_gpu_inference=False,
        risk_thresholds=[.06,.10,.14,.18,.24],target_risk=.05,min_independent_calibration_scenes=10)
    write_json(root/'protocol.json',protocol)
    oracle=load_oracle(all_infos,'data/nuscenes')
    predictions=read_prediction_subset(args.predictions,{i['token'] for i in all_infos})
    for name,chosen,split in specs:
        if any(i['token'] not in predictions for i in chosen): raise ValueError('Missing historical model output')
        frames=[frame_from_info(i,'data/nuscenes') for i in chosen]
        methods=['new_full'] if split=='calibration' else args.methods.split(',')
        for method in methods:
            cfg=replace(M1Config.load('configs/worldstream_m1_refinement.yaml'),logging=False,visualization=False,**METHODS[method])
            path=root/'scene_results'/name
            if (path/method/'state.jsonl').exists() and not (path/method/'summary.json').exists():
                raise ValueError('Partial run exists; choose fresh output or investigate before rerunning: '+str(path/method))
            run_scene(path,frames,predictions,oracle,cfg,method,split)

if __name__=='__main__': main()
