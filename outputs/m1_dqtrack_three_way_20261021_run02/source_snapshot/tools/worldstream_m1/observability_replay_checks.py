"""Actual nuScenes ROI mask audit and independently replayed ablation check."""
from dataclasses import replace
from pathlib import Path
import json
import cv2
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1 import surface_visibility_constraints as constraints
from worldstream.m1.mask_codec import decode_mask,encode_mask
from worldstream.m1.geometry import project_anchor,region_slices
from worldstream.m1.dense_support import swept_dynamic_region
from .evidence_experiments import load_inputs,append_json
from .dataset import frame_from_info
from .coverage_oracle import world_lidar,prepare_pair,score_camera
from .io_utils import write_json
ROOT=Path('outputs/module1_physical_observability_optimization')


def main():
    cv2.setNumThreads(1);specs,infos,outputs,oracle=load_inputs('calibration')
    old_events=json.loads((ROOT/'critical_roi_diagnosis.json').read_text())['events']
    roi_scenes={r['scene'] for r in old_events};support_scene='c0d3796765e2452d990309b2568adac5'
    old_cfg=replace(M1Config.load('configs/worldstream_m1_independent.yaml'),dense_region_policy='surface_consensus',dense_guard_radius_px=2,surface_boundary_guard=False)
    cfg=M1Config.load('configs/worldstream_m1_physical_observability.yaml')
    actual_apply=constraints.apply_surface_constraints
    def ablation(*args):
        actual_apply(*args);e=args[2]
        e.stable_mask=constraints.candidate_masks(e.constraint_measurements)['constraints_two_view_only']
        e.stable_candidate_mask=e.stable_mask.copy()
    pairs=[];compared=0;differences=0
    for spec in specs:
        if spec['scene_id'] not in roi_scenes|{support_scene}:continue
        expected={}
        for r in map(json.loads,(ROOT/'traces/development'/spec['scene_id']/'pixels.jsonl').read_text().splitlines()):
            if r['method'] in ('surface_constraints','constraints_two_view_only'):expected[r['method'],r['frame'],r['camera']]=r
        managers={}
        if spec['scene_id'] in roi_scenes:managers.update(surface_consensus=WorldStateManager(old_cfg),surface_constraints=WorldStateManager(cfg))
        if spec['scene_id']==support_scene:managers['constraints_two_view_only']=WorldStateManager(cfg)
        previous=None;oldpoints=None
        for token in spec['samples']:
            frame=frame_from_info(infos[token],'data/nuscenes');points=world_lidar(infos[token])
            refs=prepare_pair(previous,frame,oldpoints,points,oracle[previous.frame_id],oracle[token],cfg) if previous else None
            results={}
            for name,m in managers.items():
                constraints.apply_surface_constraints=ablation if name=='constraints_two_view_only' else actual_apply
                try:r=m.precheck(frame)
                finally:constraints.apply_surface_constraints=actual_apply
                if name!='surface_consensus':
                    for cid,e in m.last_evidence.items():
                        old=expected[name,token,cid];meta=r.invalidation_regions[cid]
                        compared+=1;differences+=old['effective_pixels']!=int((~m.last_masks[cid]).sum()) or old['stable_pixels']!=int(e.stable_mask.sum())
                        assert np.array_equal(decode_mask(meta['exact_invalidation_mask']),m.last_masks[cid])
                results[name]=r
            for old in old_events:
                if old['frame']!=token:continue
                cid=old['camera'];cam=frame.cameras[cid];m=managers['surface_consensus'];n=managers['surface_constraints']
                ref=dict(refs[cid]);ref['scorable']&=ref['annotated']
                score=score_camera(ref,n.last_masks[cid],n.last_evidence[cid],cfg)
                event=next(e for e in score['events'] if e['instance']==old['event']['instance'])
                sl=region_slices(event['region'],cam.image.shape);guard=np.zeros(cam.image.shape[:2],bool)
                # Precheck prunes expired sources on its private fork. Old
                # public-store entries must not fabricate an active guard.
                for a in m._pending_store.anchors.values():
                    if a.anchor_type!='object' or a.semantic_class in ('barrier','traffic_cone'):continue
                    p=project_anchor(a,cam,old_cfg)
                    if p.region is not None:guard[region_slices(swept_dynamic_region(a,cam,p,old_cfg),cam.image.shape)]=True
                assert not (guard&m.last_evidence[cid].stable_mask).any()
                tile=np.zeros_like(guard)
                for box in results['surface_consensus'].invalidation_regions[cid]['regions']:tile[region_slices(box,cam.image.shape)]=True
                assert (tile|~m.last_masks[cid]).all()
                record=dict(scene=frame.scene_id,frame=token,camera=cid,event=event,old_exact_refresh=old['event']['invalidation_coverage'],
                    new_exact_refresh=event['invalidation_coverage'],old_tile_envelope_refresh=float(tile[sl].mean()),
                    predicted_dynamic_sweep_coverage=float(guard[sl].mean()),
                    old_stable_pixels_in_roi=int((~m.last_masks[cid][sl]).sum()),new_stable_pixels_in_roi=int((~n.last_masks[cid][sl]).sum()),
                    masks={name:encode_mask(manager.last_masks[cid]) for name,manager in [('old',m),('new',n)]},
                    source='Actual RGB/pose/original cached predictions. Annotation ROI scored after precheck only.',
                    mapping_bug_found=False,physical_root_cause='Unknown_Root_Cause',GT_online=False)
                pairs.append(record)
                panels=[]
                for name,manager in [('old',m),('new',n)]:
                    image=cam.image.copy();stable=~manager.last_masks[cid]
                    image[stable]=(.35*image[stable]+.65*np.array([0,220,0])).astype(np.uint8)
                    x0,y0,x1,y1=np.rint(event['region']).astype(int);cv2.rectangle(image,(x0,y0),(x1,y1),(255,40,40),1)
                    cv2.putText(image,name+' exact stable (green); offline event ROI (red)',(8,24),0,.42,(255,255,255),1);panels.append(image)
                path=ROOT/'visualizations/roi_before_after'/f'{token}_{cid}.png';path.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(path),cv2.cvtColor(np.concatenate(panels,axis=1),cv2.COLOR_RGB2BGR));record['visualization']=str(path)
            for name,m in managers.items():m.commit(frame,outputs[token],results[name])
            previous=frame;oldpoints=points
        print('ROI/ablation replay',spec['scene_id'],'complete',flush=True)
    assert not differences,(compared,differences)
    write_json(ROOT/'roi_exact_mask_audit.json',dict(events=pairs,mask_to_tile_bug_found=False,GT_used_online=False))
    write_json(ROOT/'independent_ablation_replay_check.json',dict(compared_camera_rows=compared,differences=differences,
        ablation='constraints_two_view_only',ablation_actual_separate_manager=True,
        explanation='Same actual committed images and predictions; regional masks never change original perception or shadow identity association.'))


if __name__=='__main__':main()
