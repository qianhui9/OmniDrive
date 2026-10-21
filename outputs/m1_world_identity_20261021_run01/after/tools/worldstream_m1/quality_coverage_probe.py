"""Measure the unmodified baseline's censored RGB and dense acceptance gates."""
from collections import defaultdict
from dataclasses import replace
import json
from types import SimpleNamespace
import cv2
import numpy as np

from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m1.association import associate
from worldstream.m1.dense_support import ray_geometry
from worldstream.m1.foreground_correspondence import triangulate_correspondences
from worldstream.m1.geometry import project_anchor
from worldstream.m1.mask_codec import decode_mask
from .dataset import frame_from_info
from .evidence_experiments import append_json, load_inputs
from .quality_coverage_audit import ROOT, breakdown, sha
from .io_utils import write_json


def measure_dense(previous, current, history):
    """Each condition is counted alone and incrementally on the same population."""
    from worldstream.m1.temporal_visibility import gray_image
    a,b=gray_image(previous),gray_image(current); h,w=b.shape
    yy,xx=np.mgrid[:h,:w].astype(np.float32);target=np.stack((xx,yy),-1)
    f=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(a,b,None)
    r=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(b,a,None);source=target+r
    def sample(im,coords=source):
        return cv2.remap(im,coords[...,0],coords[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    fb=np.linalg.norm(r+sample(f),axis=2)
    dx=np.gradient(source,axis=1);dy=np.gradient(source,axis=0)
    jac=dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
    af=a.astype(np.float32)/255;bf=b.astype(np.float32)/255
    grad=np.zeros_like(af)
    for axis in (0,1):
        grad+=np.abs(sample(cv2.Sobel(af,cv2.CV_32F,int(axis==0),int(axis==1),ksize=3)/8)-
                     cv2.Sobel(bf,cv2.CV_32F,int(axis==0),int(axis==1),ksize=3)/8)
    conditions={}
    conditions['valid_flow']=np.isfinite(source).all(2)&(source[...,0]>=4)&(source[...,0]<w-4)&(source[...,1]>=4)&(source[...,1]<h-4)
    conditions['fb_consistency']=np.isfinite(fb)&(fb<=.35)&(jac>=.5)&(jac<=2)
    conditions['photo_and_texture']=(np.abs(sample(af)-bf)<=.02)&(grad<=.04)&(sample(cv2.cornerMinEigenVal(af,7,ksize=3))>=1e-5)&(cv2.cornerMinEigenVal(bf,7,ksize=3)>=1e-5)
    safe=np.isfinite(source).all(2);geometry=np.zeros((h,w),bool)
    geometry[safe]=ray_geometry(previous,current,source[safe],target[safe],.5)
    conditions['ray_geometry']=geometry
    prefix=np.logical_and.reduce(list(conditions.values()))
    conditions['complete_neighbourhood']=cv2.erode(prefix.astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    # The actual surface function is invoked ONLY on the eroded dense candidate
    # population. Standalone three-view failures must use that same population;
    # triangulating millions of already-rejected image pixels adds no evidence.
    safe=conditions['complete_neighbourhood']
    xyz,angle,err,_,valid=triangulate_correspondences(previous,current,SimpleNamespace(source_points=source[safe],current_points=target[safe]))
    adjacent=np.zeros((h,w),bool);adjacent[safe]=valid&(angle>=.5)&(err<=.5)
    conditions['adjacent_observability']=adjacent
    long=np.zeros((h,w),bool);loop=long.copy();ray=long.copy()
    if history is not None:
        older,_,prev_source=history;old=gray_image(older)
        lf=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(old,b,None)
        lr=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(b,old,None);ls=target+lr
        lfb=np.linalg.norm(lr+sample(lf,ls),axis=2)
        lp=np.linalg.norm(ls-sample(prev_source),axis=2)
        ldx=np.gradient(ls,axis=1);ldy=np.gradient(ls,axis=0)
        lj=ldx[...,0]*ldy[...,1]-ldx[...,1]*ldy[...,0]
        _,la,le,_,lv=triangulate_correspondences(older,current,SimpleNamespace(source_points=ls[safe],current_points=target[safe]))
        long[safe]=lv&(la>=.5)&(le<=.5)
        long&=(ls[...,0]>=4)&(ls[...,0]<w-4)&(ls[...,1]>=4)&(ls[...,1]<h-4)&(lfb<=.35)&(lj>=.5)&(lj<=2)
        loop=lp<=.5
        cam=xyz@older.world_to_camera[:3,:3].T+older.world_to_camera[:3,3];uvh=cam@older.intrinsic.T
        with np.errstate(invalid='ignore',divide='ignore'):uv=uvh[:,:2]/uvh[:,2:]
        ray[safe]=(np.linalg.norm(uv-ls[safe],axis=1)<=.5)&(cam[:,2]>.5)
    conditions['long_baseline_match']=long;conditions['trajectory_loop']=loop;conditions['third_ray_reprojection']=ray
    live=np.ones((h,w),bool);records={}
    for name,mask in conditions.items():
        before=int(live.sum());live&=mask
        population=safe if name in ('adjacent_observability','long_baseline_match','trajectory_loop','third_ray_reprojection') else np.ones((h,w),bool)
        records[name]=dict(total_pixels=h*w,standalone_population=int(population.sum()),alone_rejected=int((population&~mask).sum()),incremental_rejected=before-int(live.sum()),retained=int(live.sum()))
    return records


def main():
    cv2.setNumThreads(1)
    if (ROOT/'baseline_probe_complete.json').exists(): raise RuntimeError('Preserve completed probe')
    specs,infos,outputs,oracle=load_inputs('calibration')
    cfg=M1Config.load('configs/worldstream_m1_independent.yaml')
    allrows=[];totals=defaultdict(CounterLike);matches=0
    for spec in specs:
        m=WorldStateManager(replace(cfg,identity_continuity=True));surface=WorldStateManager(replace(cfg,dense_guard_radius_px=2,dense_region_policy='surface_visibility'))
        truth={};previous=None
        from .evidence_oracle import identity_population
        for token in spec['samples']:
            frame=frame_from_info(infos[token],'data/nuscenes');known,_=identity_population(outputs[token].anchors,oracle[token])
            result=m.precheck(frame);sr=surface.precheck(frame)
            old=list(m.store.anchors.values())
            # Reconstruct selection exactly as the manager, then use diagnostics after commit.
            m.commit(frame,outputs[token],result)
            diag=m.last_commit_diagnostics;selected=diag['selected_prediction_indices'];preds=[outputs[token].anchors[j] for j in selected]
            pairs,costs=associate(old,preds,frame.timestamp,cfg)
            for i,j in pairs:
                anchor=old[i]
                if anchor.anchor_type!='object':continue
                raw=selected[j];new=known.get(raw);oldtruth=truth.get(anchor.anchor_id)
                item=dict(scene=frame.scene_id,frame=token,anchor_id=anchor.anchor_id,prediction=raw,
                    correct=(new['instance_id']==oldtruth['instance_id']) if new and oldtruth else None,
                    displacement_m=float(np.linalg.norm(preds[j].geometry_world.center-anchor.geometry_world.center)),
                    legal_competitors=int(np.isfinite(np.delete(costs[i],j)).sum()+np.isfinite(np.delete(costs[:,j],i)).sum()),views=[])
                for cid,cam in frame.cameras.items():
                    obs=anchor.observation_support.get(cid);e=m.last_evidence.get(cid)
                    current=replace(anchor,geometry_world=preds[j].geometry_world,source_timestamp=frame.timestamp,
                                    velocity_world=preds[j].velocity_world,velocity_reliable=preds[j].velocity_reliable)
                    p=project_anchor(current,cam,cfg)
                    if obs is None or e is None or not p.valid:continue
                    a,b=e.source_points,e.current_points;r=np.asarray(obs.region);s=np.asarray(p.region)
                    inside=(a[:,0]>=r[0])&(a[:,0]<r[2])&(a[:,1]>=r[1])&(a[:,1]<r[3])&(b[:,0]>=s[0])&(b[:,0]<s[2])&(b[:,1]>=s[1])&(b[:,1]<s[3])
                    rawmask=e.raw_correspondence_supported if e.raw_correspondence_supported is not None else e.supported
                    good=inside&rawmask&e.correspondence_valid&e.geometry_valid
                    count=int(good.sum());total=int(inside.sum());quadrants=len(np.unique(np.clip(((a[good]-r[:2])/np.maximum(r[2:]-r[:2],1)*2).astype(int),0,1),axis=0)) if count else 0
                    residual=m.validator.residual(obs.descriptor,m.validator.descriptor(cam.image,p.region))
                    reasons=[]
                    if abs(e.source_timestamp-obs.timestamp)>1e-5:reasons.append('source_time_mismatch')
                    if count<cfg.support_min_points:reasons.append('RGB_tracks_missing')
                    if not total or count/total<cfg.support_point_fraction:reasons.append('crop_track_fraction')
                    if quadrants/4<cfg.support_quadrant_fraction:reasons.append('spatial_quadrants')
                    if residual>cfg.residual_threshold:reasons.append('appearance_descriptor')
                    item['views'].append(dict(camera=cid,tracks=count,total=total,quadrants=quadrants,residual=residual,reasons=reasons))
                item['observable_without_population_gates']=bool(item['views']) and all(not v['reasons'] for v in item['views'])
                allrows.append(item);append_json(ROOT/'traces/baseline_identity_probe.jsonl',item)
            for raw,aid in zip(selected,diag['committed_anchor_ids']):
                truth.pop(aid,None)
                if raw in known:truth[aid]=known[raw]
            truth={a:t for a,t in truth.items() if a in m.store.anchors}
            if previous:
                for cid,cam in frame.cameras.items():
                    rec=measure_dense(previous.cameras[cid],cam,surface._surface_history.get(cid))
                    e=surface.last_evidence[cid]
                    rec['dynamic_occlusion_and_changed']=dict(retained=int(decode_mask(sr.invalidation_regions[cid]['observation_stable_mask']).sum()))
                    rec['effective_after_refresh']=dict(retained=int((~surface.last_masks[cid]).sum()))
                    append_json(ROOT/'traces/baseline_surface_funnel.jsonl',dict(scene=frame.scene_id,frame=token,camera=cid,gates=rec,history_available=cid in surface._surface_history,forced_refresh=sr.full_refresh_hint))
                    for gate,counts in rec.items():
                        for k,v in counts.items():totals[gate][k]+=v
            surface.commit(frame,outputs[token],sr)
            previous=frame;matches+=1
        print('baseline probe',spec['scene_id'],matches,flush=True)
    details=json.loads((ROOT/'identity_rejection_diagnostics.json').read_text())
    details['uncensored_RGB_probe']=dict(population=breakdown(allrows),
        outside_half_metre=breakdown([r for r in allrows if r['displacement_m']>.5]),
        observable_outside_half_metre=breakdown([r for r in allrows if r['displacement_m']>.5 and r['observable_without_population_gates']]),
        observable_with_competition=breakdown([r for r in allrows if r['legal_competitors'] and r['observable_without_population_gates']]),
        measured_reasons={reason:breakdown([r for r in allrows if any(reason in v['reasons'] for v in r['views'])]) for reason in ('source_time_mismatch','RGB_tracks_missing','crop_track_fraction','spatial_quadrants','appearance_descriptor')},
        no_common_view=breakdown([r for r in allrows if not r['views']]),
        note='Observable crop tracks are feasibility evidence, not foreground ownership or true identity')
    write_json(ROOT/'identity_rejection_diagnostics.json',details)
    write_json(ROOT/'surface_acceptance_funnel.json',dict(baseline=dict(totals),
        unit='Pixel-camera pairs on non-startup Development frames; standalone rejects overlap, incremental rejects form a funnel',
        final_steps='Dynamic masks/changed and periodic refresh are measured final outputs, not independent acceptors',
        raw='traces/baseline_surface_funnel.jsonl'))
    write_json(ROOT/'baseline_probe_complete.json',dict(frames=matches,core_sources=json.loads((ROOT/'baseline_source_manifest.json').read_text()),
        identity_probe_sha256=sha(ROOT/'traces/baseline_identity_probe.jsonl'),surface_probe_sha256=sha(ROOT/'traces/baseline_surface_funnel.jsonl')))
    with (ROOT/'root_cause_audit.md').open('a') as handle:
        handle.write('\n## Measured uncensored baseline probe\n\n'+json.dumps(details['uncensored_RGB_probe'],ensure_ascii=False,indent=2)+'\n\n## Measured surface funnel\n\n'+json.dumps(dict(totals),indent=2)+'\n')


class CounterLike(defaultdict):
    def __init__(self):super().__init__(int)


if __name__=='__main__': main()
