"""Offline independent LiDAR/annotation checks. Never imported by online M1.

LiDAR samples constrain sparse image locations, not all pixels or neural features.
Unannotated points are assumed static, with previous/current LiDAR depth checks.
"""
import cv2
import numpy as np
from worldstream.m1.geometry import project_points, transform_points, region_slices, project_anchor
from worldstream.m1 import WorldAnchor
from worldstream.m1.omnidrive_adapter import quaternion_transform
from .dataset import resolve_image


def world_lidar(info):
    raw=np.fromfile(resolve_image(info['lidar_path'],'data/nuscenes'),dtype=np.float32).reshape(-1,5)[:,:3]
    pose=quaternion_transform(info['ego2global_rotation'],info['ego2global_translation'])
    sensor=quaternion_transform(info['lidar2ego_rotation'],info['lidar2ego_translation'])
    return transform_points(raw,pose@sensor)


def box_basis(g):
    return np.stack([g.points[4]-g.points[0],g.points[2]-g.points[0],g.points[1]-g.points[0]],axis=1)/g.size


def depth_map(points,camera):
    uv,z,front=project_points(points,camera.intrinsic,camera.world_to_camera)
    h,w=camera.image.shape[:2];good=front & np.isfinite(uv).all(1)
    xy=np.rint(np.nan_to_num(uv,nan=-10)).astype(int)
    good&=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    depth=np.full((h,w),np.inf,np.float32)
    np.minimum.at(depth,(xy[good,1],xy[good,0]),z[good])
    depth=cv2.erode(depth,np.ones((3,3),np.uint8))
    return uv,z,xy,good,depth


def prepare_pair(previous,current,previous_points,current_points,previous_anns,current_anns,config):
    """Compute reference correspondence and critical events once, outside timing."""
    prior={a['instance_id']:a for a in previous_anns}
    source=current_points.copy();known=np.ones(len(source),bool);moving=np.zeros(len(source),bool)
    annotation_seen=np.zeros(len(source),bool)
    semantic=np.full(len(source),"unannotated_background",dtype=object)
    for ann in current_anns:
        g=ann['geometry'];basis=box_basis(g);local=(current_points-g.center)@basis
        inside=(np.abs(local)<=g.size/2+.05).all(1)
        annotation_seen|=inside
        semantic[inside]=ann["class"]
        old=prior.get(ann['instance_id'])
        if old is None:
            known[inside]=False;moving[inside]=True
        else:
            source[inside]=local[inside]@box_basis(old['geometry']).T+old['geometry'].center
            moving[inside]=np.linalg.norm(source[inside]-current_points[inside],axis=1)>.1
    cameras={}
    for cid,cam in current.cameras.items():
        oldcam=previous.cameras[cid]
        uv,z,xy,inside,depth=depth_map(current_points,cam)
        vis=inside.copy();ids=np.flatnonzero(inside)
        vis[ids]&=z[ids]<=depth[xy[ids,1],xy[ids,0]]+.5
        olduv,oldz,oldxy,oldinside,_=depth_map(source,oldcam)
        *_,olddepth=depth_map(previous_points,oldcam)
        visible_old=np.zeros(len(source),bool);ids=np.flatnonzero(oldinside)
        ref=olddepth[oldxy[ids,1],oldxy[ids,0]]
        visible_old[ids]=np.isfinite(ref)&(np.abs(oldz[ids]-ref)<=.5)
        scorable=vis & known & visible_old
        # Confirmed disocclusion: an independently measured closer old LiDAR surface.
        disoccluded=np.zeros(len(source),bool)
        disoccluded[ids]=np.isfinite(ref)&(oldz[ids]>ref+1.)
        critical=vis & (moving | ~known | disoccluded)
        events=[]
        oldvisible={}
        for ann in previous_anns:
            p=project_anchor(WorldAnchor('oracle',previous.scene_id,'object',ann['class'],ann['geometry'],previous.timestamp),oldcam,config)
            if p.valid:oldvisible[ann['instance_id']]=ann
        for ann in current_anns:
            p=project_anchor(WorldAnchor('oracle',current.scene_id,'object',ann['class'],ann['geometry'],current.timestamp),cam,config)
            if not p.valid:continue
            old=prior.get(ann['instance_id']);reasons=[]
            if ann['instance_id'] not in oldvisible:reasons.append('newly_visible')
            if old is not None and np.linalg.norm(ann['geometry'].center-old['geometry'].center)>.25:reasons.append('moving_object')
            if old is not None and int(ann['visibility'])<int(old['visibility']):reasons.append('visibility_drop')
            if reasons:events.append(dict(instance=ann['instance_id'],semantic_class=ann['class'],region=p.region,reasons=reasons))
        cameras[cid]=dict(xy=xy,source_uv=olduv,scorable=scorable,critical=critical,moving=moving,
            known=known,events=events,annotated=annotation_seen,depth_m=z,semantic_class=semantic,
            target_uv=uv,visibility_supported=vis,previous_visibility_supported=visible_old)
    return cameras


def sparse_map(evidence,shape,config):
    h,w=shape;mapping=np.full((h,w,2),np.nan,np.float32)
    if evidence is None:return mapping
    rad=config.support_patch_radius_px
    for p,q in zip(evidence.source_points[evidence.supported],evidence.current_points[evidence.supported]):
        x,y=np.rint(q).astype(int);x0,x1=max(0,x-rad),min(w,x+rad+1);y0,y1=max(0,y-rad),min(h,y+rad+1)
        yy,xx=np.mgrid[y0:y1,x0:x1];mapping[y0:y1,x0:x1]=np.stack([xx,yy],axis=-1)+p-q
    return mapping


def score_camera(reference,invalid,evidence,config):
    xy=reference['xy'];idx=np.flatnonzero(reference['scorable']);critical=np.flatnonzero(reference['critical'])
    accepted=~invalid[xy[idx,1],xy[idx,0]]
    chosen=idx[accepted]
    source=(evidence.dense_source_map if evidence is not None and evidence.dense_source_map is not None
            else sparse_map(evidence,invalid.shape,config))
    predicted=source[xy[chosen,1],xy[chosen,0]]
    error=np.linalg.norm(predicted-reference['source_uv'][chosen],axis=1)
    # Non-finite support mappings remain errors; never improve the denominator by dropping them.
    false=(~np.isfinite(error))|(error>2.)|reference['moving'][chosen]
    missed=(~invalid[xy[critical,1],xy[critical,0]]) if len(critical) else np.empty(0,bool)
    events=[]
    for event in reference['events']:
        coverage=float(invalid[region_slices(event['region'],invalid.shape)].mean())
        events.append(dict(event,invalidation_coverage=coverage,recalled=coverage>=.9))
    return dict(scorable_points=len(idx),accepted_points=len(chosen),false_points=int(false.sum()),
        accepted_annotated_points=int(reference['annotated'][chosen].sum()),
        accepted_background_points=int((~reference['annotated'][chosen]).sum()),
        critical_points=len(critical),missed_critical_points=int(missed.sum()),
        correspondence_error_sum=float(np.where(np.isfinite(error),error,0).sum()),events=events)
