"""Rejection-only RGB triangulation tied to the actual historical 3D box.

Static epipolar agreement alone leaves depth unobservable. Require bidirectional
tracks, measurable parallax and points inside both source and predicted boxes.
This is same-sensor evidence, never independent foreground or task truth.
"""
import cv2
import numpy as np
from .geometry import valid_projection_camera,predict_geometry,oriented_box_membership


def occlusion_table(projections,camera_ids,config):
    ids=list(projections);result={aid:{} for aid in ids}
    for cid in camera_ids:
        selected=[i for i,aid in enumerate(ids) if projections[aid][cid].region is not None]
        if not selected:
            for aid in ids:result[aid][cid]=False
            continue
        boxes=np.array([projections[ids[i]][cid].region for i in selected],dtype=float)
        depths=np.array([projections[ids[i]][cid].depth for i in selected])
        overlap=np.maximum(0,np.minimum(boxes[:,None,2],boxes[None,:,2])-np.maximum(boxes[:,None,0],boxes[None,:,0]))
        overlap*=np.maximum(0,np.minimum(boxes[:,None,3],boxes[None,:,3])-np.maximum(boxes[:,None,1],boxes[None,:,1]))
        area=np.maximum(0,boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1])
        mask=(depths[None,:]<depths[:,None]-config.occlusion_depth_margin_m)&(overlap>config.occlusion_overlap_fraction*area[:,None])
        np.fill_diagonal(mask,False);values=mask.any(1);lookup={ids[i]:bool(v) for i,v in zip(selected,values)}
        for aid in ids:result[aid][cid]=lookup.get(aid,False)
    return result


def verify_anchor_geometry(anchor,previous,current,evidence,projection,config):
    d=dict(source='same_RGB_bidirectional_triangulation',independent_sensor=False,
           foreground_ownership_certified=False,passed=False,reason='missing_source',points=[])
    if anchor.anchor_type!='object':d['reason']='map_depth_not_certified';return d
    if anchor.semantic_class not in ('barrier','traffic_cone'):
        # Static-scene triangulation is invalid for independently moving objects.
        d['reason']='dynamic_geometry_requires_independent_motion';return d
    old=anchor.observation_support.get(current.camera_id)
    if (previous is None or evidence is None or old is None
            or abs(previous.timestamp-old.timestamp)>1e-5
            or abs(evidence.source_timestamp-old.timestamp)>1e-5
            or not all(valid_projection_camera(c) and c.timing_reliable for c in (previous,current))):return d
    p=np.asarray(evidence.source_points);q=np.asarray(evidence.current_points);r=np.asarray(old.region)
    if not len(p):d['reason']='no_correspondence';return d
    inside=(p[:,0]>=r[0])&(p[:,0]<r[2])&(p[:,1]>=r[1])&(p[:,1]<r[3])
    s=np.asarray(projection.region)
    inside&=(q[:,0]>=s[0])&(q[:,0]<s[2])&(q[:,1]>=s[1])&(q[:,1]<s[3])
    chosen=np.flatnonzero(inside & evidence.correspondence_valid & evidence.geometry_valid & evidence.supported)
    d['available_points']=len(chosen)
    if not len(chosen):d['reason']='no_bidirectional_box_support';return d
    p=p[chosen];q=q[chosen]
    rays0=np.c_[p,np.ones(len(p))]@np.linalg.inv(previous.intrinsic).T
    rays1=np.c_[q,np.ones(len(q))]@np.linalg.inv(current.intrinsic).T
    rays0=rays0@previous.world_to_camera[:3,:3];rays1=rays1@current.world_to_camera[:3,:3]
    cos=np.sum(rays0*rays1,1)/(np.linalg.norm(rays0,axis=1)*np.linalg.norm(rays1,axis=1))
    angles=np.degrees(np.arccos(np.clip(cos,-1,1)))
    homogeneous=cv2.triangulatePoints(previous.intrinsic@previous.world_to_camera[:3],
        current.intrinsic@current.world_to_camera[:3],p.T.copy(),q.T.copy())
    with np.errstate(invalid='ignore',divide='ignore'):xyz=(homogeneous[:3]/homogeneous[3:]).T
    finite=np.isfinite(xyz).all(1);errors=[];front=np.ones(len(xyz),bool)
    for cam,observed in ((previous,p),(current,q)):
        c=xyz@cam.world_to_camera[:3,:3].T+cam.world_to_camera[:3,3]
        uvh=c@cam.intrinsic.T
        with np.errstate(invalid='ignore',divide='ignore'):uv=uvh[:,:2]/uvh[:,2:]
        errors.append(np.linalg.norm(uv-observed,axis=1));front&=c[:,2]>.5
    geom=predict_geometry(anchor,current.timestamp)
    padding=config.depth_padding_m
    source_inside=oriented_box_membership(xyz,anchor.geometry_world,padding)
    target_inside=oriented_box_membership(xyz,geom,padding)
    valid=finite&front&(angles>=config.min_triangulation_angle_deg)&(np.maximum(*errors)<=config.flow_epipolar_max_px)
    supported=valid&source_inside&target_inside
    count=int(supported.sum());fraction=count/len(chosen)
    d.update(supported_points=count,support_fraction=fraction,observable_points=int(valid.sum()),
        membership='validated_oriented_cuboid_local_frame',
        passed=count>=config.support_min_points and fraction>=config.support_point_fraction,
        reason='ok' if count>=config.support_min_points and fraction>=config.support_point_fraction else
            'depth_unobservable' if not valid.any() else 'historical_3d_correspondence_mismatch',
        points=[dict(source_uv=a.tolist(),current_uv=b.tolist(),world_xyz=c.tolist(),angle_deg=float(angle),
            fb_error_px=float(evidence.fb_error_px[index]),patch_residual=float(evidence.patch_residual[index]),
            reprojection_error_px=float(max(e0,e1)),source_box=bool(si),current_box=bool(ti),accepted=bool(ok))
            for a,b,c,angle,index,e0,e1,si,ti,ok in zip(p,q,xyz,angles,chosen,*errors,source_inside,target_inside,supported)])
    return d
