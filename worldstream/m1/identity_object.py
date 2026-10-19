"""Conditional object-surface tracks separated from measured background.

Predicted box translation defines a testable motion hypothesis, not ownership
truth. Motion-compensated rays must triangulate within both predicted boxes,
and an actually tracked exterior ring must be behind them in both views.
Unobservable/indistinguishable backgrounds return Unknown. No GT is consumed.
"""
from dataclasses import replace
from types import SimpleNamespace
import cv2
import numpy as np
from .foreground_correspondence import triangulate_correspondences
from .geometry import oriented_box_membership,region_slices,valid_projection_camera

PRODUCER='finite_actual_object_surface_identity_v1'
POPULATION='finite_motion_compensated_box_surface_and_background_ring_identity'
DEPENDENCY='Actual_RGB_tracks_predicted_translation_box_membership_and_observed_background_depth_not_ownership_truth'


def surface_geometry(anchor,target,old,cam,p,q,rp,rq,config):
    delta=target.geometry_world.center-anchor.geometry_world.center
    motion=np.eye(4);motion[:3,3]=delta
    effective=replace(cam,world_to_camera=cam.world_to_camera@motion)
    xyz,angle,error,depth,valid=triangulate_correspondences(old,effective,SimpleNamespace(source_points=p,current_points=q))
    belongs=valid&(angle>=.5)&(error<=.5)&oriented_box_membership(xyz,anchor.geometry_world,0)&oriented_box_membership(xyz+delta,target.geometry_world,0)
    _,ra,re,rd,rv=triangulate_correspondences(old,cam,SimpleNamespace(source_points=rp,current_points=rq))
    ring=rv&(ra>=.5)&(re<=.5)
    gap=None
    if belongs.sum()>=4 and ring.sum()>=4:
        gap=np.percentile(rd[ring],25,axis=0)-np.percentile(depth[belongs],90,axis=0)
    return belongs,ring,gap,dict(source_world_points=xyz,parallax_deg=angle,reprojection_error_px=error)


def measure_object_view(anchor,target,old,cam,validator,config,baseline):
    from .identity_temporal import _inside,_lk,quadrants
    from .temporal_visibility import gray_image
    from .geometry_support import measured_patch_support
    from .geometry import project_anchor
    v=dict(baseline,status='unknown',reason='object_surface_unobservable',observation_model='translation_compensated_box_surface_ring')
    obs=anchor.observation_support[cam.camera_id];projection=project_anchor(target,cam,config)
    if not projection.valid:return v
    r=np.asarray(obs.region);s=np.asarray(projection.region);a,b=gray_image(old),gray_image(cam)
    pad=np.maximum((r[2:]-r[:2])*.5,12);outer=np.r_[r[:2]-pad,r[2:]+pad]
    mask=np.zeros(a.shape,np.uint8);mask[region_slices(outer,a.shape)]=255
    points=cv2.goodFeaturesToTrack(a,maxCorners=192,qualityLevel=.01,minDistance=4,mask=mask,blockSize=5)
    if points is None:return dict(v,reason='object_and_ring_tracks_missing')
    p=points[:,0];q,fb,valid=_lk(a,b,p)
    _,_,photo,residual=measured_patch_support(a,b,p,q,valid,config)
    measured=valid&photo&(residual<=.04)
    inside=_inside(p,r)&_inside(q,s);pad1=np.maximum((s[2:]-s[:2])*.5,12)
    ring=measured&~_inside(p,r)&~_inside(q,s)&_inside(q,np.r_[s[:2]-pad1,s[2:]+pad1])
    foreground=measured&inside
    ip=np.flatnonzero(foreground);ir=np.flatnonzero(ring)
    belongs,background,gap,geometry=surface_geometry(anchor,target,old,cam,p[ip],q[ip],p[ir],q[ir],config)
    selected=ip[belongs];ridx=ir[background]
    v.update(total_tracks=len(p),measured_tracks=int(measured.sum()),crop_tracks=len(ip),object_surface_tracks=len(selected),
        ring_tracks=len(ridx),background_gap_m=gap.tolist() if gap is not None else None)
    if len(selected)<4:return dict(v,reason='object_surface_geometry_missing')
    if len(ridx)<4:return dict(v,reason='independent_ring_geometry_missing')
    if gap is None or (gap<config.foreground_depth_gap_m).any():return dict(v,reason='object_background_depth_indistinguishable')
    if quadrants(p[selected],r)<3 or quadrants(q[selected],s)<3:return dict(v,reason='object_surface_spatial_support_missing')
    return dict(v,status='supported',reason='observed_motion_compensated_surface_and_background',tracks=len(selected),
        source_region=r.tolist(),current_region=s.tolist(),source_timestamp=old.timestamp,current_timestamp=cam.timestamp,
        source_coordinate_space=old.coordinate_space,current_coordinate_space=cam.coordinate_space,
        source_descriptor=obs.descriptor.tolist(),source_points=p[selected].tolist(),current_points=q[selected].tolist(),
        fb_error_px=fb[selected].tolist(),patch_residual=residual[selected].tolist(),
        ring_source_points=p[ridx].tolist(),ring_current_points=q[ridx].tolist(),
        ring_fb_error_px=fb[ridx].tolist(),ring_patch_residual=residual[ridx].tolist(),
        source_intrinsic=old.intrinsic.tolist(),source_world_to_camera=old.world_to_camera.tolist(),
        source_camera_snapshot_sha256=__import__(__package__+'.identity_temporal',fromlist=['camera_snapshot_digest']).camera_snapshot_digest(old),
        conditional_foreground_hypothesis=True,foreground_ownership_certified=False)


def validate_object_view(anchor,current,cam,v,config,measurements):
    from .identity_temporal import _inside,quadrants
    from .state_types import CameraFrame
    p=np.asarray(v['source_points'],float);q=np.asarray(v['current_points'],float)
    rp=np.asarray(v['ring_source_points'],float);rq=np.asarray(v['ring_current_points'],float)
    r=np.asarray(v['source_region']);s=np.asarray(v['current_region'])
    if (len(rp)<4 or rp.shape!=(len(rp),2) or rq.shape!=rp.shape or not np.isfinite(rp).all() or not np.isfinite(rq).all()
            or _inside(rp,r).any() or _inside(rq,s).any() or quadrants(p,r)<3 or quadrants(q,s)<3):return False
    # A signed numeric ring still has to be drawable inside the actual images
    # and the producer's measured exterior ROI. Off-image rays are not samples.
    h,w=cam.image.shape[:2];bounds=np.array([0,0,w,h])
    pad=np.maximum((r[2:]-r[:2])*.5,12);outer=np.r_[r[:2]-pad,r[2:]+pad]
    ys,xs=region_slices(outer,cam.image.shape);source_outer=[xs.start,ys.start,xs.stop,ys.stop]
    pad1=np.maximum((s[2:]-s[:2])*.5,12);target_outer=np.r_[s[:2]-pad1,s[2:]+pad1]
    if not (_inside(rp,bounds).all() and _inside(rq,bounds).all()
            and _inside(rp,source_outer).all() and _inside(rq,target_outer).all()):return False
    for key in ('ring_fb_error_px','ring_patch_residual'):
        values=np.asarray(v[key],float)
        if values.shape!=(len(rp),) or not np.isfinite(values).all() or (values<0).any() or (values>(.75 if 'fb' in key else .04)).any():return False
    if (v.get('source_camera_snapshot_sha256')!=measurements['source_camera_sha256'].get(cam.camera_id)
            or len(np.unique(rp,axis=0))!=len(rp) or len(np.unique(rq,axis=0))!=len(rq)):return False
    old=CameraFrame(cam.camera_id,cam.image,np.asarray(v['source_intrinsic']),np.asarray(v['source_world_to_camera']),
        v['source_timestamp'],cam.coordinate_space,timing_reliable=True)
    if not valid_projection_camera(old):return False
    belongs,ring,gap,_=surface_geometry(anchor,current,old,cam,p,q,rp,rq,config)
    return bool(belongs.all() and ring.all() and gap is not None and (gap>=config.foreground_depth_gap_m).all()
        and np.allclose(gap,v['background_gap_m'],atol=1e-5))
