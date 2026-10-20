"""Measured ray hypotheses across actual images, without ownership truth.

This module only measures. A finite patch search cannot prove global uniqueness;
same-image agreement is explicitly distinguished from a different acquisition.
No reference labels, predicted boxes or depth sensors enter these functions.
"""
from types import SimpleNamespace
import cv2
import numpy as np
from .geometry import valid_projection_camera
from .foreground_correspondence import triangulate_correspondences


def gray(camera):
    return cv2.cvtColor(camera.image,cv2.COLOR_RGB2GRAY) if camera.image.ndim==3 else camera.image


def project(world,camera):
    with np.errstate(divide='ignore',invalid='ignore'):
        p=np.asarray(world)@camera.world_to_camera[:3,:3].T+camera.world_to_camera[:3,3]
        h=p@camera.intrinsic.T
        uv=h[...,:2]/h[...,2:]
    return uv,p[...,2]


def camera_center(camera):
    return -camera.world_to_camera[:3,:3].T@camera.world_to_camera[:3,3]


def baseline_angle(world,first,second):
    a=np.asarray(world)-camera_center(first);b=np.asarray(world)-camera_center(second)
    with np.errstate(divide='ignore',invalid='ignore'):
        cosine=np.sum(a*b,axis=-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))
    return np.degrees(np.arccos(np.clip(cosine,-1,1)))


def image_contract(first,second,config,temporal=False):
    if not all(valid_projection_camera(c) and c.timing_reliable for c in (first,second)):
        return 'invalid_or_unreliable_camera'
    if first.coordinate_space!=second.coordinate_space:return 'coordinate_space_mismatch'
    if not np.isfinite([first.timestamp,second.timestamp]).all():return 'invalid_timestamp'
    if temporal:
        if first.camera_id!=second.camera_id or not 0<first.timestamp-second.timestamp<=min(config.max_state_age_s,1.):
            return 'invalid_temporal_source'
    elif first.camera_id==second.camera_id or abs(first.timestamp-second.timestamp)>config.max_camera_skew_s:
        return 'no_synchronous_distinct_camera'
    return None


class PatchObservation:
    """Actual projective patch samples of a local frontoparallel depth hypothesis.

    The patch plane is a probe, not a fitted surface. Failure does not establish
    occlusion. The thresholds and aperture reuse the existing observation tests.
    """
    def __init__(self,camera,config):
        self.camera=camera;self.config=config;self.image=gray(camera).astype(np.float32)/255
        radius=config.support_patch_radius_px
        yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
        self.offsets=np.c_[xx.ravel(),yy.ravel()];self.side=2*radius+1

    def compare(self,other,uv,world,inverse_depth_slopes=None):
        uv=np.asarray(uv,float);world=np.asarray(world,float);n=len(uv)
        grid=uv[:,None,:]+self.offsets
        rays=np.concatenate((grid,np.ones((*grid.shape[:2],1))),axis=2)@np.linalg.inv(self.camera.intrinsic).T
        _,depth=project(world,self.camera)
        if inverse_depth_slopes is None:
            cam_points=rays*depth[:,None,None]
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                inverse_depth=1/depth[:,None]+np.asarray(inverse_depth_slopes)@self.offsets.T
                cam_points=rays/inverse_depth[...,None]
        patch_world=(cam_points-self.camera.world_to_camera[:3,3])@self.camera.world_to_camera[:3,:3]
        projected,z=project(patch_world,other.camera)
        def sample(observation,points):
            h,w=observation.image.shape
            valid=np.isfinite(points).all(2)&(points[...,0]>=1)&(points[...,0]<w-1)&(points[...,1]>=1)&(points[...,1]<h-1)
            coords=np.where(np.isfinite(points),points,-10).astype(np.float32)
            a=cv2.remap(observation.image,coords[...,0],coords[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            return a,valid.all(1)
        a,va=sample(self,grid);b,vb=sample(other,projected)
        offset=np.clip(np.median(a-b,axis=1),-.03,.03)
        difference=np.abs(a-b-offset[:,None]);texture=np.minimum(a.std(1),b.std(1))
        aa=a.reshape(n,self.side,self.side);bb=b.reshape(n,self.side,self.side)
        gradient=(np.abs(np.diff(aa,axis=1)-np.diff(bb,axis=1)).mean((1,2))+
                  np.abs(np.diff(aa,axis=2)-np.diff(bb,axis=2)).mean((1,2)))
        valid=va&vb&np.isfinite(z).all(1)&(z>.5).all(1)&(z<200).all(1)&np.isfinite(depth)&(depth>.5)&(depth<200)
        support=valid&(texture>=self.config.min_texture_std)&((difference<=self.config.support_residual_max).mean(1)>=.9)&(gradient<=self.config.support_gradient_max)
        return dict(supported=support,observed=valid,photo=difference.mean(1),gradient=gradient,texture=texture,
            predicted_uv=project(world,other.camera)[0])


def measured_world(previous,current,source,target):
    with np.errstate(divide='ignore',invalid='ignore'):
        xyz,angle,error,depth,valid=triangulate_correspondences(previous,current,
            SimpleNamespace(source_points=np.asarray(source,float),current_points=np.asarray(target,float)))
    return dict(world=xyz,angle=angle,error=error,depths=depth,valid=valid&(angle>=.5)&(error<=.5))


def alternative_rays(previous,current,source,target,config):
    """Finite, observed epipolar aperture; neither GT nor best-residual selection.

    Every geometrically valid supported explanation is retained. A +/-4px
    aperture (the existing search radius) is not an exhaustive depth search.
    """
    source=np.asarray(source,float);target=np.asarray(target,float)
    relative=previous.world_to_camera@np.linalg.inv(current.world_to_camera)
    t=relative[:3,3];cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    fundamental=np.linalg.inv(previous.intrinsic).T@cross@relative[:3,:3]@np.linalg.inv(current.intrinsic)
    line=np.c_[target,np.ones(len(target))]@fundamental.T
    norm=np.linalg.norm(line[:,:2],axis=1)
    safe=np.maximum(norm,1e-12)
    distance=(np.sum(line[:,:2]*source,axis=1)+line[:,2])/safe
    base=source-distance[:,None]*line[:,:2]/safe[:,None]
    direction=np.c_[-line[:,1],line[:,0]]/safe[:,None]
    offsets=np.arange(-config.search_radius_px,config.search_radius_px+.25,.5)
    points=base[:,None,:]+direction[:,None,:]*offsets[None,:,None]
    q=np.broadcast_to(target[:,None,:],points.shape)
    measured=measured_world(previous,current,points.reshape(-1,2),q.reshape(-1,2))
    valid=measured['valid'].reshape(len(source),-1)&(norm[:,None]>1e-12)
    return points,q,measured['world'].reshape(len(source),len(offsets),3),valid


def corroborating_views(current,cameras,history,config):
    """List actual legal observations; no assumed six-camera co-visibility."""
    result=[];rejected={}
    if history is not None:
        old,stamp,_=history
        reason=image_contract(current,old,config,temporal=True)
        if reason is None and np.isfinite(stamp) and old.timestamp<stamp<current.timestamp:
            result.append(('additional_time',old))
        else:rejected['additional_time']=reason or 'history_order_invalid'
    else:rejected['additional_time']='not_observed'
    for cid,camera in cameras.items():
        if cid==current.camera_id:continue
        reason=image_contract(current,camera,config)
        if reason is None:result.append(('overlap_camera',camera))
        else:rejected[cid]=reason
    return result,rejected


def observe_points(previous,current,source,target,config,cameras=None,history=None):
    """Auditable measurements; supported != independent physical truth."""
    source=np.asarray(source,float);target=np.asarray(target,float)
    m=measured_world(previous,current,source,target);n=len(target)
    current_patch=PatchObservation(current,config);old_patch=PatchObservation(previous,config)
    direct=current_patch.compare(old_patch,target,m['world'])
    ap,aq,aw,av=alternative_rays(previous,current,source,target,config)
    alt=current_patch.compare(old_patch,aq.reshape(-1,2),aw.reshape(-1,3))
    alt_support=alt['supported'].reshape(av.shape)&av
    separation=np.linalg.norm(ap-source[:,None,:],axis=2)
    alternatives=alt_support&(separation>2.)
    views,rejected=corroborating_views(current,cameras or {},history,config);measurements=[]
    for kind,view in views:
        observation=current_patch.compare(PatchObservation(view,config),target,m['world'])
        angle=baseline_angle(m['world'],current,view)
        usable=m['valid']&(angle>=.5)&observation['observed']
        observation.update(kind=kind,camera_id=view.camera_id,timestamp=view.timestamp,angle=angle,
            usable=usable,supported=observation['supported']&usable)
        # Check alternative explanations against this actual different image,
        # rather than following a shared flow chain through it.
        other_alt=current_patch.compare(PatchObservation(view,config),aq.reshape(-1,2),aw.reshape(-1,3))
        angle_alt=baseline_angle(aw,current,view)
        observation['alternative_supported']=(other_alt['supported'].reshape(av.shape)&av&(angle_alt>=.5)&alt_support)
        measurements.append(observation)
    m.update(two_view_patch=direct,alternative_source_uv=ap,alternative_world=aw,
        alternative_supported=alt_support,alternative_incompatible=alternatives,alternative_photo=alt['photo'].reshape(av.shape),
        independent_retracking_dependency='Current->source LK uses the same two RGB acquisitions; not an independent sensor',
        extra_views=measurements,unavailable_views=rejected)
    return m
