"""Dense measured RGB support; no semantic labels, rectangle fills or NN inference.

DIS proposes correspondences only. Every admitted pixel separately passes reverse
consistency, calibrated ray geometry, texture and photometric checks. Unsupported
neighbourhoods are eroded (never dilated) to protect boundaries and disocclusions.
This is observational evidence, not certification of neural dependency equality.
"""
from time import perf_counter
import cv2
import numpy as np
from .geometry import valid_projection_camera


def ray_geometry(previous, current, source, target, tolerance=.5):
    """Vectorized two-ray least squares, with positive depth and reprojection gates."""
    if not all(valid_projection_camera(c) and c.timing_reliable for c in (previous,current)):
        return np.zeros(len(source),bool)
    inv0=np.linalg.inv(previous.intrinsic); inv1=np.linalg.inv(current.intrinsic)
    relative=current.world_to_camera @ np.linalg.inv(previous.world_to_camera)
    r,t=relative[:3,:3],relative[:3,3]
    x=np.c_[source,np.ones(len(source))] @ inv0.T
    y=np.c_[target,np.ones(len(target))] @ inv1.T
    a=x @ r.T
    if np.linalg.norm(t)<.02:
        uv=a @ current.intrinsic.T
        with np.errstate(invalid='ignore',divide='ignore'):
            error=np.linalg.norm(uv[:,:2]/uv[:,2:]-target,axis=1)
        return (a[:,2]>0)&np.isfinite(error)&(error<=tolerance)
    aa=np.einsum('ij,ij->i',a,a);bb=np.einsum('ij,ij->i',y,y);ab=np.einsum('ij,ij->i',a,y)
    at=np.einsum('ij,j->i',a,t);bt=np.einsum('ij,j->i',y,t);det=aa*bb-ab*ab
    # Rays closer than half a pixel cannot establish a well-conditioned finite depth.
    focal=max(abs(previous.intrinsic[0,0]),abs(current.intrinsic[0,0]))
    conditioned=det/(aa*bb)>(.5/focal)**2
    safe=np.maximum(det,1e-15)
    z0=(ab*bt-bb*at)/safe;z1=(aa*bt-ab*at)/safe
    point1=a*z0[:,None]+t
    point0=(y*z1[:,None]-t) @ r
    u1=point1 @ current.intrinsic.T;u0=point0 @ previous.intrinsic.T
    with np.errstate(invalid='ignore',divide='ignore'):
        e1=np.linalg.norm(u1[:,:2]/u1[:,2:]-target,axis=1)
        e0=np.linalg.norm(u0[:,:2]/u0[:,2:]-source,axis=1)
    return conditioned & (z0>.5)&(z0<200)&(z1>.5)&(z1<200)&(e0<=tolerance)&(e1<=tolerance)


def dense_support(previous,current,config):
    start=perf_counter();h,w=current.image.shape[:2]
    empty=np.zeros((h,w),bool)
    diag=dict(dense_flow_ms=0.,dense_validation_ms=0.,dense_support_area_ratio=0.)
    if (previous is None or previous.image.shape[:2]!=(h,w)
            or not 0<current.timestamp-previous.timestamp<=config.max_state_age_s
            or not all(valid_projection_camera(c) and c.timing_reliable for c in (previous,current))
            or min(h,w)<32):
        return empty,None,diag
    def gray(c):
        return cv2.cvtColor(c.image,cv2.COLOR_RGB2GRAY) if c.image.ndim==3 else c.image
    old,new=gray(previous),gray(current)
    forward=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(old,new,None)
    reverse=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(new,old,None)
    yy,xx=np.mgrid[:h,:w].astype(np.float32);target=np.stack((xx,yy),axis=-1)
    source=target+reverse
    def sample(im):return cv2.remap(im,source[...,0],source[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    fb=np.linalg.norm(reverse+sample(forward),axis=2)
    boundary=((source[...,0]>=4)&(source[...,0]<w-4)&(source[...,1]>=4)&(source[...,1]<h-4))
    finite=np.isfinite(source).all(2)&np.isfinite(fb)
    good=boundary&finite&(fb<=.35)
    # A folding/compressing warp is not evidence for a visible, unique surface.
    dx=np.gradient(source,axis=1);dy=np.gradient(source,axis=0)
    jacobian=dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
    good&=(jacobian>=.5)&(jacobian<=2.)
    diag['dense_flow_ms']=(perf_counter()-start)*1000;t=perf_counter()
    oldf=old.astype(np.float32)/255.;newf=new.astype(np.float32)/255.;warped=sample(oldf)
    difference=np.abs(warped-newf)
    grad=np.zeros_like(newf)
    for axis in (0,1):
        g0=cv2.Sobel(oldf,cv2.CV_32F,int(axis==0),int(axis==1),ksize=3)/8
        g1=cv2.Sobel(newf,cv2.CV_32F,int(axis==0),int(axis==1),ksize=3)/8
        grad+=np.abs(sample(g0)-g1)
    photo=(difference<=min(config.support_residual_max,.02))&(grad<=min(config.support_gradient_max,.04))
    # Both images must have locally observable 2-D texture; no flat background fills.
    eigen0=cv2.cornerMinEigenVal(oldf,7,ksize=3);eigen1=cv2.cornerMinEigenVal(newf,7,ksize=3)
    textured=(sample(eigen0)>=1e-5)&(eigen1>=1e-5)
    good&=photo&textured
    geometry=np.zeros_like(good);idx=np.flatnonzero(good)
    if len(idx):
        geometry.flat[idx]=ray_geometry(previous,current,source.reshape(-1,2)[idx],target.reshape(-1,2)[idx],.5)
    good&=geometry
    # Require complete local evidence; erosion only removes unverified neighbourhoods.
    rad=config.dense_guard_radius_px
    stable=cv2.erode(good.astype(np.uint8),np.ones((2*rad+1,2*rad+1),np.uint8),
                     borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    diag.update(dense_validation_ms=(perf_counter()-t)*1000,dense_support_area_ratio=float(stable.mean()),
                dense_fb_area_ratio=float((boundary&finite&(fb<=.35)).mean()),
                dense_geometry_photo_area_ratio=float(good.mean()),dense_source='bidirectional_DIS_per_pixel_ray_photo_texture_neighbourhood')
    return stable,source.astype(np.float32),diag


def swept_dynamic_region(anchor, camera, projection, config):
    """Unclipped directional sweep plus the existing metric motion uncertainty.

    Velocity is used only to reject potentially occupied pixels, never to accept
    a moving state. Expanding an already clipped off-image box would fabricate
    a large on-image rejection region, so always use original projected corners.
    """
    dt = camera.timestamp - anchor.source_timestamp
    uv = projection.uv
    depth = projection.depth
    if anchor.velocity_world is not None:
        points = anchor.geometry_world.points + np.asarray(anchor.velocity_world) * dt
        pose = camera.world_to_camera
        xyz = points @ pose[:3, :3].T + pose[:3, 3]
        if np.any(xyz[:, 2] <= .1):
            h, w = camera.image.shape[:2]
            return (0., 0., float(w), float(h))
        projected = xyz @ camera.intrinsic.T
        uv = np.concatenate([uv, projected[:, :2] / projected[:, 2:]], axis=0)
        depth = min(depth, float(xyz[:, 2].min()))
    uncertainty = anchor.geometry_uncertainty + config.velocity_uncertainty_mps * abs(dt)
    margin = config.uncertainty_px + abs(camera.intrinsic[0, 0]) * uncertainty / max(depth, .1)
    return (float(uv[:, 0].min()) - margin, float(uv[:, 1].min()) - margin,
            float(uv[:, 0].max()) + margin, float(uv[:, 1].max()) + margin)
