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


def dense_support(previous,current,config,exclusion=None,surface_history=None,gray_pair=None):
    start=perf_counter();h,w=current.image.shape[:2]
    empty=np.zeros((h,w),bool)
    diag=dict(dense_flow_ms=0.,dense_validation_ms=0.,dense_support_area_ratio=0.)
    if (previous is None or previous.image.shape[:2]!=(h,w)
            or not 0<current.timestamp-previous.timestamp<=config.max_state_age_s
            or not all(valid_projection_camera(c) and c.timing_reliable for c in (previous,current))
            or min(h,w)<32):
        return empty,None,diag
    roi=None
    if exclusion is not None:
        if np.asarray(exclusion).shape!=(h,w):
            raise ValueError("Dense exclusion shape mismatch")
        if np.all(exclusion):
            diag['dense_skipped_fully_excluded']=True
            return empty,None,diag
        rad=config.dense_guard_radius_px
        # The guard at a retained pixel needs all neighbours, even excluded ones.
        # Halo expansion schedules verification; it never expands the stable mask.
        roi=cv2.dilate((~np.asarray(exclusion,dtype=bool)).astype(np.uint8),
                      np.ones((2*rad+1,2*rad+1),np.uint8)).astype(bool)
    def gray(c):
        return cv2.cvtColor(c.image,cv2.COLOR_RGB2GRAY) if c.image.ndim==3 else c.image
    old,new=gray_pair if gray_pair is not None else (gray(previous),gray(current))
    forward=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(old,new,None)
    reverse=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(new,old,None)
    yy,xx=np.mgrid[:h,:w].astype(np.float32);target=np.stack((xx,yy),axis=-1)
    source=target+reverse
    def sample(im):return cv2.remap(im,source[...,0],source[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    fb=np.linalg.norm(reverse+sample(forward),axis=2)
    boundary=((source[...,0]>=4)&(source[...,0]<w-4)&(source[...,1]>=4)&(source[...,1]<h-4))
    finite=np.isfinite(source).all(2)&np.isfinite(fb)
    good=boundary&finite&(fb<=.35)
    funnel={}
    live=np.ones((h,w),bool)
    def gate(name,condition,population=None):
        nonlocal live
        before=int(live.sum());live=live&condition
        pop=np.ones_like(live) if population is None else population
        funnel[name]=dict(total_pixels=h*w,standalone_population=int(pop.sum()),
            alone_rejected=int((pop&~condition).sum()),incremental_rejected=before-int(live.sum()),retained=int(live.sum()))
    gate('valid_flow',boundary&finite)
    gate('fb_consistency',fb<=.35)
    # A folding/compressing warp is not evidence for a visible, unique surface.
    dx=np.gradient(source,axis=1);dy=np.gradient(source,axis=0)
    jacobian=dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
    good&=(jacobian>=.5)&(jacobian<=2.)
    gate('unique_unfolded_warp',(jacobian>=.5)&(jacobian<=2.))
    diag['dense_flow_ms']=(perf_counter()-start)*1000;t=perf_counter()
    from .measurement_cache import gray_measurement
    oldf=gray_measurement(old);newf=gray_measurement(new);warped=sample(oldf)
    difference=np.abs(warped-newf)
    grad=np.zeros_like(newf)
    for axis in (0,1):
        g0=gray_measurement(old,axis)
        g1=gray_measurement(new,axis)
        grad+=np.abs(sample(g0)-g1)
    photo=(difference<=min(config.support_residual_max,.02))&(grad<=min(config.support_gradient_max,.04))
    # Both images must have locally observable 2-D texture; no flat background fills.
    eigen0=cv2.cornerMinEigenVal(oldf,7,ksize=3);eigen1=cv2.cornerMinEigenVal(newf,7,ksize=3)
    textured=(sample(eigen0)>=1e-5)&(eigen1>=1e-5)
    good&=photo&textured
    gate('photo_and_texture',photo&textured)
    if roi is not None:good &= roi
    if roi is not None:gate('scheduled_exclusion_halo',roi)
    geometry=np.zeros_like(good);idx=np.flatnonzero(good)
    if len(idx):
        geometry.flat[idx]=ray_geometry(previous,current,source.reshape(-1,2)[idx],target.reshape(-1,2)[idx],.5)
    gate('ray_geometry',geometry,good.copy())
    good&=geometry
    # Require complete local evidence; erosion only removes unverified neighbourhoods.
    rad=config.dense_guard_radius_px
    stable=cv2.erode(good.astype(np.uint8),np.ones((2*rad+1,2*rad+1),np.uint8),
                     borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    if config.dense_region_policy=='surface_consensus':
        funnel['complete_neighbourhood_baseline_diagnostic']=dict(total_pixels=h*w,
            standalone_population=int(good.sum()),alone_rejected=int((good&~stable).sum()),
            incremental_rejected=0,retained=int(good.sum()),baseline_retained=int(stable.sum()),
            applied=False)
    else:gate('complete_neighbourhood',stable)
    if config.dense_region_policy=='local_visibility':
        # Reject locally around measured visibility/depth-motion discontinuities.
        # The dynamic swept footprint supplied by the manager is still applied.
        rejected,local_diag=local_visibility_rejection(previous,current,source,forward,reverse,fb,
                                                       jacobian,difference,textured,config)
        stable &= ~rejected
        diag.update(local_diag)
        diag['_local_rejection_mask']=rejected  # consumed before JSON diagnostics
    if config.dense_region_policy=='surface_consensus':
        from .surface_consensus import visible_surface_support
        visible,surface_diag=visible_surface_support(previous,current,source,good,config,surface_history)
        # Each pixel retains its own measured support; a missing neighbour does
        # not invalidate an otherwise observed surface. Never fill the hull.
        stable=good&visible
        gate('visible_surface_consensus',stable,good.copy())
        diag.update(surface_diag)
    elif config.dense_region_policy=='surface_local_plane':
        from .local_surface import local_surface_support
        visible,surface_diag=local_surface_support(previous,current,source,stable,config,surface_history)
        gate('local_plane_surface',visible,stable.copy())
        stable &= visible
        diag.update(surface_diag)
    elif config.dense_region_policy.startswith('surface_') and config.dense_region_policy!='surface_constraints':
        from .surface_visibility import surface_visibility_support
        visible,surface_diag=surface_visibility_support(previous,current,source,stable,surface_history,
            two_view_only=config.dense_region_policy=='surface_observable_ablation')
        stable &= visible
        diag.update(surface_diag)
    diag.update(dense_validation_ms=(perf_counter()-t)*1000,dense_support_area_ratio=float(stable.mean()),
                dense_fb_area_ratio=float((boundary&finite&(fb<=.35)).mean()),
                dense_geometry_photo_area_ratio=float(good.mean()),dense_source='bidirectional_DIS_per_pixel_ray_photo_texture_neighbourhood',
                acceptance_funnel=funnel)
    if config.surface_diagnostics:
        diag['_dense_measurements']=dict(forward=forward,reverse=reverse,fb=fb,photo=difference,
            gradient=grad,texture=np.minimum(sample(eigen0),eigen1),jacobian=jacobian,
            per_pixel_candidate=good,geometry=geometry)
    if config.change_first_guard:
        # Risk from measured, locally trackable photometric disagreement. This
        # rejects candidates; it asserts neither foreground ownership nor truth.
        diag['_measured_change_risk_mask'] = (boundary & finite & (fb<=config.flow_fb_max_px)
            & textured & (difference>max(.08,config.support_residual_max*2)))
    return stable,source.astype(np.float32),diag


def local_visibility_rejection(previous,current,source,forward,reverse,fb,jacobian,difference,textured,config):
    """Reject uncertainty near measured boundaries; no GT or full-image guard.

    Rotation-compensated parallax gradients expose depth/motion discontinuities.
    These are uncertainty seeds, NOT evidence of foreground ownership or semantic
    change. A local dilation of rejection cannot create stable pixels.
    """
    h,w=source.shape[:2]
    yy,xx=np.mgrid[:h,:w]
    relative=current.world_to_camera@np.linalg.inv(previous.world_to_camera)
    rays=np.c_[source.reshape(-1,2),np.ones(h*w)]@np.linalg.inv(previous.intrinsic).T
    rotated=rays@relative[:3,:3].T
    uvh=rotated@current.intrinsic.T
    with np.errstate(invalid='ignore',divide='ignore'):
        rotation_uv=(uvh[:,:2]/uvh[:,2:]).reshape(h,w,2)
    parallax=np.stack((xx,yy),axis=-1)-rotation_uv
    finite=np.isfinite(parallax).all(2)&np.isfinite(fb)
    parallax=np.nan_to_num(parallax,nan=0.,posinf=0.,neginf=0.)
    jump=np.maximum(np.linalg.norm(np.gradient(parallax,axis=0),axis=2),
                    np.linalg.norm(np.gradient(parallax,axis=1),axis=2))
    depth_boundary=finite&(jump>config.local_parallax_jump_px)
    fold=finite&((jacobian<.5)|(jacobian>2.))
    ambiguous_flow=finite&(fb>config.flow_fb_max_px)
    strong_residual=finite&textured&(difference>max(.08,config.support_residual_max*2))
    seeds=depth_boundary|fold|ambiguous_flow|strong_residual|~finite
    radius=config.local_boundary_radius_px
    rejected=cv2.dilate(seeds.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
    return rejected,dict(local_boundary_seed_area=float(seeds.mean()),
        local_depth_motion_boundary_area=float(depth_boundary.mean()),
        local_visibility_rejection_area=float(rejected.mean()),
        local_policy='measured_parallax_fb_fold_photo_boundary_rejection_only')


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
