"""Local plane interpolation inside measured support hulls, never ROI filling.

For a plane in camera coordinates n.X=1, inverse depth is rho(u)=n.K^-1 u.
Fit rho on one spatial subset; check its induced reprojection on withheld
correspondences. Inside their convex hull, each pixel must still agree with the
measured warp, texture, photometry and visibility tests. This is a conditional
RGB surface hypothesis, requiring independent quality validation, not ownership.
"""
from collections import Counter
from time import perf_counter
from types import SimpleNamespace
import cv2
import numpy as np
from .foreground_correspondence import triangulate_correspondences
from .geometry import valid_projection_camera


def _project(world, camera):
    points=world@camera.world_to_camera[:3,:3].T+camera.world_to_camera[:3,3]
    uvh=points@camera.intrinsic.T
    with np.errstate(invalid='ignore',divide='ignore'):uv=uvh[:,:2]/uvh[:,2:]
    return uv,points[:,2]


def _history_tracks(previous,current,source,history,target):
    if history is None:return None
    older,stamp,mapping=history;h,w=source.shape[:2]
    if (older.camera_id!=current.camera_id or older.coordinate_space!=current.coordinate_space
            or not valid_projection_camera(older) or not older.timing_reliable
            or older.image.shape[:2]!=(h,w) or mapping is None or mapping.shape!=source.shape
            or not np.isfinite([older.timestamp,stamp,previous.timestamp,current.timestamp]).all()
            or stamp!=previous.timestamp or not 0<previous.timestamp-older.timestamp<=1
            or not 0<current.timestamp-previous.timestamp<=1):return None
    from .temporal_visibility import gray_image
    a,b=gray_image(older),gray_image(current)
    f=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(a,b,None)
    r=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(b,a,None);long=target+r
    def sample(im,coords):
        return cv2.remap(im,coords[...,0],coords[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))
    fb=np.linalg.norm(r+sample(f,long),axis=2)
    loop=np.linalg.norm(long-sample(mapping,source),axis=2)
    dx=np.gradient(long,axis=1);dy=np.gradient(long,axis=0);jac=dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
    observable=np.isfinite(long).all(2)&(long[...,0]>=4)&(long[...,0]<w-4)&(long[...,1]>=4)&(long[...,1]<h-4)&(fb<=.35)&(jac>=.5)&(jac<=2)
    valid=observable&(loop<=.5)
    return older,long,valid,observable&np.isfinite(loop)&(loop>.5)


def local_surface_support(previous,current,source,candidate,config,history=None):
    tick=perf_counter();h,w=candidate.shape;result=np.zeros_like(candidate)
    diag=dict(surface_claim='local_plane_RGB_observation_support_not_ownership',surface_plane_units_accepted=0,
        surface_two_view_pixels=0,surface_long_baseline_pixels=0,surface_candidate_pixels=int(candidate.sum()))
    if (previous.camera_id!=current.camera_id or previous.coordinate_space!=current.coordinate_space
            or previous.image.shape[:2]!=(h,w) or source.shape!=(h,w,2)
            or not np.isfinite([previous.timestamp,current.timestamp]).all()
            or not 0<current.timestamp-previous.timestamp<=1
            or not all(valid_projection_camera(c) and c.timing_reliable for c in (previous,current))):
        return result,dict(diag,surface_reason='camera_contract_mismatch',surface_ms=(perf_counter()-tick)*1000)
    if not candidate.any():return result,dict(diag,surface_reason='no_dense_candidates',surface_ms=(perf_counter()-tick)*1000)
    yy,xx=np.mgrid[:h,:w];target=np.stack((xx,yy),-1).astype(np.float32)
    keys=candidate&(yy%3==0)&(xx%3==0)
    indices=np.flatnonzero(keys);q=target.reshape(-1,2)[indices];p=source.reshape(-1,2)[indices]
    xyz,angle,error,_,valid=triangulate_correspondences(previous,current,SimpleNamespace(source_points=p,current_points=q))
    good=valid&(angle>=.5)&(error<=.5)
    key_world=np.full((h*w,3),np.nan);key_world[indices[good]]=xyz[good]
    key_ok=np.zeros(h*w,bool);key_ok[indices[good]]=True;key_ok=key_ok.reshape(h,w)
    long=_history_tracks(previous,current,source,history,target) if config.surface_use_history else None
    long_world=np.full((h*w,3),np.nan);long_ok=np.zeros((h,w),bool)
    if long is not None:
        older,uv,visible,conflict=long;selected=np.flatnonzero(keys&visible)
        xyz,angle,error,_,valid=triangulate_correspondences(older,current,SimpleNamespace(source_points=uv.reshape(-1,2)[selected],current_points=target.reshape(-1,2)[selected]))
        pp,z=_project(xyz,previous)
        valid&=(angle>=.5)&(error<=.5)&(z>.5)&(z<200)&(np.linalg.norm(pp-source.reshape(-1,2)[selected],axis=1)<=.5)
        long_world[selected[valid]]=xyz[valid];long_ok.flat[selected[valid]]=True
        candidate=candidate&~conflict
    rays=np.c_[target.reshape(-1,2),np.ones(h*w)]@np.linalg.inv(current.intrinsic).T
    boundary=np.zeros((h,w),bool)
    if config.surface_boundary_guard:
        # Reject locally measured folds, discontinuities or competing source
        # occupancy. Never shrink the manager's dynamic swept footprint.
        rel=current.world_to_camera@np.linalg.inv(previous.world_to_camera)
        r0=np.c_[source.reshape(-1,2),np.ones(h*w)]@np.linalg.inv(previous.intrinsic).T
        uvh=(r0@rel[:3,:3].T)@current.intrinsic.T
        with np.errstate(invalid='ignore',divide='ignore'):rotation=(uvh[:,:2]/uvh[:,2:]).reshape(h,w,2)
        parallax=target-rotation
        jump=np.maximum(np.linalg.norm(np.gradient(parallax,axis=0),axis=2),np.linalg.norm(np.gradient(parallax,axis=1),axis=2))
        # Rounded source collisions count all mappings, including rejected ones;
        # a visible point cannot borrow support from a colliding rejected flow.
        finite=np.isfinite(source).all(2)
        cells=np.rint(np.nan_to_num(source)*4).astype(int)
        inside=finite&(cells[...,0]>=0)&(cells[...,0]<4*w)&(cells[...,1]>=0)&(cells[...,1]<4*h)
        counts=np.zeros((4*h,4*w),np.int32);np.add.at(counts,(cells[...,1][inside],cells[...,0][inside]),1)
        collision=np.zeros((h,w),bool);collision[inside]=counts[cells[...,1][inside],cells[...,0][inside]]>1
        seeds=~np.isfinite(jump)|(jump>config.local_parallax_jump_px)|collision
        boundary=cv2.dilate(seeds.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    failures=Counter();unit=config.surface_unit_px
    for y in range(0,h,unit):
        for x in range(0,w,unit):
            region=np.s_[y:min(y+unit,h),x:min(x+unit,w)]
            eligible=candidate[region]&~boundary[region]
            if int(eligible.sum())<8:continue
            accepted=False
            for mode,mask,points in [('two_view',key_ok,key_world),('long_baseline',long_ok,long_world)]:
                sy,sx=np.nonzero(mask[region]&eligible);sy=sy+y;sx=sx+x
                if len(sx)<8:failures['key_support_missing']+=1;continue
                quadrants=np.unique(np.c_[(sx-x)>=unit/2,(sy-y)>=unit/2],axis=0)
                if len(quadrants)<3:failures['key_spatial_support']+=1;continue
                ids=sy*w+sx;world=points[ids];cam=world@current.world_to_camera[:3,:3].T+current.world_to_camera[:3,3]
                # Spatial checkerboard partition: withheld points never fit n.
                train=((sx//3+sy//3)%2)==0
                if train.sum()<4 or (~train).sum()<4:failures['withheld_support_missing']+=1;continue
                design=rays[ids];rho=1/cam[:,2]
                normal,_,rank,_=np.linalg.lstsq(design[train],rho[train],rcond=None)
                if rank<3 or not np.isfinite(normal).all():failures['degenerate_plane']+=1;continue
                all_y,all_x=np.nonzero(eligible);all_y+=y;all_x+=x;all_ids=all_y*w+all_x
                inverse=rays[all_ids]@normal
                if not np.isfinite(inverse).all() or (inverse<=1/200).any() or (inverse>=2).any():
                    failures['plane_depth_invalid']+=1;continue
                cc=rays[all_ids]/inverse[:,None]
                ww=(cc-current.world_to_camera[:3,3])@current.world_to_camera[:3,:3]
                predicted,z=_project(ww,previous)
                residual=np.linalg.norm(predicted-source[all_y,all_x],axis=1)
                support=(z>.5)&(z<200)&np.isfinite(residual)&(residual<=.5)
                # All measured keys (including withheld) must explain the same
                # plane; do not fit one surface and discard competing keypoints.
                key_cc=design/(design@normal)[:,None]
                key_ww=(key_cc-current.world_to_camera[:3,3])@current.world_to_camera[:3,:3]
                key_pred,kz=_project(key_ww,previous)
                if ((kz<=.5).any() or not np.isfinite(key_pred).all() or (np.linalg.norm(key_pred-source[sy,sx],axis=1)>.5).any()):
                    failures['withheld_reprojection_conflict']+=1;continue
                if mode=='long_baseline':
                    old_pred,oz=_project(key_ww,long[0])
                    if (oz<=.5).any() or (np.linalg.norm(old_pred-long[1][sy,sx],axis=1)>.5).any():
                        failures['third_view_plane_conflict']+=1;continue
                hull=cv2.convexHull(np.c_[sx-x,sy-y].astype(np.int32));inside=np.zeros(eligible.shape,np.uint8)
                cv2.fillConvexPoly(inside,hull,1)
                support&=inside[all_y-y,all_x-x].astype(bool)
                # A final small erosion confines the evidence to the hull's
                # interior. It never bridges an unsupported pixel or boundary.
                local=np.zeros(eligible.shape,np.uint8);local[all_y-y,all_x-x]=support
                local=cv2.erode(local,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
                if not local.any():failures['no_supported_hull_interior']+=1;continue
                result[region]|=local;diag['surface_'+('two_view_pixels' if mode=='two_view' else 'long_baseline_pixels')]+=int(local.sum())
                diag['surface_plane_units_accepted']+=1;accepted=True;break
            if not accepted:failures['unit_unknown']+=1
    diag.update(surface_supported_pixels=int(result.sum()),surface_boundary_rejected_pixels=int((candidate&boundary).sum()),
        surface_plane_failure_counts=dict(failures),surface_reason='measured_local_plane_hull',
        surface_history_available=long is not None,surface_temporal_conflict_pixels=int(long[3].sum()) if long is not None else 0,
        surface_ms=(perf_counter()-tick)*1000)
    return result,diag
