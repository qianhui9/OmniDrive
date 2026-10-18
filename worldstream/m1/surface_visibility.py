"""Three-view ray/correspondence closure, a selective RGB observation test.

Two rays can fit a wrong along-epipolar match. Require that reconstructed surface
to explain a separately measured longer-baseline track and the composed track.
Missing history/parallax is Unknown, never affirmative surface evidence.
This cannot prove ownership in repetitive/occluded RGB without quality validation.
"""
from time import perf_counter
from types import SimpleNamespace
import cv2
import numpy as np
from .foreground_correspondence import triangulate_correspondences


def surface_visibility_support(previous,current,source,candidate,history=None,two_view_only=False):
    start=perf_counter();h,w=candidate.shape;result=np.zeros_like(candidate)
    diag=dict(surface_candidate_pixels=int(candidate.sum()),surface_supported_pixels=0,
              surface_claim='three_view_RGB_ray_and_correspondence_closure_not_pixel_ownership',surface_reason='missing_history')
    idx=np.flatnonzero(candidate)
    if not len(idx):return result,dict(diag,surface_ms=(perf_counter()-start)*1000)
    yy,xx=np.mgrid[:h,:w];target=np.stack((xx,yy),-1).astype(np.float32)
    p=source.reshape(-1,2)[idx];q=target.reshape(-1,2)[idx]
    xyz,angle,error,_,valid=triangulate_correspondences(previous,current,SimpleNamespace(source_points=p,current_points=q))
    valid&=(angle>=.5)&(error<=.5)
    diag['surface_observable_pixels']=int(valid.sum())
    if two_view_only:
        result.flat[idx]=valid
        return result,dict(diag,surface_supported_pixels=int(valid.sum()),surface_reason='two_view_observability_ablation',surface_ms=(perf_counter()-start)*1000)
    if history is None:return result,dict(diag,surface_ms=(perf_counter()-start)*1000)
    older,history_current_timestamp,previous_source=history
    if (history_current_timestamp!=previous.timestamp or older.image.shape[:2]!=(h,w)
            or previous_source is None or previous_source.shape!=(h,w,2)
            or not 0<previous.timestamp-older.timestamp<=1 or not 0<current.timestamp-previous.timestamp<=1):
        return result,dict(diag,surface_reason='history_time_or_shape_mismatch',surface_ms=(perf_counter()-start)*1000)
    def gray(c):return cv2.cvtColor(c.image,cv2.COLOR_RGB2GRAY) if c.image.ndim==3 else c.image
    forward=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(gray(older),gray(current),None)
    reverse=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(gray(current),gray(older),None)
    long_source=target+reverse
    def sample(im,coords):return cv2.remap(im,coords[...,0],coords[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))
    fb=np.linalg.norm(reverse+sample(forward,long_source),axis=2)
    composed=sample(previous_source,source)
    loop=np.linalg.norm(long_source-composed,axis=2)
    dx=np.gradient(long_source,axis=1);dy=np.gradient(long_source,axis=0)
    jac=dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
    observed=long_source.reshape(-1,2)[idx]
    _,long_angle,long_error,_,long_valid=triangulate_correspondences(older,current,SimpleNamespace(source_points=observed,current_points=q))
    c=xyz@older.world_to_camera[:3,:3].T+older.world_to_camera[:3,3]
    uvh=c@older.intrinsic.T
    with np.errstate(invalid='ignore',divide='ignore'):projection=uvh[:,:2]/uvh[:,2:]
    ray_closure=np.linalg.norm(projection-observed,axis=1)
    inside=(observed[:,0]>=4)&(observed[:,0]<w-4)&(observed[:,1]>=4)&(observed[:,1]<h-4)
    long_checks=(long_valid&(long_angle>=.5)&(long_error<=.5)&inside&(c[:,2]>.5)&(ray_closure<=.5)
                 &(fb.flat[idx]<=.35)&(loop.flat[idx]<=.5)&(jac.flat[idx]>=.5)&(jac.flat[idx]<=2))
    passed=valid&long_checks
    result.flat[idx]=passed
    diag.update(surface_supported_pixels=int(passed.sum()),surface_loop_pass_pixels=int((loop.flat[idx]<=.5).sum()),
        surface_ray_closure_pass_pixels=int((ray_closure<=.5).sum()),surface_reason='measured_three_view_closure',
        surface_ms=(perf_counter()-start)*1000)
    return result,diag
