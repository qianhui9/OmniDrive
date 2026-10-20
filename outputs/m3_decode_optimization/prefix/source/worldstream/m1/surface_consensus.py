"""Visible inverse-depth consensus with disjoint spatial validation.

Scheduling tiles do not define surfaces. Connected measured support and training
consensus do; held-out correspondences never fit or rank a model. Multiple
incompatible surfaces have separate hulls. All accepted pixels still require
their own measured ray/flow/photo/texture support and unambiguous reprojection.
This conditional RGB hypothesis cannot identify coherently incorrect flow.
"""
from collections import Counter
from itertools import combinations
from time import perf_counter
from types import SimpleNamespace
import cv2
import numpy as np
from .foreground_correspondence import triangulate_correspondences
from .geometry import valid_projection_camera
from .local_surface import _project, _history_tracks


def plane_projection(rays, normal, current, previous):
    inverse=rays@normal
    with np.errstate(invalid='ignore',divide='ignore'):
        xyz=rays/inverse[:,None]
    world=(xyz-current.world_to_camera[:3,3])@current.world_to_camera[:3,:3]
    uv,z=_project(world,previous)
    valid=np.isfinite(uv).all(1)&np.isfinite(inverse)&(inverse>1/200)&(inverse<2)&(z>.5)&(z<200)
    return uv,valid


def training_models(design,rho,source,current,previous,split=True):
    """Deterministic train-only consensus; no held-out result ranks proposals."""
    remaining=np.arange(len(rho));models=[]
    for _ in range(3 if split else 1):
        if len(remaining)<4:break
        r=design[remaining];d=rho[remaining]
        # Fixed spatially distributed seed schedule, independent of image truth.
        seeds=np.unique(np.linspace(0,len(r)-1,min(9,len(r))).astype(int))
        groups=[np.arange(len(r))]
        if split:groups.extend(np.asarray(c) for c in combinations(seeds,3))
        best=None;best_count=0
        for group in groups:
            n,_,rank,_=np.linalg.lstsq(r[group],d[group],rcond=None)
            if rank<3 or not np.isfinite(n).all():continue
            uv,valid=plane_projection(r,n,current,previous)
            good=valid&(np.linalg.norm(uv-source[remaining],axis=1)<=.5)
            count=int(good.sum())
            if count>best_count:best_count=count;best=good
            if count==len(r):break
        if best is None or best_count<4:break
        ids=remaining[best]
        n,_,rank,_=np.linalg.lstsq(design[ids],rho[ids],rcond=None)
        uv,valid=plane_projection(design[ids],n,current,previous)
        if rank<3 or not (valid&(np.linalg.norm(uv-source[ids],axis=1)<=.5)).all():
            remaining=remaining[~best];continue
        models.append((n,ids));remaining=remaining[~best]
    return models


def visible_surface_support(previous,current,source,candidate,config,history=None):
    tick=perf_counter();h,w=candidate.shape;result=np.zeros((h,w),bool);counts=Counter();trace=[]
    diag=dict(surface_claim='visible_disjoint_consensus_RGB_not_physical_ownership',surface_candidate_pixels=int(candidate.sum()))
    if (previous.camera_id!=current.camera_id or previous.coordinate_space!=current.coordinate_space
            or previous.image.shape[:2]!=(h,w) or source.shape!=(h,w,2)
            or not np.isfinite([previous.timestamp,current.timestamp]).all()
            or not 0<current.timestamp-previous.timestamp<=1
            or not all(valid_projection_camera(c) and c.timing_reliable for c in (previous,current))):
        return result,dict(diag,surface_reason='camera_contract_mismatch')
    yy,xx=np.mgrid[:h,:w];target=np.stack((xx,yy),-1).astype(np.float32)
    # The original 0.5 degree / 0.5 px / >=8 measured-key requirements stay.
    keys=candidate&(yy%3==0)&(xx%3==0);indices=np.flatnonzero(keys)
    xyz,angle,error,_,valid=triangulate_correspondences(previous,current,
        SimpleNamespace(source_points=source.reshape(-1,2)[indices],current_points=target.reshape(-1,2)[indices]))
    valid&=(angle>=.5)&(error<=.5)
    worlds=np.full((h*w,3),np.nan);worlds[indices[valid]]=xyz[valid]
    key_ok=np.zeros(h*w,bool);key_ok[indices[valid]]=True;key_ok=key_ok.reshape(h,w)
    long=_history_tracks(previous,current,source,history,target) if config.surface_use_history else None
    if long is not None:
        older,long_uv,visible,conflict=long;candidate=candidate&~conflict
        # Recover otherwise unobservable keys with a real longer baseline. Its
        # world point must also predict the actual immediate source measurement.
        ids=np.flatnonzero(keys&visible&~key_ok)
        xyz,a,e,_,v=triangulate_correspondences(older,current,
            SimpleNamespace(source_points=long_uv.reshape(-1,2)[ids],current_points=target.reshape(-1,2)[ids]))
        uv,z=_project(xyz,previous)
        v&=(a>=.5)&(e<=.5)&(z>.5)&(z<200)&(np.linalg.norm(uv-source.reshape(-1,2)[ids],axis=1)<=.5)
        worlds[ids[v]]=xyz[v];key_ok.flat[ids[v]]=True
    # Reject measured discontinuities before connectivity; no semantic fill.
    rel=current.world_to_camera@np.linalg.inv(previous.world_to_camera)
    sr=np.c_[source.reshape(-1,2),np.ones(h*w)]@np.linalg.inv(previous.intrinsic).T
    uvh=(sr@rel[:3,:3].T)@current.intrinsic.T
    with np.errstate(invalid='ignore',divide='ignore'):parallax=target-(uvh[:,:2]/uvh[:,2:]).reshape(h,w,2)
    jump=np.maximum(np.linalg.norm(np.gradient(parallax,axis=0),axis=2),np.linalg.norm(np.gradient(parallax,axis=1),axis=2))
    boundary=~np.isfinite(jump)|(jump>config.local_parallax_jump_px)
    if config.surface_boundary_guard:candidate=candidate&~boundary
    rays=np.c_[target.reshape(-1,2),np.ones(h*w)]@np.linalg.inv(current.intrinsic).T
    # Two overlapping 16px cells schedule one 32px support domain. Surface
    # membership is measured connectivity, not the rectangular scheduling tile.
    step=config.surface_unit_px;size=step*2
    for y in range(0,h,step):
        for x in range(0,w,step):
            stopx=min(x+size,w);stopy=min(y+size,h);sl=np.s_[y:stopy,x:stopx];local=candidate[sl]
            if not local.any():continue
            _,labels=cv2.connectedComponents(local.astype(np.uint8),connectivity=4)
            if not config.surface_consensus_components:labels=local.astype(np.int32)
            for label in np.unique(labels[local]):
                mask=labels==label;sy,sx=np.nonzero(mask&key_ok[sl]);sx+=x;sy+=y
                unit=dict(bounds=[x,y,stopx,stopy],candidate_pixels=int(mask.sum()),key_count=len(sx),reason='key_support_missing',models=[])
                def reject(reason):unit['reason']=reason;counts[reason]+=1
                if config.surface_diagnostics:trace.append(unit)
                if len(sx)<8:reject('key_support_missing');continue
                ids=sy*w+sx;design=rays[ids]
                camera=worlds[ids]@current.world_to_camera[:3,:3].T+current.world_to_camera[:3,3]
                train=((sx//3+sy//3)%2)==0
                if train.sum()<4 or (~train).sum()<4:reject('withheld_support_missing');continue
                if not config.surface_consensus_withheld:train[:]=True
                models=training_models(design[train],1/camera[train,2],source[sy[train],sx[train]],current,previous,config.surface_consensus_split)
                if not models:reject('training_consensus_missing');continue
                ay,ax=np.nonzero(mask);ay+=y;ax+=x;allids=ay*w+ax
                predictions=[];hypotheses=[]
                for normal,fitids in models:
                    fit=np.flatnonzero(train)[fitids];fp=np.c_[sx[fit],sy[fit]]
                    # Fit support must span a genuine 2D area, independently of
                    # withheld locations. Eight keys on a line do not qualify.
                    if cv2.contourArea(cv2.convexHull(fp.astype(np.float32)))<=0:continue
                    uv,valid=plane_projection(design,normal,current,previous)
                    residual=np.linalg.norm(uv-source[sy,sx],axis=1)
                    inlier=valid&(residual<=.5)
                    held=~train&inlier
                    if config.surface_consensus_withheld and held.sum()<4:continue
                    # Unsupported keys do not validate the plane; each retained
                    # model needs independent held-out 2D spatial coverage.
                    if config.surface_consensus_withheld and cv2.contourArea(cv2.convexHull(np.c_[sx[held],sy[held]].astype(np.float32)))<=0:continue
                    supported=np.zeros(len(sx),bool);supported[fit]=True;supported|=held
                    hull=cv2.convexHull(np.c_[sx[supported]-x,sy[supported]-y].astype(np.int32))
                    interior=np.zeros(local.shape,np.uint8);cv2.fillConvexPoly(interior,hull,1)
                    # A contradictory key inside this support means a genuinely
                    # competing surface; carve its neighbourhood, never bridge it.
                    badkeys=~inlier
                    conflicts=np.zeros(local.shape,np.uint8);conflicts[sy[badkeys]-y,sx[badkeys]-x]=1
                    conflicts=cv2.dilate(conflicts,np.ones((7,7),np.uint8)).astype(bool)
                    predicted,pvalid=plane_projection(rays[allids],normal,current,previous)
                    ok=pvalid&(np.linalg.norm(predicted-source[ay,ax],axis=1)<=.5)&interior[ay-y,ax-x].astype(bool)&~conflicts[ay-y,ax-x]
                    if not config.surface_consensus_expand:ok&=key_ok[ay,ax]
                    if long is not None:
                        olduv,oldvalid=plane_projection(rays[allids],normal,current,long[0])
                        measured=long[2][ay,ax]
                        ok&=~measured|(oldvalid&(np.linalg.norm(olduv-long[1][ay,ax],axis=1)<=.5))
                    hypotheses.append(ok);predictions.append(predicted)
                    unit['models'].append(dict(normal=normal.tolist(),fit_current_uv=fp.tolist(),
                        withheld_current_uv=np.c_[sx[held],sy[held]].tolist(),withheld_errors=residual[held].tolist(),pixels=int(ok.sum())))
                if not hypotheses:reject('independent_consensus_missing');continue
                support=np.stack(hypotheses);accepted=support.any(0)
                for a in range(len(hypotheses)):
                    for b in range(a):accepted&=~(support[a]&support[b]&(np.linalg.norm(predictions[a]-predictions[b],axis=1)>.5))
                # Retain only support components connected to measured keys.
                out=np.zeros(local.shape,np.uint8);out[ay-y,ax-x]=accepted
                if config.surface_consensus_components:
                    _,labs=cv2.connectedComponents(out,connectivity=4)
                    live=np.unique(labs[(key_ok[sl])&out.astype(bool)]);live=live[live!=0]
                    out=np.isin(labs,live).astype(np.uint8)
                result[sl]|=out.astype(bool);unit['reason']='accepted' if out.any() else 'pixel_support_missing';counts[unit['reason']]+=1
    result&=candidate
    diag.update(surface_supported_pixels=int(result.sum()),surface_plane_failure_counts=dict(counts),
        surface_reason='connected_train_consensus_disjoint_validation_measured_pixels',surface_history_available=long is not None,
        surface_boundary_rejected_pixels=int((keys&boundary).sum()),surface_ms=(perf_counter()-tick)*1000)
    if config.surface_diagnostics:diag['_surface_unit_trace']=trace;diag['_surface_boundary_mask']=boundary
    return result,diag
