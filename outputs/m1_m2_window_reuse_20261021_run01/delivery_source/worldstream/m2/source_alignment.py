"""Measured M1 links -> token proposals; partial support never grants reuse."""
from dataclasses import dataclass
import numpy as np


@dataclass
class SourceTokenMap:
    source_indices: np.ndarray
    source_coordinates: np.ndarray
    support_count: np.ndarray
    support_ratio: np.ndarray
    translation_error_px: np.ndarray
    conflict: np.ndarray
    complete: np.ndarray
    dirty_ratio: np.ndarray
    unknown_ratio: np.ndarray
    camera_id: object
    source_frame_id: object
    current_frame_id: object


def token_sources(link, candidate, dirty, unknown, patch_size, expected_source_frame,
                  camera_id, current_frame, tolerance_px=.5):
    """Count unique actual pixel centres; no centre/box dilation or GT access.

    Fractional coordinates are retained for diagnostic content interpolation.
    Admission requires an entire observed patch and consistent integer-token
    translation. Competing source pixels/tokens are rejected, never averaged
    into an acceptance score. The mean displacement is a diagnostic location.
    """
    h,w=candidate.shape;ph,pw=(patch_size,patch_size) if isinstance(patch_size,int) else patch_size
    th,tw=h//ph,w//pw;n=th*tw
    count=np.zeros(n,np.int32);conflict=np.zeros(n,bool);delta=np.zeros((n,2),np.float64)
    low=np.full((n,2),np.inf);high=np.full((n,2),-np.inf)
    xygrid=np.column_stack((np.tile(np.arange(tw),th),np.repeat(np.arange(th),tw))).astype(float)
    bound=(link.get('source_frame_id')==expected_source_frame and link.get('source_camera_id')==camera_id
        and link.get('coordinate_space')=='preprocessed_image' and link.get('validity_lease',{}).get('source_valid') is True)
    p=np.asarray(link.get('source_points',np.empty((0,2))),float);q=np.asarray(link.get('current_points',np.empty((0,2))),float)
    if p.shape!=q.shape or p.ndim!=2 or p.shape[1:]!=(2,):bound=False
    if bound and len(p):
        finite=np.isfinite(p).all(1)&np.isfinite(q).all(1);p,q=p[finite],q[finite]
        xy=np.rint(q).astype(np.int64)
        inside=(xy[:,0]>=0)&(xy[:,0]<tw*pw)&(xy[:,1]>=0)&(xy[:,1]<th*ph)
        inside&=(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)
        p,q,xy=p[inside],q[inside],xy[inside]
        eligible=candidate[xy[:,1],xy[:,0]]&~dirty[xy[:,1],xy[:,0]]&~unknown[xy[:,1],xy[:,0]]
        p,q,xy=p[eligible],q[eligible],xy[eligible]
        pixel=xy[:,1]*w+xy[:,0]
        unique,inverse=np.unique(pixel,return_inverse=True)
        mins=np.full((len(unique),2),np.inf);maxs=np.full_like(mins,-np.inf)
        np.minimum.at(mins,inverse,p-q);np.maximum.at(maxs,inverse,p-q)
        centres=np.column_stack((unique%w,unique//w))
        token=centres[:,1]//ph*tw+centres[:,0]//pw
        bad=np.any(maxs-mins>tolerance_px,axis=1)
        np.logical_or.at(conflict,token,bad)
        good=~bad;token=token[good];d=(mins[good]+maxs[good])/2
        np.add.at(count,token,1);np.add.at(delta,token,d)
        np.minimum.at(low,token,d);np.maximum.at(high,token,d)
    mean=delta/np.maximum(count[:,None],1)
    coordinates=xygrid+mean/np.array([pw,ph])
    integer=np.rint(coordinates).astype(np.int64)
    source=integer[:,1]*tw+integer[:,0]
    in_bounds=(integer[:,0]>=0)&(integer[:,0]<tw)&(integer[:,1]>=0)&(integer[:,1]<th)
    offset=(integer-xygrid)*np.array([pw,ph])
    error=np.maximum(np.max(np.abs(low-offset),axis=1),np.max(np.abs(high-offset),axis=1))
    error[count==0]=np.inf
    # A source token proposed for multiple destinations has ambiguous occupancy.
    observed=(count>0)&in_bounds
    _,inv,frequency=np.unique(source[observed],return_inverse=True,return_counts=True)
    conflict[np.flatnonzero(observed)[frequency[inv]>1]]=True
    dr=dirty[:th*ph,:tw*pw].reshape(th,ph,tw,pw).mean(axis=(1,3)).ravel()
    ur=unknown[:th*ph,:tw*pw].reshape(th,ph,tw,pw).mean(axis=(1,3)).ravel()
    complete=(count==ph*pw)&~conflict&in_bounds&(error<=tolerance_px)&(dr==0)&(ur==0)&bound
    source[~observed]=-1
    return SourceTokenMap(source.reshape(th,tw),coordinates.reshape(th,tw,2),count.reshape(th,tw),
        (count/(ph*pw)).reshape(th,tw),error.reshape(th,tw),conflict.reshape(th,tw),complete.reshape(th,tw),
        dr.reshape(th,tw),ur.reshape(th,tw),camera_id,expected_source_frame,current_frame)


def source_windows(mapping, window_size):
    """Whole closed windows only; source layout and real padding must match."""
    h,w=mapping.complete.shape;nh,nw=(h+window_size-1)//window_size,(w+window_size-1)//window_size
    result=np.full(nh*nw,-1,np.int64)
    for wy in range(nh):
        for wx in range(nw):
            y,x=wy*window_size,wx*window_size;hh,ww=min(window_size,h-y),min(window_size,w-x)
            if not mapping.complete[y:y+hh,x:x+ww].all():continue
            sources=mapping.source_indices[y:y+hh,x:x+ww]
            sy,sx=np.divmod(sources,w);dy,dx=sy-np.arange(y,y+hh)[:,None],sx-np.arange(x,x+ww)[None,:]
            if not (np.all(dy==dy[0,0]) and np.all(dx==dx[0,0])):continue
            y0,x0=int(sy[0,0]),int(sx[0,0])
            if y0%window_size or x0%window_size:continue
            if min(window_size,h-y0)!=hh or min(window_size,w-x0)!=ww:continue
            result[wy*nw+wx]=y0//window_size*nw+x0//window_size
    return result


def probe_features(previous,current,mapping):
    """Unqualified errors for measured partial-token proposals, not admission."""
    import torch
    import torch.nn.functional as F
    h,w,c=current.shape
    coords=torch.as_tensor(mapping.source_coordinates,device=current.device,dtype=current.dtype)
    grid=torch.stack((2*(coords[...,0]+.5)/w-1,2*(coords[...,1]+.5)/h-1),-1)
    warped=F.grid_sample(previous.permute(2,0,1)[None],grid[None],mode='bilinear',align_corners=False)[0].permute(1,2,0)
    valid=torch.as_tensor((mapping.support_count>0)&~mapping.conflict,device=current.device)
    valid&=(coords[...,0]>=0)&(coords[...,0]<=w-1)&(coords[...,1]>=0)&(coords[...,1]<=h-1)
    valid&=torch.isfinite(warped).all(-1)
    if not valid.any():return dict(tokens=0,status='no_measured_token_proposal',reuse_authorized=False)
    diff=(warped-current).float()[valid];ref=current.float()[valid]
    norms=diff.norm(dim=-1)
    return dict(tokens=int(valid.sum()),max_abs=float(diff.abs().max()),relative_l2=float(diff.norm()/ref.norm().clamp_min(1e-12)),
        token_error_p50=float(norms.median()),token_error_p95=float(torch.quantile(norms,.95)),
        complete_supported_tokens=int(mapping.complete.sum()),reuse_authorized=False,
        positions='Patch content is before absolute position; contextual errors include historical position/RoPE and are diagnostic only.')
