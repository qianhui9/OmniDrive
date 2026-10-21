"""Bounded candidate evaluation. Predictions propose; actual RGB gates select.

No weighted score trades away a necessary gate. Conflicting fully supported
candidates remain Unknown; a candidate failing a necessary gate cannot veto a
different fully measured candidate. Static geometry is inappropriate for dynamic
identity views and must be explicitly disabled there, not implicitly bypassed.
"""
import cv2
import numpy as np
from .geometry_support import measured_patch_support, calibrated_correspondence


def retain_depth_points(points, history, old, config, region=None):
    """Append actual last-commit tracked pixels, never relocate/interpolate depth.

    Keep EVERY newly detected corner so the original support denominator remains
    visible. Add at most flow_max_points tracked depth rays. All appended points
    also count in the actual Anchor denominator; no difficult corners are removed.
    """
    p=np.asarray(points,np.float32).reshape(-1,2)
    d=dict(status='Not Run',redetected_points=len(p),depth_records=0,retained_points=0,
           duplicate_or_near_existing=0,original_points_preserved=True)
    if not config.persistent_depth_tracks:return p,d
    d['status']='no_depth_history'
    if history is None:return p,d
    from .identity_temporal import camera_snapshot_digest
    if history.get('camera_sha256')!=camera_snapshot_digest(old) or history.get('timestamp')!=old.timestamp:
        d['status']='source_binding_rejected';return p,d
    hp=np.asarray(history['pixels'],np.float32).reshape(-1,2);hz=np.asarray(history['depths']);hu=np.asarray(history['uncertainty_m'])
    d['depth_records']=len(hp)
    if len(hp)!=len(hz) or len(hu)!=len(hp):raise ValueError('Malformed tracked depth history')
    good=np.isfinite(hp).all(1)&np.isfinite(hz)&np.isfinite(hu)&(hu>=0)&(hz-hu>=.5)&(hz+hu<=200)
    h,w=old.image.shape[:2];good&=(hp[:,0]>=3)&(hp[:,0]<w-3)&(hp[:,1]>=3)&(hp[:,1]<h-3)
    if region is not None:
        good&=(hp[:,0]>=region[0])&(hp[:,0]<region[2])&(hp[:,1]>=region[1])&(hp[:,1]<region[3])
    if len(p):
        from scipy.spatial import cKDTree
        distance,_=cKDTree(p).query(hp)
        d['duplicate_or_near_existing']=int((good&(distance<=config.flow_fb_max_px)).sum())
        good&=distance>config.flow_fb_max_px
    # Stable actual track order, no sorting on current target/labels/acceptance.
    retained=hp[good][:config.flow_max_points]
    d.update(status='retained' if len(retained) else 'no_distinct_valid_depth_rays',retained_points=len(retained))
    return np.concatenate([p,retained]),d


def select_candidates(old,new,points,candidates,config,static_geometry=True):
    from .temporal_visibility import gray_image
    a,b=gray_image(old),gray_image(new);p=np.asarray(points,float);n=len(p)
    evaluated=[];records={}
    for name,c in candidates.items():
        q,fb,lk=c['points'],c['fb'],c['valid']
        geometric=(calibrated_correspondence(old,new,p,q,config.flow_epipolar_max_px)
                   if static_geometry else np.ones(n,bool))
        _,_,photo,residual=measured_patch_support(a,b,p,q,lk&geometric,config)
        accepted=lk&geometric&photo&(residual<=config.support_residual_max)
        evaluated.append((name,c,accepted))
        records[name]=dict(points=q,fb_error_px=fb,lk_valid=lk,geometry_valid=geometric,
            patch_supported=photo,patch_residual=residual,fully_supported=accepted,
            attempted=c.get('attempted',np.ones(n,bool)),prediction_residual_px=c.get('prediction_residual_px'))
    # Stable order preserves original raw solution when the fully valid
    # measurements agree. This is not a score or an appearance-only identity.
    q=candidates['raw']['points'].copy();fb=candidates['raw']['fb'].copy()
    good=np.zeros(n,bool);chosen=np.full(n,'unknown',dtype=object);conflict=np.zeros(n,bool)
    for name,c,accepted in evaluated:
        clash=good&accepted&(np.linalg.norm(q-c['points'],axis=1)>config.flow_fb_max_px)
        conflict|=clash
        take=~good&accepted
        q[take]=c['points'][take];fb[take]=c['fb'][take];chosen[take]=name;good|=accepted
    good&=~conflict;chosen[conflict]='ambiguous'
    _,inverse,counts=np.unique(np.rint(q/2).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    duplicates=good&(counts[inverse]>1);good&=~duplicates;chosen[duplicates]='duplicate'
    totals={name:dict(attempted=int(np.asarray(r['attempted']).sum()),lk_valid=int(r['lk_valid'].sum()),
        geometric=int((r['lk_valid']&r['geometry_valid']).sum()),local_supported=int(r['fully_supported'].sum()),
        selected=int((good&(chosen==name)).sum())) for name,r in records.items()}
    return q,fb,good,dict(candidates=records,selected_candidate=chosen.tolist(),
        candidate_funnel=totals,ambiguity_rejected=int(conflict.sum()),duplicate_rejected=int(duplicates.sum()),
        static_geometry_required=static_geometry)


def measured_aligned_descriptor(old,previous,current,projection,evidence,validator):
    """Only measured affine agreement permits resampling a historical crop.

    Same residual/texture threshold as the original descriptor; fit and withheld
    spatial witnesses both obey the existing 0.75px FB budget. No box-only warp.
    """
    from .temporal_visibility import gray_image
    c=validator.config;p=evidence.source_points;q=evidence.current_points;r=old.region;s=projection.region
    inside=(p[:,0]>=r[0])&(p[:,0]<r[2])&(p[:,1]>=r[1])&(p[:,1]<r[3])
    support=inside&evidence.supported&(q[:,0]>=s[0])&(q[:,0]<s[2])&(q[:,1]>=s[1])&(q[:,1]<s[3])
    idx=np.flatnonzero(support);d=dict(status='unknown',reason='insufficient_measured_alignment',total_source_points=int(inside.sum()),points=len(idx))
    if len(idx)<max(8,c.support_min_points) or len(idx)/max(1,int(inside.sum()))<c.support_point_fraction:return None,d
    if (previous is None or previous.timestamp!=old.timestamp or evidence.source_timestamp!=old.timestamp
            or evidence.timestamp!=current.timestamp or previous.camera_id!=current.camera_id):return None,dict(d,reason='alignment_source_mismatch')
    if not np.array_equal(validator.descriptor(previous.image,r),old.descriptor):return None,dict(d,reason='source_descriptor_mismatch')
    def quadrants(x,region):
        bits=np.clip(((x-np.asarray(region[:2]))/np.maximum(np.asarray(region[2:])-region[:2],1)*2).astype(int),0,1)
        return len(np.unique(bits,axis=0))/4
    if min(quadrants(p[idx],r),quadrants(q[idx],s))<c.support_quadrant_fraction:return None,dict(d,reason='alignment_spatial_support')
    fit,hold=idx[::2],idx[1::2];design=np.c_[p[fit],np.ones(len(fit))]
    matrix,_,rank,_=np.linalg.lstsq(design,q[fit],rcond=None);matrix=matrix.T
    error=np.linalg.norm(np.c_[p[idx],np.ones(len(idx))]@matrix.T-q[idx],axis=1)
    det=np.linalg.det(matrix[:,:2])
    if rank<3 or not np.isfinite(matrix).all() or not .5<=det<=2 or (error>c.flow_fb_max_px).any():return None,dict(d,reason='alignment_fit_or_withheld_error')
    # Pixel validity must cover the actual descriptor crop, no border synthesis.
    gray=gray_image(current);h,w=gray.shape
    valid=cv2.warpAffine(np.ones((h,w),np.uint8),matrix,(w,h),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)
    from .geometry import region_slices
    crop=valid[region_slices(r,(h,w))]
    if not crop.size or crop.mean()<c.min_coverage:return None,dict(d,reason='alignment_outside_view')
    warped=cv2.warpAffine(gray,matrix,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    return validator.descriptor(warped,r),dict(d,status='measured',reason='withheld_measured_affine',
        matrix=matrix.tolist(),max_fit_and_withheld_error_px=float(error.max()),withheld_points=len(hold),
        foreground_certified=False,metric_geometry_certified=False)
