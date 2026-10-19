"""Measured centre recovery and dependency-scoped support; no reuse authority."""
import cv2
import numpy as np


def motion_risk_masks(anchors,previous,current,evidence,config,scene_id,signature):
    """Finite committed-source occupancy envelopes, for rejection only.

    Box hulls describe possible occupancy, never foreground ownership. Dirty
    additionally requires an actually changed same-coordinate gray sample.
    Conflicting source occupancy outside the predicted envelope is Unknown.
    """
    from .geometry import valid_projection_camera, project_points
    shape=current.image.shape[:2];dirty=np.zeros(shape,bool);unknown=np.zeros(shape,bool)
    details=dict(anchors_considered=0,envelopes=[],discarded=[],GT_used=False,
        policy='committed_source_occupancy_plus_actual_input_change_rejection_only')
    if (previous is None or previous.image.shape[:2]!=current.image.shape[:2] or not previous.timing_reliable
            or not current.timing_reliable or not all(valid_projection_camera(c) for c in (previous,current))
            or not 0<current.timestamp-previous.timestamp<=config.max_state_age_s):
        details['reason']='history_camera_time_unavailable';return dirty,unknown,details
    # Do not lose measured changes when the legacy dynamic exclusion later
    # clears changed_mask. This separate rejection evidence cannot grant links.
    dirty |= evidence.changed_mask
    if evidence.measured_change_risk_mask is not None:dirty |= evidence.measured_change_risk_mask
    from .temporal_visibility import gray_image
    # The committed observation store intentionally retains grayscale only.
    # This is measured rejection evidence, never a replacement Exact RGB gate.
    actual_change=gray_image(current)!=gray_image(previous)
    q=evidence.current_points
    finite=np.isfinite(q).all(1)&np.isfinite(evidence.fb_error_px)
    failed=finite&~evidence.correspondence_valid
    xy=np.rint(q[failed]).astype(int)
    xy=xy[(xy[:,0]>=0)&(xy[:,0]<shape[1])&(xy[:,1]>=0)&(xy[:,1]<shape[0])]
    unknown[xy[:,1],xy[:,0]]=True
    for anchor in anchors:
        if anchor.anchor_type!='object' or anchor.semantic_class in ('barrier','traffic_cone'):continue
        details['anchors_considered']+=1
        obs=anchor.observation_support.get(current.camera_id)
        age=current.timestamp-anchor.source_timestamp
        bound=(anchor.scene_id==scene_id and anchor.source_execution_signature==signature
            and anchor.source_frame_id and obs is not None and obs.timestamp==previous.timestamp
            and anchor.source_timestamp<=previous.timestamp and 0<age<=config.max_state_age_s)
        velocity=None if anchor.velocity_world is None else np.asarray(anchor.velocity_world)
        if (not bound or velocity is None or velocity.shape!=(3,) or not np.isfinite(velocity).all()
                or np.linalg.norm(velocity)<.05):
            details['discarded'].append(dict(anchor_id=anchor.anchor_id,reason='unbound_stale_or_no_measured_motion'));continue
        g=anchor.geometry_world
        points=np.concatenate((g.points,g.points+velocity*age))
        uv,z,front=project_points(points,current.intrinsic,current.world_to_camera)
        if not front.all() or not np.isfinite(uv).all():
            details['discarded'].append(dict(anchor_id=anchor.anchor_id,reason='sweep_crosses_camera_plane_unbounded'));continue
        hull=cv2.convexHull(uv.astype(np.float32)).reshape(-1,2)
        hull=cv2.convexHull(np.clip(hull,[-shape[1]*2,-shape[0]*2],[shape[1]*3,shape[0]*3]).astype(np.float32))
        envelope=np.zeros(shape,np.uint8);cv2.fillConvexPoly(envelope,np.rint(hull).astype(np.int32),1)
        envelope=envelope.astype(bool)
        # A source sample within a historical possible occupancy crop remains
        # physically trackable; that fact does not make changed pixels reusable.
        occupancy=np.zeros(shape,bool)
        if evidence.dense_source_map is not None:
            p=evidence.dense_source_map;r=obs.region
            occupancy=np.isfinite(p).all(2)&(p[...,0]>=r[0])&(p[...,0]<r[2])&(p[...,1]>=r[1])&(p[...,1]<r[3])
        risk=(envelope|occupancy)&actual_change
        dirty |= risk&envelope
        unknown |= risk&~envelope
        details['envelopes'].append(dict(anchor_id=anchor.anchor_id,source_frame=anchor.source_frame_id,
            source_timestamp=anchor.source_timestamp,source_state_epoch=anchor.source_state_epoch,
            semantic_class=anchor.semantic_class,velocity_world=velocity.tolist(),velocity_reliable=anchor.velocity_reliable,
            source_region=list(obs.region),hull=hull.reshape(-1,2).tolist(),envelope_pixels=int(envelope.sum()),
            changed_envelope_pixels=int((risk&envelope).sum()),ambiguous_source_pixels=int((risk&~envelope).sum()),
            foreground_ownership=False))
    unknown &= ~dirty
    return dirty,unknown,details


def change_guard(dirty, evidence, config):
    seeds=dirty.copy()
    if evidence.measured_change_risk_mask is not None:
        seeds |= evidence.measured_change_risk_mask
    radius=config.support_patch_radius_px  # measured local support footprint
    guarded=cv2.dilate(seeds.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
    return guarded,dict(seed_pixels=int(seeds.sum()),guard_radius_px=radius,
        added_dirty_pixels=int((guarded&~dirty).sum()),online_measurements_only=True)


def measured_links(old, camera, evidence, candidate, dirty, config):
    """Dense pixels plus separately checked sparse centres, with dedup/conflicts."""
    from .temporal_visibility import gray_image
    h,w=candidate.shape
    dense=candidate.copy() if evidence.dense_source_map is not None else np.zeros_like(candidate)
    sparse=np.zeros_like(candidate);conflict=np.zeros_like(candidate)
    valid=evidence.supported & evidence.correspondence_valid & evidence.geometry_valid
    p=evidence.source_points[valid];q=evidence.current_points[valid]
    finite=np.isfinite(p).all(1)&np.isfinite(q).all(1)
    p,q=p[finite],q[finite]
    xy=np.rint(q).astype(int)
    inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    inside &= (p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)
    p,q,xy=p[inside],q[inside],xy[inside]
    excluded=evidence.observation_exclusion_mask
    eligible=~dirty[xy[:,1],xy[:,0]]
    if excluded is not None:eligible &= ~excluded[xy[:,1],xy[:,0]]
    old_gray=gray_image(old).astype(np.float32)/255.
    current_gray=gray_image(camera).astype(np.float32)/255.
    def sample(im, points):
        if not len(points):return np.empty(0,np.float32)
        # OpenCV remap dimensions are signed short; dense links can exceed 32k.
        return np.concatenate([cv2.remap(im,q[:,0].reshape(-1,1).astype(np.float32),
            q[:,1].reshape(-1,1).astype(np.float32),cv2.INTER_LINEAR)[:,0]
            for q in (points[i:i+16384] for i in range(0,len(points),16384))])
    center_residual=np.abs(sample(old_gray,p)-sample(current_gray,q))
    eligible &= np.isfinite(center_residual)&(center_residual<=config.support_residual_max)
    p,q,xy=p[eligible],q[eligible],xy[eligible]
    chosen={}
    for a,b,c in zip(p,q,xy):
        x,y=map(int,c);key=(x,y)
        if key in chosen and np.linalg.norm(chosen[key][0]-a)>config.flow_fb_max_px:
            conflict[y,x]=True
        elif dense[y,x] and np.linalg.norm(evidence.dense_source_map[y,x]-a)>config.flow_fb_max_px:
            conflict[y,x]=True
        else:chosen.setdefault(key,(a,b));sparse[y,x]=True
    both=dense&sparse&~conflict
    merged=(dense|sparse)&~conflict&~dirty
    yy,xx=np.nonzero(merged)
    target=np.column_stack((xx,yy)).astype(np.float32)
    source=evidence.dense_source_map[yy,xx].copy() if evidence.dense_source_map is not None else np.empty_like(target)
    origins=np.zeros(len(target),np.uint8)  # 0=dense, 1=sparse, 2=both
    locations=np.flatnonzero(merged)
    for (x,y),(a,b) in chosen.items():
        key=y*w+x;i=int(np.searchsorted(locations,key))
        if i==len(locations) or locations[i]!=key:continue
        if both[y,x]:origins[i]=2
        else:source[i]=a;target[i]=b;origins[i]=1
    residual=np.abs(sample(old_gray,source)-sample(current_gray,target))
    diagnostics=dict(dense_only=int((dense&~sparse&~conflict&~dirty).sum()),
        sparse_only=int((sparse&~dense&~conflict&~dirty).sum()),both=int(both.sum()),
        measurement_conflicts=int(conflict.sum()),invalid_coordinates=int((~finite).sum()+(~inside).sum()),
        supported_centres_considered=int(valid.sum()),new_real_correspondences=int((sparse&~dense&merged).sum()),
        new_wrong_correspondences=None,physical_quality='unscorable_without_offline_reference')
    return merged,source,target,residual,origins,diagnostics,conflict


def scope_support(candidate, dirty, footprint=None, complete_scope=True):
    """Count observed input pixels, including exact strided/dilated Conv footprints."""
    h,w=candidate.shape
    if footprint is None:
        supported=int(candidate.sum());conflicts=int(dirty.sum());total=h*w
    else:
        x0,y0,x1,y1=footprint['token_region']
        kh,kw=footprint['kernel'];sh,sw=footprint['stride']
        ph,pw=footprint['padding'];dh,dw=footprint['dilation']
        yy=np.unique((np.arange(y0,y1)[:,None]*sh-ph+np.arange(kh)*dh).ravel())
        xx=np.unique((np.arange(x0,x1)[:,None]*sw-pw+np.arange(kw)*dw).ravel())
        yy=yy[(yy>=0)&(yy<h)];xx=xx[(xx>=0)&(xx<w)]
        total=len(yy)*len(xx)
        supported=int(candidate[np.ix_(yy,xx)].sum());conflicts=int(dirty[np.ix_(yy,xx)].sum())
    ratio=supported/total if total else 0.
    return dict(observation_support_count=supported,observation_support_ratio=ratio,
        observed_conflict_count=conflicts,unobserved_dependency_ratio=1-ratio,
        dependency_pixel_count=total,observation_valid=bool(total and supported==total and conflicts==0 and complete_scope),
        observed_conflict=bool(conflicts),partial_observation_only=bool(supported and (supported<total or not complete_scope)),
        complete_dependency_observation=bool(total and supported==total and conflicts==0 and complete_scope),
        transformer_dependency_certified=False)


def source_centre_integrals(source, evidence, dirty):
    """One measured source centre per raster cell; never a filled source patch."""
    shape=dirty.shape;h,w=shape
    support=np.zeros(shape,np.uint8);conflict=np.zeros(shape,np.uint8)
    def raster(points):
        p=points[np.isfinite(points).all(1)]
        xy=np.rint(p).astype(int)
        return xy[(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)]
    xy=raster(source);support[xy[:,1],xy[:,0]]=1
    if evidence is not None:
        valid=evidence.correspondence_valid&evidence.geometry_valid
        p=evidence.source_points[valid];q=evidence.current_points[valid]
        finite=np.isfinite(p).all(1)&np.isfinite(q).all(1)
        p,q=p[finite],q[finite];xy=np.rint(q).astype(int)
        inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        p,xy=p[inside],xy[inside]
        bad=dirty[xy[:,1],xy[:,0]]
        xy=raster(p[bad]);conflict[xy[:,1],xy[:,0]]=1
    support[conflict.astype(bool)]=0
    return cv2.integral(support),cv2.integral(conflict)


def descriptor_scope_support(region, integrals, shape, direct_reference=True):
    """Historical descriptor's actual crop dependency, not its projected box fill."""
    from .geometry import region_slices
    yy,xx=region_slices(region,shape);x0,x1=xx.start,xx.stop;y0,y1=yy.start,yy.stop
    def count(a):return int(a[y1,x1]-a[y0,x1]-a[y1,x0]+a[y0,x0])
    total=max(0,y1-y0)*max(0,x1-x0)
    supported=count(integrals[0]) if direct_reference and total else 0
    conflicts=count(integrals[1]) if direct_reference and total else 0
    ratio=supported/total if total else 0.
    return dict(observation_support_count=supported,observation_support_ratio=ratio,
        observed_conflict_count=conflicts,unobserved_dependency_ratio=1-ratio,
        dependency_pixel_count=total,observation_valid=False,observed_conflict=bool(conflicts),
        partial_observation_only=bool(supported),complete_dependency_observation=False,
        support_count_semantics='unique_measured_source_centres_rasterized_once',
        support_coordinate_frame='historical_descriptor_input',
        observation_reference_frame_mismatch=not direct_reference,
        source_region=[[x0,y0,x1,y1]],dependency_scope='local_observation',
        transformer_dependency_certified=False)
