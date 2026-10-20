"""Online, same-sensor temporal visibility evidence from the last committed image."""
from time import perf_counter
import cv2
import numpy as np
from .state_types import OnlineVisibilityEvidence
from .geometry_support import calibrated_correspondence, measured_patch_support


def gray_image(camera):
    from .measurement_cache import MEASUREMENTS
    cache = MEASUREMENTS.get()
    if cache is not None and id(camera) in cache['gray']:
        owner, gray = cache['gray'][id(camera)]
        if owner is camera:return gray
    image = camera.image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
    if cache is not None:cache['gray'][id(camera)] = (camera,gray)
    return gray


def build_evidence(previous, current, config, dense_exclusion=None, surface_history=None,
                   validate_dense=True, depth_history=None):
    start = perf_counter(); gray = gray_image(current)
    h,w = gray.shape
    points = np.empty((0,2), np.float32)
    evidence = OnlineVisibilityEvidence(current.camera_id, previous.timestamp if previous else float('nan'),
        current.timestamp, points, points.copy(), np.empty(0), np.empty(0,bool), np.empty(0,bool),
        np.empty(0,bool), np.empty(0), np.zeros((h,w),bool), np.zeros((h,w),bool))
    d = evidence.diagnostics
    d.update(reason='no_history', temporal_correspondence_ms=0., visibility_verification_ms=0., geometry_support_ms=0.)
    if previous is None: return evidence
    dt = current.timestamp-previous.timestamp
    if (not np.isfinite(dt) or not 0 < dt <= config.max_state_age_s or previous.image.shape[:2] != (h,w)
            or not previous.timing_reliable or not current.timing_reliable):
        d['reason']='incompatible_history'; return evidence
    old_gray = gray_image(previous)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=config.flow_max_points, qualityLevel=.01,
                                 minDistance=config.flow_min_distance_px, blockSize=5)
    if p0 is None:p0=np.empty((0,1,2),np.float32)
    from .correspondence_recovery import retain_depth_points
    source,seed_diagnostics=retain_depth_points(p0[:,0],depth_history,previous,config)
    d['depth_feature_seeding']=seed_diagnostics
    p0=source.reshape(-1,1,2)
    if not len(p0):d['reason']='insufficient_texture';return evidence
    if config.ego_motion_mode != 'off':
        from .ego_motion import track_with_prediction
        source = p0[:,0]
        target,fb,good,motion = track_with_prediction(previous,current,source,config,depth_history)
        evidence.motion_measurements = motion
        d['ego_motion'] = dict(motion['info'], raw_valid=int(motion['raw_valid'].sum()),
            recovered=int(motion['seed_recovered'].sum()), conflicts=int(motion['initialization_conflict'].sum()))
        if 'candidate_funnel' in motion:d['candidate_funnel']=motion['candidate_funnel']
        collision = np.zeros(len(source),bool)  # uniqueness already checked after merge
    else:
        p1, s1, error = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, winSize=(21,21), maxLevel=3,
                                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        if p1 is None: d['reason']='forward_failed'; return evidence
        # Sanitize failed coordinates before the reverse OpenCV call.
        p1 = np.where(np.isfinite(p1),p1,p0)
        back, s0, _ = cv2.calcOpticalFlowPyrLK(gray, old_gray, p1, None, winSize=(21,21), maxLevel=3,
                                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        if back is None: d['reason']='backward_failed'; return evidence
        source, target = p0[:,0], p1[:,0]
        fb = np.linalg.norm(back[:,0]-source,axis=1)
        good = s1[:,0].astype(bool) & s0[:,0].astype(bool) & (fb <= config.flow_fb_max_px)
        good &= np.isfinite(target).all(1) & (error[:,0] <= config.flow_lk_max_error)
        good &= (target[:,0] >= 0)&(target[:,0]<w)&(target[:,1]>=0)&(target[:,1]<h)
        # Many-to-one correspondences are possible occlusion, never affirmative evidence.
        cells = np.rint(target/2).astype(np.int64)
        _, inverse, counts = np.unique(cells,axis=0,return_inverse=True,return_counts=True)
        collision = counts[inverse] > 1
        good &= ~collision
    d['temporal_correspondence_ms']=(perf_counter()-start)*1000
    t=perf_counter()
    geom = calibrated_correspondence(previous,current,source,target,config.flow_epipolar_max_px)
    d['visibility_verification_ms']=(perf_counter()-t)*1000
    t=perf_counter()
    stable, changed, supported, residual = measured_patch_support(old_gray,gray,source,target,good&geom,config)
    d['geometry_support_ms']=(perf_counter()-t)*1000
    evidence.source_points=source; evidence.current_points=target; evidence.fb_error_px=fb
    evidence.correspondence_valid=good; evidence.geometry_valid=geom; evidence.supported=supported
    evidence.patch_residual=residual; evidence.stable_mask=stable; evidence.changed_mask=changed
    if config.observation_diagnostics:
        from .state_types import json_value
        m=evidence.motion_measurements or {}
        d['point_measurements']=json_value(dict(source_points=source,current_points=target,
            fb_error_px=fb,correspondence_valid=good,geometry_valid=geom,patch_supported=supported,
            patch_residual=residual,**{k:v for k,v in m.items() if k!='source_points'}))
    if config.foreground_correspondence or config.identity_continuity:
        # Preserve measured proposal points separately from rejection masks.
        # They cannot grant ordinary observation/pixel support to dynamic ROIs.
        evidence.raw_correspondence_supported=supported.copy()
    if config.dense_local_support and validate_dense:
        from .dense_support import dense_support
        dense,source_map,diagnostics=dense_support(previous,current,config,exclusion=dense_exclusion,surface_history=surface_history,
            gray_pair=(old_gray,gray) if config.efficient_state else None)
        evidence.local_rejection_mask=diagnostics.pop('_local_rejection_mask',None)
        evidence.surface_units=diagnostics.pop('_surface_unit_trace',None)
        evidence.surface_boundary_mask=diagnostics.pop('_surface_boundary_mask',None)
        evidence.dense_measurements=diagnostics.pop('_dense_measurements',None)
        evidence.measured_change_risk_mask=diagnostics.pop('_measured_change_risk_mask',None)
        evidence.stable_mask=dense & ~changed
        evidence.dense_source_map=source_map
        evidence.source='same_rgb_dense_bidirectional_ray_photo_neighbourhood'
        d.update(diagnostics)
    d.update(reason='measured', points=len(source), bidirectional_points=int(good.sum()),
             geometric_points=int((good&geom).sum()), supported_points=int(supported.sum()),
             collision_count=int(collision.sum()), source='same_RGB_not_independent_sensor')
    return evidence


def anchor_visibility(evidence, old, projection, config, anchor=None, previous=None, current=None):
    """Certify the observed crop support, not foreground ownership or full box surface."""
    if evidence is None or old is None or abs(evidence.source_timestamp-old.timestamp)>1e-5:
        return False, 'temporal_source_missing', {}
    p,q = evidence.source_points,evidence.current_points
    r,s = old.region,projection.region
    inside = (p[:,0]>=r[0])&(p[:,0]<r[2])&(p[:,1]>=r[1])&(p[:,1]<r[3])
    # A static/rationally predicted box provides a bounded per-point affine prediction.
    size = np.maximum(np.array(r[2:])-r[:2],1)
    expected = (p-np.array(r[:2]))/size*(np.array(s[2:])-s[:2])+np.array(s[:2])
    prediction_kind='legacy_box_affine'
    prediction_valid=np.ones(len(p),bool)
    residual=np.linalg.norm(q-expected,axis=1)
    if config.ego_motion_mode != 'off':
        from .ego_motion import point_predictions,static_box_predictions,segment_residual
        prediction_kind='pose_rotation_depth_unknown'
        if previous is None or current is None:
            prediction_valid[:]=False
        else:
            expected,prediction_valid,spread,_,_=point_predictions(previous,current,p,config)
            residual=np.linalg.norm(q-expected,axis=1)+spread
            prediction_valid&=np.isfinite(spread)
            m=evidence.motion_measurements
            if config.ego_motion_mode=='depth' and m is not None:
                depth_valid=np.isfinite(m['source_depth_m'])&m['prediction_valid']&np.isfinite(m['prediction_spread_px'])
                prior_residual=np.linalg.norm(q-m['predicted_points'],axis=1)+m['prediction_spread_px']
                residual=np.where(depth_valid,prior_residual,residual)
                prediction_valid|=depth_valid
                prediction_kind='prior_measured_per_point_conditional_static_depth'
            if config.ego_motion_mode=='depth' and anchor is not None:
                a,b,box_valid=static_box_predictions(anchor,previous,current,p,config)
                residual=np.where(box_valid,segment_residual(q,a,b),residual)
                prediction_valid|=box_valid
                prediction_kind='per_ray_box_depth_interval_conditional_static'
    bounded = prediction_valid & (residual <= projection.search_radius_px)
    in_current = (q[:,0]>=s[0])&(q[:,0]<s[2])&(q[:,1]>=s[1])&(q[:,1]<s[3])
    accepted=inside & evidence.supported & bounded & in_current
    total=int(inside.sum()); count=int(accepted.sum())
    quadrants=np.minimum(((p[accepted]-r[:2])/size*2).astype(int),1)
    coverage=len(np.unique(quadrants,axis=0))/4 if count else 0.
    fraction=count/total if total else 0.
    d=dict(selected_points=total,supported_points=count,support_fraction=fraction,quadrant_fraction=coverage,
           source=evidence.source,foreground_ownership_certified=False,
           prediction_kind=prediction_kind,metric_geometry_certified=False,
           prediction_valid_points=int((inside&prediction_valid).sum()),
           prediction_residual_px=residual[inside].tolist())
    ok=(count>=config.support_min_points and fraction>=config.support_point_fraction and coverage>=config.support_quadrant_fraction)
    return bool(ok), 'ok' if ok else 'temporal_support_insufficient', d
