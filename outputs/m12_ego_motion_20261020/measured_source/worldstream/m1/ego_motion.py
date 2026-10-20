"""Camera-time world predictions for SEARCH, never independent metric/identity proof.

Rotation is depth-free. Translation needs depth: prior measured RGB triangulation
is explicitly conditional on static rays, with finite depth uncertainty. No GT,
detector velocity, current target measurement or global homography supplies depth.
"""
import cv2
import numpy as np
from .geometry import valid_projection_camera, transform_points, project_points, immutable_array


def camera_pair_reason(old, new, config):
    if old is None:
        return 'source_camera_missing'
    if (old.camera_id != new.camera_id or old.coordinate_space != new.coordinate_space
            or old.image.shape[:2] != new.image.shape[:2]):
        return 'camera_coordinate_mismatch'
    if not old.timing_reliable or not new.timing_reliable:
        return 'camera_time_unknown'
    if not np.isfinite([old.timestamp, new.timestamp]).all() or not 0 < new.timestamp-old.timestamp <= config.max_state_age_s:
        return 'camera_time_outside_window'
    if not all(valid_projection_camera(c) for c in (old, new)):
        return 'invalid_calibration_or_preprocessing'
    # A valid homography must map raw pixel rays to the supplied image K. Input
    # adapters bind that mapping to execution signatures; malformed maps reject.
    for c in (old, new):
        if c.raw_to_image is not None:
            a = np.asarray(c.raw_to_image)
            if a.shape != (3, 3) or not np.isfinite(a).all() or abs(np.linalg.det(a)) < 1e-10:
                return 'invalid_preprocessing_mapping'
    return None


def project_source_depth(old, new, points, depths):
    """Optical z depth, column-vector SE(3); returned points are current pixels."""
    p = np.asarray(points, float).reshape(-1, 2)
    z = np.broadcast_to(np.asarray(depths, float), (len(p),))
    rays = np.c_[p, np.ones(len(p))] @ np.linalg.inv(old.intrinsic).T
    xyz = rays * z[:, None]
    world = transform_points(xyz, np.linalg.inv(old.world_to_camera))
    uv, target_z, front = project_points(world, new.intrinsic, new.world_to_camera)
    valid = front & np.isfinite(uv).all(1) & np.isfinite(z) & (z >= .5) & (z <= 200)
    return uv, valid, world, target_z


def point_predictions(old, new, points, config, history=None):
    p = np.asarray(points, float).reshape(-1, 2);n = len(p)
    q = p.copy();valid = np.zeros(n, bool);depth_valid = valid.copy()
    spread = np.full(n, np.inf);depths = np.full(n, np.nan)
    reason = camera_pair_reason(old, new, config)
    info = dict(reason=reason or 'rotation_search_hypothesis', depth_kind='unavailable',
                metric_geometry_certified=False, static_assumption_certified=False)
    if reason:return q, valid, spread, depths, info
    relative = new.world_to_camera @ np.linalg.inv(old.world_to_camera)
    rays = np.c_[p, np.ones(n)] @ np.linalg.inv(old.intrinsic).T
    rotated = rays @ relative[:3, :3].T @ new.intrinsic.T
    valid = np.isfinite(rotated).all(1) & (rotated[:, 2] > 1e-6)
    q[valid] = rotated[valid, :2] / rotated[valid, 2:]
    # Rotation is an initializer under translation, not a bounded true prediction.
    if np.linalg.norm(relative[:3, 3]) < .02:spread[valid] = 0.
    if config.ego_motion_mode == 'depth' and history is not None:
        from .identity_temporal import camera_snapshot_digest
        if (history.get('camera_sha256') != camera_snapshot_digest(old)
                or history.get('timestamp') != old.timestamp):
            info['depth_kind'] = 'source_depth_binding_mismatch'
        else:
            hp = np.asarray(history['pixels']);hz = np.asarray(history['depths'])
            hu = np.asarray(history['uncertainty_m'])
            if len(hp):
                from scipy.spatial import cKDTree
                distance, index = cKDTree(hp).query(p)
                # No depth interpolation across an object boundary. Two source
                # corners must coincide within the existing FB tolerance.
                match = distance <= config.flow_fb_max_px
                nearest = np.minimum(index, len(hp)-1)
                z = hz[nearest];u = hu[nearest]
                predicted, ok, _, _ = project_source_depth(old, new, p, z)
                lower, lok, _, _ = project_source_depth(old, new, p, z-u)
                upper, uok, _, _ = project_source_depth(old, new, p, z+u)
                depth_valid = match & ok & lok & uok
                q[depth_valid] = predicted[depth_valid]
                spread[depth_valid] = np.maximum(np.linalg.norm(lower-predicted, axis=1),
                    np.linalg.norm(upper-predicted, axis=1))[depth_valid]
                depths[depth_valid] = z[depth_valid]
                info['depth_kind'] = 'prior_RGB_triangulation_conditional_static'
    h,w = new.image.shape[:2]
    valid &= np.isfinite(q).all(1) & (q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    info.update(predicted_points=int(valid.sum()), depth_points=int(depth_valid.sum()),
        rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(relative[:3,:3])-1)/2,-1,1)))),
        camera_translation_m=float(np.linalg.norm(relative[:3,3])))
    return q, valid, spread, depths, info


def lk_measure(old_gray, gray, points, config, initial=None):
    p = np.asarray(points, np.float32).reshape(-1,1,2);n=len(p)
    if not n:return p[:,0], np.empty(0), np.zeros(0,bool)
    args=dict(winSize=(21,21),maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    q, status, error = cv2.calcOpticalFlowPyrLK(old_gray,gray,p,
        None if initial is None else np.asarray(initial,np.float32).reshape(-1,1,2).copy(),
        flags=0 if initial is None else cv2.OPTFLOW_USE_INITIAL_FLOW, **args)
    if q is None:return p[:,0].copy(),np.full(n,np.inf),np.zeros(n,bool)
    finite = np.isfinite(q).all((1,2));q=np.where(np.isfinite(q),q,p).astype(np.float32)
    back, rev, _ = cv2.calcOpticalFlowPyrLK(gray,old_gray,q,None,**args)
    if back is None:return q[:,0],np.full(n,np.inf),np.zeros(n,bool)
    fb=np.linalg.norm(back[:,0]-p[:,0],axis=1);q=q[:,0];h,w=gray.shape
    good=status[:,0].astype(bool)&rev[:,0].astype(bool)&finite&np.isfinite(fb)
    good&=(fb<=config.flow_fb_max_px)&(error[:,0]<=config.flow_lk_max_error)
    good&=(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    _, inverse, counts=np.unique(np.rint(q/2).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    good&=counts[inverse]==1
    return q,fb,good


def track_with_prediction(old, new, p, config, history=None):
    from .temporal_visibility import gray_image
    raw,raw_fb,raw_good=lk_measure(gray_image(old),gray_image(new),p,config)
    pred,pvalid,spread,depths,info=point_predictions(old,new,p,config,history)
    seeded,seed_fb,seed_good=lk_measure(gray_image(old),gray_image(new),p,config,
        initial=np.where(pvalid[:,None],pred,p))
    seed_good &= pvalid
    disagreement=np.linalg.norm(raw-seeded,axis=1)
    conflict=raw_good&seed_good&(disagreement>config.flow_fb_max_px)
    use=~raw_good&seed_good
    q=np.where(use[:,None],seeded,raw);fb=np.where(use,seed_fb,raw_fb)
    good=(raw_good|seed_good)&~conflict
    # Recheck uniqueness after merging tracks from the two initializations.
    _,inv,counts=np.unique(np.rint(q/2).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    good &= counts[inv]==1
    m=dict(source_points=np.asarray(p),raw_points=raw,raw_fb_error_px=raw_fb,raw_valid=raw_good,
        seeded_points=seeded,seeded_valid=seed_good,predicted_points=pred,prediction_valid=pvalid,
        prediction_spread_px=spread,source_depth_m=depths,prediction_residual_px=np.linalg.norm(q-pred,axis=1),
        initialization_conflict=conflict,seed_recovered=use&good,info=info)
    return q,fb,good,m


def measured_depth_history(old, new, evidence, config, binding, additional_exclusion=None):
    """Only actual previous->current tracks; consumed starting at NEXT precheck.

    Reject dynamic/occluded footprints. Background stationarity is conditional,
    so the result can initialize search, never certify metric geometry.
    """
    if old is None or not len(evidence.source_points):return None
    from .foreground_correspondence import triangulate_correspondences
    from .identity_temporal import camera_snapshot_digest
    p,q=evidence.source_points,evidence.current_points
    world,angle,reprojection,_,valid=triangulate_correspondences(old,new,evidence)
    keep=evidence.supported&evidence.geometry_valid&evidence.correspondence_valid&valid
    keep&=(angle>=config.min_triangulation_angle_deg)&(reprojection<=config.flow_epipolar_max_px)
    z=(world@new.world_to_camera[:3,:3].T+new.world_to_camera[:3,3])[:,2]
    # Perturb measured disparity by the existing epipolar residual budget, and
    # include a 0.25 m floor. Poor parallax remains unavailable.
    u=np.maximum(config.depth_padding_m,z*config.flow_epipolar_max_px/
        (min(old.intrinsic[0,0],new.intrinsic[0,0])*np.maximum(np.sin(np.radians(angle)),1e-6)))
    keep&=np.isfinite(z)&np.isfinite(u)&(z-u>=.5)&(z+u<=200)
    if evidence.observation_exclusion_mask is not None:
        xy=np.rint(q).astype(int);h,w=evidence.observation_exclusion_mask.shape
        xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1)
        keep&=~evidence.observation_exclusion_mask[xy[:,1],xy[:,0]]
    if additional_exclusion is not None:
        xy=np.rint(q).astype(int);h,w=additional_exclusion.shape
        xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1)
        keep&=~additional_exclusion[xy[:,1],xy[:,0]]
    return dict(**binding,timestamp=new.timestamp,camera_sha256=camera_snapshot_digest(new),
        pixels=immutable_array(q[keep]),depths=immutable_array(z[keep]),uncertainty_m=immutable_array(u[keep]),
        evidence_kind='conditional_static_prior_RGB_triangulation',metric_geometry_certified=False)


def static_box_predictions(anchor, old, new, points, config):
    """Ray/OBB depth interval, conditional on the supplied static geometry.

    A class prior is not stationarity truth. This bounded search model retains
    that assumption explicitly and cannot be used by the physical certifier.
    """
    p=np.asarray(points,float);n=len(p)
    empty=(np.full((n,2),np.nan),np.full((n,2),np.nan),np.zeros(n,bool))
    if camera_pair_reason(old,new,config):return empty
    if anchor.anchor_type!='object' or anchor.semantic_class not in {'barrier','traffic_cone'}:
        return empty
    from .geometry import oriented_box_membership
    g=anchor.geometry_world;corners=np.asarray(g.points)
    # Reuse the strict oriented-box contract, including malformed corner refusal.
    if not oriented_box_membership(np.asarray([g.center]),g).all():return empty
    from itertools import combinations
    delta=corners[1:]-corners[0];axes=None
    for ix in combinations(range(7),3):
        edges=delta[list(ix)];length=np.linalg.norm(edges,axis=1)
        if np.any(length<1e-6):continue
        basis=edges/length[:,None]
        bits=(corners-corners[0])@basis.T/length
        if np.allclose(basis@basis.T,np.eye(3),atol=1e-5) and np.allclose(bits,np.rint(bits),atol=1e-4) and (bits>=-1e-4).all() and (bits<=1.0001).all():
            axes=basis;break
    if axes is None:return empty
    inv=np.linalg.inv(old.world_to_camera);origin=inv[:3,3]
    rays=np.c_[p,np.ones(n)]@np.linalg.inv(old.intrinsic).T@inv[:3,:3].T
    local_origin=(origin-corners[0])@axes.T;direction=rays@axes.T
    lo=np.full(n,.5);hi=np.full(n,200.)
    for k in range(3):
        parallel=np.abs(direction[:,k])<1e-10
        with np.errstate(divide='ignore',invalid='ignore'):
            a=-local_origin[k]/direction[:,k];b=(length[k]-local_origin[k])/direction[:,k]
        lo=np.maximum(lo,np.where(parallel,-np.inf,np.minimum(a,b)))
        hi=np.minimum(hi,np.where(parallel,np.inf,np.maximum(a,b)))
        lo=np.where(parallel & ((local_origin[k]<0)|(local_origin[k]>length[k])),np.inf,lo)
    valid=np.isfinite(lo)&np.isfinite(hi)&(hi>=lo)
    # Clamp only for safe projection; invalid rays remain invalid.
    near,nok,_,_=project_source_depth(old,new,p,np.clip(lo,.5,200))
    far,fok,_,_=project_source_depth(old,new,p,np.clip(hi,.5,200))
    return near,far,valid&nok&fok


def segment_residual(q, a, b):
    ab=b-a
    t=np.sum((q-a)*ab,axis=1)/np.maximum(np.sum(ab*ab,axis=1),1e-12)
    closest=a+np.clip(t,0,1)[:,None]*ab
    return np.linalg.norm(q-closest,axis=1)
