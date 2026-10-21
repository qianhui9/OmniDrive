"""Committed association-only tracklets; never physical motion or reusable state.

Robust motion consumes the original distance budget. Appearance is a necessary
component gate, not an additive score. Dormant records own no feature or tensor.
"""
import copy
import json
import sys
import numpy as np
from .geometry import state_geometry_version, project_anchor, immutable_array
from .state_types import WorldAnchor, Geometry, json_value
from .association import legal_assignment, association_ambiguities


def bound(record, anchor, frame, epoch):
    if not record or not record.get('samples'):
        return False
    h = record['samples'][-1]
    return (record['scene_id'] == frame.scene_id == anchor.scene_id
            and record['class'] == anchor.semantic_class
            and record['epoch'] == epoch == anchor.source_state_epoch
            and record['signature'] == frame.execution_signature == anchor.source_execution_signature
            and record['origin'] == anchor.evidence_origin
            and h['version'] == state_geometry_version(anchor)
            and h['frame_id'] == anchor.source_frame_id
            and h['timestamp'] == anchor.source_timestamp
            and np.array_equal(h['center'], anchor.geometry_world.center))


def predict(record, timestamp, config):
    h = record['samples']
    if len(h) < 3:
        return None
    t = np.asarray([r['timestamp'] for r in h], float)
    x = np.asarray([r['center'] for r in h], float)
    dt = np.diff(t)
    age = timestamp-t[-1]
    if (not np.isfinite(t).all() or not np.isfinite(x).all() or not 0 < age <= config.association_track_ttl_s
            or (dt <= 0).any() or (dt > config.max_frame_gap_s).any()
            or len({r['frame_id'] for r in h}) != len(h)):
        return None
    # Median pairwise slopes resist a bad centre; all residuals still consume
    # the budget rather than silently discarding inconvenient observations.
    slopes = [(x[j]-x[i])/(t[j]-t[i]) for i in range(len(h)) for j in range(i+1,len(h))]
    velocity = np.median(slopes, axis=0)
    intercept = np.median(x-(t-t[-1])[:,None]*velocity, axis=0)
    residual = np.linalg.norm(x-(intercept+(t-t[-1])[:,None]*velocity),axis=1).max()
    step_velocity = np.diff(x,axis=0)/dt[:,None]
    instability = np.linalg.norm(step_velocity-velocity,axis=1).max()
    uncertainty = float(residual+instability*age)
    if not np.isfinite(uncertainty) or uncertainty >= config.association_gate_m:
        return None
    return intercept+velocity*age, uncertainty


def candidates(store, frame, config, origin):
    anchors = list(store.anchors.values())
    priors, dormant = {}, set()
    for aid, record in sorted(store.association_tracks.items()):
        if record['origin'] != origin:
            continue
        old = store.anchors.get(aid)
        if old is None:
            h = record['samples'][-1]
            old = WorldAnchor(aid, record['scene_id'], 'object', record['class'],
                Geometry(np.asarray(h['points'],dtype=h['dtypes'][0]), np.asarray(h['center'],dtype=h['dtypes'][1]),
                         np.asarray(h['size'],dtype=h['dtypes'][2])), h['timestamp'],
                velocity_world=np.asarray(h['velocity'],dtype=h['velocity_dtype']) if h['velocity'] is not None else None,
                velocity_reliable=h['velocity_reliable'],
                source_frame_id=h['frame_id'], source_state_epoch=record['epoch'],
                source_execution_signature=record['signature'], evidence_origin=record['origin'])
        if not bound(record, old, frame, store.generation):
            continue
        motion = predict(record, frame.timestamp, config)
        if motion is None:
            continue
        if aid in store.anchors:
            i = next(i for i,a in enumerate(anchors) if a.anchor_id == aid)
        else:
            i = len(anchors); anchors.append(old)
        if old.source_timestamp < (store.last_timestamp or -np.inf):
            dormant.add(i)
        priors[i] = motion
    return anchors, priors, dormant


def appearance(record, target, frame, config, validator, cache, key):
    """Compare actual object ROIs in matching cameras; missing/view change unknown."""
    measurements = []
    for cid, cam in frame.cameras.items():
        if not cam.timing_reliable or abs(cam.timestamp-frame.timestamp) > config.max_camera_skew_s:
            continue
        ck = (key,cid)
        if ck not in cache:
            p = project_anchor(target, cam, config)
            desc = validator.descriptor(cam.image,p.region) if p.valid else None
            # Reject heavily occluded predicted regions in caller-provided mask.
            cache[ck] = (p, desc)
        p, desc = cache[ck]
        if desc is None or not p.valid:
            continue
        ray = target.geometry_world.center-np.linalg.inv(cam.world_to_camera)[:3,3]
        ray /= max(np.linalg.norm(ray),1e-9)
        values = []
        for sample in record['samples'][-3:]:
            o = sample['rois'].get(cid)
            if (o is None or o['space'] != cam.coordinate_space
                    or not 0 < frame.timestamp-o['timestamp'] <= config.association_track_ttl_s
                    or np.dot(ray,o['ray']) < np.cos(np.deg2rad(15))):
                continue
            area = (p.region[2]-p.region[0])*(p.region[3]-p.region[1])
            if not .5 <= area/o['area'] <= 2:
                continue
            values.append(validator.residual(np.asarray(o['descriptor']),desc))
        if values:
            measurements.append(all(np.isfinite(v) and v <= config.residual_threshold for v in values))
    return None if not measurements else all(measurements)


def refine(anchors, predictions, costs, accepted, dormant, store, frame, config, validator):
    """Prune only measured conflicts; solve with explicit unmatched dummy nodes.

    An entire competing component needs a unique measured surviving bijection.
    Unknown edges remain competitors and never become positive evidence.
    """
    legal = np.isfinite(costs); compatible = legal.copy(); measured = {}; cache = {}
    from .geometry import intersect_area, region_area
    for i,j in zip(*np.nonzero(legal)):
        if anchors[i].anchor_type != 'object':
            continue
        if i not in dormant and legal[i].sum() == legal[:,j].sum() == 1:
            continue
        record = store.association_tracks.get(anchors[i].anchor_id)
        if not bound(record,anchors[i],frame,store.generation):
            continue
        pred = predictions[j]
        target = WorldAnchor('',frame.scene_id,pred.anchor_type,pred.semantic_class,pred.geometry_world,frame.timestamp)
        # Box projection alone is not foreground support. Obvious nearer ROI
        # overlap makes appearance unknown, never a positive correspondence.
        occluded = False
        for cid,cam in frame.cameras.items():
            p = project_anchor(target,cam,config)
            if not p.valid: continue
            for k,other in enumerate(predictions):
                if k == j or other.anchor_type != 'object': continue
                z = cam.world_to_camera[:3,:3]@other.geometry_world.center+cam.world_to_camera[:3,3]
                tz = cam.world_to_camera[:3,:3]@target.geometry_world.center+cam.world_to_camera[:3,3]
                if z[2]+config.occlusion_depth_margin_m >= tz[2]: continue
                a = WorldAnchor('',frame.scene_id,other.anchor_type,other.semantic_class,other.geometry_world,frame.timestamp)
                q = project_anchor(a,cam,config)
                if q.valid and intersect_area(p.region,q.region) > config.occlusion_overlap_fraction*region_area(p.region):
                    occluded = True; break
            if occluded: break
        value = None if occluded else appearance(record,target,frame,config,validator,cache,j)
        if value is not None:
            measured[i,j] = value
            if not value: compatible[i,j] = False
    result = [(i,j) for i,j in accepted if compatible[i,j] and i not in dormant]
    changed = set(); seen = set()
    for start in range(len(anchors)):
        if start in seen or not legal[start].any(): continue
        rows = {start}; cols = set()
        while True:
            nc = set(np.flatnonzero(legal[list(rows)].any(0)))
            nr = set(np.flatnonzero(legal[:,list(nc)].any(1)))
            if nr == rows and nc == cols: break
            rows,cols = nr,nc
        seen |= rows
        sub = compatible[np.ix_(sorted(rows),sorted(cols))]
        if not (len(rows)==len(cols) and np.all(sub.sum(0)==1) and np.all(sub.sum(1)==1)): continue
        edges = {(sorted(rows)[a],sorted(cols)[b]) for a,b in zip(*np.nonzero(sub))}
        if not all(measured.get(e) is True for e in edges): continue
        result = [(i,j) for i,j in result if i not in rows and j not in cols]+sorted(edges)
        changed |= edges
    # Reject endpoint-forced matches before a one-to-one solver with private
    # unmatched columns. No maximum-cardinality objective can rescue weak edges.
    gated = np.full_like(costs,np.inf)
    for e in result: gated[e] = costs[e]
    if gated.size:
        from scipy.optimize import linear_sum_assignment
        n,m = gated.shape
        dummy = np.full((n,n),np.inf); np.fill_diagonal(dummy,config.association_gate_m+1.)
        matrix = np.c_[gated,dummy]
        ri,cj = linear_sum_assignment(matrix)
        result = [(int(i),int(j)) for i,j in zip(ri,cj) if j<m and np.isfinite(gated[i,j])]
    return result,changed,dict(measured_edges=len(measured),mismatches=sum(not v for v in measured.values()),
        recovered=[anchors[i].anchor_id for i,j in result if i in dormant],certified=False)


def publish(store, current, prior, frame, config, reliable):
    """Called inside the existing private commit worker, never precheck."""
    if current.anchor_type != 'object' or current.geometry_world.size is None: return
    old = store.association_tracks.get(current.anchor_id)
    history = copy.deepcopy(old['samples']) if prior is not None and reliable and bound(old,prior,frame,store.generation) else []
    if prior is not None and not reliable:
        store.association_tracks.pop(current.anchor_id,None)
        return
    rois = {}
    for cid,o in current.observation_support.items():
        cam = frame.cameras[cid]
        if (not cam.timing_reliable or abs(o.timestamp-frame.timestamp)>config.max_camera_skew_s
                or o.coverage<config.min_coverage or not np.isfinite(o.descriptor).all()): continue
        ray = current.geometry_world.center-np.linalg.inv(cam.world_to_camera)[:3,3]
        ray /= max(np.linalg.norm(ray),1e-9)
        rois[cid] = dict(descriptor=immutable_array(o.descriptor),timestamp=o.timestamp,space=cam.coordinate_space,
            region=list(o.region),area=float((o.region[2]-o.region[0])*(o.region[3]-o.region[1])),ray=ray.tolist())
    g=current.geometry_world
    history.append(dict(center=g.center.tolist(),points=g.points.tolist(),size=g.size.tolist(),
        orientation_axes=(g.points[1:4]-g.points[0]).tolist(),timestamp=frame.timestamp,
        dtypes=[str(a.dtype) for a in (g.points,g.center,g.size)],
        velocity=current.velocity_world.tolist() if current.velocity_world is not None else None,
        velocity_dtype=str(current.velocity_world.dtype) if current.velocity_world is not None else None,
        velocity_reliable=current.velocity_reliable,
        frame_id=frame.frame_id,version=state_geometry_version(current),quality='unambiguous_association' if prior else 'seed_observation',
        ambiguous=False,rois=rois))
    store.association_tracks[current.anchor_id] = dict(scene_id=frame.scene_id,epoch=store.generation,
        signature=frame.execution_signature,origin=current.evidence_origin,**{'class':current.semantic_class},
        samples=history[-config.association_track_frames:])


def prune(store,timestamp,config):
    records = [(k,v) for k,v in store.association_tracks.items()
               if 0 <= timestamp-v['samples'][-1]['timestamp'] <= config.association_track_ttl_s]
    records.sort(key=lambda kv:(-kv[1]['samples'][-1]['timestamp'],kv[0]))
    def memory_bytes(value):
        size=sys.getsizeof(value)
        if isinstance(value,dict):
            size+=sum(memory_bytes(k)+memory_bytes(v) for k,v in value.items())
        elif isinstance(value,(list,tuple)):
            size+=sum(memory_bytes(v) for v in value)
        elif isinstance(value,np.ndarray):
            # Bytes-backed immutable arrays report a small header; charge their
            # owned payload too. Shared objects are deliberately overcounted.
            size+=value.nbytes
        return size
    kept={};used=0;memory=0
    for key,value in records[:config.association_track_limit]:
        count=len(json.dumps(json_value({key:value}),allow_nan=False).encode())
        owned=memory_bytes({key:value})
        if max(used+count,memory+owned)>config.association_track_bytes: continue
        kept[key]=value;used+=count;memory+=owned
    store.association_tracks=kept
    return dict(tracks=len(kept),serialized_bytes=used,accounted_memory_bytes=memory,physical_certification=False)
