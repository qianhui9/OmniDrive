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
        if not 0 < frame.timestamp-old.source_timestamp <= config.association_track_ttl_s:
            continue
        if aid in store.anchors:
            i = next(i for i,a in enumerate(anchors) if a.anchor_id == aid)
        else:
            i = len(anchors); anchors.append(old)
        if old.source_timestamp < (store.last_timestamp or -np.inf):
            dormant.add(i)
        if motion is not None:
            priors[i] = motion
    return anchors, priors, dormant


def appearance(record, target, frame, config, validator, cache, key):
    """Legacy metadata-only API: descriptors alone can never prove or veto identity.

    Actual aligned measurements are provided by AlignmentSession in refine().
    """
    return None


def refine(anchors, predictions, costs, accepted, dormant, store, frame, config, validator,
           history=(), priors=None, selected=None, origin=None):
    from .association_alignment import AlignmentSession
    from .association_diagnostics import edge_gates, summarize
    priors=priors or {}; selected=selected or list(enumerate(predictions))
    origin=origin or (anchors[0].evidence_origin if anchors else 'actual_perception')
    session=AlignmentSession(history,store,frame,config,validator,predictions)
    graph=costs.copy(); measured={}; attempts=set()
    # Probe the nearest bounded set of same-class candidates even outside the
    # old distance gate. They cannot be accepted without measured metric motion.
    for i,old in enumerate(anchors):
        if old.anchor_type!='object':continue
        potential=[]
        for j,new in enumerate(predictions):
            d=edge_gates(old,new,frame,store.generation,origin,config,priors.get(i),i in dormant)
            if all(d['gates'][k] for k in ('category','size','time','scene','epoch','signature','source')):
                potential.append((d['prediction_residual_m'],j))
        targets=set(np.flatnonzero(np.isfinite(costs[i])))|{j for _,j in sorted(potential)[:4]}
        for j in sorted(targets,key=lambda j:(not np.isfinite(costs[i,j]),float(np.linalg.norm(predictions[j].geometry_world.center-old.geometry_world.center)),j)):
            record=store.association_tracks.get(old.anchor_id)
            if not bound(record,old,frame,store.generation):continue
            prior_gate=edge_gates(old,predictions[j],frame,store.generation,origin,config,priors.get(i),i in dormant)
            prior_supported=i in priors and prior_gate['gates']['geometry']
            require_metric=not np.isfinite(costs[i,j]) and not prior_supported
            # An ordinary isolated edge is still checked for reliable conflict;
            # unknown evidence preserves its unqualified geometric semantics.
            value=session.measure(old,j,require_metric=require_metric)
            measured[i,j]=value;attempts.add((i,j))
            if not np.isfinite(costs[i,j]) and value['status']=='supported':
                if require_metric and value['metric_supported']:
                    graph[i,j]=value['metric_budget_m']
                elif prior_supported:
                    graph[i,j]=prior_gate['prediction_residual_m']
    compatible=np.isfinite(graph)
    for edge,value in measured.items():
        if value['status']=='contradicted':compatible[edge]=False
    result=[(i,j) for i,j in accepted if compatible[i,j] and i not in dormant]
    changed=set();seen=set();competition_rejected=0
    for start in range(len(anchors)):
        if start in seen or not np.isfinite(graph[start]).any():continue
        rows={start};cols=set()
        while True:
            nc=set(np.flatnonzero(np.isfinite(graph[list(rows)]).any(0)))
            nr=set(np.flatnonzero(np.isfinite(graph[:,list(nc)]).any(1)))
            if nr==rows and nc==cols:break
            rows,cols=nr,nc
        seen|=rows
        sub=compatible[np.ix_(sorted(rows),sorted(cols))]
        edges={(sorted(rows)[a],sorted(cols)[b]) for a,b in zip(*np.nonzero(sub))}
        unique=(len(rows)==len(cols) and np.all(sub.sum(0)==1) and np.all(sub.sum(1)==1))
        # Unknown candidate relations tested outside the original graph remain
        # competitors when they share either endpoint with the proposed change.
        outside_unknown={e for e,v in measured.items() if v['status']=='unknown'
                         and not np.isfinite(graph[e]) and (e[0] in rows or e[1] in cols)}
        evidence=all(measured.get(e,{}).get('status')=='supported' for e in edges)
        if not unique or not evidence or outside_unknown:
            competition_rejected+=int(any(i in dormant or not np.isfinite(costs[i,j]) for i,j in edges))
            continue
        result=[(i,j) for i,j in result if i not in rows and j not in cols]+sorted(edges)
        changed|=edges
    # Explicit unmatched columns; no weak edge is introduced by cardinality.
    if graph.size:
        from scipy.optimize import linear_sum_assignment
        n,m=graph.shape;gated=np.full_like(graph,np.inf)
        for e in result:gated[e]=graph[e]
        dummy=np.full((n,n),np.inf);np.fill_diagonal(dummy,config.association_gate_m+1.)
        ii,jj=linear_sum_assignment(np.c_[gated,dummy])
        result=[(int(i),int(j)) for i,j in zip(ii,jj) if j<m and np.isfinite(gated[i,j])]
    # Measurement receipts are also bounded, independent of diagnostic toggle.
    # Raw correspondence arrays and image data never enter the commit log.
    receipts=[];receipt_bytes=0
    for (i,j),value in measured.items():
        row=dict(anchor_id=anchors[i].anchor_id,prediction_index=selected[j][0],
                 status=value['status'],reason=value['reason'],metric_supported=value['metric_supported'])
        count=len(json.dumps(row).encode())
        if len(receipts)<config.association_diagnostic_limit and receipt_bytes+count<=config.association_diagnostic_bytes:
            receipts.append(row);receipt_bytes+=count
    audit=summarize(anchors,predictions,frame,store,origin,config,priors,dormant,graph,result,measured,selected,
                    byte_budget=max(0,config.association_diagnostic_bytes-receipt_bytes),session=session)
    return result,changed,dict(measured_edges=session.calls,
        supported=sum(v['status']=='supported' for v in measured.values()),
        mismatches=sum(v['status']=='contradicted' for v in measured.values()),
        unknown=sum(v['status']=='unknown' for v in measured.values()),
        recovered=[anchors[i].anchor_id for i,j in result if i in dormant],certified=False,
        dormant_candidates=sum(i in dormant for i,j in attempts),
        dormant_supported=sum(i in dormant and v['status']=='supported' for (i,j),v in measured.items()),
        dormant_appearance_conflicts=sum(i in dormant and v['status']=='contradicted' for (i,j),v in measured.items()),
        competition_rejected=competition_rejected,bootstrap_supported=sum(v.get('metric_supported',False) for v in measured.values()),
        measurements=receipts,measurement_receipt_bytes=receipt_bytes,audit=audit)


def publish(store, current, prior, frame, config, reliable, frame_digest_value=None):
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
        orientation_axes=(g.points[[4,2,1]]-g.points[0]).tolist(),timestamp=frame.timestamp,
        frame_digest=frame_digest_value,commit_version=store.commit_count+1,
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
