"""Bounded commit-only identity history and the single object predictor.

Records own geometry metadata only: no image, descriptor, feature, certificate,
or verified lease. A dormant record is just another input to associate().
"""
import copy
import json
import sys
import numpy as np
from .geometry import state_geometry_version
from .state_types import WorldAnchor, Geometry


def bound(record, anchor, frame, epoch):
    if not record or not record.get('samples'):return False
    h=record['samples'][-1]
    return (record['scene_id']==anchor.scene_id==frame.scene_id
        and record['epoch']==anchor.source_state_epoch==epoch
        and record['signature']==anchor.source_execution_signature==(getattr(frame,'execution_signature',None) or getattr(frame,'source_execution_signature',None))
        and record['origin']==anchor.evidence_origin and record['class']==anchor.semantic_class
        and h['version']==state_geometry_version(anchor) and h['frame_id']==anchor.source_frame_id
        and h['timestamp']==anchor.source_timestamp and np.array_equal(h['center'],anchor.geometry_world.center))


def predict(anchor, timestamp, config, record=None):
    """One prediction entry: robust history, or last observed centre if short.

    Recognized fit/scatter and declared localization errors consume the original
    distance budget. These are engineering bounds, not calibrated probabilities.
    Invalid history never invokes a second predictor or search.
    """
    age=timestamp-anchor.source_timestamp
    limit=config.association_track_ttl_s if config.association_track_prior else config.max_state_age_s
    # Unmodelled velocity error grows with observation age even for a good
    # constant-velocity fit. The same budget applies to active and dormant IDs.
    sigma=anchor.geometry_uncertainty+config.velocity_uncertainty_mps*age
    if not np.isfinite([age,sigma]).all() or not 0<age<=limit or sigma<0:return None
    if record is not None:
        if not bound(record,anchor,anchor,anchor.source_state_epoch):return None
        h=record['samples'];t=np.asarray([s['timestamp'] for s in h],float)
        x=np.asarray([s['center'] for s in h],float);steps=np.diff(t)
        if (not np.isfinite(x).all() or not np.isfinite(t).all() or (steps<=0).any()
                or (steps>config.max_frame_gap_s).any() or len({s['frame_id'] for s in h})!=len(h)):
            return None
        if len(h)>=3:
            slopes=[(x[j]-x[i])/(t[j]-t[i]) for i in range(len(h)) for j in range(i+1,len(h))]
            velocity=np.median(slopes,axis=0)
            intercept=np.median(x-(t-t[-1])[:,None]*velocity,axis=0)
            residual=np.linalg.norm(x-(intercept+(t-t[-1])[:,None]*velocity),axis=1).max()
            scatter=np.linalg.norm(np.diff(x,axis=0)/steps[:,None]-velocity,axis=1).max()
            uncertainty=float(sigma+residual+scatter*age)
            if not np.isfinite(uncertainty) or uncertainty>config.association_gate_m:return None
            return intercept+velocity*age,uncertainty
    if config.dynamic_motion_prediction:
        from .world_motion import prediction
        physical=prediction(anchor,timestamp)
        if physical is not None:
            center,covariance=physical;u=float(3*np.sqrt(np.linalg.eigvalsh(covariance).max()))
            return (center,u) if u<=config.association_gate_m else None
    # No detector velocity and no guessed startup velocity. Age-dependent
    # uncertainty consumes the same spatial residual gate for every object.
    return (anchor.geometry_world.center.copy(),float(sigma)) if sigma<=config.association_gate_m else None


def _anchor(record,aid):
    s=record['snapshot'];h=record['samples'][-1]
    def array(name):return np.asarray(s[name],dtype=s['dtypes'][name]) if s[name] is not None else None
    return WorldAnchor(aid,record['scene_id'],'object',record['class'],
        Geometry(array('points'),array('center'),array('size')),h['timestamp'],
        velocity_world=array('velocity'),velocity_reliable=s['velocity_reliable'],
        geometry_uncertainty=s['geometry_uncertainty'],source_frame_id=h['frame_id'],
        source_state_epoch=record['epoch'],source_execution_signature=record['signature'],evidence_origin=record['origin'])


def candidates(store,frame,config,origin):
    anchors=list(store.anchors.values());indices={a.anchor_id:i for i,a in enumerate(anchors)}
    dormant=set();priors={}
    if config.association_track_prior:
        for aid,r in sorted(store.association_tracks.items()):
            if aid not in indices:
                old=_anchor(r,aid)
                if bound(r,old,frame,store.generation) and r['origin']==origin and 0<frame.timestamp-old.source_timestamp<=config.association_track_ttl_s:
                    indices[aid]=len(anchors);anchors.append(old)
    for i,old in enumerate(anchors):
        if old.anchor_type!='object':continue
        if (old.scene_id!=frame.scene_id or old.source_state_epoch!=store.generation
                or old.source_execution_signature!=frame.execution_signature or old.evidence_origin!=origin):
            priors[i]=None;continue
        record=store.association_tracks.get(old.anchor_id) if config.association_track_prior else None
        if record is not None and not bound(record,old,frame,store.generation):
            priors[i]=None;continue
        priors[i]=predict(old,frame.timestamp,config,record)
        if old.source_timestamp < (store.last_timestamp if store.last_timestamp is not None else old.source_timestamp):dormant.add(i)
    return anchors,priors,dormant


def publish(store,current,prior,frame,config,reliable=True,frame_digest_value=None):
    if current.anchor_type!='object' or current.geometry_world.size is None:return
    old=store.association_tracks.get(current.anchor_id)
    history=copy.deepcopy(old['samples']) if prior is not None and reliable and bound(old,prior,frame,store.generation) else []
    if not reliable:
        store.association_tracks.pop(current.anchor_id,None);return
    if history and frame.timestamp-history[-1]['timestamp']>config.max_frame_gap_s:history=[]
    h=dict(center=current.geometry_world.center.tolist(),timestamp=frame.timestamp,frame_id=frame.frame_id,
           version=state_geometry_version(current),commit_version=store.commit_count+1)
    history.append(h)
    g=current.geometry_world
    arrays=dict(points=g.points,center=g.center,size=g.size,velocity=current.velocity_world)
    snapshot={k:v.tolist() if v is not None else None for k,v in arrays.items()}
    snapshot.update(dtypes={k:str(v.dtype) if v is not None else None for k,v in arrays.items()},
        velocity_reliable=current.velocity_reliable,geometry_uncertainty=current.geometry_uncertainty)
    store.association_tracks[current.anchor_id]=dict(scene_id=frame.scene_id,epoch=store.generation,
        signature=frame.execution_signature,origin=current.evidence_origin,**{'class':current.semantic_class},
        samples=history[-config.association_track_frames:],snapshot=snapshot)


def prune(store,timestamp,config):
    def owned(value):
        n=sys.getsizeof(value)
        if isinstance(value,dict):n+=sum(owned(k)+owned(v) for k,v in value.items())
        elif isinstance(value,(list,tuple)):n+=sum(owned(v) for v in value)
        return n
    rows=sorted(((aid,r) for aid,r in store.association_tracks.items()
        if 0<=timestamp-r['samples'][-1]['timestamp']<=config.association_track_ttl_s),
        key=lambda kv:(-kv[1]['samples'][-1]['timestamp'],kv[0]))
    kept={};serialized=0;memory=0
    for aid,r in rows[:config.association_track_limit]:
        item={aid:r};size=len(json.dumps(item,allow_nan=False).encode());n=owned(item)
        if max(serialized+size,memory+n)>config.association_track_bytes:continue
        kept[aid]=r;serialized+=size;memory+=n
    store.association_tracks=kept
    return dict(tracks=len(kept),serialized_bytes=serialized,accounted_memory_bytes=memory)
