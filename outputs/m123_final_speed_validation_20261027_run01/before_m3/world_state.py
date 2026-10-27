"""Task queries over a frozen M1 precheck, captured BEFORE current perception commit.

Physical/observation validity is not a Transformer or task-quality certificate.
No ground truth, current detector output, or anchor-to-query mapping is accepted.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import re
import numpy as np
from .task_conditioner import task_type


@dataclass(frozen=True)
class VerifiedWorldState:
    scene_id: str
    frame_id: str
    timestamp: float
    state_epoch: int
    execution_signature: str
    records: tuple
    diagnostics: dict
    state_version: int = 0
    content_sha256: str = ''

    def __post_init__(self):
        object.__setattr__(self,'content_sha256',snapshot_digest(self))


def snapshot_digest(state):
    payload=(state.scene_id,state.frame_id,state.timestamp,state.state_epoch,
             state.execution_signature,state.state_version,state.records,state.diagnostics)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


def freeze_verified_state(manager, frame):
    result=manager.last_result;store=getattr(manager,'_pending_store',None)
    records=[]
    if (store is None or result is None or result.scene_id!=frame.scene_id or result.frame_id!=frame.frame_id
            or result.timestamp!=frame.timestamp or store.scene_id!=frame.scene_id
            or getattr(manager,'_pending',None)!=(frame.scene_id,frame.frame_id,frame.timestamp,frame.execution_signature)
            or result.diagnostics.get('execution_signature')!=frame.execution_signature
            or manager.store.source_digest()!=manager._pending_source_digest
            or manager._history_digest()!=manager._pending_history_digest):
        return VerifiedWorldState(frame.scene_id,frame.frame_id,frame.timestamp,manager.store.generation,
            frame.execution_signature,(),dict(reason='stale_or_missing_precheck'))
    from worldstream.m1.geometry import predict_geometry,valid_transform
    if not valid_transform(frame.ego_to_world):
        return VerifiedWorldState(frame.scene_id,frame.frame_id,frame.timestamp,store.generation,
            frame.execution_signature,(),dict(reason='invalid_ego_pose'))
    world_to_ego=np.linalg.inv(frame.ego_to_world)
    valid_features={v['feature_id'] for v in result.valid_feature_candidates}
    for aid in result.valid_anchor_ids:
        anchor=store.anchors.get(aid);check=result.diagnostics.get('anchors',{}).get(aid,{})
        if (anchor is None or anchor.scene_id!=frame.scene_id or not check.get('valid')
                or anchor.last_verified_timestamp!=frame.timestamp
                or not 0<frame.timestamp-anchor.source_timestamp<=manager.config.max_state_age_s):continue
        # Observation geometry cannot legalize an annotation/oracle commit.
        # Unknown historical origins stay unknown and are not task inputs.
        if anchor.evidence_origin not in ('actual_perception','historical_full_model_detection_replay'):continue
        if anchor.source_execution_signature!=frame.execution_signature:continue
        if anchor.source_state_epoch!=store.generation:continue
        cameras=sorted(cid for cid,c in check.get('cameras',{}).items() if c.get('valid'))
        if not cameras:continue
        g=predict_geometry(anchor,frame.timestamp)
        center=(world_to_ego@np.r_[g.center,1])[:3]
        if not np.isfinite(center).all():continue
        velocity=None
        if anchor.velocity_reliable and anchor.velocity_world is not None:
            v=world_to_ego[:3,:3]@anchor.velocity_world
            if np.isfinite(v).all():velocity=v.tolist()
        features=[]
        for fid in anchor.associated_feature_ids:
            f=store.features.get(fid)
            if fid in valid_features and f is not None and f.last_verified_timestamp==frame.timestamp:
                features.append(dict(feature_id=fid,version=f.computation_version,
                    computation_timestamp=f.computation_timestamp,execution_signature=f.execution_signature,
                    dependency_scope=f.dependency_scope,transformer_dependency_certified=False))
        records.append(dict(anchor_id=aid,anchor_type=anchor.anchor_type,semantic_class=anchor.semantic_class,
            center_ego_m=center.tolist(),geometry_ego_m=((world_to_ego[:3,:3]@g.points.T).T+world_to_ego[:3,3]).tolist(),
            velocity_ego_mps=velocity,source_timestamp=anchor.source_timestamp,
            geometry_kind='historical_observation_prior_verified_by_current_temporal_support',
            physical_geometry_certified=False,current_visual_required=True,
            verified_timestamp=frame.timestamp,source_frame=anchor.source_frame_id,
            source_frame_reason=None if anchor.source_frame_id is not None else 'Unknown historical source frame',
            source_state_epoch=anchor.source_state_epoch,
            source_execution_signature=anchor.source_execution_signature,evidence_origin=anchor.evidence_origin,
            source_cameras=sorted(anchor.observation_support),verified_cameras=cameras,
            feature_versions=features,history_observations=anchor.history_observations))
    return VerifiedWorldState(frame.scene_id,frame.frame_id,frame.timestamp,store.generation,
        frame.execution_signature,tuple(records),dict(reason='frozen_historical_precheck',
        source='M1 history before current perception commit',thresholds_provisional=result.diagnostics.get('thresholds_provisional',True),
        residual_threshold_calibrated=result.diagnostics.get('residual_threshold_calibrated',False),
        total_history_anchors=len(store.anchors),verified_history_anchors=len(records),
        complete_scene_coverage=False,task_quality_certified=False,
        max_state_age_s=manager.config.max_state_age_s,
        m1_protocol_sha256=hashlib.sha256(json.dumps(manager.config.to_dict(),sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()),state_version=store.commit_count)


def query_world_state(state,question,context):
    """Simple dependency filters; partial history never establishes absence/counts."""
    task=task_type(question);q=question.lower()
    state_valid=False
    def reject(reason):return dict(admitted=False,reason=reason,task=task,records=[],text='',state_valid=state_valid)
    if state is None:return reject('no_frozen_verified_history')
    try:
        if state.content_sha256!=snapshot_digest(state):return reject('historical_snapshot_modified')
    except (ValueError,TypeError):return reject('historical_snapshot_invalid')
    if (state.scene_id!=str(context.get('scene_id')) or state.frame_id!=str(context.get('frame_id'))
            or state.timestamp!=context.get('timestamp')
            or state.state_epoch!=context.get('state_epoch')
            or state.state_version!=context.get('state_version')
            or state.execution_signature!=context.get('state_execution_signature')):return reject('stale_state_context')
    # A bound empty snapshot is valid but never establishes complete coverage.
    # This diagnostic is independent of task suitability and text quality.
    try:
        state_valid=all(r.get('source_state_epoch')==state.state_epoch
            and r.get('source_execution_signature')==state.execution_signature
            and r.get('source_frame') and r['source_frame']!=state.frame_id
            and r.get('verified_timestamp')==state.timestamp
            and r.get('evidence_origin') in ('actual_perception','historical_full_model_detection_replay')
            and isinstance(r.get('source_timestamp'),(int,float))
            and 0<state.timestamp-r['source_timestamp']<=state.diagnostics.get('max_state_age_s',1.)
            for r in state.records)
    except (TypeError,ValueError):state_valid=False
    if task in ('global_safety','unknown'):return reject('full_scene_task_requires_visual')
    if re.search(r'\b(how many|any|no |all |count|number|none|absence)\b',q):return reject('partial_history_cannot_establish_count_or_absence')
    records=[r for r in state.records if r['anchor_type']==('map' if task=='map' else 'object')
        and r.get('source_state_epoch')==state.state_epoch
        and r.get('source_execution_signature')==state.execution_signature
        and r.get('source_frame') and r['source_frame']!=state.frame_id
        and r.get('verified_timestamp')==state.timestamp
        and isinstance(r.get('source_timestamp'),(int,float))
        and 0<state.timestamp-r['source_timestamp']<=state.diagnostics.get('max_state_age_s',1.0)]
    if re.search(r'\b(speed|moving|motion|velocity)\b',q):records=[r for r in records if r['velocity_ego_mps'] is not None]
    classes={'pedestrian':'pedestrian','truck':'truck','bus':'bus','car':'car','bicycle':'bicycle','motorcycle':'motorcycle'}
    requested={v for k,v in classes.items() if re.search(r'\b'+k+r's?\b',q)}
    if requested:records=[r for r in records if r['semantic_class'] in requested]
    elif re.search(r'\bvehicles?\b',q):records=[r for r in records if r['semantic_class'] in ('car','truck','bus','trailer','construction_vehicle','motorcycle','bicycle')]
    if 'left' in q:records=[r for r in records if r['center_ego_m'][1]>0]
    if 'right' in q:records=[r for r in records if r['center_ego_m'][1]<0]
    if 'front' in q:records=[r for r in records if r['center_ego_m'][0]>0]
    if 'behind' in q or 'rear' in q:records=[r for r in records if r['center_ego_m'][0]<0]
    if not records:return reject('no_task_relevant_verified_history')
    # Truncation cannot masquerade as complete coverage. Keep deterministic IDs.
    records=sorted(records,key=lambda r:r['anchor_id'])[:16]
    payload=dict(source='verified historical M1 observations; incomplete scene coverage',
        frame_id=state.frame_id,timestamp=state.timestamp,state_epoch=state.state_epoch,
        execution_signature=state.execution_signature,records=records)
    return dict(admitted=True,reason='verified_history_candidate_requires_task_quality',task=task,records=records,state_valid=state_valid,
        text='\nHistorical world-state context (partial observations, unknown is not absent):\n'+json.dumps(payload,separators=(',',':'))+'\n',
        complete_scene_coverage=False,task_quality_certified=False,
        m1_protocol_sha256=state.diagnostics.get('m1_protocol_sha256'))


def insert_state_context(ids,text,tokenizer,max_added_tokens=512):
    """Insert text after the first image marker, preserving every original ID."""
    import torch
    flat=ids.reshape(-1);positions=(flat==-200).nonzero().flatten()
    if len(positions)!=1:raise ValueError('World state requires exactly one native image marker')
    tokens=tokenizer.encode(text,add_special_tokens=False)
    if not tokens:raise ValueError('Empty state context encoding')
    if len(tokens)>max_added_tokens:raise ValueError('State context exceeds explicit token budget')
    index=int(positions[0])+1
    return torch.cat((flat[:index],flat.new_tensor(tokens),flat[index:]))
