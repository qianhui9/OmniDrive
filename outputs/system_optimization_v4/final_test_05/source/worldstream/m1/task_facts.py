"""Minimal offline task-fact evidence extraction, with explicit Unknown.

An M1 observation-valid anchor does not certify any of these physical facts.
This extractor deliberately has no annotation/GT or detector-score-to-error
conversion input. Available M1 evidence lacks independent identity/foreground
and metric/motion error bounds, so the current implementation cannot certify
a fact. It is not connected to M3 or to EVA scheduling.
"""
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .geometry import predict_geometry, valid_transform


@dataclass(frozen=True)
class TaskFact:
    fact_id: str
    anchor_id: str
    scene_id: str
    source_frame: Optional[str]
    source_timestamp: float
    verified_timestamp: Optional[float]
    semantic_class: str
    fact_type: str
    fact_value: Any
    coordinate_frame: str
    uncertainty: dict
    evidence: dict
    supporting_cameras: tuple
    required_dependencies: tuple
    computation_source: dict
    rejection_reason: str
    certified: bool = False


def extract_task_facts(anchor, frame, observation_check):
    """Keep candidate value separate from an independently certified value."""
    center=None;velocity=None
    geometry=predict_geometry(anchor,frame.timestamp)
    if valid_transform(frame.ego_to_world):
        pose=np.linalg.inv(frame.ego_to_world)
        point=(pose@np.r_[geometry.center,1.])[:3]
        if np.isfinite(point).all():center=point
        if anchor.velocity_reliable and anchor.velocity_world is not None:
            value=pose[:3,:3]@anchor.velocity_world
            if np.isfinite(value).all():velocity=value
    fresh=(anchor.scene_id==frame.scene_id and 0<frame.timestamp-anchor.source_timestamp<=1.
        and anchor.evidence_origin in ('actual_perception','historical_full_model_detection_replay')
        and anchor.source_execution_signature in (None,frame.execution_signature))
    cameras=tuple(sorted(cid for cid,c in observation_check.get('cameras',{}).items() if c.get('valid')))
    source=dict(origin=anchor.evidence_origin,source_execution_signature=anchor.source_execution_signature,
        source_state_epoch=anchor.source_state_epoch,current_execution_signature=frame.execution_signature)
    evidence=dict(observation_consistent=bool(observation_check.get('valid')),history_observations=anchor.history_observations,
        detection_score=anchor.source_score,last_innovation_m=anchor.last_innovation_m,
        independent_identity=False,independent_foreground_ownership=False)
    uncertainty=dict(independent_localization_error_bound_m=None,independent_motion_error_bound_mps=None,
        declared_geometry_uncertainty_m=anchor.geometry_uncertainty,
        declared_uncertainty_is_calibrated=False,direction_interval=None)
    specs=[('object_persistence',True if observation_check.get('valid') else None,
        ('independent_identity','current_foreground_observation'),'identity_and_foreground_unverified'),
        ('relative_direction',dict(longitudinal=int(np.sign(center[0])),lateral=int(np.sign(center[1]))) if center is not None else None,
        ('independent_identity','signed_direction_interval_excludes_zero'),'signed_interval_and_referent_unverified'),
        ('metric_distance',float(np.linalg.norm(center)) if center is not None else None,
        ('independent_identity','independent_localization_error_bound'),'independent_metric_error_bound_missing'),
        ('motion_relation',velocity.tolist() if velocity is not None else None,
        ('independent_identity','time_baseline','independent_object_motion_error_bound'),'independent_motion_evidence_missing')]
    return tuple(TaskFact(f'{anchor.anchor_id}:{frame.frame_id}:{kind}',anchor.anchor_id,frame.scene_id,
        anchor.source_frame_id,anchor.source_timestamp,None,anchor.semantic_class,kind,value,
        'current_ego',dict(uncertainty),dict(evidence),cameras,deps,dict(source),
        reason if fresh and anchor.anchor_type=='object' and center is not None else 'stale_unsupported_provenance_or_geometry')
        for kind,value,deps,reason in specs)
