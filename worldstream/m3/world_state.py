from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CommittedAnchor:
    anchor_id: str
    anchor_type: str
    semantic_class: str
    center_ego: tuple
    velocity_ego: tuple
    observed: bool
    cameras: tuple
    identity_qualified: bool
    association_unambiguous: bool
    source_frame: str
    source_timestamp: float
    state_version: str
    evidence_origin: str
    score: object
    observation_kind: str = 'current_photometric_support'

@dataclass(frozen=True)
class CurrentCompetitionEvidence:
    candidate_id: str
    semantic_class: str
    center_ego: tuple
    score: float
    cameras: tuple
    source_frame: str
    source_timestamp: float
    evidence_origin: str = 'actual_current_decoded_perception'
    answer_witness_eligible: bool = False

@dataclass(frozen=True)
class CommittedWorldState:
    scene_id: str
    frame_id: str
    timestamp: float
    epoch: int
    version: int
    execution_signature: str
    anchors: tuple
    ego_to_world: tuple
    complete_scene_coverage: bool = False
    competition_evidence: tuple = ()

    def validate(self, context):
        for (key, expected) in (('scene_id', self.scene_id), ('frame_id', self.frame_id), ('timestamp', self.timestamp), ('state_epoch', self.epoch)):
            if context.get(key) != expected:
                raise ValueError('Committed world context mismatch: ' + key)
        if context.get('current_state_version', self.version) != self.version:
            raise ValueError('Committed world version mismatch')
        if context.get('state_execution_signature', self.execution_signature) != self.execution_signature:
            raise ValueError('Committed world execution mismatch')

def freeze_committed_state(manager, frame, qualifier=None, current_detection_support=False, competition_candidates=None):
    """Read current published store only; immutable scalar/tuple observations.

    Identity qualification is separate from current detector evidence. Missing
    current support remains UNKNOWN; no retained prediction is called observed.
    """
    store = manager.store
    if getattr(manager, '_pending', None) is not None or store.scene_id != frame.scene_id or store.last_timestamp != frame.timestamp or (frame.frame_id not in store.committed_frame_ids) or (manager._previous_frame_id != frame.frame_id) or (store.execution_signature != frame.execution_signature) or (manager.last_commit_diagnostics.get('transaction_status') == 'rejected'):
        raise ValueError('World diff requires matching completed M1 commit')
    from worldstream.m1.geometry import valid_transform
    if not valid_transform(frame.ego_to_world):
        raise ValueError('Committed snapshot requires actual ego pose')
    qualified = set()
    transform = np.linalg.inv(frame.ego_to_world)
    geometric_cameras = []
    if current_detection_support or competition_candidates is not None:
        from worldstream.m1.geometry import valid_projection_camera
        geometric_cameras = [(cid, cam) for (cid, cam) in frame.cameras.items() if cam.timing_reliable and abs(cam.timestamp - frame.timestamp) <= manager.config.max_state_age_s and valid_projection_camera(cam)]
    records = []
    for a in sorted(store.anchors.values(), key=lambda a: a.anchor_id):
        if a.evidence_origin != 'actual_perception':
            continue
        cameras = tuple(sorted((k for (k, v) in a.observation_support.items() if k in frame.cameras and v.timestamp == frame.cameras[k].timestamp)))
        observation_kind = 'current_photometric_support'
        if current_detection_support and a.anchor_type == 'object' and (not cameras) and (a.source_frame_id == frame.frame_id) and (a.source_timestamp == frame.timestamp):
            projected = []
            point = np.asarray(a.geometry_world.center)
            for (cid, cam) in geometric_cameras:
                c = cam.world_to_camera[:3, :3] @ point + cam.world_to_camera[:3, 3]
                uvh = cam.intrinsic @ c
                if c[2] > 0.1:
                    (u, v) = uvh[:2] / uvh[2]
                    if 0 <= u < cam.image.shape[1] and 0 <= v < cam.image.shape[0]:
                        projected.append(cid)
            cameras = tuple(sorted(projected))
            if cameras:
                observation_kind = 'current_detector_center_projection_not_reuse_certificate'
        observed = a.source_frame_id == frame.frame_id and a.source_timestamp == frame.timestamp and bool(cameras)
        center = transform[:3, :3] @ a.geometry_world.center + transform[:3, 3]
        velocity = transform[:3, :3] @ a.velocity_world if a.velocity_reliable and a.velocity_world is not None else None
        if not np.isfinite(center).all():
            continue
        last = a.association_history[-1] if a.association_history else {}
        records.append(CommittedAnchor(a.anchor_id, a.anchor_type, a.semantic_class, tuple(map(float, center)), tuple(map(float, velocity)) if velocity is not None and np.isfinite(velocity).all() else (), observed, cameras, bool(a.identity_link_qualified or a.anchor_id in qualified), bool(last.get('source_is_previous_frame') and last.get('no_competing_edge')), a.source_frame_id or '', float(a.source_timestamp), a.state_version, a.evidence_origin, float(a.source_score) if a.source_score is not None and np.isfinite(a.source_score) else None, observation_kind))
    competition = []
    return CommittedWorldState(frame.scene_id, frame.frame_id, frame.timestamp, store.generation, store.commit_count, store.execution_signature, tuple(records), tuple(map(float, frame.ego_to_world.reshape(-1))), competition_evidence=tuple(competition))
