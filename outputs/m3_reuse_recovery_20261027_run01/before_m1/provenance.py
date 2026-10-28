from dataclasses import asdict
from .state_types import ComputationalProvenance
import copy
import math


def provenance_for(record, frame):
    return ComputationalProvenance(
        record.feature_id, list(record.source_anchor_ids), list(record.source_camera_ids),
        dict(record.source_region), record.source_layer, record.dependency_scope,
        coordinate_space={k: v.coordinate_space for k, v in frame.cameras.items()})


def export_provenance(store):
    return {key: asdict(value) for key, value in store.provenance.items()}


def export_state_audit(manager):
    """Defensive committed-state export: observations, candidates and claims differ."""
    from .geometry import state_geometry_version
    store=manager.store
    certificates=manager.get_current_certifications()
    qualified={c.anchor_id:c for c in certificates.certifications}
    result=manager.last_result
    historical = result is not None and result.diagnostics.get('transaction_status')=='committed'
    rejected={}
    for c in certificates.rejected:
        rejected.setdefault(c['anchor_id'],[]).extend(c['rejection_reasons'])
    rows=[]
    for aid,a in store.anchors.items():
        certificate=qualified.get(aid)
        rows.append(dict(anchor_id=aid, semantic_class=a.semantic_class,anchor_type=a.anchor_type,
            center_world=a.geometry_world.center.tolist(), source_frame_id=a.source_frame_id,
            source_timestamp=a.source_timestamp, source_epoch=a.source_state_epoch,
            state_version=state_geometry_version(a), execution_signature=a.source_execution_signature,
            cameras={cid:dict(timestamp=o.timestamp,region=list(o.region)) for cid,o in a.observation_support.items()},
            observed=bool(a.observation_support), candidate=bool(historical and aid in result.association_candidate_ids),
            historical_observation_supported=bool(historical and aid in result.observation_supported_anchor_ids),
            observation_validation_phase='precheck_historical_version_only',
            historical_observation_source=(result.anchor_provenance.get(aid) if historical else None),
            identity_status='identity_certified' if certificate else ('rejected' if aid in rejected else 'unknown'),
            identity_rejection_reasons=sorted(set(rejected.get(aid,[]))),
            identity_certified=certificate is not None, physical_or_geometry_certified=False,
            certified_claim=certificate.certified_claim if certificate else None,
            status='identity_certified' if certificate else ('observed' if a.observation_support else 'unknown'),
            association_basis=a.association_basis, motion_prediction=a.motion_prediction,
            motion_history=a.motion_history))
    for row in rows:
        external = store.anchors[row['anchor_id']].external_tracking
        if external:
            row['external_tracking'] = copy.deepcopy(external)
    return copy.deepcopy(dict(scene_id=store.scene_id,log_token=manager._committed_log_token,frame_id=manager._previous_frame_id,
        timestamp=store.last_timestamp,state_epoch=store.generation,source_digest=store.source_digest(),
        execution_signature=store.execution_signature,transaction_status='committed' if store.last_timestamp is not None else 'empty',
        anchors=rows,
        features=store.feature_versions(), certifications=certificates.to_dict(),
        last_transaction=manager.last_commit_diagnostics,
        invalidations={} if result is None else {k:result.diagnostics.get(k) for k in
            ('reset_reason','expired_anchor_ids','input_rejection','transaction_status')},
        no_gpu_computation_skipped=True))
