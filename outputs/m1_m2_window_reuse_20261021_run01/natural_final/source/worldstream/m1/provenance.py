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


def feature_source_snapshot(record):
    """Only metadata: do not extend a large tensor allocation's lifetime."""
    if record is None:return None
    return {name:copy.deepcopy(getattr(record,name)) for name in (
        'computation_version','computation_timestamp','source_frame_id','source_state_epoch',
        'source_camera_ids','execution_signature','computation_sources')}


def preserve_camera_computation_source(store,frame,diagnostics,previous):
    """M2 may verify old EVA values; current heads still compute new carriers.

    No M1 physical/observation verification is inferred from this M2 witness.
    Mixed camera outputs carry each actual source and a conservative oldest
    computation time. Missing provenance is Unknown, never the current frame.
    """
    record=store.features.get('backbone')
    if record is None or not diagnostics.get('actual_reused_cameras'):return None
    sources=copy.deepcopy(diagnostics.get('output_camera_provenance',[]))
    expected=set(record.source_camera_ids)
    known=(len(sources)==len(expected) and {s.get('camera_id') for s in sources}==expected)
    for source in sources:
        timestamp=source.get('computation_timestamp')
        known=known and bool(source.get('computation_frame')) and isinstance(timestamp,(int,float)) and math.isfinite(timestamp) and timestamp<=frame.timestamp
    if not known:
        record.source_frame_id=None;record.source_state_epoch=None
        record.computation_timestamp=None;record.computation_version=0
        record.computation_sources=sources or [dict(source='Unknown cached EVA computation source')]
    else:
        frames={s['computation_frame'] for s in sources}
        record.source_frame_id=next(iter(frames)) if len(frames)==1 else None
        record.computation_timestamp=min(s['computation_timestamp'] for s in sources)
        record.computation_sources=sources
        def identity(values):
            return sorted((s['camera_id'],s['computation_frame'],s['computation_timestamp']) for s in values)
        old=[]
        if previous and previous['execution_signature']==frame.execution_signature:
            old=previous['computation_sources']
            if not old and previous['source_frame_id'] is not None and previous['computation_timestamp'] is not None:
                old=[dict(camera_id=cid,computation_frame=previous['source_frame_id'],
                    computation_timestamp=previous['computation_timestamp']) for cid in previous['source_camera_ids']]
        same=bool(old and all('camera_id' in s for s in old) and identity(old)==identity(sources))
        if same:
            record.computation_version=previous['computation_version']
            record.source_state_epoch=previous['source_state_epoch']
        else:
            record.source_state_epoch=None
    for fid in ('object_carrier','map_carrier'):
        provenance=store.provenance.get(fid)
        if provenance is not None:
            if known:provenance.upstream_versions['backbone']=record.computation_version
            else:provenance.upstream_versions.pop('backbone',None)
    return dict(actual_camera_sources_known=bool(known),computation_sources=record.computation_sources,
        source_frame_id=record.source_frame_id,computation_timestamp=record.computation_timestamp,
        computation_version=record.computation_version,physical_state_certified=False)


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
        anchors=rows, depth_history={cid:{k:v for k,v in h.items() if k not in ('pixels','depths','uncertainty_m')} | {'points':len(h['pixels'])} for cid,h in manager._depth_history.items() if h is not None},
        features=store.feature_versions(), certifications=certificates.to_dict(),
        last_transaction=manager.last_commit_diagnostics,
        invalidations={} if result is None else {k:result.diagnostics.get(k) for k in
            ('reset_reason','expired_anchor_ids','input_rejection','transaction_status')},
        no_gpu_computation_skipped=True))
