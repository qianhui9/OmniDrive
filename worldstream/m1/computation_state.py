"""Committed computation lineage and revocable, pixel-scoped RGB links.

The lease uses the existing state horizon. It does not extend a computation
source lifetime when a later request verifies an older cached value.
"""
from dataclasses import asdict
import copy
import cv2
import numpy as np
from time import perf_counter
from .geometry import valid_projection_camera
from .mask_codec import encode_mask
from .state_types import ComputationStateBundle, ValidityLease

def validity_lease(source_timestamp, observed_timestamp, now, horizon, compatible=True, conflict=False):
    reason = None
    if not compatible:
        reason = 'source_incompatible'
    elif conflict:
        reason = 'observed_conflict'
    elif any((t is None or not np.isfinite(t) for t in (source_timestamp, observed_timestamp, now))):
        reason = 'source_missing'
    elif not source_timestamp <= observed_timestamp < now:
        reason = 'source_time_order'
    elif now - source_timestamp > horizon or now - observed_timestamp > horizon:
        reason = 'source_expired'
    return ValidityLease(observed_timestamp, observed_timestamp, horizon, reason is None, reason)

def build_bundle(manager, frame, result, validate_observation=True):
    bundle = ComputationStateBundle(frame.scene_id, frame.frame_id, frame.timestamp, manager.store.generation, frame.execution_signature)
    receipt = manager._computation_receipt or {}
    cache = receipt.get('cache_context') or {}
    bundle.diagnostics = dict(observed_source_frame_id=manager._previous_frame_id, observed_source_timestamp=manager.store.last_timestamp, source_cache_version=receipt.get('cache_version') if receipt.get('cache_saved') else None, cache_available=bool(receipt.get('cache_saved')), cache_frame_id=cache.get('frame_id'), full_refresh_hint=result.full_refresh_hint, validation_requested=validate_observation, transformer_dependency_certified=False, validity_lease_export_enabled=manager.config.validity_lease, lease_policy='existing_max_state_age_s_no_independent_lifecycle', state_service='maintain', observation_only=True)
    bundle.diagnostics['transaction_status'] = 'pending'
    for (cid, camera) in frame.cameras.items():
        shape = camera.image.shape[:2]
        old = manager._previous_cameras.get(cid)
        evidence = manager.last_evidence.get(cid)
        lease = validity_lease(old.timestamp if old else None, old.timestamp if old else None, camera.timestamp, manager.config.max_state_age_s, compatible=bool(old and valid_projection_camera(old) and valid_projection_camera(camera) and old.timing_reliable and camera.timing_reliable and (old.coordinate_space == camera.coordinate_space) and (old.image.shape[:2] == camera.image.shape[:2])))
        candidate = np.zeros(shape, bool)
        dirty = np.zeros(shape, bool)
        source = np.empty((0, 2), np.float32)
        target = np.empty((0, 2), np.float32)
        residual = np.empty(0, np.float32)
        origins = None
        link_diagnostics = {}
        guard_diagnostics = {}
        if evidence is not None:
            dirty |= evidence.changed_mask
            if validate_observation and lease.source_valid and (not result.full_refresh_hint):
                candidate |= evidence.stable_mask
                candidate &= ~dirty
                (yy, xx) = np.nonzero(candidate)
                if evidence.dense_source_map is not None:
                    target = np.column_stack((xx, yy)).astype(np.float32)
                    source = evidence.dense_source_map[yy, xx].copy()
                else:
                    candidate[:] = False
                    keep = evidence.supported & evidence.correspondence_valid & evidence.geometry_valid
                    source = evidence.source_points[keep].copy()
                    target = evidence.current_points[keep].copy()
                    xy = np.rint(target).astype(int)
                    inside = (xy[:, 0] >= 0) & (xy[:, 0] < shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < shape[0])
                    (source, target, xy) = (source[inside], target[inside], xy[inside])
                    keep = ~dirty[xy[:, 1], xy[:, 0]]
                    (source, target, xy) = (source[keep], target[keep], xy[keep])
                    candidate[xy[:, 1], xy[:, 0]] = True
                if len(source):
                    values = []
                    from .temporal_visibility import gray_image
                    old_gray = gray_image(old).astype(np.float32) / 255.0
                    current_gray = gray_image(camera).astype(np.float32) / 255.0
                    for offset in range(0, len(source), 16384):
                        (p, q) = (source[offset:offset + 16384], target[offset:offset + 16384])
                        old_rgb = cv2.remap(old_gray, p[:, 0].reshape(-1, 1), p[:, 1].reshape(-1, 1), cv2.INTER_LINEAR)[:, 0]
                        current_rgb = cv2.remap(current_gray, q[:, 0].reshape(-1, 1), q[:, 1].reshape(-1, 1), cv2.INTER_LINEAR)[:, 0]
                        values.append(np.abs(old_rgb - current_rgb))
                    residual = np.concatenate(values)
        if not lease.source_valid and old is not None:
            dirty[:] = True
            candidate[:] = False
        if result.full_refresh_hint:
            candidate[:] = False
        motion_diagnostics = {}
        link_candidate = candidate
        unknown = ~(dirty | candidate)
        strict = result.invalidation_regions[cid]
        encode_start = perf_counter()
        contract = dict(coordinate_space=camera.coordinate_space, image_shape=list(shape), dirty_hint_mask=encode_mask(dirty), reuse_candidate_mask=encode_mask(candidate), unknown_mask=encode_mask(unknown), refresh_required_mask=copy.deepcopy(strict['refresh_required_mask']), exact_invalidation_mask=copy.deepcopy(strict['exact_invalidation_mask']), full_refresh_hint=result.full_refresh_hint, dirty_area_ratio=float(dirty.mean()), candidate_area_ratio=float(candidate.mean()), unknown_area_ratio=float(unknown.mean()))
        bundle.diagnostics.setdefault('mask_encoding_copy_ms', 0.0)
        bundle.diagnostics['mask_encoding_copy_ms'] += (perf_counter() - encode_start) * 1000
        bundle.refresh_contract[cid] = contract
        bundle.observation_links[cid] = dict(source_frame_id=manager._previous_frame_id, source_camera_id=cid, source_timestamp=old.timestamp if old else None, observation_timestamp=camera.timestamp, coordinate_space=camera.coordinate_space, source_points=source, current_points=target, photometric_residual=residual, residual_space='RGB_derived_gray_intensity_0_1', validity_lease=asdict(lease), local_support_only=True, ambiguity='two_view_RGB_alternatives_not_exhaustively_resolved', unsupported_area='unknown', transformer_dependency_certified=False)
        bundle.observation_links[cid].update(measurement_origins=origins, link_recovery=link_diagnostics, change_first_guard=guard_diagnostics)
        records = [(fid, r) for (fid, r) in manager.store.features.items() if cid in r.source_camera_ids]
        if not records:
            records = [(None, None)]
        for fid,record in records:
            available=bool(record is not None and record.tensor is not None and record.source_frame_id
                and record.computation_timestamp is not None and np.isfinite(record.computation_timestamp)
                and record.computation_timestamp<frame.timestamp and record.execution_signature==frame.execution_signature)
            bundle.source_records.append(dict(feature_id=fid,source_camera_id=cid,source_frame_id=manager._previous_frame_id,source_timestamp=old.timestamp if old else None,
                computation_frame_id=record.source_frame_id if record else None,
                computation_timestamp=record.computation_timestamp if record else None,
                computation_version=record.computation_version if record else None,
                computation_version_namespace='m1_feature_record',
                execution_signature=record.execution_signature if record else frame.execution_signature,
                computation_available=available,source_cache_version=None))
        if receipt.get('cache_saved') and receipt.get('cache_boundary') == 'window' and (cache.get('frame_id') == manager._previous_frame_id) and (cache.get('execution_signature') == frame.execution_signature):
            for (block, units) in receipt.get('output_window_provenance', {}).items():
                geometry = receipt.get('prefix_window_geometry', {}).get(block)
                if geometry is None:
                    geometry = receipt.get('prefix_window_geometry', {}).get(str(block))
                if not geometry:
                    continue
                (ph, pw) = geometry
                cols = (shape[1] + pw - 1) // pw
                per_camera = (shape[0] + ph - 1) // ph * cols
                for (index, unit) in enumerate(units):
                    if unit.get('camera_id') != cid:
                        continue
                    position = index % per_camera
                    (y, x) = (position // cols * ph, position % cols * pw)
                    region = (x, y, min(x + pw, shape[1]), min(y + ph, shape[0]))
                    timestamp = unit.get('computation_timestamp')
                    available = bool(unit.get('computation_frame_id') and timestamp is not None and np.isfinite(timestamp) and (timestamp < frame.timestamp))
                    bundle.source_records.append(dict(feature_id=f'eva_prefix:{block}:{index}', producer_layer=block, source_frame_id=manager._previous_frame_id, source_camera_id=cid, source_timestamp=old.timestamp if old else None, computation_frame_id=unit.get('computation_frame_id'), computation_timestamp=timestamp, computation_version=unit.get('computation_version'), source_cache_version=receipt['cache_version'], computation_version_namespace='m2_window_execution', execution_signature=frame.execution_signature, dependency_scope='closed_window_prefix', source_region=[region], current_region=[region], observation_timestamp=camera.timestamp, observation_valid=bool(candidate[y:region[3], x:region[2]].any()), observed_conflict=bool(dirty[y:region[3], x:region[2]].any()), computation_available=available, transformer_dependency_certified=False))
    return bundle
