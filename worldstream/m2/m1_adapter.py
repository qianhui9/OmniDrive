"""Validate source-linked M1 hints; exact Transformer equality belongs to M2."""
import numpy as np
from worldstream.m1.mask_codec import decode_mask

def computation_admission(bundle, context, images, cache, force_refresh=False, details=None, window_footprints=None):
    """Validate metadata; return dirty hints only, NEVER an equality mask.

    Unknown is eligible for full independent Exact comparison. Candidate pixels
    get exactly the same full comparison. No warp enters the execution path.
    """
    from worldstream.m1.novelty_detector import validate_refresh_partition
    if bundle is None:
        return (None, 'missing_computation_state')
    if bundle.diagnostics.get('transaction_status') == 'aborted':
        return (None, 'm1_aborted_transaction')
    if not isinstance(bundle.timestamp, (int, float)) or not np.isfinite(bundle.timestamp):
        return (None, 'm1_source_time_mismatch')
    if bundle.scene_id != context.get('scene_id') or bundle.frame_id != context.get('frame_id') or bundle.execution_signature != context.get('execution_signature') or (bundle.timestamp != context.get('timestamp')):
        return (None, 'm1_context_mismatch')
    if context.get('state_epoch') is not None and context['state_epoch'] != bundle.state_epoch:
        return (None, 'm1_state_epoch_mismatch')
    if force_refresh or bundle.diagnostics.get('full_refresh_hint'):
        return (None, 'm1_full_refresh')
    if context.get('coordinate_space') != 'preprocessed_image':
        return (None, 'preprocessing_geometry_unavailable')
    if not np.isfinite(bundle.timestamp):
        return (None, 'm1_source_time_mismatch')
    if cache.context is not None:
        if bundle.diagnostics.get('cache_frame_id') != cache.context['frame_id'] or bundle.diagnostics.get('source_cache_version') != cache.version or bundle.diagnostics.get('observed_source_frame_id') != cache.context['frame_id']:
            return (None, 'm1_cache_source_mismatch')
        if cache.context.get('state_epoch') is not None and cache.context['state_epoch'] != bundle.state_epoch:
            return (None, 'm1_state_epoch_mismatch')
    else:
        return (None, 'no_valid_cache')
    if not bundle.diagnostics.get('cache_available'):
        return (None, 'm1_computation_unavailable')
    window_sources = [s for values in cache.sources.values() for s in values]
    horizons = [link.get('validity_lease', {}).get('max_observation_age') for link in bundle.observation_links.values()]
    if not horizons or any((not isinstance(v, (float, int)) or not np.isfinite(v) or v < 0 for v in horizons)):
        return (None, 'm1_invalid_lease')
    camera_reasons = {}
    local_expired = np.zeros((len(images), *images.shape[-2:]), dtype=bool)
    source_positions = {id(source): (block, index) for (block, units) in cache.sources.items() for (index, source) in enumerate(units)}
    for source in window_sources:
        cid = source.get('camera_id')
        if cid not in context['camera_ids']:
            return (None, 'm1_computation_source_mismatch')
        horizon = bundle.observation_links.get(cid, {}).get('validity_lease', {}).get('max_observation_age')
        if not isinstance(horizon, (int, float)) or not np.isfinite(horizon) or horizon < 0:
            return (None, 'm1_invalid_lease')
        (block, index) = source_positions[id(source)]
        footprints = window_footprints.get(block, [])
        if not footprints or len(cache.sources[block]) != len(images) * len(footprints) or context['camera_ids'][index // len(footprints)] != cid or (source.get('producer_layer') != block) or (not isinstance(source.get('computation_version'), int)) or (not 0 < source['computation_version'] <= cache.version):
            return (None, 'm1_computation_source_mismatch')
        timestamp = source.get('computation_timestamp')
        if not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp) or (not 0 < bundle.timestamp - timestamp):
            return (None, 'm1_computation_source_expired')
        if bundle.timestamp - timestamp > horizon and (not False):
            camera_index = index // len(footprints)
            (x0, y0, x1, y1) = footprints[index % len(footprints)]['input_region']
            local_expired[camera_index, y0:y1, x0:x1] = True
    masks = []
    if set(bundle.refresh_contract) != set(context.get('camera_ids', [])):
        return (None, 'm1_camera_order_mismatch')
    for cid in context['camera_ids']:
        contract = bundle.refresh_contract[cid]
        link = bundle.observation_links.get(cid, {})
        lease = link.get('validity_lease', {})
        if link.get('source_frame_id') != cache.context['frame_id'] or link.get('source_camera_id') != cid:
            return (None, 'm1_invalid_lease')
        if not lease.get('source_valid'):
            camera_reasons[cid] = 'm1_invalid_lease'
        source_time = link.get('source_timestamp')
        horizon = lease.get('max_observation_age')
        if not isinstance(source_time, (int, float)) or not np.isfinite(source_time) or (not isinstance(horizon, (int, float))) or (not np.isfinite(horizon)) or (horizon < 0) or (not 0 < bundle.timestamp - source_time):
            return (None, 'm1_invalid_lease')
        if bundle.timestamp - source_time > horizon:
            camera_reasons[cid] = 'm1_invalid_lease'
        for record in bundle.source_records:
            if record.get('source_camera_id') != cid or not record.get('computation_available'):
                continue
            computed_time = record.get('computation_timestamp')
            if True and str(record.get('feature_id', '')).startswith('eva_prefix:'):
                try:
                    (_, block, index) = record['feature_id'].split(':')
                    producer = cache.sources[int(block)][int(index)]
                except (ValueError, KeyError, IndexError, TypeError):
                    return (None, 'm1_computation_source_mismatch')
                if record.get('computation_version_namespace') != 'm2_window_execution' or producer.get('camera_id') != cid or record.get('computation_frame_id') != producer.get('computation_frame_id') or (record.get('computation_version') != producer.get('computation_version')) or (computed_time != producer.get('computation_timestamp')):
                    return (None, 'm1_computation_source_mismatch')
            if record.get('execution_signature') != bundle.execution_signature or record.get('computation_frame_id') == bundle.frame_id or (not isinstance(computed_time, (int, float))) or (not np.isfinite(computed_time)) or (not 0 < bundle.timestamp - computed_time) or (record.get('source_cache_version') not in (None, cache.version)):
                return (None, 'm1_computation_source_mismatch')
            if bundle.timestamp - computed_time > horizon and (not False) and (not (True and str(record.get('feature_id', '')).startswith('eva_prefix:'))):
                camera_reasons[cid] = 'm1_computation_source_expired'
        if contract.get('full_refresh_hint'):
            return (None, 'm1_full_refresh')
        if contract.get('coordinate_space') != 'preprocessed_image':
            return (None, 'preprocessing_geometry_unavailable')
        try:
            (dirty, candidate, unknown) = [decode_mask(contract[key]) for key in ('dirty_hint_mask', 'reuse_candidate_mask', 'unknown_mask')]
        except Exception:
            return (None, 'invalid_refresh_contract')
        if dirty.shape != tuple(images.shape[-2:]) or not validate_refresh_partition(dirty, candidate, unknown):
            return (None, 'invalid_refresh_contract')
        if True and cid in camera_reasons:
            dirty = np.ones_like(dirty)
        elif True and dirty.any():
            camera_reasons[cid] = 'm1_dirty_camera_observation'
        dirty = dirty | local_expired[len(masks)]
        masks.append(dirty)
    if details is not None:
        details.update(camera_reasons=camera_reasons, force_recompute_cameras=[bool(m.any()) for m in masks], expired_source_pixels=int(local_expired.sum()), )
    return (np.stack(masks), None)
