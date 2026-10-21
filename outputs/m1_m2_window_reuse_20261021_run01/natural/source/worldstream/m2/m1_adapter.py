"""M1 observation candidates are never accepted as transformer equality certificates."""
import numpy as np
import torch
from functools import lru_cache
from worldstream.m1.mask_codec import decode_mask


@lru_cache(maxsize=16)
def _rejection_samples(height,width,window_h,window_w):
    rows=[];cols=[]
    for y in range(0,height,window_h):
        for x in range(0,width,window_w):
            yy=(y,(y+min(y+window_h,height)-1)//2,min(y+window_h,height)-1)
            xx=(x,(x+min(x+window_w,width)-1)//2,min(x+window_w,width)-1)
            rows.append([a for a in yy for b in xx]);cols.append([b for a in yy for b in xx])
    return np.asarray(rows),np.asarray(cols)


def invalidation_mask(result, state, context, images, device=True, rejection_footprint=None):
    """Accept only model-input coordinates; unknown preprocessing fails closed.

    The live M1 model adapter uses these exact preprocessed coordinates. Old 640x360
    raw replays cannot be resized to 640x640 masks without the crop/interpolation
    footprint, and deliberately trigger a full forward instead.
    """
    if result is None:
        return None, None  # Controlled exact input comparison, separately labelled.
    if (result.scene_id != context['scene_id'] or result.frame_id != context['frame_id']
            or result.diagnostics.get('execution_signature') != context.get('execution_signature')):
        return None, 'm1_context_mismatch'
    if result.full_refresh_hint:
        return None, 'm1_full_refresh'
    coordinate = context.get('coordinate_space')
    if coordinate != 'preprocessed_image':
        return None, 'preprocessing_geometry_unavailable'
    masks = []
    for cid in context['camera_ids']:
        region = result.invalidation_regions.get(cid, {})
        if region.get('coordinate_space') != 'preprocessed_image':
            return None, 'preprocessing_geometry_unavailable'
        encoded = region.get('exact_invalidation_mask')
        if encoded is None:
            return None, 'missing_invalidation_mask'
        mask = (state.last_masks.get(cid) if state is not None and state.last_result is result
                else None)
        if mask is None:
            mask = decode_mask(encoded)
        if mask.shape != tuple(images.shape[-2:]):
            return None, 'preprocessing_geometry_unavailable'
        masks.append(mask)
    # dense_source_map / feature_versions / provenance remain M1 evidence only.
    # Strict equality is independently established on the retained actual input.
    if rejection_footprint is not None:
        ph,pw,window_size=rejection_footprint
        h,w=images.shape[-2:]
        rows,cols=_rejection_samples(h//ph*ph,w//pw*pw,ph*window_size,pw*window_size)
        # One invalid sampled pixel proves a window cannot be reused. Conversely,
        # no sampled invalid pixel is NEVER an acceptance: the full reduction follows.
        if all(mask[rows,cols].any(axis=1).all() for mask in masks):
            return None,'insufficient_stable_windows'
    masks = np.stack(masks)
    return (torch.as_tensor(masks, device=images.device) if device else masks), None


def cpu_window_opportunity(invalid, projection, window_size):
    """Exact footprint reduction on existing CPU M1 masks; no CUDA transfer/launch."""
    b,h,w = invalid.shape
    ph,pw = projection.kernel_size
    th,tw = h//ph,w//pw
    patches = invalid[:,:th*ph,:tw*pw].reshape(b,th,ph,tw,pw).any(axis=(2,4))
    hp,wp = (th+window_size-1)//window_size*window_size,(tw+window_size-1)//window_size*window_size
    padded = np.pad(patches,((0,0),(0,hp-th),(0,wp-tw)),constant_values=False)
    return ~padded.reshape(b,hp//window_size,window_size,wp//window_size,window_size).any(axis=(2,4))


def computation_admission(bundle, context, images, cache, force_refresh=False,
                          selective_camera=False, separate_computation_age=False, details=None, window_footprints=None):
    """Validate metadata; return dirty hints only, NEVER an equality mask.

    Unknown is eligible for full independent Exact comparison. Candidate pixels
    get exactly the same full comparison. No warp enters the execution path.
    """
    from worldstream.m1.novelty_detector import validate_refresh_partition
    if bundle is None:
        return None, "missing_computation_state"
    if bundle.diagnostics.get('transaction_status') == 'aborted':
        return None, 'm1_aborted_transaction'
    if (bundle.scene_id != context.get("scene_id") or bundle.frame_id != context.get("frame_id")
            or bundle.execution_signature != context.get("execution_signature")
            or bundle.timestamp != context.get("timestamp")):
        return None, "m1_context_mismatch"
    if force_refresh or bundle.diagnostics.get("full_refresh_hint"):
        return None, "m1_full_refresh"
    if context.get("coordinate_space") != "preprocessed_image":
        return None, "preprocessing_geometry_unavailable"
    if not np.isfinite(bundle.timestamp):
        return None, "m1_source_time_mismatch"
    if cache.context is not None:
        if (bundle.diagnostics.get("cache_frame_id") != cache.context["frame_id"]
                or bundle.diagnostics.get("source_cache_version") != cache.version
                or bundle.diagnostics.get("observed_source_frame_id") != cache.context["frame_id"]):
            return None, "m1_cache_source_mismatch"
        if (cache.context.get('state_epoch') is not None
                and cache.context['state_epoch'] != bundle.state_epoch):
            return None, "m1_state_epoch_mismatch"
    else:
        return None, "no_valid_cache"
    if not bundle.diagnostics.get("cache_available"):
        return None, "m1_computation_unavailable"
    camera_sources = cache.sources if isinstance(cache.sources,list) else [s for values in cache.sources.values() for s in values]
    horizons = [link.get('validity_lease',{}).get('max_observation_age') for link in bundle.observation_links.values()]
    if not horizons or any(not isinstance(v,(float,int)) or not np.isfinite(v) or v < 0 for v in horizons):
        return None, 'm1_invalid_lease'
    horizon = min(horizons)
    camera_reasons = {}
    local_expired = np.zeros((len(images), *images.shape[-2:]), dtype=bool)
    window_mode = window_footprints is not None and isinstance(cache.sources, dict)
    source_positions = {id(source): (block, index) for block, units in cache.sources.items()
                        for index, source in enumerate(units)} if window_mode else {}
    for source in camera_sources:
        cid = source.get('camera_id')
        if cid not in context['camera_ids']:
            return None, 'm1_computation_source_mismatch'
        horizon = bundle.observation_links.get(cid, {}).get('validity_lease', {}).get('max_observation_age')
        if not isinstance(horizon, (int, float)) or not np.isfinite(horizon) or horizon < 0:
            return None, 'm1_invalid_lease'
        timestamp = source.get('computation_timestamp')
        if (not isinstance(timestamp,(int,float)) or not np.isfinite(timestamp)
                or not 0 < bundle.timestamp-timestamp):
            return None, 'm1_computation_source_expired'
        if bundle.timestamp-timestamp > horizon and not separate_computation_age:
            if window_mode:
                block, index = source_positions[id(source)]
                footprints = window_footprints.get(block, [])
                if not footprints or len(cache.sources[block]) != len(images)*len(footprints):
                    return None, 'm1_computation_source_mismatch'
                camera_index = index//len(footprints)
                if context['camera_ids'][camera_index] != cid:
                    return None, 'm1_computation_source_mismatch'
                x0,y0,x1,y1 = footprints[index % len(footprints)]['input_region']
                local_expired[camera_index,y0:y1,x0:x1] = True
            else:
                if not selective_camera:return None, 'm1_computation_source_expired'
                camera_reasons[cid] = 'm1_computation_source_expired'
    masks = []
    if set(bundle.refresh_contract) != set(context.get("camera_ids", [])):
        return None, "m1_camera_order_mismatch"
    for cid in context["camera_ids"]:
        contract = bundle.refresh_contract[cid]
        link = bundle.observation_links.get(cid, {})
        lease = link.get("validity_lease", {})
        if (link.get("source_frame_id") != cache.context["frame_id"]
                or link.get("source_camera_id") != cid):
            return None, "m1_invalid_lease"
        if not lease.get('source_valid'):
            if not selective_camera:return None, 'm1_invalid_lease'
            camera_reasons[cid] = 'm1_invalid_lease'
        source_time = link.get('source_timestamp')
        horizon = lease.get('max_observation_age')
        if (not isinstance(source_time,(int,float)) or not np.isfinite(source_time)
                or not isinstance(horizon,(int,float)) or not np.isfinite(horizon) or horizon < 0
                or not 0 < bundle.timestamp-source_time):
            return None, 'm1_invalid_lease'
        if bundle.timestamp-source_time > horizon:
            if not selective_camera:return None, 'm1_invalid_lease'
            camera_reasons[cid] = 'm1_invalid_lease'
        for record in bundle.source_records:
            if record.get('source_camera_id') != cid or not record.get('computation_available'):
                continue
            computed_time = record.get('computation_timestamp')
            if isinstance(cache.sources,list) and record.get('feature_id')=='backbone':
                producer=next((s for s in cache.sources if s.get('camera_id')==cid),{})
                if (record.get('computation_version_namespace')!='m2_camera_execution'
                        or record.get('computation_frame_id')!=producer.get('computation_frame')
                        or record.get('computation_version')!=producer.get('computation_version')
                        or computed_time!=producer.get('computation_timestamp')):
                    return None, 'm1_computation_source_mismatch'
            if window_mode and str(record.get('feature_id','')).startswith('eva_prefix:'):
                try:
                    _, block, index = record['feature_id'].split(':')
                    producer = cache.sources[int(block)][int(index)]
                except (ValueError, KeyError, IndexError, TypeError):
                    return None, 'm1_computation_source_mismatch'
                if (record.get('computation_version_namespace') != 'm2_window_execution'
                        or producer.get('camera_id') != cid
                        or record.get('computation_frame_id') != producer.get('computation_frame_id')
                        or record.get('computation_version') != producer.get('computation_version')
                        or computed_time != producer.get('computation_timestamp')):
                    return None, 'm1_computation_source_mismatch'
            if (record.get('execution_signature') != bundle.execution_signature
                    or record.get('computation_frame_id') == bundle.frame_id
                    or not isinstance(computed_time,(int,float)) or not np.isfinite(computed_time)
                    or not 0 < bundle.timestamp-computed_time
                    or record.get('source_cache_version') not in (None,cache.version)):
                return None, 'm1_computation_source_mismatch'
            if (bundle.timestamp-computed_time > horizon and not separate_computation_age
                    and not (window_mode and str(record.get('feature_id','')).startswith('eva_prefix:'))):
                if not selective_camera:return None, 'm1_computation_source_mismatch'
                camera_reasons[cid] = 'm1_computation_source_expired'
        if contract.get("full_refresh_hint"):
            return None, "m1_full_refresh"
        if contract.get("coordinate_space") != "preprocessed_image":
            return None, "preprocessing_geometry_unavailable"
        try:
            dirty, candidate, unknown = [decode_mask(contract[key]) for key in
                ("dirty_hint_mask", "reuse_candidate_mask", "unknown_mask")]
        except Exception:
            return None, "invalid_refresh_contract"
        if dirty.shape != tuple(images.shape[-2:]) or not validate_refresh_partition(dirty, candidate, unknown):
            return None, "invalid_refresh_contract"
        if selective_camera and cid in camera_reasons:
            dirty = np.ones_like(dirty)
        elif selective_camera and dirty.any():
            camera_reasons[cid] = 'm1_dirty_camera_observation'
        if window_mode:
            dirty = dirty | local_expired[len(masks)]
        masks.append(dirty)
    if details is not None:
        details.update(camera_reasons=camera_reasons,
            force_recompute_cameras=[bool(m.any()) for m in masks],
            expired_source_pixels=int(local_expired.sum()),
            observation_lease_separated=separate_computation_age)
    return np.stack(masks), None
