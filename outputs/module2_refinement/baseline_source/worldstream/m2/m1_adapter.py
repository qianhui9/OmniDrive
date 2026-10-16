"""M1 observation candidates are never accepted as transformer equality certificates."""
import numpy as np
import torch
from worldstream.m1.mask_codec import decode_mask


def invalidation_mask(result, state, context, images):
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
        mask = decode_mask(encoded)
        if mask.shape != tuple(images.shape[-2:]):
            return None, 'preprocessing_geometry_unavailable'
        masks.append(mask)
    # dense_source_map / feature_versions / provenance remain M1 evidence only.
    # Strict equality is independently established on the retained actual input.
    return torch.as_tensor(np.stack(masks), device=images.device), None
