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
