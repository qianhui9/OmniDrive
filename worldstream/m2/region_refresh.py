"""WSGRR: one explainable selector; no network, image warp or identity inference."""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from .packing import partition, window_mask
from .dependency_planner import patch_updates


MUST_EVENTS = frozenset(('appeared', 'disappeared', 'reappeared', 'state_changed',
    'association_invalid', 'state_jump', 'source_invalid', 'state_expired'))


def pooled_descriptors(x, size):
    windows, _ = partition(x.float(), size)
    valid, _ = partition(torch.ones_like(x[..., :1], dtype=torch.float32), size)
    return windows.sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)


def cosine_drift(current, previous):
    # Equal zero vectors have zero drift; a zero/nonzero pair has distance one.
    drift = (1 - F.cosine_similarity(current, previous, dim=-1)).clamp(0, 2)
    return torch.where((current == previous).all(-1), torch.zeros_like(drift), drift)


def region_tokens(region, camera, shape, projection, image_hw, margin=0, device='cpu'):
    """Intersect the actual dilated Conv2d sampling lattice with a half-open box.

    No dense image allocation, fixed patch divisor, or bounding receptive-field
    approximation: even gaps between dilated kernel samples are respected.
    """
    n, h, w = shape
    result = torch.zeros(shape, dtype=torch.bool, device=device)
    x0, y0, x1, y1 = region
    kh, kw = projection.kernel_size; sh, sw = projection.stride
    ph, pw = projection.padding; dh, dw = projection.dilation
    height, width = image_hw
    x0, y0 = max(0, x0-margin), max(0, y0-margin)
    x1, y1 = min(width, x1+margin), min(height, y1+margin)
    if x1 <= x0 or y1 <= y0:
        return result
    # Pixel p occupies [p,p+1); the input contract uses pixel region bounds.
    xs = torch.arange(w, device=device)[:, None]*sw-pw + torch.arange(kw, device=device)*dw
    ys = torch.arange(h, device=device)[:, None]*sh-ph + torch.arange(kh, device=device)*dh
    xx = ((xs >= math.floor(x0)) & (xs < math.ceil(x1)) & (xs >= 0) & (xs < width)).any(1)
    yy = ((ys >= math.floor(y0)) & (ys < math.ceil(y1)) & (ys >= 0) & (ys < height)).any(1)
    result[camera] = yy[:, None] & xx[None, :]
    return result


def select_regions(x0, cache, bundle, context, projection, size, config,
                   uncertainty_pixels, mandatory_tokens, *, events=True, budget=True):
    shape = x0.shape[:3]; device = x0.device
    descriptions = pooled_descriptors(x0, size)
    drift = cosine_drift(descriptions, cache.descriptors)
    ages = cache.window_ages[0] + 1
    mandatory = window_mask(mandatory_tokens, size)
    # Average local evidence, including only real pixels/tokens at padded edges.
    pixels = torch.as_tensor(uncertainty_pixels, device=device, dtype=torch.float32)
    ph, pw = projection.padding
    token_evidence = F.avg_pool2d(F.pad(pixels[:, None], (pw,pw,ph,ph)),
        projection.kernel_size, projection.stride)[:, 0] if projection.dilation == (1,1) else None
    if token_evidence is None:
        token_evidence = F.unfold(pixels[:,None], projection.kernel_size,
            projection.dilation, projection.padding, projection.stride).mean(1).reshape(shape)
    evidence = pooled_descriptors(token_evidence[...,None], size).flatten()
    old = {(r['camera_id'], r['track_id']): r for r in cache.region_states}
    snapshot = copy.deepcopy(old)
    event_rows = []
    cameras = {cid: i for i, cid in enumerate(context['camera_ids'])}
    def mask_for(r):
        return region_tokens(r['bbox_xyxy'], cameras[r['camera_id']], shape,
                             projection, uncertainty_pixels.shape[-2:], config.region_margin_px, device)
    def add_event(key, names, current=None):
        nonlocal mandatory
        affected = torch.zeros(shape, device=device, dtype=torch.bool)
        for r in (old.get(key), current):
            if r is not None:
                affected |= mask_for(r)
        windows = window_mask(affected, size)
        mandatory |= windows
        event_rows.append(dict(camera_id=key[0], track_id=key[1], events=sorted(names),
            previous_bbox=copy.deepcopy(old.get(key,{}).get('bbox_xyxy')),
            current_bbox=copy.deepcopy(current.get('bbox_xyxy')) if current is not None else None,
            class_id=(current or old.get(key,{})).get('class_id'),
            windows=windows.nonzero().flatten().tolist()))
    if events:
        current_keys={(r['camera_id'],r['track_id']) for r in bundle.region_states}
        for key,prior in old.items():
            if key not in current_keys and prior.get('valid_until', math.inf)<bundle.timestamp:
                add_event(key,['state_expired'])
                snapshot.pop(key,None)
        for event in bundle.region_events:
            for key in list(old):
                if (key[1] == event['track_id'] and event['event'] in MUST_EVENTS
                        and event.get('camera_id',key[0])==key[0]):
                    add_event(key, [event['event']])
                    if event['event'] in ('state_expired','disappeared'):
                        snapshot.pop(key, None)
    for row in bundle.region_states:
        key = (row['camera_id'], row['track_id'])
        prior = old.get(key)
        names = set(row.get('event_flags', ())) & MUST_EVENTS
        if events:
            if prior is None:
                birth=row.get('track_birth_timestamp')
                names.add('appeared' if birth is not None and birth >= cache.context['timestamp'] else 'region_entered')
            else:
                if row['state_signature'] != prior['state_signature']:
                    names.add('state_changed')
                delta = np.max(np.abs(np.asarray(row['bbox_xyxy'])-prior['bbox_xyxy']))
                # A centre crossing a window boundary is position-sensitive;
                # changing a box edge by a pixel alone is ordinary uncertainty.
                def centre_window(r):
                    x0,y0,x1,y1 = r['bbox_xyxy']
                    kh,kw=projection.kernel_size;sh,sw=projection.stride
                    ph,pw=projection.padding;dh,dw=projection.dilation
                    return (math.floor(((x0+x1)/2+pw-dw*(kw-1)/2)/sw)//size,
                            math.floor(((y0+y1)/2+ph-dh*(kh-1)/2)/sh)//size)
                if delta >= config.movement_threshold_px or centre_window(row) != centre_window(prior):
                    names.add('position_changed')
                if row.get('visible') is True and prior.get('occluded') is True:
                    names.add('reappeared')
                if prior.get('identity_link_qualified') and not row.get('identity_link_qualified'):
                    names.add('association_invalid')
            if names:
                add_event(key, names, row)
        if events and names & {'disappeared','state_expired'}:
            snapshot.pop(key,None)
        else:
            snapshot[key] = copy.deepcopy(row)
    score = (config.drift_weight*drift + config.age_weight*(ages.float()/config.max_reuse_frames).clamp(max=1)
             + config.uncertainty_weight*evidence)
    selected = mandatory.clone()
    per_camera = selected.numel()//len(x0)
    # CPU ordering is deterministic for score/age ties, including CUDA inputs.
    scores, age_values = score.tolist(), ages.tolist()
    for camera in range(len(x0)):
        ids = [i for i in range(camera*per_camera,(camera+1)*per_camera)
               if not mandatory[i] and scores[i] >= config.score_threshold]
        ids.sort(key=lambda i: (-scores[i], -age_values[i], i))
        if budget:
            ids = ids[:config.ordinary_budget]
        selected[ids] = True
    return selected, descriptions, list(snapshot.values()), dict(
        mandatory_windows=int(mandatory.sum()), ordinary_windows=int((selected & ~mandatory).sum()),
        mandatory_window_ids=mandatory.nonzero().flatten().tolist(),
        ordinary_window_ids=(selected & ~mandatory).nonzero().flatten().tolist(),
        drift=drift.tolist(), ages=age_values, uncertainty=evidence.tolist(), scores=scores,
        event_regions=event_rows, region_state_count=len(bundle.region_states),
        event_affected_windows=sum(len(e['windows']) for e in event_rows), event_missed_windows=0)
