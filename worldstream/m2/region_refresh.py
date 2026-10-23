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
                   uncertainty_pixels, mandatory_tokens, *, events=True, budget=True,window_targets=None):
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
    # Selection only borrows immutable rows. StateCache.commit owns the single
    # defensive metadata copy at the atomic publication boundary.
    snapshot = dict(old)
    if not events:
        snapshot={key:r for key,r in snapshot.items() if r.get('valid_until',math.inf)>=bundle.timestamp}
    event_rows = []
    # Region geometry and state events are CPU data. Rasterize their tiny Token
    # grids on CPU instead of issuing arange/nonzero/D2H work per object on CUDA.
    kh,kw=projection.kernel_size;sh,sw=projection.stride
    ph,pw=projection.padding;dh,dw=projection.dilation
    height,width=uncertainty_pixels.shape[-2:]
    ys=np.arange(shape[1])[:,None]*sh-ph+np.arange(kh)*dh
    xs=np.arange(shape[2])[:,None]*sw-pw+np.arange(kw)*dw
    per_camera=((shape[1]+size-1)//size)*((shape[2]+size-1)//size)
    region_required=np.zeros(len(x0)*per_camera,dtype=bool)
    regional_uncertainty=np.zeros_like(region_required)
    cameras = {cid: i for i, cid in enumerate(context['camera_ids'])}
    mapped_regions={}
    def mask_for(r):
        key=(r['camera_id'],None if r.get('scope')=='camera' else tuple(r['bbox_xyxy']))
        if key in mapped_regions:return mapped_regions[key]
        result=np.zeros(shape[0]*per_camera,dtype=bool)
        mapped_regions[key]=result
        offset=cameras[r['camera_id']]*per_camera
        if r.get('scope')=='camera':
            result[offset:offset+per_camera]=True
            return result
        x0,y0,x1,y1=r['bbox_xyxy'];m=config.region_margin_px
        x0,y0=max(0,x0-m),max(0,y0-m);x1,y1=min(width,x1+m),min(height,y1+m)
        if x1<=x0 or y1<=y0:return result
        xx=((xs>=math.floor(x0))&(xs<math.ceil(x1))&(xs>=0)&(xs<width)).any(1)
        yy=((ys>=math.floor(y0))&(ys<math.ceil(y1))&(ys>=0)&(ys<height)).any(1)
        tokens=yy[:,None]&xx[None,:]
        hp,wp=(shape[1]+size-1)//size*size,(shape[2]+size-1)//size*size
        tokens=np.pad(tokens,((0,hp-shape[1]),(0,wp-shape[2])))
        windows=tokens.reshape(hp//size,size,wp//size,size).any((1,3)).reshape(-1)
        result[offset:offset+per_camera]=windows
        return result
    uncertainty_rows=[]
    for row in bundle.uncertainty_regions:
        affected=mask_for(row)
        regional_uncertainty |= affected
        if row['mandatory']:region_required |= affected
        uncertainty_rows.append(dict(camera_id=row['camera_id'],
            source_track_id=row.get('source_track_id'),scope=row['scope'],
            bbox_xyxy=row['bbox_xyxy'],reason=row.get('reason'),mandatory=row['mandatory'],
            windows=np.flatnonzero(affected).tolist(),world_identity_authorized=False))
    uncertainty_required=region_required.copy()
    def add_event(key, names, current=None):
        affected = np.zeros(len(x0)*per_camera,dtype=bool)
        for r in (old.get(key), current):
            if r is not None:
                affected |= mask_for(r)
        region_required[:] |= affected
        event_rows.append(dict(camera_id=key[0], track_id=key[1], events=sorted(names),
            previous_bbox=copy.deepcopy(old.get(key,{}).get('bbox_xyxy')),
            current_bbox=copy.deepcopy(current.get('bbox_xyxy')) if current is not None else None,
            class_id=(current or old.get(key,{})).get('class_id'),
            windows=np.flatnonzero(affected).tolist()))
    if events:
        current_keys={(r['camera_id'],r['track_id']) for r in bundle.region_states}
        old_by_track={}
        for key in old:old_by_track.setdefault(key[1],[]).append(key)
        for key,prior in old.items():
            if key not in current_keys and prior.get('valid_until', math.inf)<bundle.timestamp:
                add_event(key,['state_expired'])
                snapshot.pop(key,None)
        for event in bundle.region_events:
            for key in old_by_track.get(event['track_id'],()):
                if (key[1] == event['track_id'] and event['event'] in MUST_EVENTS
                        and event.get('camera_id',key[0])==key[0]):
                    add_event(key, [event['event']])
                    if event['event'] in ('state_expired','disappeared'):
                        snapshot.pop(key, None)
                    elif event['event'] in ('association_invalid','source_invalid') and key not in current_keys:
                        snapshot.pop(key,None)
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
            snapshot[key] = row
    # Binary local risk has the same [0,1] scale as dirty/unknown image evidence.
    # Do not dilute a small, explicitly unverified ROI across an entire window.
    evidence=torch.maximum(evidence,torch.as_tensor(regional_uncertainty,device=device,dtype=evidence.dtype))
    score = (config.drift_weight*drift + config.age_weight*(ages.float()/config.max_reuse_frames).clamp(max=1)
             + config.uncertainty_weight*evidence)
    mandatory |= torch.as_tensor(region_required,device=device)
    # CPU ordering is deterministic for score/age ties, including CUDA inputs.
    scores,age_values,required,drift_values,evidence_values=torch.stack([
        score,ages.float(),mandatory.float(),drift,evidence]).cpu().tolist()
    age_values=list(map(int,age_values));required=np.asarray(required,dtype=bool)
    selected=required.copy()
    for camera in range(len(x0)):
        ids = [i for i in range(camera*per_camera,(camera+1)*per_camera)
               if not required[i] and (window_targets is not None or regional_uncertainty[i]
                                       or scores[i] >= config.score_threshold)]
        ids.sort(key=lambda i: (-scores[i], -age_values[i], i))
        if window_targets is not None:
            ids=ids[:max(0,window_targets[camera]-int(required[camera*per_camera:(camera+1)*per_camera].sum()))]
        elif budget:
            ids = ids[:config.ordinary_budget]
        selected[ids] = True
    return torch.as_tensor(selected,device=device), descriptions, list(snapshot.values()), dict(
        mandatory_windows=int(required.sum()), ordinary_windows=int((selected & ~required).sum()),
        mandatory_window_ids=np.flatnonzero(required).tolist(),
        ordinary_window_ids=np.flatnonzero(selected & ~required).tolist(),
        drift=drift_values, ages=age_values, uncertainty=evidence_values, scores=scores,
        event_regions=event_rows, region_state_count=len(bundle.region_states),
        uncertainty_regions=uncertainty_rows,
        uncertainty_window_ids=np.flatnonzero(regional_uncertainty).tolist(),
        uncertainty_required_window_ids=np.flatnonzero(uncertainty_required).tolist(),
        offline_total_window_targets=window_targets,
        event_affected_windows=sum(len(e['windows']) for e in event_rows), event_missed_windows=0)
