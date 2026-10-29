"""One nonnegative residual-invalidation score; no task labels or online models."""
import math
import torch
from .state_cache import NonFiniteCacheError


def world_signal(bundle, refresh, total_windows):
    """Aggregate existing projected-track events, never TrackID-to-token identity."""
    events=refresh.get('event_regions',())
    tracks={e['track_id'] for e in events}
    count=len({r['track_id'] for r in bundle.region_states})
    mandatory=refresh.get('mandatory_windows',total_windows)
    uncertainty=len(refresh.get('uncertainty_window_ids',()))
    values=(mandatory/max(1,total_windows),min(1.,len(tracks)/max(1,count)),uncertainty/max(1,total_windows))
    return dict(score=.5*values[0]+.25*values[1]+.25*values[2],
        mandatory_ratio=values[0],event_track_ratio=values[1],uncertainty_ratio=values[2],
        event_tracks=len(tracks),projected_tracks=count,
        source='existing_m1_wsgrr_projected_region_events_not_current_detections')


def feature_statistics(current, previous, tail_fraction=.10, *, include_local=True):
    """Shared FP32 delta; one compact D2H read for all camera statistics.

    Token changes use the camera anchor RMS as denominator, not individual
    near-zero tokens. Camera maxima protect small critical views. The high
    fraction is a statistic (change > 1 anchor RMS), never a separate admission.
    """
    old=previous.float();delta=current.float()-old
    power=delta.square()
    denom=old.square().flatten(1).mean(1).clamp_min(1e-12)
    global_drift=(power.flatten(1).mean(1)/denom).sqrt()
    if not include_local:
        # The deployed RMS/Risk0 configurations have no local contribution.
        # Keep the identical global reduction; do not build token scores, topk,
        # hot masks, or a stacked multi-statistic transfer for unused features.
        values=global_drift.detach().cpu().tolist()
        if any(not math.isfinite(v) for v in values):
            raise NonFiniteCacheError('nonfinite_global_risk_statistics')
        return dict(global_drift=max(values),per_camera_global=values)
    token=(power.mean(-1)/denom[:,None,None]).sqrt().flatten(1)
    k=max(1,math.ceil(token.shape[1]*tail_fraction))
    tail=token.topk(k,dim=1,sorted=False).values.mean(1)
    hot=(token>1.).float().mean(1)
    rows=torch.stack((global_drift,tail,hot)).detach().cpu().tolist()
    if any(not math.isfinite(v) for row in rows for v in row):
        raise NonFiniteCacheError('nonfinite_global_risk_statistics')
    return dict(global_drift=max(rows[0]),local_tail=max(rows[1]),hot_fraction=max(rows[2]),
        per_camera_global=rows[0],per_camera_tail=rows[1],per_camera_hot_fraction=rows[2])


def risk_score(stats, world_change, age, config):
    local=stats['local_tail']*(1.+stats['hot_fraction']) if config.needs_local_risk else 0.
    features=(stats['global_drift'],local,
              world_change,age/config.global_anchor_interval)
    normalized=tuple(v/s for v,s in zip(features,config.global_risk_scales))
    risk=sum(w*v for w,v in zip(config.global_risk_weights,normalized))
    if not math.isfinite(risk) or not 0<=world_change<=1:raise ValueError('invalid_global_risk')
    return risk,features,normalized
