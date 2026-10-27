"""One global state update: full anchor suffix or approximate temporal residual.

The suffix has global attention dependencies. Only the prefix uses independent
windows; no suffix tokens or attention KV are treated as independent units.
"""
from dataclasses import dataclass, replace
from time import perf_counter
import torch
from .anchor_risk import feature_statistics, risk_score


@dataclass(frozen=True)
class GlobalAnchor:
    scene_id: str
    frame_id: str
    timestamp: float
    state_epoch: int
    signature: tuple
    camera_ids: tuple
    prefix_boundary: int
    shape: tuple
    stride: tuple
    residual: torch.Tensor
    h2: torch.Tensor
    age: int
    generation: int
    publication_version: int
    world_change: float = 0.

    def tensors(self):
        return (self.residual, self.h2)

    def metadata(self):
        return tuple(getattr(self, key) for key in self.__dataclass_fields__
                     if key not in ('residual', 'h2', 'signature')), self.signature


def update_global(x, blocks, prefix, cache, context, signature, config, observer=None):
    """Decide BEFORE suffix execution using one fixed nonnegative risk sum.

    interval=2 means anchors on frames 0,2,4 (one reuse between anchors).
    Global drift, local change tail and hot-token fraction share a float32
    delta and one compact host read. M1 events accumulate since the anchor.
    Weights/scales/threshold are fixed offline parameters. Necessary initial,
    maximum-age and explicit M1 whole-frame updates use this same suffix.
    """
    start = perf_counter()
    anchor = cache.global_anchor
    age = anchor.age + 1 if anchor is not None else 0
    drift = None
    stats=None;risk=None;features=None;normalized=None
    world=min(1.,(anchor.world_change if anchor is not None else 0.)+context.get('_global_world_signal',{}).get('score',0.))
    if anchor is None:
        reason = 'initial_anchor'
    else:
        if (anchor.scene_id != context['scene_id'] or anchor.state_epoch != context['state_epoch']
                or anchor.signature != signature or anchor.camera_ids != tuple(context['camera_ids'])
                or anchor.prefix_boundary != len(prefix) or anchor.shape != tuple(x.shape)
                or anchor.stride != tuple(x.stride()) or anchor.h2.dtype != x.dtype
                or anchor.h2.device != x.device or anchor.residual.shape != x.shape
                or anchor.residual.dtype != x.dtype or anchor.residual.device != x.device
                or anchor.publication_version != cache.version):
            raise ValueError('global_anchor_source_mismatch')
        stats=feature_statistics(x,anchor.h2,config.global_tail_fraction,
                                 include_local=config.needs_local_risk)
        drift=stats['global_drift']
        risk,features,normalized=risk_score(stats,world,age,config)
        reason = ('m1_full_refresh' if context.get('_global_full_refresh') else
                  'anchor_expired' if age >= config.global_anchor_interval else
                  'feature_drift' if risk >= config.risk_threshold else None)
    decision_ms = (perf_counter()-start)*1000
    tick = perf_counter()
    suffix_count = len(blocks)-len(prefix)
    stages = []
    if reason is not None:
        # Own H2 before invoking arbitrary suffix modules. Only one anchor is
        # retained; this private candidate is published with the window state.
        h2 = x.detach().clone(memory_format=torch.preserve_format)
        for i in range(len(prefix), len(blocks)):
            x = blocks[i](x)
            stages.append(dict(block=i, mode='full', reason='global_anchor_update'))
            if observer is not None: observer(i, x)
        if (x.shape != h2.shape or x.dtype != h2.dtype or x.device != h2.device
                or x.stride() != h2.stride()):
            raise ValueError('global_suffix_contract_incompatible')
        residual = x-h2
        candidate = GlobalAnchor(context['scene_id'], context['frame_id'], context['timestamp'],
            context['state_epoch'], signature, tuple(context['camera_ids']), len(prefix),
            tuple(h2.shape), tuple(h2.stride()), residual, h2, 0,
            1 if anchor is None else anchor.generation+1, cache.version+1)
    else:
        # Current visual input always participates. Preserve native layout even
        # if the backend's elementwise kernel chooses another dense layout.
        result = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
        torch.add(x, anchor.residual, out=result)
        x = result
        candidate = replace(anchor, age=age, publication_version=cache.version+1,world_change=world)
        stages = [dict(block=i, mode='approx_global_residual_reuse',
                       reason='temporal_anchor_residual') for i in range(len(prefix),len(blocks))]
    update_ms = (perf_counter()-tick)*1000
    diagnostics = dict(global_update_mode='anchor_update' if reason else 'approx_global_residual_reuse',
        approx_global_residual_reuse=reason is None, global_anchor_refresh_reason=reason,
        global_anchor_age=age, global_anchor_age_after=candidate.age,
        global_anchor_frame_id=candidate.frame_id, global_anchor_generation=candidate.generation,
        global_anchor_scene_id=candidate.scene_id, global_anchor_timestamp=candidate.timestamp,
        global_anchor_state_epoch=candidate.state_epoch, global_anchor_publication_version=candidate.publication_version,
        global_prefix_boundary=candidate.prefix_boundary, global_feature_shape=candidate.shape,
        global_feature_stride=tuple(x.stride()), global_feature_dtype=str(x.dtype),
        global_anchor_bytes=sum(t.numel()*t.element_size() for t in candidate.tensors()),
        global_feature_drift=drift, global_drift_definition='max_camera_relative_h2_rms',
        global_risk_score=risk,global_risk_threshold=config.risk_threshold,
        global_risk_features=features,global_risk_normalized=normalized,
        global_risk_weights=config.global_risk_weights,global_risk_scales=config.global_risk_scales,
        global_local_statistics=stats,global_world_change=world,
        global_world_signal=context.get('_global_world_signal'),
        global_anchor_interval=config.global_anchor_interval, global_drift_threshold=config.global_drift_threshold,
        global_suffix_executed_blocks=suffix_count if reason else 0,
        global_suffix_skipped_blocks=0 if reason else suffix_count,
        global_suffix_reuse_frames=int(reason is None), global_suffix_executions=int(reason is not None),
        global_decision_ms=decision_ms, global_update_host_ms=update_ms,
        global_anchor_update_host_ms=update_ms if reason else 0.,
        global_residual_update_host_ms=0. if reason else update_ms,
        global_state_approximate=reason is None, global_quality_certified=False)
    return x, candidate, stages, diagnostics
