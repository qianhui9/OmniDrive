"""Ordered token deletion is approximate. Learned carriers have no box index."""
from dataclasses import dataclass
import math
import torch
from .task_conditioner import task_type


@dataclass
class Selection:
    indices: torch.Tensor
    diagnostics: dict


def select_carriers(carriers, layout, question, method='full', ratio=1., seed=0,
                    world_state=None, context=None):
    if carriers.ndim != 3 or carriers.shape[0] != 1 or carriers.shape[1] != layout.total:
        raise ValueError('Carrier shape does not match audited layout')
    total = layout.total
    indices = torch.arange(total, device=carriers.device)
    task = task_type(question)
    reason = 'full_carrier'
    eligible = method != 'full'
    if method == 'task':
        # Branch-level provenance only. No invented object salience/confidence.
        # All object and ego tokens remain; map deletion is an offline hypothesis.
        current = context or {}
        state_current = (world_state is None or (
                         getattr(world_state, 'scene_id', None) == current.get('scene_id') and
                         getattr(world_state, 'last_timestamp', None) == current.get('timestamp')))
        if task != 'object_relation':
            eligible = False
            reason = 'global_map_or_unknown_requires_full_context'
        elif not state_current:
            eligible = False
            reason = 'stale_or_different_scene_world_state'
        else:
            keep_map = max(1, math.ceil(layout.map_tokens * ratio)) if layout.map_tokens else 0
            indices = indices[:layout.object_tokens + layout.ego_tokens + keep_map]
            reason = 'branch_provenance_only_no_anchor_to_query_mapping'
    elif method in ('topk', 'random'):
        # Untrained query norms are not proposal confidence. This is a control.
        ego = torch.arange(layout.object_tokens, layout.object_tokens + layout.ego_tokens, device=carriers.device)
        candidates = indices[(indices < layout.object_tokens) | (indices >= layout.object_tokens + layout.ego_tokens)]
        count = max(0, math.ceil(total * ratio) - len(ego))
        if method == 'topk':
            scores = carriers[0, candidates].float().norm(dim=-1)
            chosen = candidates[scores.topk(min(count, len(candidates))).indices]
            reason = 'norm_topk_control_not_detection_confidence'
        else:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            chosen = candidates[torch.randperm(len(candidates), generator=generator)[:count].to(carriers.device)]
            reason = 'seeded_random_control'
        indices = torch.cat((ego, chosen)).sort().values
    elif method != 'full':
        raise ValueError('Unknown selection method')
    return Selection(indices, dict(method=method, task=task, original_tokens=total,
        selected_tokens=len(indices), indices=indices.tolist(), eligible=eligible,
        approximate=len(indices) != total, reason=reason,
        retained_object_tokens=int((indices < layout.object_tokens).sum()),
        retained_ego_tokens=int(((indices >= layout.object_tokens) & (indices < layout.object_tokens + layout.ego_tokens)).sum()),
        retained_map_tokens=int((indices >= layout.object_tokens + layout.ego_tokens).sum()),
        source=dict(context or {})))
