"""Dependency closure for the actual EVA non-shifted window/global topology."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .packing import window_mask, partition, unpartition


def patch_updates(invalid_pixels, projection):
    """Any invalid pixel in the full Conv2d receptive field invalidates a token."""
    return F.max_pool2d(invalid_pixels[:, None].float(), projection.kernel_size,
                        projection.stride, projection.padding, projection.dilation)[:, 0].bool()


def exact_pixel_changes(current, previous):
    # NaNs are always invalid. No tolerance, image warp or centre-point test.
    return (current != previous).any(1) | ~torch.isfinite(current).all(1)


@dataclass
class StagePlan:
    block: int
    window_size: int
    updated: torch.Tensor
    tokens: torch.Tensor
    reason: str


class DependencyPlanner:
    def __init__(self, backbone):
        self.backbone = backbone

    def plan(self, updates):
        plans = []
        for i, block in enumerate(self.backbone.blocks):
            size = block.window_size
            if size:
                changed = window_mask(updates, size)
                packed, padded = partition(updates[..., None], size)
                packed = changed[:, None, None, None].expand_as(packed)
                updates = unpartition(packed, size, padded, updates.shape[1:])[..., 0]
                reason = 'window_dependency'
                if block.use_residual_block:
                    # Actual EVA bottleneck has one stride-1 3x3 convolution.
                    updates = F.max_pool2d(updates[:, None].float(), 3, 1, 1)[:, 0].bool()
                    reason = 'cross_window_residual'
            else:
                changed = updates.flatten(1).any(1)
                updates = changed[:, None, None].expand_as(updates)
                reason = 'global_dependency'
            plans.append(StagePlan(i, size, changed, updates, reason))
        return plans

    def cacheable_prefix(self):
        result = []
        for i, block in enumerate(self.backbone.blocks):
            if not block.window_size or block.use_residual_block:
                break
            result.append(i)
        return result
