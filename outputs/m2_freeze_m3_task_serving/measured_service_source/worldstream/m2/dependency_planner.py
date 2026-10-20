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

    def plan(self, updates, stop_after=None):
        plans = []
        last_size = None
        for i, block in enumerate(self.backbone.blocks[:stop_after]):
            size = block.window_size
            if size:
                if size == last_size and plans[-1].reason == 'window_dependency':
                    plans.append(StagePlan(i, size, plans[-1].updated, updates, 'window_dependency'))
                    continue
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
            last_size = size
        return plans

    def cacheable_prefix(self):
        result = []
        for i, block in enumerate(self.backbone.blocks):
            if not block.window_size or block.use_residual_block:
                break
            result.append(i)
        return result

    def input_footprints(self, height, width):
        """Input dependency of each prefix window, including earlier windows/Conv2d.

        Virtual window/Conv padding has no observed input pixels. Global and
        cross-window residual blocks terminate the executable cacheable prefix.
        """
        p=self.backbone.patch_embed.proj
        kh,kw=p.kernel_size;sh,sw=p.stride;ph,pw=p.padding;dh,dw=p.dilation
        th=(height+2*ph-dh*(kh-1)-1)//sh+1
        tw=(width+2*pw-dw*(kw-1)-1)//sw+1
        result={}
        for i in self.cacheable_prefix():
            ws=self.backbone.blocks[i].window_size;units=[]
            for y in range(0,th,ws):
                for x in range(0,tw,ws):
                    y0,x0,y1,x1=y,x,min(y+ws,th),min(x+ws,tw)
                    for block in reversed(self.backbone.blocks[:i+1]):
                        size=block.window_size
                        y0=y0//size*size;x0=x0//size*size
                        y1=min(th,(y1+size-1)//size*size);x1=min(tw,(x1+size-1)//size*size)
                    units.append(dict(token_region=[x0,y0,x1,y1],
                        input_region=[max(0,x0*sw-pw),max(0,y0*sh-ph),
                            min(width,(x1-1)*sw-pw+dw*(kw-1)+1),
                            min(height,(y1-1)*sh-ph+dh*(kh-1)+1)],
                        kernel=[kh,kw],stride=[sh,sw],padding=[ph,pw],dilation=[dh,dw]))
            result[i]=units
        return result
