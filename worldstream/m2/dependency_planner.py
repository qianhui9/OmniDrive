"""Dependency closure for the actual EVA non-shifted window/global topology."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .packing import window_mask, partition, unpartition


def patch_updates(invalid_pixels, projection):
    """Any invalid pixel in the full Conv2d receptive field invalidates a token."""
    if invalid_pixels.ndim != 3 or invalid_pixels.dtype != torch.bool:
        raise ValueError('Expected boolean NHW preprocessed-image mask')
    # Explicit padding handles valid Conv2d configurations whose dilated
    # receptive field exceeds max_pool2d's stricter padding limit.
    ph,pw=projection.padding
    pixels=F.pad(invalid_pixels[:,None].float(),(pw,pw,ph,ph))
    return F.max_pool2d(pixels,projection.kernel_size,projection.stride,
                        0,projection.dilation)[:,0].bool()


def exact_pixel_changes(current, previous):
    # NaNs are always invalid. No tolerance, image warp or centre-point test.
    return (current != previous).any(1) | ~torch.isfinite(current).all(1)


def block0_input(backbone, images):
    """The actual native BHWC boundary, computed exactly once per request."""
    from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
    x = backbone.patch_embed(images)
    if backbone.pos_embed is not None:
        x = x + get_abs_pos(backbone.pos_embed, backbone.pretrain_use_cls_token, x.shape[1:3])
    return x


def exact_token_changes(current, previous):
    """Exact equality at Block 0; non-finite tokens always need computation."""
    if (current.ndim != 4 or current.shape != previous.shape or
            current.device != previous.device or current.dtype != previous.dtype):
        raise ValueError('Expected matching BHWC Block 0 inputs')
    return (current != previous).any(-1) | ~torch.isfinite(current).all(-1)


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

    @staticmethod
    def windows_to_tokens(selected, shape, size):
        n,h,w=shape
        hp,wp=(h+size-1)//size*size,(w+size-1)//size*size
        if selected.dtype != torch.bool or selected.numel()!=n*(hp//size)*(wp//size):
            raise ValueError('Selected windows do not match the actual token grid')
        packed=selected[:,None,None,None].expand(-1,size,size,1)
        return unpartition(packed,size,(hp,wp),(h,w))[...,0]

    def plan_windows(self, selected, shape, stop_after=None):
        # Always pass through the same real topology closure, including different
        # window sizes, residual boundaries and global barriers.
        return self.plan(self.windows_to_tokens(selected,shape,self.backbone.blocks[0].window_size),stop_after)

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
        # Window flags alone do not prove independence: a replacement norm or
        # attention module can mix spatial windows while retaining the weights.
        from projects.mmdet3d_plugin.models.backbones.eva_vit import (
            EVAViT,PatchEmbed,Block,Attention,SwiGLU,DropPath,VisionRotaryEmbeddingFast)
        from projects.mmdet3d_plugin.models.utils.attention import FlashAttention
        self.rejection_reason=None
        def reject():
            self.rejection_reason='unsupported_model_topology'
            return []
        def plain(module, allowed):
            for child in module.modules():
                if (type(child) not in allowed or
                        getattr(child.forward,'__func__',None) is not type(child).forward):
                    return False
                if isinstance(child,torch.nn.LayerNorm) and len(child.normalized_shape)!=1:
                    return False
            return True
        if (type(self.backbone) is not EVAViT or
                getattr(self.backbone.forward,'__func__',None) is not EVAViT.forward or
                not plain(self.backbone.patch_embed,{PatchEmbed,torch.nn.Conv2d}) or
                self.backbone.patch_embed.proj.padding_mode!='zeros'):
            return reject()
        for block in self.backbone.blocks:
            if (type(block) is not Block or type(block.attn) is not Attention or
                    type(block.mlp) is not SwiGLU or
                    type(block.norm1) is not torch.nn.LayerNorm or
                    type(block.norm2) is not torch.nn.LayerNorm):
                return reject()
        allowed={Block,Attention,SwiGLU,DropPath,VisionRotaryEmbeddingFast,FlashAttention,
            torch.nn.Linear,torch.nn.LayerNorm,torch.nn.Identity,torch.nn.SiLU,torch.nn.Dropout}
        result = []
        for i, block in enumerate(self.backbone.blocks):
            if not block.window_size or block.use_residual_block:
                break
            if (not plain(block,allowed) or
                    getattr(block._forward,'__func__',None) is not Block._forward):
                return reject()
            result.append(i)
        return result

    def input_footprints(self, height, width, *, verified_prefix=None):
        """Input dependency of each prefix window, including earlier windows/Conv2d.

        Virtual window/Conv padding has no observed input pixels. Global and
        cross-window residual blocks terminate the executable cacheable prefix.
        """
        p=self.backbone.patch_embed.proj
        kh,kw=p.kernel_size;sh,sw=p.stride;ph,pw=p.padding;dh,dw=p.dilation
        th=(height+2*ph-dh*(kh-1)-1)//sh+1
        tw=(width+2*pw-dw*(kw-1)-1)//sw+1
        result={}
        for i in self.cacheable_prefix() if verified_prefix is None else verified_prefix:
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
