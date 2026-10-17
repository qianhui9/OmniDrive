"""Offline-only patch representation probe; never called by production Auto."""
import numpy as np
import torch
import torch.nn.functional as F


def align_patch_features(previous,current,source_map,stable_pixels,patch_size=16):
    """Probe geometry, not an equivalence certificate.

    Sample a pixel-derived map at token centres and interpolate content features
    BEFORE absolute positions; current position embedding must be added afterwards.
    RoPE-dependent contextual features are deliberately not warped here.
    """
    h,w,c=current.shape
    stable=np.asarray(stable_pixels,dtype=bool)[:h*patch_size,:w*patch_size]
    mask=stable.reshape(h,patch_size,w,patch_size).all(axis=(1,3))
    if source_map is None or not mask.any():
        return current,dict(accepted_patches=0,total_patches=h*w,executed=False,
                            reason='no_complete_observation_supported_patch')
    centre=(patch_size-1)/2
    import cv2
    yy,xx=np.meshgrid(np.arange(h,dtype=np.float32)*patch_size+centre,
                      np.arange(w,dtype=np.float32)*patch_size+centre,indexing='ij')
    coordinates=cv2.remap(np.asarray(source_map,dtype=np.float32),xx,yy,cv2.INTER_LINEAR)
    source=(torch.as_tensor(coordinates,device=current.device,dtype=current.dtype)-centre)/patch_size
    grid=torch.stack((2*(source[...,0]+.5)/w-1,2*(source[...,1]+.5)/h-1),-1)
    warped=F.grid_sample(previous.permute(2,0,1)[None],grid[None],mode='bilinear',padding_mode='zeros',align_corners=False)[0].permute(1,2,0)
    valid=(torch.as_tensor(mask,device=current.device)&torch.isfinite(warped).all(-1)&
           (source[...,0]>=0)&(source[...,0]<=w-1)&(source[...,1]>=0)&(source[...,1]<=h-1))
    difference=(warped-current).abs()[valid]
    candidate=torch.where(valid[...,None],warped,current)
    return candidate,dict(accepted_patches=int(valid.sum()),total_patches=h*w,executed=True,
        max_abs=float(difference.max()) if difference.numel() else None,
        relative_l2=float((warped-current)[valid].norm()/current[valid].norm().clamp_min(1e-12)),
        absolute_position='current positions must be added after alignment',
        rope='early contextual feature warp not certified',transformer_dependency_certified=False)
