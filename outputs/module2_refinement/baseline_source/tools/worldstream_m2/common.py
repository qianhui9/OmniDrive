"""Experiment-only model loading; no alternate implementation of EVA."""
import json
from pathlib import Path
import torch


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def backbone(checkpoint='ckpts/iter_10548.pth', device='cuda', flash=True):
    from mmcv import Config
    from projects.mmdet3d_plugin.models.backbones.eva_vit import EVAViT
    cfg = Config.fromfile('projects/configs/OmniDrive/mask_eva_lane_det_vlm.py')
    kwargs = dict(cfg.model.img_backbone); kwargs.pop('type')
    kwargs.update(with_cp=False, flash_attn=flash)
    model = EVAViT(**kwargs).eval()
    weights = torch.load(checkpoint, map_location='cpu')
    weights = weights.get('state_dict', weights)
    weights = {k[len('img_backbone.'):]: v for k, v in weights.items() if k.startswith('img_backbone.')}
    model.load_state_dict(weights, strict=True)
    del weights
    # Deployed config keeps FP32 weights; its FlashAttention wrapper casts Q/K/V
    # to FP16 and returns FP32. Globally calling half() breaks the original block.
    return model.to(device=device)
