"""Measure feasibility and A/B/C numerical attribution before adding reuse boundaries."""
import argparse
import json
from time import perf_counter
import torch
from tools.worldstream_m2.benchmark import kernel_profile
from worldstream.m2.profiler import measure
from worldstream.m2.window_executor import WindowExecutor
from worldstream.m2.packing import partition,unpartition
from .common import ROOT,backbone,write,error_location,stats,reference


class PackedDense:
    """Diagnostic B path: all windows updated through the exact deployed executor."""
    def __init__(self,model):
        self.model=model;self.stages={};self.ex=WindowExecutor()
    def __call__(self,images,restore_mlp=False,skip=0):
        from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
        x=self.model.patch_embed(images)
        if self.model.pos_embed is not None:x=x+get_abs_pos(self.model.pos_embed,self.model.pretrain_use_cls_token,x.shape[1:3])
        for i,b in enumerate(self.model.blocks):
            if i<skip:continue
            if i<2:
                if restore_mlp:
                    # Same original order: norm -> padded attention -> UNPAD -> residual/MLP.
                    normalized,pad=partition(b.norm1(x),b.window_size)
                    attention=unpartition(b.attn(normalized),b.window_size,pad,x.shape[1:3])
                    x=x+b.drop_path(attention)
                    x=x+b.drop_path(b.mlp(b.norm2(x)))
                else:
                    windows,_=partition(x,b.window_size)
                    ids=torch.arange(len(windows),device=x.device)
                    x,_=self.ex.execute(b,x,windows,ids)
            else:x=b(x)
            self.stages[i]=x.detach().clone()
        return [x.permute(0,3,1,2)]
