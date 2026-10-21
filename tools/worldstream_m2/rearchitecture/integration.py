"""Full original OmniDrive, current temporal heads and real greedy language generation."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import resource
from time import perf_counter
import numpy as np
import torch
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.camera_executor import camera_cost_key
from worldstream.m2.profiler import measure
from tools.worldstream_m2.refine_integration import planning_metric
from tools.worldstream_m2.refinement_common import window_statistics
from .common import ROOT,write,arrays,memory,compare,quality_pass,reference,stats
from .phase_ab import PackedDense


class PipelineTimer:
    def __init__(self):self.records=[]
    def wrap(self,module,name,dynamic=None):
        original=module.forward
        def forward(*args,**kw):
            label=dynamic(args,kw) if dynamic else name
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            tick=perf_counter();a.record()
            with torch.autograd.profiler.record_function(label):result=original(*args,**kw)
            b.record();self.records.append((label,a,b,(perf_counter()-tick)*1000))
            return result
        module.forward=forward
    def collect(self):
        result={}
        for name,a,b,cpu in self.records:
            r=result.setdefault(name,dict(gpu_ms=0.,cpu_enqueue_wall_ms=0.,calls=0))
            r['gpu_ms']+=a.elapsed_time(b);r['cpu_enqueue_wall_ms']+=cpu;r['calls']+=1
        self.records.clear();return result


def load():
    from mmcv import Config
    from mmcv.runner import load_checkpoint,wrap_fp16_model
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_coverage_observe.py')
    cfg.model.m1_enable=False;cfg.model.train_cfg=None;cfg.model.pretrained=None;cfg.data.test.test_mode=True
    dataset=build_dataset(cfg.data.test)
    model=build_model(cfg.model,test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):wrap_fp16_model(model)
    load_checkpoint(model,'ckpts/iter_10548.pth',map_location='cpu');model.cuda().eval()
    generate=model.lm_head.generate
    def greedy(*a,**kw):
        kw.update(do_sample=False);kw.pop('temperature',None);kw.pop('top_p',None)
        return generate(*a,**kw)
    model.lm_head.generate=greedy
    return model,dataset






def error_location_cpu(a,b):
    from .common import error_location
    if a.shape!=b.shape:
        return dict(computed_shape=list(a.shape),full_shape=list(b.shape),
            comparison='camera_subbatch_intermediate_not_full_batch; final assembled features compared separately')
    return error_location(torch.from_numpy(a),torch.from_numpy(b))
