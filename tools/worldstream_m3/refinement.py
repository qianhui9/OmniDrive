"""Paired real continuous OmniDrive refinement protocol; no answer-file reuse."""
import argparse
import copy
from contextlib import nullcontext
from dataclasses import replace
import gc
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import subprocess
from functools import wraps
import sys
from time import perf_counter
import numpy as np
import torch
from worldstream.m3 import M3Runtime,expand_input
from worldstream.m3.visual_state import base_model
from worldstream.m3.numerics import precision_probe
from .experiment import Timer,write,ground_truth,chosen_questions,typed_digest


def monitor_gpu(function):
    @wraps(function)
    def wrapped(args):
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        log=path.parent/'logs'/f'{path.name}_gpu.csv';log.parent.mkdir(exist_ok=True)
        with log.open('w') as handle:
            process=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                '--format=csv,noheader,nounits','--loop-ms=1000'],stdout=handle,stderr=subprocess.STDOUT)
            try:return function(args)
            finally:
                process.terminate();process.wait(timeout=10)
    return wrapped


class RecordedRuntime:
    """No instance-bound closure or hidden retention of a previous runtime."""
    def __init__(self,runtime,recorded,sampling=False,legacy=False,policy='native'):
        self.inner=runtime;self.recorded=recorded;self.config=runtime.config;self.sampling=sampling;self.legacy=legacy
        self.policy=policy
    def reset(self):self.inner.reset()
    def generate(self,*a,**kw):
        if not self.sampling:
            kw.update(do_sample=False);kw.pop('temperature',None);kw.pop('top_p',None)
        if self.legacy:
            kw.pop('verified_state',None);kw.pop('tokenizer',None)
        lm=kw.get('lm_head',a[0] if a else None)
        if self.policy=='fp32_qkv':
            from worldstream.m3.projection_precision import fp32_qkv_prefill
            probe=fp32_qkv_prefill(base_model(lm))
        else:probe=precision_probe(base_model(lm),self.policy)
        with probe:
            answers,diag=self.inner.generate(*a,**kw)
        self.recorded.extend(x.detach() for x in answers)
        return answers,diag


def legacy_runtime():
    path=Path('outputs/module3/clean_multi_20261016_220113/source/worldstream/m3')
    spec=importlib.util.spec_from_file_location('m3_frozen_legacy',path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.M3Runtime




def snapshot_source(dest):
    manifest={}
    paths=list(Path('worldstream/m3').glob('*.py'))+list(Path('tools/worldstream_m3').glob('*.py'))
    paths+=[Path('projects/mmdet3d_plugin/models/detectors/petr3d.py')]
    for p in paths:
        target=dest/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    write(dest/'source_manifest.json',manifest)


def freeze_sample_questions(sample,frozen):
    """Replay only native question IDs/text, never historical image/carrier data.
    Run on prepared CPU samples before scatter and all service timing.
    """
    from mmcv.parallel import DataContainer
    assert len(sample['input_ids'])==len(sample['img_metas'])==1
    sample['input_ids']=[DataContainer([x.detach().clone() for x in frozen['ids']],stack=False)]
    sample['img_metas'][0].data['vlm_labels']=DataContainer(list(frozen['questions']),stack=False)
    return sample
