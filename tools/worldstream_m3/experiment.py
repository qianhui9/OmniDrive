"""Real weights, real nuScenes, unchanged question/GT, unique language paths.

Run in OmniDrive: python -m tools.worldstream_m3.experiment --output outputs/module3/run01
No output-file skip is used, even for the original baseline.
"""
import argparse
import copy
import gc
from dataclasses import replace
from functools import wraps
import hashlib
import inspect
import json
import random
import subprocess
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch

from worldstream.m3 import M3Runtime, CarrierLayout, expand_input
from worldstream.m3.visual_state import base_model
from worldstream.m3.prefix_matcher import exact_lcp
from worldstream.m3.task_conditioner import task_type


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def stats(xs):
    return dict(p50=float(np.percentile(xs, 50)), p95=float(np.percentile(xs, 95)), count=len(xs)) if xs else None


class Timer:
    def __init__(self):
        self.events = []

    def wrap(self, module, name, dynamic=False):
        original = module.forward
        @wraps(original)
        def forward(*a, **kw):
            label = name
            if dynamic:
                x = kw.get('inputs_embeds')
                if x is None: x = kw.get('input_ids')
                if x is None and a: x = a[0]
                label = 'prefill' if kw.get('inputs_embeds') is not None or kw.get('past_key_values') is None else 'decode'
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); tick = perf_counter()
            with torch.autograd.profiler.record_function('m3_probe_' + label):
                result = original(*a, **kw)
            end.record()
            self.events.append((label, start, end, (perf_counter()-tick)*1000))
            return result
        module.forward = forward

    def collect(self):
        output = {}
        for label, a, b, wall in self.events:
            r = output.setdefault(label, dict(gpu_ms=0., cpu_enqueue_ms=0., calls=0))
            r['gpu_ms'] += a.elapsed_time(b); r['cpu_enqueue_ms'] += wall; r['calls'] += 1
        self.events.clear()
        return output

    def wrap_method(self, obj, attribute, name):
        original = getattr(obj, attribute)
        @wraps(original)
        def method(*a, **kw):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); tick = perf_counter()
            with torch.autograd.profiler.record_function('m3_probe_' + name):
                output = original(*a, **kw)
            end.record()
            self.events.append((name, start, end, (perf_counter()-tick)*1000))
            return output
        setattr(obj, attribute, method)


class ExperimentRuntime(M3Runtime):
    """A class override avoids an instance -> closure -> bound-method GPU cycle."""
    def __init__(self, config, sampling, recorded):
        super().__init__(config)
        self.sampling = sampling
        self.recorded = recorded

    def generate(self,*a,**kw):
        if not self.sampling:
            kw.update(do_sample=False);kw.pop('temperature',None);kw.pop('top_p',None)
        answers,diagnostics=super().generate(*a,**kw)
        self.recorded.extend(x.detach().cpu().tolist() for x in answers)
        return answers,diagnostics


def ground_truth(token):
    result = {}
    for folder in ('conv/val', 'vqa/val'):
        path = Path('data/nuscenes') / folder / (token+'.json')
        if path.exists():
            for pair in json.loads(path.read_text()):
                result[pair['question']] = pair['answer']
    return result


def typed_digest(value):
    if hasattr(value,'tensor'):value=value.tensor
    if isinstance(value,torch.Tensor):
        tensor=value.detach().cpu().contiguous()
        return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value,np.ndarray):
        return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,dict):return {k:typed_digest(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [typed_digest(v) for v in value]
    return value


def chosen_questions(questions, limit):
    # Deterministic task coverage from question text only, never from ground truth.
    selected = [0]
    for task in ('object_relation', 'map', 'global_safety', 'unknown'):
        found = next((i for i, q in enumerate(questions) if i not in selected and task_type(q) == task), None)
        if found is not None: selected.append(found)
    selected += [i for i in range(len(questions)) if i not in selected]
    return sorted(selected[:limit]) if limit else list(range(len(questions)))
