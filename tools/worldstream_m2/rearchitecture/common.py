"""Frozen references and experiment inputs; never writes earlier artifacts."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
from tools.worldstream_m2.common import write, backbone
from tools.worldstream_m2.refinement_common import arrays, numeric_diff, detection_diff, error_location, memory_alignment, map_alignment

ROOT = Path('outputs/module2_rearchitecture')
PYTHON = '/home/qh/miniconda3/envs/OmniDrive/bin/python'


def reference():
    name = 'm2_rearchitecture_reference'
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT/'baseline_source/worldstream/m2/__init__.py'
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stats(values):
    return dict(p50=float(np.percentile(values,50)), p95=float(np.percentile(values,95)), samples=values)


def signature(model):
    return tuple((n,t.data_ptr(),t._version,str(t.dtype),str(t.device),tuple(t.shape))
                 for n,t in list(model.named_parameters())+list(model.named_buffers()))


def memory(model):
    return arrays({name:{k:v for k,v in vars(head).items() if k.startswith('memory_') or k=='sample_time'}
                   for name,head in [('det',model.pts_bbox_head),('map',model.map_head)]})


def compare(actual, baseline):
    return dict(feature=error_location(torch.from_numpy(actual['feature']),torch.from_numpy(baseline['feature'])),
        detection=detection_diff(actual['output'],baseline['output']),
        map=numeric_diff(actual['output'][0]['lane_results'],baseline['output'][0]['lane_results']),
        map_matched=map_alignment(actual['output'][0]['lane_results'],baseline['output'][0]['lane_results']),
        output=numeric_diff(actual['output'],baseline['output']),
        memory=numeric_diff(actual['memory'],baseline['memory']),
        memory_matched=memory_alignment(actual['memory'],baseline['memory']),
        text_equal=actual['text']==baseline['text'],
        planning_equal=actual['planning'].get('trajectory')==baseline['planning'].get('trajectory'),
        planning=actual['planning'],baseline_planning=baseline['planning'])


def quality_pass(q):
    # Same frozen targets as the preceding round, with missing evidence failing closed.
    coordinates=[x['max_abs'] for x in q['map']['leaves'] if x['path']=='root[0][0]']
    return bool(q['detection'].get('unmatched',1)==0 and
        q['detection'].get('max_center_difference_m',float('inf'))<=.01 and
        not q['map']['shape_mismatches'] and max(coordinates,default=float('inf'))<=.01 and
        q['text_equal'] and q['planning_equal'] and q['planning'].get('parsed',False))
