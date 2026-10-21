"""Balanced multi-scene actual OmniDrive plus nonzero controlled sparse quality checks."""
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import copy
import json
import random
import re
import resource
import numpy as np
import torch
from .common import write
from .refinement_common import ROOT,reference,select_protocol,arrays,numeric_diff,detection_diff,error_location,window_statistics,memory_alignment,map_alignment
from tools.worldstream_m1.model_smoke import digest
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.profiler import measure
from worldstream.m2.cost_model import cost_key
from worldstream.m2.packing import partition,unpartition
from worldstream.m2.window_executor import WindowExecutor


def memory(model):
    return arrays({name:{k:v for k,v in vars(head).items() if k.startswith('memory_') or k=='sample_time'}
                   for name,head in [('det',model.pts_bbox_head),('map',model.map_head)]})


def planning_metric(text,info):
    answers=' '.join(str(a) for qa in (text or []) for a in qa.get('A',[]))
    pairs=re.findall(r'\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)',answers)
    if len(pairs)<6:return dict(parsed=False)
    xy=np.asarray(pairs[:6],float);gt=np.asarray(info['gt_planning'])[0,:,:2]
    mask=np.asarray(info['gt_planning_mask'])[0].all(axis=-1).astype(bool)
    distance=np.linalg.norm(xy-gt,axis=1)
    return dict(parsed=True,trajectory=xy.tolist(),ade_m=float(distance[mask].mean()) if mask.any() else None,
        fde_m=float(distance[-1]) if mask[-1] else None)


@torch.no_grad()
def supplement_costs(model):
    # Block 0 full output and packed reconstruction can have different strides.
    # Profile the second actual layout independently instead of sharing/interpolating costs.
    path=ROOT/'break_even_results.json';table=json.loads(path.read_text())
    block=model.img_backbone.blocks[1];x=torch.randn(6,40,40,1024,device='cuda')
    reference()
    from m2_refinement_reference.window_executor import WindowExecutor as OriginalExecutor
    original=OriginalExecutor()
    ex=WindowExecutor();cache=partition(block(x),16)[0]
    prior_path=ROOT/'layout_cost_supplement.json'
    rows=json.loads(prior_path.read_text()) if prior_path.exists() else []
    for count in (0,1,6,36,54):
        key=cost_key(block,x,1)
        if any(r['key']==key and r['updated']==count for r in table['records']):continue
        ids=torch.randperm(54,device='cuda')[:count]
        wins,pad=partition(x,16);wins[ids]+=.03;current=unpartition(wins,16,pad,(40,40))
        full=measure(lambda:block(current),warmup=5,repeats=20)
        packed=measure(lambda:ex.execute(block,current,cache,ids),warmup=5,repeats=20)
        prior=measure(lambda:original.execute(block,current,cache,ids),warmup=5,repeats=20)
        spatial=packed['gpu_ms']<prior['gpu_ms'] and packed['wall_ms']<prior['wall_ms']
        chosen=packed if spatial else prior
        err=error_location(ex.execute(block,current,cache,ids)[0],block(current))
        qualified=table['whole_backbone_counts'].get(str(count),False) and err['max_abs']<=.02 and err['relative_l2']<=.002
        table['records'].append(dict(key=key,block=1,updated=count,incremental_gpu_ms=chosen['gpu_ms'],
            packing_backend='spatial' if spatial else 'original',
            incremental_wall_ms=chosen['wall_ms'],full_gpu_ms=full['gpu_ms'],full_wall_ms=full['wall_ms'],
            backbone_quality_certified=qualified,
            quality_certified=qualified and count==0 and bool(table.get('formal_quality_policy'))))
        rows.append(dict(block=1,updated=count,full=full,packed=packed,original=prior,error=err))
    write(ROOT/'layout_cost_supplement.json',rows);write(path,table)
