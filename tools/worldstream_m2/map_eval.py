"""Parallel, content-bound caching of UNCHANGED official per-frame functions.

All AP/topology/F-score aggregation still runs in the original evaluator, in its
original frame order. No prediction filter, threshold or matching rule is changed.
"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import importlib
import json
import multiprocessing
import pickle
from unittest.mock import patch
import numpy as np

_geometry_cache={}


def has_geometry_cache():return bool(_geometry_cache)


def geometry_key(name,xs,ys,mask=None,relax=False,categories=()):
    digest=hashlib.sha256(json.dumps([name,bool(relax),len(xs),len(ys)]).encode())
    for value in [*xs,*ys,*categories]+([mask] if mask is not None else []):
        array=np.ascontiguousarray(value)
        digest.update(json.dumps([str(array.dtype),list(array.shape),array.nbytes]).encode())
        digest.update(array.tobytes())
    digest.update(b'no_mask' if mask is None else b'with_mask')
    return digest.hexdigest()


def _frame_compute(payload):
    gt,pred,maximum=payload
    from openlanev2.centerline.evaluation.distance import pairwise,chamfer_distance,frechet_distance,iou_distance
    from openlanev2.centerline.evaluation.f_score import LaneEval
    x=[r['points'] for r in gt['lane_centerline']];y=[r['points'] for r in pred['lane_centerline']]
    chamfer=pairwise(x,y,chamfer_distance,relax=True);mask=chamfer<maximum
    frechet=pairwise(x,y,frechet_distance,mask=mask,relax=True)
    tx=[r['points'] for r in gt['traffic_element']];ty=[r['points'] for r in pred['traffic_element']]
    iou=pairwise(tx,ty,iou_distance)
    px=[np.int8(1) for _ in y];gx=[np.int8(1) for _ in x]
    score=LaneEval().bench(y,px,x,gx)
    return [(geometry_key('chamfer_distance',x,y,relax=True),chamfer),
        (geometry_key('frechet_distance',x,y,mask=mask,relax=True),frechet),
        (geometry_key('iou_distance',tx,ty),iou),
        (geometry_key('LaneEval.bench',y,x,categories=(px,gx)),score)]


def evaluate_map(ground_truth,predictions,workers=0):
    module=importlib.import_module('openlanev2.centerline.evaluation.evaluate')
    if not workers:return module.evaluate(ground_truth,predictions,verbose=False)
    if isinstance(ground_truth,str):
        with open(ground_truth,'rb') as handle:ground_truth=pickle.load(handle)
    # Validate format BEFORE computation, with the original checker.
    module.check_results(predictions)
    if module.f1.dist_th!=1.5 or module.f1.ratio_th!=.75 or not np.array_equal(module.f1.x_samples,np.linspace(-50,50,100,endpoint=False)):
        raise ValueError('Official F-score parameters changed; cache precomputation is not compatible')
    payloads=((value['annotation'],predictions['results'][key]['predictions'],module.THRESHOLDS_FRECHET[-1])
              for key,value in ground_truth.items())
    cache=_geometry_cache if workers<0 else {}
    # Spawn avoids inheriting the model's CUDA context. Children perform CPU
    # evaluation only and inherit taskset's affinity, excluding faulty cores.
    if workers>0:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for count,result in enumerate(pool.map(_frame_compute,payloads,chunksize=8),1):
                for key,value in result:
                    if key in cache and not np.array_equal(np.asarray(cache[key]),np.asarray(value),equal_nan=True):
                        raise ValueError('Nondeterministic result for identical official geometry inputs')
                    cache[key]=value
                if count%250==0:print('OFFICIAL_MAP_PRECOMPUTE',count,'/',len(ground_truth),flush=True)
        _geometry_cache.clear();_geometry_cache.update(cache)
    def pairwise(xs,ys,distance_function,mask=None,relax=False):
        key=geometry_key(distance_function.__name__,xs,ys,mask,relax)
        if key not in cache:raise ValueError('Official geometry cache input mismatch')
        return cache[key].copy()
    def bench(pred_lanes,pred_category,gt_lanes,gt_category):
        key=geometry_key('LaneEval.bench',pred_lanes,gt_lanes,categories=(pred_category,gt_category))
        if key not in cache:raise ValueError('Official F-score cache input mismatch')
        return cache[key]
    with ExitStack() as stack:
        stack.enter_context(patch.object(module,'pairwise',pairwise))
        stack.enter_context(patch.object(module.f1,'bench',bench))
        return module.evaluate(ground_truth,predictions,verbose=False)
