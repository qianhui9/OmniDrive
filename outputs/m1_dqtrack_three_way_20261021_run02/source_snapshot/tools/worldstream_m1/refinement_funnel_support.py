"""Offline paired reference support at rejected old-plane units.

This measures the unmodified flow at candidate sparse reference pixels. It does
not call a rejected pixel an accepted error, nor claim a rejection rule alone
causally prevents that error. New-method ablations provide paired decisions.
"""
from collections import Counter,defaultdict
from dataclasses import replace
from pathlib import Path
import json
import cv2
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager
from .dataset import frame_from_info
from .coverage_oracle import world_lidar,prepare_pair
from .evidence_experiments import load_inputs
from .io_utils import write_json

ROOT=Path('outputs/module1_quality_coverage_refinement')


def scene(spec):
    infos,outputs,oracle=_INPUTS
    cfg=replace(M1Config.load('configs/worldstream_m1_independent.yaml'),dense_region_policy='surface_local_plane',
        dense_guard_radius_px=2,surface_diagnostics=True)
    m=WorldStateManager(cfg);previous=None;oldpoints=None;counts=defaultdict(Counter)
    for token in spec['samples']:
        frame=frame_from_info(infos[token],'data/nuscenes');points=world_lidar(infos[token])
        ref=prepare_pair(previous,frame,oldpoints,points,oracle[previous.frame_id],oracle[token],cfg) if previous else None
        precheck=m.precheck(frame)
        if ref:
            for cid,e in m.last_evidence.items():
                if e.dense_measurements is None or not e.surface_units:continue
                raw=cv2.erode(e.dense_measurements['per_pixel_candidate'].astype(np.uint8),np.ones((5,5),np.uint8),
                    borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
                r=ref[cid];idx=np.flatnonzero(r['scorable']&r['annotated']);xy=r['xy'][idx];p=e.dense_source_map[xy[:,1],xy[:,0]]
                error=np.linalg.norm(p-r['source_uv'][idx],axis=1);false=~np.isfinite(error)|(error>2)|r['moving'][idx]
                candidate=raw[xy[:,1],xy[:,0]]
                actual=~m.last_masks[cid][xy[:,1],xy[:,0]]
                for unit in e.surface_units:
                    x0,y0,x1,y1=unit['bounds'];inside=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)&candidate
                    c=counts[unit['primary_reason']]
                    c['measured_candidate_reference_points']+=int(inside.sum())
                    c['correct_measured_correspondences']+=int((inside&~false).sum())
                    c['incorrect_measured_correspondences']+=int((inside&false).sum())
                    c['correct_final_accepts']+=int((inside&actual&~false).sum())
                    c['false_final_accepts']+=int((inside&actual&false).sum())
                    c['correct_candidates_left_unknown_or_refreshed']+=int((inside&~actual&~false).sum())
                    c['incorrect_candidates_left_unknown_or_refreshed']+=int((inside&~actual&false).sum())
        m.commit(frame,outputs[token],precheck);previous=frame;oldpoints=points
    path=ROOT/'funnel_support'/f"{spec['scene_id']}.json";write_json(path,dict(scene=spec['scene_id'],counts=dict(counts)))
    print('funnel support',spec['scene_id'],flush=True)


def main():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    cv2.setNumThreads(1);specs,infos,outputs,oracle=load_inputs('calibration')
    global _INPUTS
    _INPUTS=(infos,outputs,oracle)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(scene,specs))
    counts=defaultdict(Counter)
    for p in (ROOT/'funnel_support').glob('*.json'):
        for reason,v in json.loads(p.read_text())['counts'].items():counts[reason].update(v)
    write_json(ROOT/'surface_rejected_reference_support.json',dict(primary_reasons=dict(counts),
        population='Original complete-neighbourhood candidates, sparse annotated reference, unchanged 2px oracle',
        interpretation='Measured correct/incorrect correspondences rejected at these units, not a causal per-rule counterfactual. Actual on/off effects are in surface_ablation.json.'))


if __name__=='__main__':main()
