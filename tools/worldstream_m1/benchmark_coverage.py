"""Preloaded interleaved warm benchmark; same input for sparse and dense paths."""
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import hashlib
import cv2
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager
from .dataset import select_infos,frame_from_info,prediction_outputs
from .io_utils import read_prediction_subset,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='outputs/module1_refinement/coverage_final/warmed_latency.json');a=p.parse_args()
    cv2.setNumThreads(1)
    infos=select_infos('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl',limit=4)
    predictions=read_prediction_subset('test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json',{i['token'] for i in infos})
    frames=[frame_from_info(i,'data/nuscenes') for i in infos]
    configs={'sparse':replace(M1Config.load('configs/worldstream_m1_balanced.yaml'),logging=False),
             'dense':replace(M1Config.load('configs/worldstream_m1_coverage.yaml'),logging=False)}
    outputs=[prediction_outputs(predictions[f.frame_id],configs['sparse']) for f in frames]
    results={n:dict(samples_ms=[],fingerprints=[]) for n in configs}
    for repeat in range(5):
        for name in (list(configs) if repeat%2==0 else list(configs)[::-1]):
            m=WorldStateManager(configs[name]);signs=[]
            for i,(f,out) in enumerate(zip(frames,outputs)):
                t=perf_counter();r=m.precheck(f);m.commit(f,out,r);elapsed=(perf_counter()-t)*1000
                if repeat and i:results[name]['samples_ms'].append(elapsed)
                signs.append(dict(valid=r.valid_anchor_ids,masks={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in m.last_masks.items()}))
            results[name]['fingerprints'].append(signs)
    for n,rec in results.items():
        rec['deterministic']=all(v==rec['fingerprints'][0] for v in rec['fingerprints'])
        assert rec['deterministic']
        rec['p50_p95_ms']=np.percentile(rec['samples_ms'],[50,95]).tolist()
    write_json(a.output,dict(results=results,warmup=1,repeats=4,samples=[f.frame_id for f in frames],
        scope='Preloaded 4-frame real replay, skip reset frame, interleaved order. No IO, oracle, model, transfer.'))

if __name__=='__main__':main()
