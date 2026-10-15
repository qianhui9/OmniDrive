"""Matched warmed CPU cost measurements; allocation tracking in a separate pass."""
import argparse
import cProfile
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
import tracemalloc
import resource
from dataclasses import replace
import cv2
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager
from .dataset import select_infos,frame_from_info,prediction_outputs
from .io_utils import read_prediction_subset,write_json


def signature(manager,result):
    return dict(valid=result.valid_anchor_ids,masks={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in manager.last_masks.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/module1_refinement/latency_profile.json')
    p.add_argument('--repeats',type=int,default=10);p.add_argument('--warmup',type=int,default=2)
    args=p.parse_args();cv2.setNumThreads(1)
    infos=select_infos('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl',limit=4)
    preds=read_prediction_subset('test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json',{i['token'] for i in infos})
    frames=[frame_from_info(i,'data/nuscenes') for i in infos]
    cfg=M1Config.load('configs/worldstream_m1_refinement.yaml')
    outputs=[prediction_outputs(preds[i['token']],cfg) for i in infos]
    results=[]
    for name,config in [('old_strict',replace(cfg,temporal_visibility=False,local_support=False)),('new_full',cfg)]:
        samples=[]; fingerprints=[]
        def once(timed=False):
            manager=WorldStateManager(config); signs=[]
            for idx,frame in enumerate(frames):
                start=perf_counter();r=manager.precheck(frame);manager.commit(frame,outputs[idx],r)
                total=(perf_counter()-start)*1000
                if timed and idx:
                    samples.append(dict(r.diagnostics['timings_ms'],state_update=manager.last_commit_diagnostics['state_update_ms'],
                        association=manager.last_commit_diagnostics['association_ms'],m1_overall_wall=total,
                        input_adaptation=0.,gpu_to_cpu_transfer=0.))
                signs.append(signature(manager,r))
            return signs
        for iteration in range(args.warmup+args.repeats):
            fingerprints.append(once(iteration>=args.warmup))
        assert all(f==fingerprints[0] for f in fingerprints)
        profiler=cProfile.Profile();profiler.enable();once();profiler.disable()
        profiler.dump_stats(str(Path(args.output).with_name(name+'_hotspots.prof')))
        import pstats
        with Path(args.output).with_name(name+'_hotspots.txt').open('w') as handle:
            pstats.Stats(profiler,stream=handle).sort_stats('cumtime').print_stats(25)
        tracemalloc.start();once();retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        results.append(dict(method=name,samples=samples,percentiles_ms={k:dict(p50=float(np.percentile([s[k] for s in samples],50)),
                           p95=float(np.percentile([s[k] for s in samples],95))) for k in samples[0]},
                           trace_peak_bytes=peak,trace_retained_bytes=retained,deterministic=True,decision_fingerprints=fingerprints[0]))
        print(name,results[-1]['percentiles_ms']['m1_overall_wall'],flush=True)
    write_json(args.output,dict(environment=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,
        cpu=platform.processor(),opencv_threads=cv2.getNumThreads()),frames=len(frames),sample_ids=[i['token'] for i in infos],
        warmup=args.warmup,repeats=args.repeats,results=results,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Preloaded real 4-frame replay, skip reset frame timing; excludes IO/model/offline oracle. CPU allocation tracing separate; OpenCV allocator pools excluded. Input adaptation/transfer not applicable here; measured in GPU experiment.'))

if __name__=='__main__':main()
