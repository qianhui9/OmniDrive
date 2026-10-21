"""Repeated real two-keyframe CPU microbenchmark; allocation tracing is a separate pass."""
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import resource
import tracemalloc
import numpy as np
from worldstream.m1 import M1Config, FullOutputs, WorldStateManager
from .dataset import select_infos, frame_from_info, prediction_outputs
from .io_utils import read_prediction_subset, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--infos', default='data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    p.add_argument('--predictions', required=True)
    p.add_argument('--data-root', default='data/nuscenes')
    p.add_argument('--output', default='outputs/module1/profile.json')
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    if args.repeats < 1: raise ValueError('repeats must be positive')
    config = M1Config()
    infos = select_infos(args.infos, limit=2)
    preds = read_prediction_subset(args.predictions, {i['token'] for i in infos})
    frames = [frame_from_info(i, args.data_root) for i in infos]
    outputs = [prediction_outputs(preds[i['token']], config) for i in infos]
    results = []
    for count in sorted({min(n, len(outputs[0].anchors)) for n in (8, 32, 64, 128)}):
        for method in ('normalized_gray', 'gradient'):
            cfg = replace(config, descriptor=method, max_anchors=count)
            measurements = []
            for repeat in range(args.repeats + 1):
                manager = WorldStateManager(cfg)
                manager.initialize(frames[0], FullOutputs(outputs[0].anchors[:count]))
                begin = perf_counter()
                result = manager.precheck(frames[1])
                manager.commit(frames[1], FullOutputs(outputs[1].anchors[:count]), result)
                total = (perf_counter()-begin)*1000
                if repeat:
                    measurements.append(dict(result.diagnostics['timings_ms'],
                        association=manager.last_commit_diagnostics['association_ms'],
                        state_update=manager.last_commit_diagnostics['state_update_ms'], total=total))
            # Allocation tracing is excluded from the timing samples above.
            tracemalloc.start()
            manager = WorldStateManager(cfg)
            manager.initialize(frames[0], FullOutputs(outputs[0].anchors[:count]))
            result = manager.precheck(frames[1])
            manager.commit(frames[1], FullOutputs(outputs[1].anchors[:count]), result)
            current_bytes, peak_bytes = tracemalloc.get_traced_memory(); tracemalloc.stop()
            results.append({'anchor_limit': count, 'actual_prior_anchor_count': count, 'descriptor': method,
                'mean_cpu_wall_ms': {k: float(np.mean([m[k] for m in measurements])) for k in measurements[0]},
                'std_total_ms': float(np.std([m['total'] for m in measurements])),
                'state_and_temporary_traced_peak_bytes': peak_bytes, 'traced_retained_bytes': current_bytes,
                'retained_tensor_bytes': manager.store.tensor_bytes})
    write_json(args.output, {'source': 'real two-keyframe replay repeated microbenchmark, not model speedup',
        'sample_ids': [i['token'] for i in infos], 'repeats': args.repeats, 'warmup': 1,
        'results': results, 'gpu_time_ms': None, 'gpu_memory_bytes': None,
        'process_peak_rss_kib_including_inputs': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'cpu_memory_scope': 'tracemalloc includes Python/NumPy-owned state and temporaries; excludes preloaded '
                            'images, model, C-library allocator pools; process RSS includes inputs'})
    print(args.output)


if __name__ == '__main__': main()
