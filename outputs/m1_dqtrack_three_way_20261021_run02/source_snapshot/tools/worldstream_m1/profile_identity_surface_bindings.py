"""Measure validation cost against HEAD on exposed, preloaded driving frames.

Run after quality replay finishes so unrelated jobs do not contaminate timing.
This measures an engineering correction, with no claimed deployment speedup.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import subprocess
from time import perf_counter

import cv2
import numpy as np

from worldstream.m1 import M1Config, WorldStateManager
from .dataset import frame_from_info, prediction_outputs
from .independent_refinement import baseline_manager, distribution
from .io_utils import read_prediction_subset, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if {4, 5} & os.sched_getaffinity(0):
        raise RuntimeError('Run with taskset -c 0-3,6-31')
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / 'baseline_source/worldstream/m1'
    snapshot.mkdir(parents=True)
    for path in Path('worldstream/m1').glob('*.py'):
        (snapshot / path.name).write_bytes(subprocess.check_output(['git', 'show', 'HEAD:' + str(path)]))
    cv2.setNumThreads(1)
    protocol = json.loads(Path('outputs/module1_independent_refinement/experiment_protocol.json').read_text())
    tokens = protocol['scenes']['development_long'][0]['samples'][:8]
    with Path('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl').open('rb') as handle:
        infos = {i['token']: i for i in pickle.load(handle)['infos'] if i['token'] in tokens}
    frames = [frame_from_info(infos[t], 'data/nuscenes') for t in tokens]
    predictions = read_prediction_subset('test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json', set(tokens))
    config = M1Config.load('configs/worldstream_m1_independent.yaml')
    outputs = [prediction_outputs(predictions[t], config) for t in tokens]
    settings = dict(conservative={}, identity_only=dict(identity_continuity=True),
        strict_only=dict(foreground_correspondence=True),
        both_producers=dict(identity_continuity=True, foreground_correspondence=True),
        surface_identity=dict(identity_continuity=True, dense_region_policy='surface_visibility', dense_guard_radius_px=2))
    rows = []
    comparisons = 0
    for repeat in range(4):
        for method, values in settings.items():
            captures = {}
            order = ['archived_HEAD', 'current'] if repeat % 2 == 0 else ['current', 'archived_HEAD']
            for version in order:
                cfg = replace(config, **values)
                manager = baseline_manager(args.output, cfg) if version == 'archived_HEAD' else WorldStateManager(cfg)
                masks, identities = [], []
                for index, (frame, output) in enumerate(zip(frames, outputs)):
                    start = perf_counter()
                    result = manager.precheck(frame)
                    pre = perf_counter()
                    manager.commit(frame, output, result)
                    end = perf_counter()
                    masks.append({cid: value.copy() for cid, value in manager.last_masks.items()})
                    diag = manager.last_commit_diagnostics
                    identities.append([{k: v[k] for k in ('anchor_id', 'status', 'rejection_reasons', 'measurements')}
                        for v in diag['identity_continuity_proposals']])
                    if repeat and index:
                        rows.append(dict(version=version, method=method, repeat=repeat, frame=frame.frame_id,
                            total_ms=(end-start)*1000, precheck_ms=(pre-start)*1000, commit_ms=(end-pre)*1000,
                            identity_ms=sum(p['producer_cpu_ms'] for p in diag['identity_continuity_proposals']),
                            foreground_ms=sum(p['producer_cpu_ms'] for p in diag['foreground_correspondence_proposals']),
                            certification_ms=diag['physical_certification_ms']))
                captures[version] = masks, identities
            assert captures['archived_HEAD'][1] == captures['current'][1]
            for left, right in zip(captures['archived_HEAD'][0], captures['current'][0]):
                for cid in left:
                    assert np.array_equal(left[cid], right[cid])
                    comparisons += 1
    write_json(args.output / 'runtime_cost_breakdown.json', dict(
        baseline_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        raw=rows, methods={method: {version: {key: distribution([r[key] for r in rows if r['method']==method and r['version']==version])
            for key in ('total_ms', 'precheck_ms', 'commit_ms', 'identity_ms', 'foreground_ms', 'certification_ms')}
            for version in ('archived_HEAD', 'current')} for method in settings},
        compared_camera_masks=comparisons, all_masks_equal=True, identity_measurements_and_decisions_equal=True,
        warmup_repetitions=1, timed_repetitions=3, unique_frames=tokens, settings=settings,
        excludes='Image IO, detection inference, output adapter, Oracle and logging',
        optimization_performed=False, independent_quality_evaluation=False, cpu_affinity=sorted(os.sched_getaffinity(0))))
    print('Isolated timing and output comparisons complete', flush=True)


if __name__ == '__main__':
    main()
