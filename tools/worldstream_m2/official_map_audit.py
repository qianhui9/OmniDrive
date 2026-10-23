"""Offline GT lane audit using the UNCHANGED OpenLane geometry and matcher.

All retained scores, original 1/2/3 m criteria and original ego relaxation are
preserved. Counts are operating points, not AP; masked distances are not errors.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
import multiprocessing
from pathlib import Path
import pickle

import numpy as np

from .common import write
from tools.worldstream_m123.benchmark.metrics import distribution


def audit_frame(payload):
    token, scene, gt, lines, scores = payload
    module = importlib.import_module('openlanev2.centerline.evaluation.evaluate')
    xs = [lane['points'] for lane in gt['lane_centerline']]
    ys = [np.asarray(line, dtype=np.float32) for line in lines]
    confidences = np.asarray(scores).reshape(-1).tolist()
    if len(ys) != len(confidences):
        raise ValueError('Prediction point/score count differs')
    chamfer = module.pairwise(xs, ys, module.chamfer_distance, relax=True)
    mask = chamfer < module.THRESHOLDS_FRECHET[-1]
    distances = module.pairwise(xs, ys, module.frechet_distance, mask=mask, relax=True)
    topology = np.asarray(gt['topology_lclc'])
    if topology.shape != (len(xs), len(xs)):
        raise ValueError('GT lane/topology identity mismatch')
    # This is a graph-degree stratum, not an independent intersection label.
    branches = (topology.sum(0) > 1) | (topology.sum(1) > 1)
    rows = []
    for threshold in module.THRESHOLDS_FRECHET:
        tp, fp, indices = module._tpfp(xs, ys, confidences, distances, threshold)
        matches = []
        taken = set()
        rejected = []
        for index in range(len(ys)):
            if tp[index]:
                matched = int(indices[index]); taken.add(matched)
                matches.append(dict(prediction_index=index, ground_truth_index=matched,
                    ground_truth_id=str(gt['lane_centerline'][matched]['id']), confidence=confidences[index],
                    official_relaxed_frechet_m=float(distances[matched, index]),
                    raw_frechet_m=float(module.frechet_distance(xs[matched], ys[index])),
                    branch_related_ground_truth=bool(branches[matched])))
            else:
                nearest = int(distances[:, index].argmin()) if xs else None
                computed = nearest is not None and bool(mask[nearest, index])
                rejected.append(dict(prediction_index=index, confidence=confidences[index],
                    nearest_ground_truth_index=nearest if computed else None,
                    nearest_official_relaxed_frechet_m=float(distances[nearest, index]) if computed else None,
                    reason=('no_ground_truth' if not xs else 'chamfer_prefilter_no_candidate' if not computed
                        else 'nearest_gt_already_covered' if distances[nearest, index] < threshold else 'distance_gate')))
        uncovered = [i for i in range(len(xs)) if i not in taken]
        covered = sorted(taken)
        rows.append(dict(threshold_m=threshold, ground_truth=len(xs), predictions=len(ys),
            tp=int(tp.sum()), fp=int(fp.sum()), fn=len(uncovered), matches=matches,
            unmatched_predictions=rejected, unmatched_ground_truth_indices=uncovered,
            branch_ground_truth=int(branches.sum()), branch_ground_truth_matched=int(branches[covered].sum()),
            ground_truth_edges=int(topology.sum()),
            edges_between_matched_ground_truth=int(topology[np.ix_(covered, covered)].sum()),
            predicted_edges=0))
    return dict(frame_id=token, scene_id=scene, thresholds=rows)


def evaluate_run(run, output, limit=40, workers=8):
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'status.json', dict(status='RUNNING'))
    cfg = Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    cfg.data.test.test_mode = True; cfg.data.test.pipeline = []
    dataset = build_dataset(cfg.data.test)
    infos = {info['token']: info for info in dataset.data_infos}
    tokens = json.loads((run/'sequence.json').read_text())['tokens']
    with (run/'predictions.pkl').open('rb') as handle:
        predictions = pickle.load(handle)
    if len(tokens) != len(predictions) or len(set(tokens)) != len(tokens):
        raise ValueError('Prediction/frame identity mismatch')
    with open(dataset.lane_anno_file, 'rb') as handle:
        ground_truth = pickle.load(handle)
    selected = list(zip(tokens, predictions))[:limit or None]
    payloads = [(token, infos[token]['scene_token'], ground_truth[tuple(infos[token]['lane_info'])]['annotation'],
        *prediction['lane_results'][0]) for token, prediction in selected]
    totals = {}; errors = {}; raw_errors = {}; scenes = {}; count = 0
    with (output/'frames.jsonl').open('w') as handle:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for row in pool.map(audit_frame, payloads, chunksize=4):
                count += 1; handle.write(json.dumps(row, allow_nan=False)+'\n')
                for value in row['thresholds']:
                    key = str(value['threshold_m']); totals.setdefault(key, {}); errors.setdefault(key, []); raw_errors.setdefault(key, [])
                    scene = scenes.setdefault(row['scene_id'], {}).setdefault(key, {})
                    for field in ('ground_truth','predictions','tp','fp','fn','branch_ground_truth','branch_ground_truth_matched',
                            'ground_truth_edges','edges_between_matched_ground_truth','predicted_edges'):
                        totals[key][field] = totals[key].get(field, 0)+value[field]
                        scene[field] = scene.get(field, 0)+value[field]
                    errors[key].extend(match['official_relaxed_frechet_m'] for match in value['matches'])
                    raw_errors[key].extend(match['raw_frechet_m'] for match in value['matches'])
                if count % 250 == 0: print('OFFICIAL_MAP_AUDIT', count, '/', len(selected), flush=True)
    summary = dict(status='COMPLETED', frames=count, scenes=len(scenes), thresholds={
        key: dict(**value, matched_official_relaxed_frechet_m=distribution(errors[key]),
            matched_raw_frechet_m=distribution(raw_errors[key])) for key, value in totals.items()},
        per_scene=scenes, prediction_run=str(run.resolve()),
        scope='Unchanged OpenLane _tpfp and geometry at official 1/2/3m thresholds, all scores. Counts are not AP. Original ego relaxation and Chamfer prefilter preserved; masked sentinel 1024 is never reported as geometric error. Graph-degree branch stratum is not an independently labeled intersection. No topology prediction head; predicted adjacency is zero.')
    write(output/'summary.json', summary)
    write(output/'status.json', dict(status='COMPLETED', frames=count))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=40); parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    evaluate_run(args.run, args.output, args.limit, args.workers)
