"""Recover completed inference from saved predictions; never rerun the model."""
import argparse
import json
import pickle
import shutil
import traceback
import fcntl
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.worldstream_m123.stateful_opportunity.common import write
from .metrics import distribution, finite_metrics, quality_summary
from .report import read_json, read_rows


def detection_metrics(raw):
    prefix = 'pts_bbox_NuScenes'
    result = {}
    errors = dict(trans_err='mATE', scale_err='mASE', orient_err='mAOE', vel_err='mAVE', attr_err='mAAE')
    for cls, values in raw['label_aps'].items():
        for key, value in values.items():
            result[f'{prefix}/{cls}_AP_dist_{key}'] = float(f'{value:.4f}')
    for cls, values in raw['label_tp_errors'].items():
        for key, value in values.items():
            result[f'{prefix}/{cls}_{key}'] = float(f'{value:.4f}')
    for key, value in raw['tp_errors'].items():
        result[f'{prefix}/{errors[key]}'] = float(f'{value:.4f}')
    result[f'{prefix}/NDS'] = raw['nd_score']
    result[f'{prefix}/mAP'] = raw['mean_ap']
    return result


def recover(dest):
    dest = Path(dest)
    lock = (dest/'recovery.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = read_json(dest/'status.json', {})
    if status.get('status') != 'FAILED' or 'Out of range float values' not in status.get('error', ''):
        return False
    sequence = read_json(dest/'sequence.json', {})
    rows = read_rows(dest/'per_frame_metrics.jsonl')
    questions = read_rows(dest/'per_question_metrics.jsonl')
    assert [r['frame_id'] for r in rows] == sequence['tokens'], 'Inference incomplete; no recovery allowed'
    assert len(questions) == len(rows) and sequence['full_validation']
    assert read_json(dest/'checkpoint_validation.json')['status'] == 'PASS'
    assert (dest/'predictions.pkl').is_file()
    history = dest/'serialization_failure_history'
    history.mkdir(exist_ok=True)
    for name in ('status.json', 'failure.log', 'map_evaluation_error.log'):
        p = dest/name
        if p.exists() and not (history/name).exists():
            shutil.copy2(p, history/name)
    write(dest/'recovery_status.json', dict(status='RUNNING', inference_rerun=False))
    raw = read_json(dest/'detection/pts_bbox/metrics_summary.json')
    assert raw is not None, 'Official detection results missing'
    outputs = pickle.load((dest/'predictions.pkl').open('rb'))
    assert len(outputs) == len(rows)
    infos = pickle.load(Path('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl').open('rb'))['infos']
    infos = sorted(infos, key=lambda i: (i['scene_token'], i['timestamp']))
    assert [i['token'] for i in infos] == sequence['tokens']
    from .worker import map_evaluate
    map_result = read_json(dest/'map_metrics.json')
    map_recomputed = map_result is None
    if map_result is None:
        map_result = map_evaluate(SimpleNamespace(data_infos=infos,
            lane_anno_file='data/nuscenes/data_dict_subset_B_val.pkl'), outputs, dest)
    identity = None
    if any('m1_commit' in row for row in rows):
        from .observe import IdentityScorer
        from tools.worldstream_m1.dataset import load_oracle
        from worldstream.m1 import M1Config, Frame
        from worldstream.m1.omnidrive_adapter import outputs_from_model, quaternion_transform
        scorer = IdentityScorer(load_oracle(infos, 'data/nuscenes'))
        config = M1Config.load('configs/worldstream_m1_final.yaml')
        for info, result, row in zip(infos, outputs, rows):
            pose = quaternion_transform(info['ego2global_rotation'], info['ego2global_translation']).astype(np.float32)
            frame = Frame(info['scene_token'], info['token'], info['timestamp']/1e6, pose, {}, 'offline-scoring-only')
            anchors = outputs_from_model(frame, [result['pts_bbox']], result['lane_results'], {}, config).anchors
            edges = scorer.update(frame, anchors, row['m1_commit'])
            assert edges == row['m1_edges'], 'Offline reconstruction disagrees with original post-commit GT scoring'
        identity = scorer.summary()
    args = read_json(dest/'invocation.json')
    summary = dict(status='COMPLETED', frames=len(rows), questions=len(questions), full_validation=True,
        warmup_frames=args.get('warmup', 3), request_ms=distribution(r['request_ms'] for r in rows),
        model_ms=distribution(r['model_ms'] for r in rows), quality=quality_summary(questions),
        m1=identity, official_detection=detection_metrics(raw), official_map=map_result, official_status='MEASURED',
        recovery=dict(inference_rerun=False, original_failure_preserved=str(history),
            source='This experiment saved predictions, committed-state rows and official evaluation artifacts',
            map_recomputed=map_recomputed))
    ledger = read_json(dest/'m1_scene_integrity.json')
    if ledger:
        summary['m1_scene_integrity'] = {s: 'PASS' if x.get('ledger', {}).get('status') == 'PASS' else 'FAIL' for s, x in ledger.items()}
    summary, unavailable = finite_metrics(summary)
    summary['undefined_metric_paths'] = unavailable
    write(dest/'metrics.json', summary)
    ledger_pass = all(v == 'PASS' for v in summary.get('m1_scene_integrity', {}).values())
    write(dest/'status.json', dict(status='COMPLETED', measured_frames=len(rows), full_validation=True,
        integrity='PASS' if ledger_pass else 'FAIL', official_status='MEASURED', recovered_after_serialization_failure=True))
    write(dest/'recovery_status.json', dict(status='COMPLETED', inference_rerun=False, undefined_metric_paths=unavailable))
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    args = parser.parse_args()
    try:
        print(recover(args.experiment))
    except Exception:
        write(Path(args.experiment)/'recovery_status.json', dict(status='FAILED', error=traceback.format_exc(), inference_rerun=False))
        raise
