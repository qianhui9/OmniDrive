"""Offline-only physical-state oracle. No claim about exact neural feature reuse."""
import argparse
import json
from pathlib import Path
import numpy as np
from worldstream.m1.association import legal_assignment, associate
from worldstream.m1.geometry import predict_geometry, project_anchor


def oracle_assignment(anchors, annotations, gate_m=2.0):
    costs = np.full((len(anchors), len(annotations)), np.inf)
    for i, anchor in enumerate(anchors):
        for j, ann in enumerate(annotations):
            if anchor.semantic_class != ann['class'] or not ann['has_sensor_support']:
                continue
            distance = np.linalg.norm(anchor.geometry_world.center - ann['geometry'].center)
            if distance <= gate_m: costs[i, j] = distance
    return {i: annotations[j]['instance_id'] for i, j in legal_assignment(costs)}


def oracle_physical_valid(anchor, current_annotations, previous_identity, timestamp, gate_m=1.0):
    if previous_identity is None:
        return None
    same = [a for a in current_annotations if a['instance_id'] == previous_identity]
    if not same:
        return False
    ann = same[0]
    # Offline GT visibility is explicitly excluded from Frame and from online precheck.
    if not ann['has_sensor_support'] or str(ann['visibility']) != '4':
        return False
    geometry = predict_geometry(anchor, timestamp)
    return bool(np.linalg.norm(geometry.center - ann['geometry'].center) <= gate_m)


def association_baselines(previous, current, frame, validator, config, pixel_gate=48., appearance_gate=1.0,
                          modes=('fixed_image', 'appearance')):
    costs = {name: np.full((len(previous), len(current)), np.inf)
             for name in modes}
    for i, old in enumerate(previous):
        for j, new in enumerate(current):
            if old.semantic_class != new.semantic_class: continue
            distances, residuals = [], []
            for cid, support in old.observation_support.items():
                camera = frame.cameras.get(cid)
                if camera is None: continue
                p = project_anchor(new, camera, config)
                if not p.valid: continue
                r, s = p.region, support.region
                distances.append(np.linalg.norm(np.array([(r[0]+r[2])/2, (r[1]+r[3])/2])
                                                - np.array([(s[0]+s[2])/2, (s[1]+s[3])/2])))
                if 'appearance' in modes:
                    residuals.append(validator.residual(support.descriptor, validator.descriptor(camera.image, r)))
            if 'fixed_image' in modes and distances and min(distances) <= pixel_gate:
                costs['fixed_image'][i, j] = min(distances)
            if 'appearance' in modes and residuals and min(residuals) <= appearance_gate:
                costs['appearance'][i, j] = min(residuals)
    return {name: legal_assignment(c) for name, c in costs.items()}


def summarize(records):
    methods = {}
    for mode in ('geometry_only', 'appearance_only', 'geometry_observation'):
        labeled = [r for r in records if r.get('oracle_valid') is not None]
        accepted = [r for r in labeled if r[mode]]
        positives = sum(r['oracle_valid'] for r in labeled)
        correct = sum(r['oracle_valid'] for r in accepted)
        methods[mode] = {
            'valid_candidate_rate': sum(r[mode] for r in records) / len(records) if records else None,
            'false_reuse_rate_physical_oracle': (len(accepted)-correct)/len(accepted) if accepted else None,
            'valid_state_recovery_rate': correct/positives if positives else None,
            'accepted_labeled': len(accepted), 'labeled_count': len(labeled),
            'oracle_valid_count': positives,
        }
    return {'oracle': 'Physical only: previous prediction matched to GT within 2m; same current instance, '
            'sensor support, visibility=4, predicted centre error<=1m. Unknown previous IDs excluded. '
            'This is NOT neural-feature reuse ground truth.',
            'methods': methods, 'historical_anchor_trials': len(records)}


def evaluate_run(path):
    path = Path(path)
    records = [json.loads(line) for line in (path/'oracle_records.jsonl').read_text().splitlines()]
    result = summarize(records)
    result['by_interval_s'] = {}
    for interval in sorted({round(r['age_s'], 1) for r in records}):
        result['by_interval_s'][str(interval)] = summarize([r for r in records if round(r['age_s'], 1) == interval])
    from .io_utils import write_json
    write_json(path/'validity_metrics.json', result)
    return result
