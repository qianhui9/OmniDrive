"""CPU-only aggregation for the R0 factual-QA and planning task benchmark."""
from collections import Counter
import math
import re

import numpy as np

FACT_GROUPS = ('vehicle_category', 'vehicle_existence', 'pedestrian_position', 'other')
PLANNING_QUESTION = 'Please provide the planning trajectory for the ego car without reasons.'


def parse_trajectory(answer):
    """Accept exactly six finite absolute XY positions inside one [PT, ...]."""
    spans = re.findall(r'\[\s*PT\s*,(.*?)\]', answer, flags=re.S)
    if len(spans) != 1:
        raise ValueError('expected_one_PT_trajectory')
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    pair = rf'\(\s*({number})\s*,\s*({number})\s*\)'
    body = spans[0].strip()
    values = re.findall(pair, body)
    if not re.fullmatch(rf'{pair}(?:\s*,\s*{pair})*', body):
        raise ValueError('invalid_PT_coordinates')
    if len(values) != 6:
        raise ValueError('expected_six_half_second_positions')
    traj = np.asarray(values, dtype=np.float64)
    if not np.isfinite(traj).all():
        raise ValueError('nonfinite_trajectory')
    return traj


def performance(rows):
    values = np.asarray([r['e2e_ms'] for r in rows], dtype=np.float64)
    if len(values) and (not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError('E2E latencies must be finite and positive')
    total = float(values.sum())
    return dict(frames=len(rows), questions=sum(len(r['questions']) for r in rows),
                mean_e2e_ms=float(values.mean()) if len(values) else None,
                p95_e2e_ms=float(np.percentile(values, 95)) if len(values) else None,
                FPS=1000*len(values)/total if total else None,
                QPS=1000*sum(len(r['questions']) for r in rows)/total if total else None,
                total_e2e_seconds=total/1000,
                latency_unit='one_visual_frame_with_all_its_questions',
                includes_cold_scene_frames=True,
                includes='image loading/preprocessing, prompt encoding, H2D, full model, answer export; CUDA synchronized',
                excludes='checkpoint initialization, GPU queue wait, offline GT/scoring and receipt serialization')


def fact_summary(events):
    def summarize(part):
        known = [r for r in part if r['quality']['known']]
        for r in known:
            s = r['quality']['score']
            if not isinstance(s, (int, float)) or not math.isfinite(s) or not 0 <= s <= 1:
                raise ValueError('Invalid finite fact score')
        n = len(known)
        score = sum(r['quality']['score'] for r in known)/n if n else None
        correct = sum(r['quality']['score'] == 1 for r in known)
        return dict(questions=len(part), scored=n, unscored=len(part)-n,
                    scoring_coverage=n/len(part) if part else None,
                    scorable_fact_accuracy=score,
                    scorable_fact_accuracy_percent=100*score if score is not None else None,
                    fully_correct_answers=correct,
                    fully_correct_accuracy=correct/n if n else None,
                    unknown_reasons=dict(Counter(
                        r['quality'].get('reason') or
                        (r['quality'].get('reference') or {}).get('reason') or 'unsupported_or_missing_GT'
                        for r in part if not r['quality']['known'])))
    return dict(overall=summarize(events),
                groups={g:summarize([e for e in events if e['group'] == g]) for g in FACT_GROUPS},
                other_metric='limited exact current-frame reference fact recall; can be fractional; not unrestricted answer accuracy',
                unknown_policy='Unscorable questions are excluded from accuracy and explicitly counted, never treated as correct.')


def planning_summary(rows):
    result = {}
    for seconds in (1, 2, 3):
        key = str(seconds)+'s'
        eligible = [r for r in rows if r['horizons'][key]['GT_valid']]
        valid = [r['horizons'][key] for r in eligible if r['horizons'][key]['prediction_valid']]
        collision = [r for r in valid if r.get('collision_evaluable', True)]
        road = [r for r in valid if r.get('road_evaluable', True)]
        mean = lambda name: float(np.mean([v[name] for v in valid])) if valid else None
        rate = lambda part, name: 100*sum(bool(v[name]) for v in part)/len(part) if part else None
        result[key] = dict(total_frames=len(rows), GT_valid_frames=len(eligible),
                           scored_trajectories=len(valid), invalid_predictions=len(eligible)-len(valid),
                           prediction_coverage=len(valid)/len(eligible) if eligible else None,
                           L2_ADE_m=mean('ADE_m'), ADE_m=mean('ADE_m'),
                           endpoint_L2_FDE_m=mean('FDE_m'),
                           collision_rate_percent=rate(collision, 'collision'),
                           raw_collision_rate_percent=rate(collision, 'raw_collision'),
                           collision_denominator=len(collision),
                           road_boundary_rate_percent=rate(road, 'road_boundary'),
                           road_boundary_denominator=len(road),
                           initially_offroad_excluded=sum(not r.get('road_evaluable', True) for r in valid))
    return dict(horizons=result, time_step_seconds=.5,
                parse_failures=sum(bool(r.get('parse_error')) for r in rows),
                parse_failure_reasons=dict(Counter(r['parse_error'] for r in rows if r.get('parse_error'))),
                L2_ADE_definition='mean Euclidean displacement error over first 2/4/6 predicted XY points; endpoint FDE also reported',
                collision_definition='any ego-box/vehicle occupancy collision up to horizon; primary excludes GT-colliding steps as original ST-P3 evaluator; raw rate also retained',
                road_boundary_definition='any rotated ego-footprint pixel outside drivable_area up to horizon; initially offroad samples excluded and counted',
                denominator_policy='GT validity uses all XY masks through each horizon. Invalid predictions excluded from quality denominators and reported as failures/coverage; no zero imputation.')


def valid_gt(info):
    gt = np.asarray(info['gt_planning'], dtype=np.float64).reshape(-1, 6, 3)[0, :, :2]
    mask = np.asarray(info['gt_planning_mask']).reshape(1, 6, -1)[0]
    valid = (mask != 0).all(axis=1) & np.isfinite(mask).all(axis=1) & np.isfinite(gt).all(axis=1)
    return gt, valid


def score_planning(answer, info, backend):
    gt, mask = valid_gt(info)
    try:
        pred = parse_trajectory(answer)
        failure = None
    except ValueError as exc:
        pred, failure = None, str(exc)
    horizons = {}
    geometry = None
    if pred is not None and any(mask[:n].all() for n in (2, 4, 6)):
        # Only valid GT prefixes enter the collision kernel; scene-tail padding
        # never leaks into scoring or causes false safe trajectories.
        steps = max(n for n in (2, 4, 6) if mask[:n].all())
        geometry = backend(info, pred[:steps], gt[:steps])
    for seconds in (1, 2, 3):
        n = 2*seconds
        row = dict(GT_valid=bool(mask[:n].all()), prediction_valid=pred is not None)
        if row['GT_valid'] and pred is not None:
            errors = np.linalg.norm(pred[:n]-gt[:n], axis=1)
            row.update(ADE_m=float(errors.mean()), FDE_m=float(errors[-1]),
                       collision=bool(any(geometry['collision'][:n])),
                       raw_collision=bool(any(geometry['raw_collision'][:n])),
                       collision_evaluable=not bool(any(geometry['out_of_grid'][:n])),
                       road_boundary=bool(any(geometry['road_boundary'][:n])),
                       road_evaluable=not geometry['initially_offroad'])
        horizons[str(seconds)+'s'] = row
    return dict(parse_error=failure, prediction_xy=pred.tolist() if pred is not None else None,
                horizons=horizons)
