"""Offline metrics: missing evidence is never encoded as a measured zero."""
import re
import math
from collections import Counter
import numpy as np


def finite_metrics(value):
    """Keep official undefined metrics null and record their exact paths."""
    unavailable = []
    def visit(x, path):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(x, np.generic):
            x = x.item()
        if isinstance(x, float) and not math.isfinite(x):
            unavailable.append(path)
            return None
        if isinstance(x, dict):
            return {k: visit(v, path+'/'+str(k)) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [visit(v, path+'/'+str(i)) for i, v in enumerate(x)]
        return x
    result = visit(value, '')
    return result, unavailable


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def distribution(values):
    values = list(values)
    if not values:
        return dict(n=0, mean=None, p50=None, p95=None, maximum=None, total=None)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite latency')
    return dict(n=len(values), mean=float(np.mean(values)),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                maximum=float(max(values)), total=float(sum(values)))


def answer_quality(answer, reference):
    if not isinstance(reference, str) or not reference.strip():
        return None
    a, b = (re.findall(r'\w+', s.lower()) for s in (answer, reference))
    overlap = sum((Counter(a) & Counter(b)).values())
    number = lambda s: [float(x) for x in re.findall(r'(?<!\w)[+-]?\d+(?:\.\d+)?', s)]
    x, y = number(answer), number(reference)
    return dict(exact_match=answer.strip().lower() == reference.strip().lower(),
                token_f1=2 * overlap / (len(a) + len(b)) if a or b else 1.,
                numeric_applicable=bool(y),
                numeric_count_match=len(x) == len(y) if y else None,
                numeric_mae=float(np.mean(np.abs(np.array(x) - y))) if y and len(x) == len(y) else None)


def quality_summary(rows):
    valid = [r['quality'] for r in rows if r.get('quality') is not None]
    numeric = [r for r in valid if r['numeric_applicable']]
    errors = [r['numeric_mae'] for r in numeric if r['numeric_mae'] is not None]
    return dict(questions=len(rows), valid_gt=len(valid), missing_gt=len(rows)-len(valid),
                answer_exact_match=ratio(sum(r['exact_match'] for r in valid), len(valid)),
                token_f1=ratio(sum(r['token_f1'] for r in valid), len(valid)),
                numeric_questions=len(numeric),
                numeric_count_match=ratio(sum(r['numeric_count_match'] for r in numeric), len(numeric)),
                numeric_mae=ratio(sum(errors), len(errors)), numeric_mae_denominator=len(errors))


def compare(base, final):
    if base is None or final is None or base <= 0 or final <= 0:
        return dict(latency_change_percent=None, speedup=None, throughput_change_percent=None)
    return dict(latency_change_percent=(base-final)/base*100, speedup=base/final,
                throughput_change_percent=(base/final-1)*100)


def paired_scene_ci(rows, baseline='B0', final='B3', key='request_ms', seed=0):
    """Bootstrap paired scene means; no pairing across different workloads."""
    groups = {}
    for r in rows:
        if r['condition'] not in (baseline, final):
            continue
        pair = (r['workload'], r['repeat'], r['scene_id'], r['frame_id'], r.get('measurement_set'))
        groups.setdefault(pair, {})[r['condition']] = r[key]
    scenes = {}
    for pair, values in groups.items():
        if baseline in values and final in values:
            scenes.setdefault(pair[2], []).append(values[final]-values[baseline])
    if len(scenes) < 2:
        return dict(status='NOT_AVAILABLE', paired_scenes=len(scenes), ci95_ms=None)
    means = np.array([np.mean(x) for x in scenes.values()])
    rng = np.random.RandomState(seed)
    boot = np.mean(rng.choice(means, (2000, len(means))), axis=1)
    return dict(status='MEASURED', paired_scenes=len(scenes),
                final_minus_baseline_ms=float(means.mean()),
                ci95_ms=np.percentile(boot, [2.5, 97.5]).tolist())


def validate_sequence(infos):
    seen, closed, previous = set(), set(), None
    for info in infos:
        scene, token, stamp = info['scene_token'], info['token'], info['timestamp']
        if token in seen:
            raise ValueError('Duplicate frame: '+token)
        if previous:
            old_scene, old_stamp = previous
            if scene == old_scene and stamp <= old_stamp:
                raise ValueError('Non-increasing scene time')
            if scene != old_scene:
                closed.add(old_scene)
                if scene in closed:
                    raise ValueError('Interleaved scene history')
        seen.add(token)
        previous = scene, stamp
    return dict(frames=len(seen), scenes=len({i['scene_token'] for i in infos}), status='PASS')
