"""Scene-cluster uncertainty: repeat/query observations never become clusters."""
from collections import defaultdict
import numpy as np

METRICS = ('factual_qa_accuracy_percent', 'stale_answer_rate_percent',
           'average_e2e_ms', 'p95_e2e_ms')


def sample_counts(frames, events):
    key = lambda e: (e['scene_id'], e['frame_id'], e['query_index'])
    scored = [e for e in events if e['scorable']]
    stale = [e for e in events if e['stale_eligible']]
    return dict(independent_scenes=len({r['scene_id'] for r in frames + events}),
        completed_repeats=len({r['repeat'] for r in frames + events}),
        query_observations=len(events),
        unique_frames=len({(r['scene_id'], r['frame_id']) for r in frames}),
        unique_queries=len({key(e) for e in events}),
        unique_scorable_queries=len({key(e) for e in scored}),
        unique_stale_eligible_query_transitions=len({key(e) for e in stale}),
        scenes_with_stale_eligible_queries=len({e['scene_id'] for e in stale}),
        unique_queries_with_any_stale_error=len({key(e) for e in stale if e['stale_answer']}))


def repeat_statistics(frames, events):
    """SD across independent inference repeats of the same scene set.

    Pooled P95 remains the main metric; mean of repeat P95s is named separately.
    """
    from .report import aggregate
    values = defaultdict(list)
    for repeat in sorted({r['repeat'] for r in frames}):
        row = aggregate([r for r in frames if r['repeat'] == repeat],
                        [e for e in events if e['repeat'] == repeat])
        for metric in METRICS:
            if row[metric] is not None:
                values[metric].append(row[metric])
    result = {}
    for metric in METRICS:
        xs = values[metric]
        result[metric + '_repeat_mean'] = float(np.mean(xs)) if xs else None
        result[metric + '_repeat_sd'] = float(np.std(xs, ddof=1)) if len(xs) > 1 else None
        result[metric + '_repeat_count'] = len(xs)
    return result


def _ratio(num, den, scale=1.):
    return np.divide(scale * num, den, out=np.full(len(den), np.nan), where=den > 0)


def _draws(frames, events, scene_ids, weights):
    index = {s: i for i, s in enumerate(scene_ids)}
    counts, elapsed = np.zeros(len(index)), np.zeros(len(index))
    known, score = np.zeros(len(index)), np.zeros(len(index))
    eligible, stale = np.zeros(len(index)), np.zeros(len(index))
    supported = {m: set() for m in METRICS}
    for r in frames:
        i = index[r['scene_id']]
        counts[i] += 1
        elapsed[i] += r['e2e_ms']
        supported['average_e2e_ms'].add(r['scene_id'])
        supported['p95_e2e_ms'].add(r['scene_id'])
    for e in events:
        i = index[e['scene_id']]
        if e['scorable']:
            known[i] += 1
            score[i] += e['score']
            supported['factual_qa_accuracy_percent'].add(e['scene_id'])
        if e['stale_eligible']:
            eligible[i] += 1
            stale[i] += bool(e['stale_answer'])
            supported['stale_answer_rate_percent'].add(e['scene_id'])
    values = dict(factual_qa_accuracy_percent=_ratio(weights @ score, weights @ known, 100.),
        stale_answer_rate_percent=_ratio(weights @ stale, weights @ eligible, 100.),
        average_e2e_ms=_ratio(weights @ elapsed, weights @ counts))
    # Resample whole scenes, preserving within-scene temporal dependence and
    # all inference repeats. Never resample individual queries or video frames.
    latency = np.array([r['e2e_ms'] for r in frames])
    scene_index = np.array([index[r['scene_id']] for r in frames], dtype=int)
    values['p95_e2e_ms'] = np.array([
        float(np.percentile(np.repeat(latency, w[scene_index]), 95))
        if frames and np.any(w[scene_index]) else np.nan for w in weights])
    boundary = dict(factual_qa_accuracy_percent=bool(known.sum() and score.sum() in (0., known.sum())),
                    stale_answer_rate_percent=bool(eligible.sum() and stale.sum() in (0., eligible.sum())))
    return values, supported, boundary


def _interval(values, clusters, boundary=False):
    valid = values[np.isfinite(values)]
    reason = ('insufficient_scene_clusters' if clusters < 2 else
              'boundary_rate_bootstrap_cannot_infer_unobserved_errors' if boundary else
              'insufficient_nonzero_denominator_draws' if len(valid) < .8 * len(values) else None)
    return dict(ci95_low=float(np.percentile(valid, 2.5)) if reason is None else None,
        ci95_high=float(np.percentile(valid, 97.5)) if reason is None else None,
        ci_status=reason or ('ok_conditional_on_nonzero_denominator' if len(valid) < len(values) else 'ok'),
        ci_valid_draws=len(valid), ci_total_draws=len(values), ci_scene_clusters=clusters,
        ci_empirically_degenerate=bool(len(valid) and np.ptp(valid) == 0))


def paired_intervals(frames_by_method, events_by_method, samples=2000, seed=20261118):
    """Use identical sampled scene multiplicities for Native and WorldStream."""
    scene_ids = sorted({r['scene_id'] for fs in frames_by_method.values() for r in fs})
    if scene_ids:
        weights = np.random.default_rng(seed).multinomial(len(scene_ids),
            np.full(len(scene_ids), 1. / len(scene_ids)), size=samples)
    else:
        weights = np.zeros((samples, 0), dtype=int)
    result, draws, supports = {}, {}, {}
    for method, frames in frames_by_method.items():
        values, supported, boundary = _draws(frames, events_by_method[method], scene_ids, weights)
        draws[method], supports[method] = values, supported
        result[method] = {}
        for metric in METRICS:
            interval = _interval(values[metric], len(supported[metric]), boundary.get(metric, False))
            result[method].update({metric + '_' + k: v for k, v in interval.items()})
    comparisons = {}
    if 'native' in draws and 'worldstream' in draws:
        for metric in METRICS:
            clusters = len(supports['native'][metric] & supports['worldstream'][metric])
            delta = draws['worldstream'][metric] - draws['native'][metric]
            comparisons[metric + '_worldstream_minus_native'] = _interval(delta, clusters)
        native_latency, world_latency = draws['native']['average_e2e_ms'], draws['worldstream']['average_e2e_ms']
        reduction = _ratio(native_latency - world_latency, native_latency, 100.)
        comparisons['e2e_latency_reduction_percent'] = _interval(reduction,
            len(supports['native']['average_e2e_ms'] & supports['worldstream']['average_e2e_ms']))
    return result, comparisons
