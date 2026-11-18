"""Fixed-duration windows and atomic semantic transitions from authoritative GT."""
from collections import Counter, defaultdict
from bisect import bisect_left
from itertools import combinations
import math
import numpy as np

from .common import EVENT_TYPES, LEVELS


def semantic_transition(before, after):
    """Count each comparable atomic fact once, including explicit set entries/exits.

    Presence means membership in the supported official annotation set. Positions
    and pair relations require visible annotated endpoints at BOTH observations.
    Attributes require nonempty authoritative labels at both observations.
    No continuous pixel/coordinate displacement is itself a semantic event.
    """
    from worldstream.m3.answer_validity import fact_directions
    old = {g['instance_id']: g for g in before}
    now = {g['instance_id']: g for g in after}
    events, valid = Counter(), Counter()
    universe, common = set(old) | set(now), set(old) & set(now)
    valid['presence'] = len(universe)
    events['appearance'] = len(set(now) - set(old))
    events['disappearance'] = len(set(old) - set(now))
    for key in common:
        a, b = old[key], now[key]
        valid['category'] += 1
        events['category'] += a['semantic_class'] != b['semantic_class']
        if a.get('attributes') and b.get('attributes'):
            valid['attribute'] += 1
            events['attribute'] += sorted(a['attributes']) != sorted(b['attributes'])
        if a['visible'] and b['visible']:
            valid['relative_position'] += 1
            events['relative_position'] += fact_directions(a['center_ego']) != fact_directions(b['center_ego'])
    visible = sorted(k for k in common if old[k]['visible'] and now[k]['visible'])
    # An unordered instance pair has one ordered direction state, oriented by ID.
    # Using every eligible pair is explicit; pair facts can dominate this atomic ratio.
    for ka, kb in combinations(visible, 2):
        def relation(objects):
            return fact_directions([objects[kb]['center_ego'][i] - objects[ka]['center_ego'][i]
                                    for i in range(3)])
        valid['spatial_relation'] += 1
        events['spatial_relation'] += relation(old) != relation(now)
    denominator, changed = sum(valid.values()), sum(events.values())
    return dict(valid_fact_transitions=denominator, changed_facts=changed,
                valid_by_type=dict(valid), events={k: int(events[k]) for k in EVENT_TYPES},
                change_rate=changed / denominator if denominator else None)


def build_windows(scenes, by, objects, seconds=5.0, max_gap=1.0):
    if seconds <= 0 or max_gap <= 0:
        raise ValueError('Window duration and transition gap must be positive')
    windows = []
    for scene in scenes:
        frames = [by[t] for t in scene['tokens']]
        timestamps = [i['timestamp'] / 1e6 for i in frames]
        origin = timestamps[0]
        cadence = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.
        bins = defaultdict(list)
        for info in frames:
            ordinal = int(math.floor((info['timestamp'] / 1e6 - origin) / seconds))
            bins[ordinal].append(info)
        predecessors = {b['token']: a for a, b in zip(frames, frames[1:])}
        for ordinal, part in sorted(bins.items()):
            start, end = origin + ordinal * seconds, origin + (ordinal + 1) * seconds
            # Coverage extends one observed median keyframe period past the last sample.
            complete = end <= timestamps[-1] + cadence + 1e-6
            changed, valid, skipped = 0, 0, 0
            counts, valid_counts = Counter(), Counter()
            for info in part:
                previous = predecessors.get(info['token'])
                if previous is None:
                    continue
                if (info['timestamp'] - previous['timestamp']) / 1e6 > max_gap:
                    skipped += 1
                    continue
                a, b = objects.get(previous['token']), objects.get(info['token'])
                if a is None or b is None:
                    skipped += 1
                    continue
                t = semantic_transition(a, b)
                changed += t['changed_facts']
                valid += t['valid_fact_transitions']
                counts.update(t['events'])
                valid_counts.update(t['valid_by_type'])
            windows.append(dict(window_id=f"{scene['scene_id']}:{ordinal:03d}", scene_id=scene['scene_id'],
                start_timestamp=start, end_timestamp=end, duration_seconds=seconds,
                complete=complete, frames=len(part), frame_ids=[i['token'] for i in part],
                valid_fact_transitions=valid, changed_facts=changed,
                change_rate=changed / valid if valid else None,
                events={k: int(counts[k]) for k in EVENT_TYPES}, valid_by_type=dict(valid_counts),
                skipped_frame_transitions=skipped))
    return windows


def fit_thresholds(windows, quantiles=(.25, .5, .75), fixed=None):
    values = [w['change_rate'] for w in windows if w['complete'] and w['change_rate'] is not None]
    if not values and fixed is None:
        raise ValueError('No complete calibration windows with comparable GT')
    thresholds = list(fixed) if fixed is not None else np.quantile(values, quantiles).tolist()
    if len(thresholds) != 3 or any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds) or thresholds != sorted(thresholds):
        raise ValueError('Expected three ordered finite change-rate thresholds in [0,1]')
    return dict(thresholds=thresholds, calibration_windows=len(values),
                calibration_rate_min=min(values) if values else None,
                calibration_rate_max=max(values) if values else None,
                quantiles=list(quantiles), source='explicit_fixed_GT_thresholds' if fixed is not None else 'calibration_GT_quantiles',
                ties='equal rates stay together in the lower bin; empty groups are retained',
                bins=['[0,q25]', '(q25,q50]', '(q50,q75]', '(q75,1]'],
                repeated_thresholds=len(set(thresholds)) < 3)


def assign_levels(windows, thresholds):
    for w in windows:
        w['stability_level'] = (LEVELS[bisect_left(thresholds, w['change_rate'])]
                                if w['complete'] and w['change_rate'] is not None else 'Unclassified')
    return windows


def distribution(windows, thresholds):
    result = []
    for i, level in enumerate((*LEVELS, 'Unclassified')):
        part = [w for w in windows if w['stability_level'] == level]
        rates = [w['change_rate'] for w in part if w['change_rate'] is not None]
        counts = Counter()
        for w in part:
            counts.update(w['events'])
        result.append(dict(stability_level=level, windows=len(part), frames=sum(w['frames'] for w in part),
            change_rate_mean=float(np.mean(rates)) if rates else None,
            change_rate_median=float(np.median(rates)) if rates else None,
            change_rate_min=min(rates) if rates else None, change_rate_max=max(rates) if rates else None,
            lower_threshold=0. if i == 0 else thresholds[i - 1] if i < 4 else None,
            upper_threshold=thresholds[i] if i < 3 else 1. if i == 3 else None,
            valid_fact_transitions=sum(w['valid_fact_transitions'] for w in part),
            changed_facts=sum(w['changed_facts'] for w in part),
            **{k + '_events': counts[k] for k in EVENT_TYPES}))
    return result
