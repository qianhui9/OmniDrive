"""Freeze annotation/availability-only scene selection before any inference."""
from collections import defaultdict
import json
from pathlib import Path
import pickle
import statistics


def choose_scenes(candidates, originals, target, per_log=3):
    """Keep all original scenes; cover dense and geometrically complex scenes.

    Short logs are exhausted, never duplicated. Remaining slots are allocated
    to the least represented eligible logs, with deterministic feature ranking.
    """
    by_id = {s['scene_id']: s for s in candidates}
    if len(by_id) != len(candidates) or len(set(originals)) != len(originals):
        raise ValueError('Duplicate scene IDs')
    if target < len(originals) or target > len(candidates):
        raise ValueError(f'Cannot select {target} unique complete scenes from {len(candidates)} candidates')
    missing = set(originals)-by_id.keys()
    if missing:
        raise ValueError('Original scenes no longer eligible: ' + repr(sorted(missing)))
    groups = defaultdict(list)
    for s in candidates:
        groups[s['log_id']].append(s)
    if set(groups) != {by_id[s]['log_id'] for s in originals}:
        raise ValueError('Candidates must stay within original logs')
    selected = dict.fromkeys(originals)
    reasons = {s: 'original_main_scene' for s in originals}

    def pick(log):
        remaining = [s for s in groups[log] if s['scene_id'] not in selected]
        current = [by_id[s] for s in selected if by_id[s]['log_id'] == log]
        # Rank within log so object counts and map primitive counts have equal
        # influence despite different units. Favor tail coverage and diversity.
        ranked = {}
        for feature in ('object_mean', 'map_mean'):
            values = sorted({s['selection_features'][feature] for s in groups[log]})
            ranked[feature] = {v: i/max(1, len(values)-1) for i, v in enumerate(values)}
        def vector(s):
            f = s['selection_features']
            return [ranked[k][f[k]] for k in ('object_mean', 'map_mean')] + [int(f[k]) for k in ('night', 'rain')]
        def score(s):
            x = vector(s)
            novelty = min(sum((a-b)**2 for a, b in zip(x, vector(t))) for t in current)
            return novelty + .25*(x[0]+x[1])
        return sorted(remaining, key=lambda s: (-score(s), s['scene_id']))[0]

    for log in sorted(groups):
        while sum(by_id[s]['log_id'] == log for s in selected) < min(per_log, len(groups[log])) and len(selected) < target:
            s = pick(log)
            selected[s['scene_id']] = None
            reasons[s['scene_id']] = 'within_log_annotation_diversity'
    while len(selected) < target:
        eligible = [log for log in groups if any(s['scene_id'] not in selected for s in groups[log])]
        log = min(eligible, key=lambda log: (sum(by_id[s]['log_id'] == log for s in selected), log))
        s = pick(log)
        selected[s['scene_id']] = None
        reasons[s['scene_id']] = 'short_log_slot_redistribution'
    return [dict(by_id[s], selection_reason=reasons[s]) for s in selected]


def build_selection(config):
    original = json.loads(Path(config['scene_manifest']).read_text())['selected']
    logs = {s['log_id'] for s in original}
    official = {s['token']: s for s in json.loads(Path(config['scene_metadata']).read_text())}
    with Path(config['infos']).open('rb') as stream:
        infos = pickle.load(stream)['infos']
    groups = defaultdict(list)
    for info in infos:
        if official[info['scene_token']]['log_token'] in logs:
            groups[info['scene_token']].append(info)
    eligible, rejected = [], []
    files = set()
    for sid, frames in sorted(groups.items()):
        scene = official[sid]
        frames.sort(key=lambda f: (f['timestamp'], f['token']))
        complete = (len(frames) == scene['nbr_samples'] and frames[0]['token'] == scene['first_sample_token']
                    and frames[-1]['token'] == scene['last_sample_token']
                    and all(a['next'] == b['token'] and b['prev'] == a['token'] and a['timestamp'] < b['timestamp']
                            for a, b in zip(frames, frames[1:])))
        reason = None if complete else 'incomplete_official_scene'
        scene_files = set()
        for f in frames:
            if len(f['cams']) != 6:
                reason = 'camera_count'
            for camera in f['cams'].values():
                path = Path(camera['data_path'])
                if not path.is_file():
                    reason = 'missing_camera_file'
                scene_files.add(str(path))
            qa = Path(config.get('data_root', 'data/nuscenes'))/'conv'/'val'/(f['token']+'.json')
            if not qa.is_file():
                reason = 'missing_query_file'
            elif len(json.loads(qa.read_text())) < 4:
                reason = 'incomplete_natural_queries'
            scene_files.add(str(qa))
        if reason:
            rejected.append(dict(scene_id=sid, reason=reason))
            continue
        description = scene['description'].lower()
        object_counts = [len(f['gt_boxes']) for f in frames]
        # Count actual map primitives from the supplied validation metadata.
        # This is a geometric complexity proxy, not a perception score.
        map_counts = [sum(len(v) for v in f.get('map_geoms', {}).values()) for f in frames]
        eligible.append(dict(scene_id=sid, log_id=scene['log_token'], tokens=[f['token'] for f in frames],
            description=scene['description'], timestamps=[f['timestamp']/1e6 for f in frames],
            selection_features=dict(object_mean=statistics.mean(object_counts), object_max=max(object_counts),
                map_mean=statistics.mean(map_counts), map_max=max(map_counts),
                night='night' in description, rain='rain' in description, location=frames[0]['location'])))
        files.update(scene_files)
    selected = choose_scenes(eligible, [s['scene_id'] for s in original],
                             config['target_scenes'], config.get('scenes_per_log', 3))
    counts = {log: sum(s['log_id'] == log for s in selected) for log in sorted(logs)}
    audit = dict(rule='Retain originals; within-log max-min rank diversity in object/map counts and night/rain; '
                     'redistribute short-log slots to least represented eligible logs; lexicographic tie breaking',
        selection_uses_model_outputs=False, target_scenes=config['target_scenes'], selected=selected,
        selected_log_counts=counts, original_scenes=[s['scene_id'] for s in original],
        eligible_features=[{k: v for k, v in s.items() if k not in ('tokens', 'timestamps')} for s in eligible],
        rejected=rejected, short_logs={log: sum(s['log_id'] == log for s in eligible) for log in logs
                                     if sum(s['log_id'] == log for s in eligible) < config.get('scenes_per_log', 3)})
    selected_tokens = {t for s in selected for t in s['tokens']}
    selected_files = {str(Path(c['data_path'])) for f in infos if f['token'] in selected_tokens for c in f['cams'].values()}
    selected_files.update(str(Path(config.get('data_root', 'data/nuscenes'))/'conv'/'val'/(t+'.json')) for t in selected_tokens)
    return selected, audit, sorted(selected_files)
