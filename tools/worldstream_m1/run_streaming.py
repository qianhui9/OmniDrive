"""Real keyframe replay; never described as live inference or high-frequency video."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import resource
from time import perf_counter
import tracemalloc
import numpy as np
from worldstream.m1 import M1Config, FullOutputs, PerceptionAnchor, WorldAnchor, WorldStateManager
from worldstream.m1.geometry import project_anchor, region_slices
from worldstream.m1.association import associate
from .dataset import (select_infos, frame_from_info, load_oracle, prediction_outputs)
from .io_utils import read_prediction_subset, write_json
from .evaluate import (oracle_assignment, oracle_physical_valid, association_baselines, evaluate_run)
from .visualize import draw_frame


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--infos', default='data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    p.add_argument('--data-root', default='data/nuscenes')
    p.add_argument('--predictions', help='Existing full-model nuScenes results JSON, world coordinates')
    p.add_argument('--source', choices=['predictions', 'annotations'], default='predictions')
    p.add_argument('--config', default='configs/worldstream_m1.yaml')
    p.add_argument('--scene')
    p.add_argument('--scene-index', type=int, default=0)
    p.add_argument('--frames', type=int, default=12)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--max-width', type=int, default=640)
    p.add_argument('--output', default='outputs/module1/replay')
    p.add_argument('--experimental-appearance', action='store_true',
                   help='Explicit ablation: do not require independent visibility; not conservative certification')
    p.add_argument('--descriptor', choices=['normalized_gray', 'gradient'])
    p.add_argument('--split', choices=['calibration', 'evaluation'], default='evaluation')
    return p


def main():
    args = args_parser().parse_args()
    if args.split == 'calibration' and not Path(args.infos).name.endswith('_train.pkl'):
        raise SystemExit('Calibration requires train infos; evaluation/test scenes must not tune thresholds')
    np.random.seed(0)
    config = M1Config.load(args.config)
    if args.descriptor: config = replace(config, descriptor=args.descriptor)
    if args.experimental_appearance:
        config = replace(config, require_visibility_evidence=False)
    if args.max_width < 32: raise ValueError('max-width must be >=32')
    if args.source == 'predictions' and not args.predictions:
        raise SystemExit('--predictions is required; use --source annotations only for explicit annotation-assisted tests')
    root = Path(args.output)
    # Do not append trials from multiple runs or silently overwrite a completed run.
    if (root/'state.jsonl').exists(): raise SystemExit('Output already contains a run; choose a new --output')
    root.mkdir(parents=True, exist_ok=True)
    infos = select_infos(args.infos, args.scene, args.frames, args.stride, args.scene_index)
    if config.calibration_path and args.split == 'evaluation':
        calibration = json.loads(Path(config.calibration_path).read_text())
        if infos[0]['scene_token'] in calibration.get('scene_ids', []):
            raise ValueError('Calibration and evaluation scenes overlap')
    print('Loading offline oracle for', len(infos), 'real keyframes', flush=True)
    oracle = load_oracle(infos, args.data_root)
    predictions = read_prediction_subset(args.predictions, {i['token'] for i in infos}) if args.predictions else {}
    if args.source == 'predictions' and any(i['token'] not in predictions for i in infos):
        raise ValueError('Selected sample(s) missing from supplied model outputs')
    fingerprint = {'path': str(args.predictions), 'size': Path(args.predictions).stat().st_size} if args.predictions else {'source': 'annotations'}
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    provenance = {'args': vars(args), 'config': config.to_dict(), 'seed': 0,
        'scene_id': infos[0]['scene_token'], 'samples': [i['token'] for i in infos],
        'timestamps_s': [i['timestamp']/1e6 for i in infos],
        'intervals_s': np.diff([i['timestamp']/1e6 for i in infos]).tolist(),
        'sampling': 'nuScenes annotated keyframes; actual intervals reported, not high-frequency video',
        'source': fingerprint, 'new_gpu_baseline_run': False, 'gpu_used': False,
        'execution_device': 'cpu', 'cuda_probe_scope': 'not probed by this CPU replay',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'annotation_use': 'Offline oracle only; annotation source mode also explicitly uses annotations in commit',
        'oracle_annotation_counts': {key: len(value) for key, value in oracle.items()},
        'info_gt_box_counts': {i['token']: len(i.get('gt_boxes', [])) for i in infos},
        'oracle_identity_source': 'sample_annotation.json sample_token reverse links, not potentially filtered sample.json anns',
        'feature_reuse_or_speedup_claim': False}
    write_json(root/'manifest.json', provenance)
    manager = WorldStateManager(config)
    identities = {}
    previous_visible_gt = {}
    timers, oracle_rows, assoc_stats = [], [], {}
    method_tracks, last_gt_tracks, next_tracks = {}, {}, {}
    novelty_counts = {'new_gt_regions': 0, 'covered_new_gt_regions': 0, 'background_pixels': 0,
                      'invalidated_background_pixels': 0, 'scope': 'annotated classes only; invalidation is an update request'}
    tracemalloc.start()
    for index, info in enumerate(infos):
        frame = frame_from_info(info, args.data_root, args.max_width, signature)
        start = perf_counter()
        result = manager.precheck(frame)  # Only historical state, current RGB/calibration. No oracle argument.
        old_anchors = list(manager.store.anchors.values())
        current_anns = oracle[frame.frame_id]
        # Everything below is an offline evaluator or a post-inference replay commit.
        for aid, check in result.diagnostics.get('anchors', {}).items():
            anchor = manager.store.anchors[aid]
            views = check['cameras'].values()
            eligible = [v for v in views if v['geometry_valid']]
            geometry_only = bool(check['within_horizon'] and check['motion_known'] and eligible)
            residual = max((v['residual'] for v in eligible), default=float('inf'))
            appearances, exposure_residuals = [], []
            appearance_start = perf_counter()
            for cid, support in anchor.observation_support.items():
                if cid not in frame.cameras: continue
                cam = frame.cameras[cid]
                desc = manager.validator.descriptor(cam.image, support.region)
                r = manager.validator.residual(support.descriptor, desc)
                # Measured illumination robustness diagnostic on real current crops.
                illuminated = np.clip(cam.image.astype(np.float32)*0.7+25, 0, 255).astype(np.uint8)
                exposure_residuals.append(manager.validator.residual(desc, manager.validator.descriptor(illuminated, support.region)))
                appearances.append(r)
                check['cameras'][cid]['historical_region'] = support.region
            appearance_ms = (perf_counter()-appearance_start)*1000
            oracle_valid = oracle_physical_valid(anchor, current_anns, identities.get(aid), frame.timestamp)
            oracle_rows.append({'scene_id': frame.scene_id, 'frame_id': frame.frame_id, 'anchor_id': aid,
                'age_s': check['age_s'], 'geometry_only': geometry_only,
                'appearance_only': bool(appearances and max(appearances) <= config.residual_threshold),
                'geometry_observation': check['valid'], 'residual': residual,
                'oracle_valid': oracle_valid, 'descriptor': config.descriptor,
                'descriptor_size': config.descriptor_size,
                'illumination_residual': float(np.mean(exposure_residuals)) if exposure_residuals else None,
                'appearance_ablation_ms': appearance_ms})
        if args.source == 'predictions':
            outputs = prediction_outputs(predictions[frame.frame_id], config)
        else:
            outputs = FullOutputs([PerceptionAnchor('object', a['class'], a['geometry'])
                                   for a in current_anns if a['has_sensor_support']], source='annotation_assisted_commit')
        # Temporary evaluator anchors, never handed to precheck.
        current = [WorldAnchor(str(j), frame.scene_id, p.anchor_type, p.semantic_class,
                               p.geometry_world, frame.timestamp, velocity_world=p.velocity_world,
                               velocity_reliable=p.velocity_reliable) for j, p in enumerate(outputs.anchors)]
        current_ids = oracle_assignment(current, current_anns)
        for name in ('fixed_image', 'appearance', 'world'):
            assoc_start = perf_counter()
            if name == 'world':
                pairs, _ = associate(old_anchors, outputs.anchors, frame.timestamp, config)
            else:
                pairs = association_baselines(old_anchors, current, frame, manager.validator, config, modes=(name,))[name]
            elapsed = (perf_counter()-assoc_start)*1000
            stats = assoc_stats.setdefault(name, {'correct': 0, 'predicted': 0, 'scorable': 0, 'possible': 0,
                                                   'id_switches': 0, 'runtime_ms': [], 'reprojection_error_px': []})
            old_ids = {i: identities.get(a.anchor_id) for i, a in enumerate(old_anchors)}
            possible = set(x for x in old_ids.values() if x) & set(current_ids.values())
            stats['possible'] += len(possible); stats['predicted'] += len(pairs)
            tracks = {}
            previous_tracks = method_tracks.get(name, {})
            next_track = next_tracks.get(name, 0)
            previous_gt = last_gt_tracks.setdefault(name, {})
            for i, j in pairs:
                identity = current_ids.get(j)
                scorable = bool(identity and old_ids.get(i))
                stats['scorable'] += int(scorable)
                correct = bool(scorable and old_ids.get(i) == identity)
                stats['correct'] += int(correct)
                # Reprojection error only for identity-confirmed links, same camera.
                if correct:
                    for cid, cam in frame.cameras.items():
                        a, b = project_anchor(old_anchors[i], cam, config), project_anchor(current[j], cam, config)
                        if a.region and b.region:
                            ca = np.array(a.region).reshape(2, 2).mean(0)
                            cb = np.array(b.region).reshape(2, 2).mean(0)
                            stats['reprojection_error_px'].append(float(np.linalg.norm(ca-cb)))
                # Track state follows GT-visible prediction identities across adjacent replay frames.
                previous_identity = old_ids.get(i)
                tracks[j] = previous_tracks.get(previous_identity) if previous_identity else None
            for j in range(len(current)):
                if tracks.get(j) is None:
                    next_track += 1; tracks[j] = next_track
                identity = current_ids.get(j)
                if identity:
                    if index and identity in previous_gt and previous_gt[identity] != tracks[j]:
                        stats['id_switches'] += 1
            method_tracks[name] = {current_ids[j]: tracks[j] for j in current_ids}
            last_gt_tracks[name].update(method_tracks[name]); next_tracks[name] = next_track
            stats['runtime_ms'].append(elapsed)
        # New GT regions are an offline coverage check, not online semantic novelty detection.
        current_regions = {cid: [] for cid in frame.cameras}
        for cid, cam in frame.cameras.items():
            annotated_mask = np.zeros(cam.image.shape[:2], bool)
            visible_gt = set()
            for j, anchor in enumerate(current):
                p = project_anchor(anchor, cam, config)
                if p.region:
                    current_regions[cid].append(p.region)
            for ann in current_anns:
                anchor = WorldAnchor('oracle', frame.scene_id, 'object', ann['class'], ann['geometry'], frame.timestamp)
                p = project_anchor(anchor, cam, config)
                if not p.valid: continue
                visible_gt.add(ann['instance_id'])
                slices = region_slices(p.region, cam.image.shape)
                annotated_mask[slices] = True
                if index and ann['instance_id'] not in previous_visible_gt.get(cid, set()):
                    novelty_counts['new_gt_regions'] += 1
                    novelty_counts['covered_new_gt_regions'] += int(manager.last_masks[cid][slices].mean() >= .9)
            if index:
                background = ~annotated_mask
                novelty_counts['background_pixels'] += int(background.sum())
                novelty_counts['invalidated_background_pixels'] += int((background & manager.last_masks[cid]).sum())
            previous_visible_gt[cid] = visible_gt
        manager.commit(frame, outputs, result)
        by_index = oracle_assignment(current, current_anns)
        committed = manager.last_commit_diagnostics['committed_anchor_ids']
        indices = manager.last_commit_diagnostics.get('selected_prediction_indices', range(len(current)))
        for j, aid in zip(indices, committed):
            identities[aid] = by_index.get(j)
        total_ms = result.diagnostics['timings_ms']['precheck_total'] + manager.last_commit_diagnostics['commit_total_ms']
        timing = dict(result.diagnostics['timings_ms'], **manager.last_commit_diagnostics)
        timing['m1_total_ms'] = total_ms
        timing['unknown_uncovered_area_ratio'] = float(np.mean([r['unknown_uncovered_area_ratio']
                                                               for r in result.invalidation_regions.values()]))
        timers.append(timing)
        with (root/'state.jsonl').open('a') as handle:
            from worldstream.m1.state_types import json_value
            handle.write(json.dumps(json_value({'precheck': result.to_dict(), 'commit': manager.last_commit_diagnostics}), allow_nan=False)+'\n')
        (root/'masks').mkdir(exist_ok=True)
        np.savez_compressed(root/'masks'/('{:03d}.npz'.format(index)), **manager.last_masks)
        if config.visualization:
            draw_frame(frame, result, root/'visualizations'/('{:03d}.jpg'.format(index)), current_regions)
        print('frame', index, 'anchors', len(manager.store.anchors), 'valid', len(result.valid_anchor_ids),
              'M1 ms', round(total_ms, 2), flush=True)
    peak_traced = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    with (root/'oracle_records.jsonl').open('w') as handle:
        from worldstream.m1.state_types import json_value
        for row in oracle_rows: handle.write(json.dumps(json_value(row), allow_nan=False)+'\n')
    association = {}
    for name, stats in assoc_stats.items():
        association[name] = {'precision': stats['correct']/stats['scorable'] if stats['scorable'] else None,
            'recall': stats['correct']/stats['possible'] if stats['possible'] else None,
            'id_switches': stats['id_switches'], 'correct': stats['correct'], 'predicted': stats['predicted'],
            'scorable': stats['scorable'], 'unscorable': stats['predicted']-stats['scorable'],
            'possible': stats['possible'], 'mean_runtime_ms': float(np.mean(stats['runtime_ms'])),
            'mean_reprojection_error_px': float(np.mean(stats['reprojection_error_px'])) if stats['reprojection_error_px'] else None}
    write_json(root/'association_metrics.json', {'methods': association,
        'scope': 'Prediction-to-GT Hungarian <=2m, same class; precision excludes unscorable IDs and reports their count. '
                 'ID switches on observed GT instance track assignment; not official nuScenes tracking score.'})
    novelty_counts['new_region_recall'] = novelty_counts['covered_new_gt_regions']/novelty_counts['new_gt_regions'] if novelty_counts['new_gt_regions'] else None
    novelty_counts['uncovered_background_update_ratio'] = novelty_counts['invalidated_background_pixels']/novelty_counts['background_pixels'] if novelty_counts['background_pixels'] else None
    novelty_counts['semantic_false_positive_rate'] = None
    novelty_counts['limitation'] = 'Unannotated background is unknown, not a negative label. '
    novelty_counts['limitation'] += 'Semantic novelty FPR unavailable; update ratio reported instead.'
    write_json(root/'novelty_metrics.json', novelty_counts)
    write_json(root/'timings.json', {'frames': timers, 'mean_m1_total_ms': float(np.mean([t['m1_total_ms'] for t in timers])),
        'peak_tracemalloc_bytes_after_input_tables': peak_traced,
        'process_peak_rss_kib_including_input_tables': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'retained_tensor_bytes': manager.store.tensor_bytes,
        'gpu_time_ms': None, 'gpu_memory_bytes': None,
        'timing_scope': 'CPU wall time; excludes input IO, visualization, and offline oracle/baseline evaluation; no GPU work in M1'})
    print(json.dumps(evaluate_run(root), indent=2))


if __name__ == '__main__': main()
