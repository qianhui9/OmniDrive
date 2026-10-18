"""Frozen M1-only real-stream evaluation; oracle labels never enter online state.

Run with python -m tools.worldstream_m1.independent_refinement. The protocol is
written before predictions/annotations are read. No test-time threshold tuning.
"""
import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import fields
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
from time import perf_counter
import types

import cv2
import numpy as np
from scipy.stats import beta
from worldstream.m1 import M1Config, WorldStateManager, WorldAnchor
from worldstream.m1.association import legal_assignment, association_ambiguities
from worldstream.m1.geometry import region_slices, project_anchor
from worldstream.m1.mask_codec import decode_mask
from .coverage_oracle import prepare_pair, world_lidar, score_camera
from .dataset import frame_from_info, prediction_outputs, load_oracle
from .io_utils import read_prediction_subset, write_json
from .refinement_common import physical_details, summarize_rows, ratio


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(count=len(values), p50=float(np.median(values)) if len(values) else None,
                p95=float(np.percentile(values, 95)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def inject_real_frame_faults(root, infos, predictions, config):
    """Real RGB/calibration and model replay, with controlled lifecycle faults."""
    from unittest.mock import patch
    from worldstream.m1.state_store import StateStore
    from worldstream.m1 import ComputedFeature
    base = frame_from_info(infos[0], 'data/nuscenes')
    following = frame_from_info(infos[1], 'data/nuscenes')
    future = frame_from_info(infos[2], 'data/nuscenes')
    first_output = prediction_outputs(predictions[base.frame_id], config)
    next_output = prediction_outputs(predictions[following.frame_id], config)
    cases = []
    for scenario in ('model_exception', 'late_commit_exception', 'precheck_exception', 'duplicate_commit',
                     'modified_frame', 'expired_pending', 'scene_change', 'timestamp_reverse', 'repeated_frame',
                     'long_gap', 'signature_change', 'invalid_pose', 'missing_camera', 'camera_reorder'):
        m = WorldStateManager(config); m.initialize(base, first_output)
        old = pickle.dumps(m.store.__dict__)
        frame = copy.deepcopy(following); expected_reset = False; expected_reject = False
        if scenario == 'scene_change': frame.scene_id += '_fault'; expected_reset = True
        if scenario == 'timestamp_reverse': frame.timestamp = base.timestamp - 1; expected_reset = True
        if scenario == 'repeated_frame': frame = copy.deepcopy(base); expected_reset = True
        if scenario == 'long_gap': frame.timestamp = base.timestamp + config.max_frame_gap_s + 1; expected_reset = True
        if scenario == 'signature_change': frame.execution_signature += '_fault'; expected_reset = True
        if scenario == 'invalid_pose': frame.ego_to_world[0,0] = np.nan; expected_reset = True
        if scenario == 'missing_camera': frame.cameras.pop(next(iter(frame.cameras))); expected_reset = True
        if scenario == 'camera_reorder': frame.cameras = dict(reversed(list(frame.cameras.items()))); expected_reset = True
        if scenario == 'precheck_exception':
            with patch.object(m.validator, 'validate', side_effect=RuntimeError('injected validation failure')):
                try: m.precheck(frame)
                except RuntimeError: expected_reject = True
            passed = expected_reject and old == pickle.dumps(m.store.__dict__)
        else:
            r = m.precheck(frame)
            if expected_reset:
                passed = bool(r.full_refresh_hint and not m.store.anchors and not r.physically_certified_anchor_ids)
            elif scenario == 'model_exception':
                m.abort_pending(); passed = old == pickle.dumps(m.store.__dict__)
            else:
                if scenario == 'modified_frame': frame.cameras[next(iter(frame.cameras))].image[0,0] ^= 255
                if scenario == 'expired_pending': m._pending_started -= config.pending_timeout_s + 1
                if scenario == 'late_commit_exception':
                    put = StateStore.put_feature; calls = []
                    def fail_late(store, feature, current):
                        calls.append(feature.feature_id)
                        if len(calls) >= 2: raise RuntimeError('injected after first feature write')
                        return put(store, feature, current)
                    extra = ComputedFeature('injected', 'carrier', (1,), 'float32', 'cpu', 'fault_injection')
                    with patch.object(StateStore, 'put_feature', fail_late):
                        try: m.commit(frame, type(next_output)(next_output.anchors, [extra], next_output.source), r)
                        except RuntimeError: expected_reject = True
                else:
                    try:
                        m.commit(frame, next_output, r)
                        if scenario == 'duplicate_commit':
                            old = pickle.dumps(m.store.__dict__)
                            m.commit(frame, next_output, r)
                    except ValueError: expected_reject = True
                passed = expected_reject and old == pickle.dumps(m.store.__dict__)
            # Prove a future valid frame can start a new conservative transaction.
            recovery = m.precheck(future)
            m.commit(future, prediction_outputs(predictions[future.frame_id], config), recovery)
            passed = passed and m.store.last_timestamp == future.timestamp
        cases.append(dict(scenario=scenario, passed=bool(passed), controlled_fault=True,
                          source_frame=base.frame_id, current_frame=following.frame_id))
    write_json(root/'transaction_correctness.json', dict(scope='Real RGB/poses + actual historical model predictions + controlled injected faults',
        all_passed=all(c['passed'] for c in cases), cases=cases))
    return cases


def independent_source_assignment(predictions, annotations):
    """OFFLINE unique instance labels; ambiguous and out-of-gate samples unscorable.

    The 2m gate censors localization errors. Never use these residuals as a
    population calibration bound. This function is not imported by runtime M1.
    """
    costs = np.full((len(predictions), len(annotations)), np.inf)
    for i, pred in enumerate(predictions):
        for j, ann in enumerate(annotations):
            if pred.semantic_class == ann['class'] and ann['has_sensor_support']:
                d = np.linalg.norm(pred.geometry_world.center - ann['geometry'].center)
                if d <= 2.:
                    costs[i, j] = d
    pairs = legal_assignment(costs)
    ambiguity = association_ambiguities(pairs, costs, .25)
    return {i: annotations[j] for i, j in pairs if not ambiguity[i, j]['ambiguous']}


def baseline_manager(root, config):
    namespace = types.ModuleType('independent_baseline_m1')
    namespace.__path__ = [str(root/'baseline_source/worldstream/m1')]
    sys.modules[namespace.__name__] = namespace
    cfg_type = importlib.import_module(namespace.__name__ + '.config').M1Config
    cls = importlib.import_module(namespace.__name__ + '.state_manager').WorldStateManager
    values = config.to_dict()
    return cls(cfg_type(**{f.name: values[f.name] for f in fields(cfg_type)}))


def render_case(path, frame, result, previous, associations, source_truth, current_truth, masks):
    """Real RGB with historical/current geometry; oracle text stays offline."""
    panels = []
    for cid, camera in frame.cameras.items():
        rgb = camera.image.copy()
        for aid, check in result.diagnostics['anchors'].items():
            r = check['cameras'][cid]['predicted_region']
            if r is None: continue
            known = source_truth.get(aid)
            match = next((j for old, j in associations if old == aid), None)
            target = current_truth.get(match)
            switch = known is not None and target is not None and known['instance_id'] != target['instance_id']
            color = (255, 30, 30) if switch else (50, 240, 70) if match is not None else (255, 180, 30)
            p, q = tuple(map(int, r[:2])), tuple(map(int, r[2:]))
            cv2.rectangle(rgb, p, q, color, 1)
            cv2.putText(rgb, aid.split(':')[-1] + (' ID SWITCH' if switch else ' candidate'), p,
                        cv2.FONT_HERSHEY_SIMPLEX, .3, color, 1)
        meta = result.invalidation_regions[cid]
        stable = decode_mask(meta['observation_stable_mask'])
        changed = decode_mask(meta['observed_changed_mask'])
        unknown = decode_mask(meta['unknown_mask'])
        partition = np.zeros_like(rgb); partition[stable] = (30, 220, 60)
        partition[changed] = (240, 40, 40); partition[unknown] = (100, 100, 130)
        overlay = ((rgb.astype(float) * .6) + partition * .4).astype(np.uint8)
        refresh = rgb.copy(); refresh[masks[cid]] = refresh[masks[cid]] // 3
        row = np.concatenate([rgb, overlay, refresh], axis=1)
        cv2.putText(row, cid + ' | projected candidate (foreground unverified) | stable/changed/unknown | refresh dark',
                    (8, 16), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        panels.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(panels, axis=0), cv2.COLOR_RGB2BGR))


def run_scene(root, infos, predictions, oracle, config, split, baseline=False):
    m = baseline_manager(root, config) if baseline else WorldStateManager(config)
    truth = {}; physical = []; candidate_rows = []; pixel_rows = []; events = []; timing = []; sizes = []
    observations = []; source_errors = []; geometry_rows = []; track_ids = defaultdict(set)
    previous = None; prev_info = None; previous_points = None; checks = Counter(); eligibility = 0
    scene = infos[0]['scene_token']; path = root/'traces'/split/scene/('baseline' if baseline else 'current')
    path.mkdir(parents=True, exist_ok=True)
    for n, info in enumerate(infos):
        frame = frame_from_info(info, 'data/nuscenes')
        before = pickle.dumps(m.store.__dict__)
        wall = perf_counter(); r = m.precheck(frame); pre_ms = (perf_counter() - wall) * 1000
        if not baseline:
            checks['precheck_isolation_checks'] += 1
            checks['precheck_isolation_failures'] += int(before != pickle.dumps(m.store.__dict__) and not r.diagnostics['reset_reason'])
        # Oracle access is AFTER online precheck, never included in frame/outputs.
        anns = oracle[frame.frame_id]
        for aid, check in r.diagnostics['anchors'].items():
            a = m.store.anchors[aid]
            ann = truth.get(aid)
            detail = physical_details(a, anns, ann['instance_id'] if ann else None, ann, frame.timestamp)
            physical.append(dict(scene=scene, frame=frame.frame_id, anchor=aid, accepted=check['valid'],
                certified=aid in getattr(r, 'physically_certified_anchor_ids', []), **detail))
            if not baseline:
                proof = check['physical_certification']
                observations.append(dict(scene=scene, frame=frame.frame_id, anchor=aid,
                    observed=check['valid'], source_frame=a.source_frame_id,
                    source_epoch=a.source_state_epoch, proof=proof,
                    reasons=check['reasons']))
                if previous is not None and a.semantic_class in ('barrier', 'traffic_cone'):
                    from worldstream.m1.bidirectional_geometry import verify_anchor_geometry
                    for cid, camera in frame.cameras.items():
                        p = project_anchor(a, camera, config)
                        if p.valid:
                            proof3d = verify_anchor_geometry(a, m._previous_cameras.get(cid), camera,
                                m.last_evidence.get(cid), p, config)
                            geometry_rows.append(dict(scene=scene, frame=frame.frame_id, anchor=aid, camera=cid,
                                **{k: v for k, v in proof3d.items() if k != 'points'}))
        points = world_lidar(info) if not baseline else None
        reference = prepare_pair(previous, frame, previous_points, points,
            oracle[previous.frame_id], anns, config) if previous is not None and not baseline else None
        for cid, camera in frame.cameras.items():
            meta = r.invalidation_regions[cid]; invalid = m.last_masks[cid]
            if not baseline:
                stable, changed, unknown = [decode_mask(meta[key]) for key in
                    ('observation_stable_mask', 'observed_changed_mask', 'unknown_mask')]
                checks['camera_mask_checks'] += 1
                checks['mask_partition_failures'] += int(not np.all(stable.astype(int) + changed.astype(int) + unknown.astype(int) == 1))
                checks['mask_encoding_failures'] += int(not np.array_equal(invalid, decode_mask(meta['refresh_required_mask'])))
                checks['refresh_ratio_failures'] += int(meta['invalidation_area_ratio'] != float(invalid.mean()))
                envelope = np.zeros_like(invalid)
                for region in meta['regions']: envelope[region_slices(region, invalid.shape)] = True
                checks['tile_envelope_failures'] += int(bool((invalid & ~envelope).any()))
            row = dict(scene=scene, frame=frame.frame_id, camera=cid, pixels=invalid.size,
                       stable=meta['verified_stable_area_ratio'], changed=meta['changed_area_ratio'],
                       unknown=meta['unknown_area_ratio'], refresh=meta['invalidation_area_ratio'])
            if reference:
                # Unannotated background is UNKNOWN for scoring. Preserve the old
                # lidar reference computation, restrict its denominator explicitly.
                ref = reference[cid]
                ref['scorable'] &= ref['annotated']
                score = score_camera(ref, invalid, m.last_evidence.get(cid), config)
                events.extend(dict(scene=scene, frame=frame.frame_id, camera=cid, **e) for e in score.pop('events'))
                row.update(score)
            pixel_rows.append(row)
        wall = perf_counter(); output = prediction_outputs(predictions[frame.frame_id], config)
        adapter_ms = (perf_counter() - wall) * 1000
        current = independent_source_assignment(output.anchors, anns)
        old_truth = dict(truth)
        wall = perf_counter(); m.commit(frame, output, r); commit_ms = (perf_counter() - wall) * 1000
        diag = m.last_commit_diagnostics
        selected = diag.get('selected_prediction_indices', list(range(min(len(output.anchors), config.max_anchors))))
        selected_truth_ids = {current[j]['instance_id'] for j in selected if j in current}
        eligibility += len(selected_truth_ids & {v['instance_id'] for aid, v in old_truth.items()
                              if aid in r.diagnostics['anchors']})
        proposed = diag.get('geometric_candidate_pairs', diag['association_pairs'])
        kept = {tuple(p) for p in diag['association_pairs']}
        for aid, j in proposed:
            old, new = old_truth.get(aid), current.get(j)
            candidate_rows.append(dict(scene=scene, frame=frame.frame_id, anchor=aid, prediction=j,
                scorable=old is not None and new is not None,
                correct=(old['instance_id'] == new['instance_id']) if old is not None and new is not None else None,
                continued=(aid, j) in kept,
                source_instance=old['instance_id'] if old else None,
                current_instance=new['instance_id'] if new else None))
        for j, aid in zip(selected, diag['committed_anchor_ids']):
            if j in current:
                truth[aid] = current[j]
                error = float(np.linalg.norm(output.anchors[j].geometry_world.center - current[j]['geometry'].center))
                source_errors.append(error); track_ids[current[j]['instance_id']].add(aid)
            else:
                truth.pop(aid, None)
        truth = {aid: ann for aid, ann in truth.items() if aid in m.store.anchors}
        timing.append(dict(scene=scene, frame=frame.frame_id, precheck_ms=pre_ms, commit_ms=commit_ms,
            prediction_adapter_ms=adapter_ms, total_cpu_ms=pre_ms+commit_ms+adapter_ms,
            **r.diagnostics['timings_ms'], association_ms=diag['association_ms']))
        sizes.append(dict(frame=frame.frame_id, anchors=len(m.store.anchors), features=len(m.store.features),
            tensor_bytes=m.store.tensor_bytes, metadata_pickle_bytes=len(pickle.dumps(m.store.__dict__)),
            history_entries=sum(len(getattr(a, 'association_history', [])) for a in m.store.anchors.values())))
        if not baseline:
            checks['source_checks'] += len(m.store.anchors)
            checks['source_failures'] += sum(a.source_state_epoch != m.store.generation or
                a.source_execution_signature != frame.execution_signature or a.source_timestamp > frame.timestamp or
                a.certified_observation_count != 0 for a in m.store.anchors.values())
            if n in (1, 5, len(infos)-1):
                render_case(root/'visualizations'/split/(scene+f'_{n:03d}.jpg'), frame, r, previous,
                    diag['association_pairs'], old_truth, current, m.last_masks)
                np.savez_compressed(path/f'masks_{n:03d}.npz', **m.last_masks)
            with (path/'state.jsonl').open('a') as handle:
                from worldstream.m1.state_types import json_value
                handle.write(json.dumps(json_value(dict(precheck=r.to_dict(), commit=diag)), allow_nan=False)+'\n')
        previous, prev_info, previous_points = frame, info, points
    data = dict(physical=physical, associations=candidate_rows, pixels=pixel_rows, events=events,
        timing=timing, sizes=sizes, checks=dict(checks), source_errors=source_errors,
        geometry=geometry_rows, observations=observations, eligible_association_links=eligibility,
        conditional_track_fragmentation=sum(max(0, len(v)-1) for v in track_ids.values()))
    write_json(path/'metrics.json', data)
    print(split, scene, 'baseline' if baseline else 'current', 'frames', len(infos), flush=True)
    return data


def aggregate(runs):
    combined = {key: sum((r[key] for r in runs), []) for key in
                ('physical', 'associations', 'pixels', 'events', 'timing', 'sizes', 'source_errors', 'geometry', 'observations')}
    counts = Counter()
    for r in runs: counts.update(r['checks'])
    combined['checks'] = dict(counts)
    combined['eligible_association_links'] = sum(r['eligible_association_links'] for r in runs)
    combined['conditional_track_fragmentation'] = sum(r['conditional_track_fragmentation'] for r in runs)
    return combined


def summarize(data):
    proposed = data['associations']; kept = [r for r in proposed if r['continued']]
    scored = [r for r in kept if r['scorable']]
    pixels = data['pixels']; events = data['events']
    accepted = sum(p.get('accepted_points', 0) for p in pixels)
    errors = sum(p.get('false_points', 0) for p in pixels)
    association = dict(proposed=len(proposed), continued=len(kept), scorable=len(scored),
        unscorable=len(kept)-len(scored), correct=sum(r['correct'] for r in scored),
        precision=ratio(sum(r['correct'] for r in scored), len(scored)),
        conditional_recall=ratio(sum(r['correct'] for r in scored), data['eligible_association_links']),
        id_switches=sum(not r['correct'] for r in scored),
        ambiguity_or_missing_rejection_rate=ratio(len(proposed)-len(kept), len(proposed)),
        correctly_rejected_false_links=sum(r['correct'] is False for r in proposed if not r['continued']),
        conditional_track_fragmentation=data['conditional_track_fragmentation'],
        source_localization_error_m=distribution(data['source_errors']),
        identity_certified_coverage=0., scope='Conditional on unique offline same-class <=2m labels; unscorable excluded, not successes')
    region = {key: float(np.average([p[key] for p in pixels], weights=[p['pixels'] for p in pixels]))
              for key in ('stable', 'changed', 'unknown', 'refresh')}
    region.update(effective_stable=1-region['refresh'], scorable_lidar_points=sum(p.get('scorable_points',0) for p in pixels),
        accepted_lidar_points=accepted, wrongly_accepted_lidar_points=errors, annotated_sample_far=ratio(errors, accepted),
        critical_events=len(events), critical_change_recall=ratio(sum(e['recalled'] for e in events), len(events)),
        critical_lidar_points=sum(p.get('critical_points',0) for p in pixels),
        missed_critical_lidar_points=sum(p.get('missed_critical_points',0) for p in pixels),
        reference_scope='Annotation-owned independent LiDAR points only; unannotated background remains unscorable; no dense pixel truth')
    region['by_event'] = {reason: dict(events=sum(reason in e['reasons'] for e in events),
        recall=ratio(sum(e['recalled'] for e in events if reason in e['reasons']),sum(reason in e['reasons'] for e in events)))
        for reason in ('newly_visible', 'moving_object', 'visibility_drop')}
    region['per_camera'] = {cid: dict(stable=float(np.mean([p['stable'] for p in pixels if p['camera']==cid])),
        accepted=sum(p.get('accepted_points',0) for p in pixels if p['camera']==cid),
        false=sum(p.get('false_points',0) for p in pixels if p['camera']==cid)) for cid in sorted({p['camera'] for p in pixels})}
    physical = summarize_rows(data['physical'], 'certified')
    physical.update(far_upper95=None, independent_accepted_scene_count=0, physical_quality_pass=False,
                    reason='Independent foreground/identity, calibrated localization and motion producers unavailable')
    costs = {key: distribution([t[key] for t in data['timing'] if key in t]) for key in
             ('precheck_ms','commit_ms','prediction_adapter_ms','total_cpu_ms','projection','association_ms',
              'observation_validation','unexplained_regions','transaction_and_frame_binding')}
    return dict(association=association, region=region, physical=physical,
        observation_support=summarize_rows(data['physical']), costs=costs, checks=data['checks'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/module1_independent_refinement')
    parser.add_argument('--config', default='configs/worldstream_m1_independent.yaml')
    parser.add_argument('--predictions', default='test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json')
    parser.add_argument('--frames', type=int, default=12)
    args = parser.parse_args(); root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    if (root/'experiment_protocol.json').exists():
        raise ValueError('Frozen protocol already exists; choose a fresh output directory')
    cv2.setNumThreads(1)
    config = M1Config.load(args.config)
    infos = pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl', 'rb'))['infos']
    scenes = sorted({i['scene_token'] for i in infos})
    exposure = json.loads((root/'prior_scene_exposure.json').read_text())
    unseen = exposure['unexposed_local_scene_indices']
    independent = len(unseen) >= 24
    chosen = unseen[:24] if independent else list(range(96,120))
    specs = {name: [dict(scene_index=i, scene_id=scenes[i],
        samples=[v['token'] for v in sorted((x for x in infos if x['scene_token']==scenes[i]),key=lambda x:x['timestamp'])[:args.frames]])
        for i in ids] for name, ids in [('calibration', chosen[:6]), ('heldout', chosen[6:12]), ('final_test', chosen[12:24])]}
    # Extra already-exposed whole-scene streams are robustness data, never final test.
    specs['development_long'] = [dict(scene_index=i, scene_id=scenes[i], samples=[v['token'] for v in
        sorted((x for x in infos if x['scene_token']==scenes[i]), key=lambda x:x['timestamp'])]) for i in (0,64)]
    protocol = dict(frozen_before_predictions_and_oracle=True, config=config.to_dict(), scenes=specs,
        local_unexposed_scenes_available=independent, independence_scope='Local recorded exposure audit; not official hidden test',
        risk=dict(far_one_sided95_max=.05, minimum_accepted_scorable=30, minimum_accepted_scenes=3,
                  zero_acceptance_far=None, scene_independence_required=True),
        region=dict(min_critical_change_recall=.998, event_refresh_fraction=.9, pixel_error_px=2.,
                    unknown_background='unscorable', image_space='aspect-preserving 640x360 raw resize, not 640x640 model input'),
        calibration=dict(online_geometry_uncertainty='uncalibrated', online_margin_m=0.,
                         positive_margin_not_calibrated=True, no_threshold_tuning=True),
        source='historical_full_model_detection_replay', offline_source_identity=dict(gate_m=2.,margin_m=.25,population_censored=True),
        cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_library_threads=1, temporal_camera_workers=config.temporal_workers,
        command=sys.argv, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_sha256={str(p):sha256(p) for p in list(Path('worldstream/m1').glob('*.py'))+[Path(__file__),Path(args.config)]})
    write_json(root/'experiment_protocol.json', protocol)
    print('protocol frozen', {s:len(v) for s,v in specs.items()}, 'local unseen', independent, flush=True)
    selected = {s: [[i for i in infos if i['token'] in set(spec['samples'])] for spec in values] for s,values in specs.items()}
    selected = {s:[sorted(v,key=lambda x:x['timestamp']) for v in streams] for s,streams in selected.items()}
    needed = [i for streams in selected.values() for stream in streams for i in stream]
    predictions = read_prediction_subset(args.predictions, {i['token'] for i in needed})
    if any(i['token'] not in predictions for i in needed): raise ValueError('Missing historical model prediction')
    oracle = load_oracle(needed, 'data/nuscenes')
    outputs = {}; raw = {}
    for split, streams in selected.items():
        runs = [run_scene(root, stream, predictions, oracle, config, split) for stream in streams]
        data = aggregate(runs); raw[split] = data; outputs[split] = summarize(data)
        write_json(root/(split+'_summary.json'), outputs[split])
    baseline = aggregate([run_scene(root, stream, predictions, oracle, config, 'development_baseline', True)
                          for stream in selected['development_long']])
    # Baseline pixel LiDAR/event scoring intentionally separate; original code measured on identical RGB/outputs.
    write_json(root/'baseline_comparison.json', dict(current=outputs['development_long'], baseline=summarize(baseline),
        scope='CPU correctness changes, not a performance optimization claim; baseline has no new oracle pixel scores'))
    write_json(root/'association_quality.json', {s:v['association'] for s,v in outputs.items()})
    write_json(root/'region_invalidation_results.json', {s:v['region'] for s,v in outputs.items()})
    write_json(root/'physical_certification_results.json', {s:v['physical'] for s,v in outputs.items()})
    write_json(root/'physical_risk_coverage.json', dict(certified=0, far=None, upper95=None,
        min_far_upper95=.05, nonzero_coverage=False, decision='No-Go', no_false_zero_risk_claim=True))
    write_json(root/'m1_cpu_profile.json', dict(scope='Full CPU prediction adapter + precheck + commit; excludes image IO, offline Oracle and logs',
        splits={s:v['costs'] for s,v in outputs.items()}, peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_scope='Whole evaluation process including dataset/Oracle; not net M1 memory'))
    write_json(root/'long_sequence_results.json', dict(scene_count=2,
        frames=len(raw['development_long']['sizes']), sizes=raw['development_long']['sizes'],
        scope='Complete real nuScenes scenes (~20s each); not hours-long endurance proof'))
    write_json(root/'source_provenance_audit.json', {s:v['checks'] for s,v in outputs.items()})
    write_json(root/'mask_consistency.json', {s:v['checks'] for s,v in outputs.items()})
    write_json(root/'object_observation_validation.json', {s:dict(summary=outputs[s]['observation_support'], records=d['observations']) for s,d in raw.items()})
    write_json(root/'geometry_observability.json', {s:dict(diagnostic_only=True, records=d['geometry']) for s,d in raw.items()})
    write_json(root/'uncertainty_calibration.json', dict(status='Uncalibrated / No-Go', finite_population_error_bound_m=None,
        independent_calibration_scenes=6, exchangeability_unit='scene; assumption not established for censored unique-2m matched population',
        localization_error={s:distribution(d['source_errors']) for s,d in raw.items()},
        missing=['uncensored source identity','independent ego/camera pose error reference','independent object motion bounds'],
        warning='Residuals are censored by offline 2m source assignment; no all-target error guarantee'))
    write_json(root/'foreground_evidence_availability.json', dict(independent_foreground_producer=False,
        independent_identity_producer=False, source='RGB, predicted boxes/velocity, camera-time poses, model features',
        roi_points_are_foreground=False, triangulated_in_box_is_foreground=False,
        minimum_next_evidence='Measured independent instance foreground with bidirectional target identity and calibrated metric/motion bounds',
        independent_component_quality_and_cost='Not Run: no such component available; do not invent cost'))
    write_json(root/'identity_proof_diagnostics.json', dict(decision='No-Go', certified=0,
        explanation='Distinct missing evidence returned for each historical anchor',
        rejection_reasons=dict(Counter(reason for d in raw.values() for row in d['observations'] for reason in row['proof']['rejection_reasons']))))
    failures = [row for d in raw.values() for row in d['associations'] if row['correct'] is False]
    write_json(root/'false_correspondence_cases.json', dict(cases=failures,
        foreground_background_pixel_ownership='Unscorable without independent instance masks; ROI appearance alone not proof',
        not_run=['dense true instance mask evaluation','independent foreground component benchmark']))
    final = outputs['final_test']
    region_ok = (independent and final['region']['stable']>0 and final['region']['critical_change_recall'] is not None
                 and final['region']['critical_change_recall']>=.998)
    write_json(root/'go_nogo_decisions.json', dict(engineering='Pending full regression/fault injection and original-model parity',
        physical_certification='No-Go', region_validation='No-Go' if not region_ok else 'Conditional sparse-reference pass; dense pixel truth unavailable',
        runtime_cost='No-Go for optimized deployment: quality gates unresolved; measured only, no P1 changes',
        local_final_scene_independence=independent))
    write_json(root/'evaluation_summary.json', outputs)
    inject_real_frame_faults(root, selected['development_long'][0], predictions, config)


if __name__ == '__main__':
    main()
