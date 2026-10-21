"""Frozen source-patch population diagnosis. Labels never enter online code."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
from worldstream.m1 import M1Config
from worldstream.m1.dense_support import dense_support
from worldstream.m1.surface_observability import PatchObservation, measured_world, gray
from .dataset import frame_from_info
from .evidence_experiments import load_inputs
from .io_utils import write_json

ROOT = Path('outputs/module1_correspondence_recovery_optimization')
PRIOR = Path('outputs/module1_physical_observability_optimization')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize():
    ROOT.mkdir(parents=True, exist_ok=False)
    files = [*Path('worldstream/m1').glob('*.py'), *Path('configs').glob('worldstream_m1*.yaml')]
    for p in files:
        dst = ROOT/'baseline_source'/p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    write_json(ROOT/'baseline_source_manifest.json', dict(
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        files={str(p): sha(p) for p in files}))
    reports = ['experiment_report.md', 'formal_go_nogo.json', 'region_method_comparison.json',
        'region_paired_error_correct_analysis.json', 'surface_acceptance_funnel.json',
        'surface_evidence_provenance.json', 'critical_roi_diagnosis.json',
        'identity_method_comparison.json', 'independent_data_adapter_audit.json',
        'roi_exact_mask_audit.json', 'data_manifest.json', 'frozen_validation_protocol.json']
    # Read and pin actual contents, including the immutable evaluation protocol.
    for name in reports:
        content = (PRIOR/name).read_text()
        if name.endswith('.json'):
            json.loads(content)
    write_json(ROOT/'historical_artifact_pins.json', {str(PRIOR/n): sha(PRIOR/n) for n in reports})
    for name in ['data_manifest.json', 'data_selection.json']:
        shutil.copy2(PRIOR/name, ROOT/name)


def sample(image, points):
    coords = np.where(np.isfinite(points), points, -10).astype(np.float32)
    return cv2.remap(image.astype(np.float32)/255, coords[..., 0], coords[..., 1], cv2.INTER_LINEAR)


def diagnose():
    cv2.setNumThreads(1)
    specs, infos, _, _ = load_inputs('calibration')
    cfg = replace(M1Config.load('configs/worldstream_m1_physical_observability.yaml'), surface_diagnostics=True)
    groups = defaultdict(list)
    for path in sorted((PRIOR/'traces/development').glob('*/point_decisions.jsonl')):
        for r in map(json.loads, path.read_text().splitlines()):
            if r['accepted']['constraints_observation_base']:
                groups[r['scene'], r['frame'], r['camera']].append(r)
    previous = {s['samples'][i]: s['samples'][i-1] for s in specs for i in range(1, len(s['samples']))}
    records = []
    frame_cache = {}
    for (scene, token, cid), rows in groups.items():
        for t in (previous[token], token):
            if t not in frame_cache:
                frame_cache[t] = frame_from_info(infos[t], 'data/nuscenes')
        old, cur = frame_cache[previous[token]].cameras[cid], frame_cache[token].cameras[cid]
        _, source_map, diag = dense_support(old, cur, cfg)
        q = np.asarray([r['pixel'] for r in rows], float)
        p = source_map[q[:, 1].astype(int), q[:, 0].astype(int)]
        m = measured_world(old, cur, p, q)
        patch = PatchObservation(cur, cfg)
        direct = patch.compare(PatchObservation(old, cfg), q, m['world'])
        direct['patch_supported_before_triangulation_gate'] = direct['supported'].copy()
        direct['supported'] &= m['valid']
        offsets = patch.offsets.astype(float)
        grid = q[:, None] + offsets
        # Actual local tracking is a diagnostic of measured deformation, not GT.
        from worldstream.m1.identity_temporal import _lk
        tracked, fb, valid = _lk(gray(cur), gray(old), grid.reshape(-1, 2).astype(np.float32))
        tracked = tracked.reshape(grid.shape); fb = fb.reshape(grid.shape[:2]); valid = valid.reshape(grid.shape[:2])
        design = np.c_[offsets, np.ones(len(offsets))]
        train = ((offsets[:, 0] + offsets[:, 1]).astype(int) % 2 == 0)
        coef = np.einsum('ij,njk->nik', np.linalg.pinv(design[train]), tracked[:, train])
        predicted = np.einsum('ij,njk->nik', design, coef)
        error = np.linalg.norm(predicted - tracked, axis=2)
        a = sample(gray(cur), grid); b = sample(gray(old), predicted)
        delta = np.abs(a-b-np.clip(np.median(a-b, axis=1), -.03, .03)[:, None])
        for i, r in enumerate(rows):
            center = len(offsets)//2
            center_agrees = bool(valid[i, center] and fb[i, center] <= .35 and np.linalg.norm(tracked[i, center]-p[i]) <= .5)
            holdout = bool(valid[i].all() and (fb[i] <= .35).all() and (error[i, ~train] <= .5).all())
            measured_support = bool((delta[i] <= cfg.support_residual_max).mean() >= .9 and min(a[i].std(), b[i].std()) >= cfg.min_texture_std)
            reasons = []
            if not direct['observed'][i]: reasons.append('observation_missing')
            if direct['texture'][i] < cfg.min_texture_std: reasons.append('measured_texture_support_insufficient')
            if not m['valid'][i]: reasons.append('measured_parallax_or_triangulation_insufficient')
            mismatch = center_agrees and holdout and measured_support and not direct['supported'][i]
            if mismatch: reasons.append('measured_local_deformation_explains_patch_frontoparallel_does_not')
            if not reasons and not direct['supported'][i]: reasons.append('Unknown_Root_Cause')
            records.append(dict(**r, source_frame=previous[token], measured_source_uv=p[i],
                current_intrinsic=cur.intrinsic, source_intrinsic=old.intrinsic,
                current_pose=cur.world_to_camera, source_pose=old.world_to_camera,
                world=m['world'][i], depths=m['depths'][i], parallax_deg=m['angle'][i], triangulation_error_px=m['error'][i],
                center_retracking_supported=center_agrees, local_holdout_supported=holdout,
                local_measured_patch_supported=measured_support, local_affine=coef[i],
                local_tracking_source_uv=tracked[i], local_tracking_valid=valid[i], local_tracking_fb=fb[i],
                holdout_residual_px=error[i, ~train], current_rgb_patch=a[i], measured_warp_patch=b[i],
                frontoparallel={k: v[i] for k, v in direct.items()},
                center_observation_residual=float(abs(a[i, center]-b[i, center])),
                complete_measured_patch_residual=float(delta[i].mean()),
                observable_explanations=reasons,
                unconfirmed_physical_causes=['multi_surface_or_depth_boundary', 'occlusion', 'exposure_change', 'true_surface_slope'],
                interpretation='Same RGB local LK and spatial holdout are conditional measurements; a threshold failure is not a physical cause'))
        # Keep memory bounded while retaining both actual frames for this group.
        frame_cache = {t: frame_cache[t] for t in (previous[token], token)}
        print('patch diagnosis', scene, token, cid, len(rows), flush=True)
    counts = {}
    for label, bad in [('correct', False), ('false', True)]:
        pop = [r for r in records if r['false'] == bad]
        counts[label] = dict(before=len(pop), after=sum(bool(r['frontoparallel']['supported']) for r in pop),
            rejected=sum(not bool(r['frontoparallel']['supported']) for r in pop),
            explanations=dict(Counter(v for r in pop if not r['frontoparallel']['supported'] for v in r['observable_explanations'])))
    assert counts['correct']['before'] == 1314 and counts['false']['before'] == 230, counts
    assert counts['correct']['after'] == 177 and counts['false']['after'] == 80, counts
    write_json(ROOT/'patch_rejection_root_cause.json', dict(population='exploratory_reused_population',
        counts=counts, records=records, GT_online=False,
        conclusion='Local observed warp comparison diagnoses model applicability only. No physical attribution from labels or failed thresholds.'))
    write_json(ROOT/'diagnostic_freeze.json', dict(files={str(ROOT/'patch_rejection_root_cause.json'): sha(ROOT/'patch_rejection_root_cause.json')},
        runtime_before_algorithm_change={str(p): sha(p) for p in Path('worldstream/m1').glob('*.py')}))
    print(counts, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--initialize', action='store_true')
    args = parser.parse_args()
    if args.initialize: initialize()
    diagnose()


if __name__ == '__main__':
    main()
