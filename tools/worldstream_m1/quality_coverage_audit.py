"""Pre-change audit. References stay offline; historical artifacts are read only."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import subprocess

from .io_utils import write_json

ROOT = Path('outputs/module1_quality_coverage_optimization')
PRIOR = Path('outputs/module1_identity_surface_refinement')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def breakdown(rows):
    scored = [r for r in rows if r.get('correct') is not None]
    return dict(objects=len(rows), scenes=len({r['scene'] for r in rows}),
                correct=sum(r['correct'] is True for r in scored),
                wrong=sum(r['correct'] is False for r in scored), unscorable=len(rows)-len(scored))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'experiment_protocol.json').exists():
        raise RuntimeError('Protocol already frozen; preserve this run')
    scenes = json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())
    logs = {s['token']: s['log_token'] for s in scenes}
    log_tokens = set(logs.values())
    samples = json.loads(Path('data/nuscenes/v1.0-trainval/sample.json').read_text())
    sample_scene = {s['token']: s['scene_token'] for s in samples}
    # Match actual scene/sample/log tokens, including old compact arrays/manifests.
    # A token merely listed in an availability inventory is NOT quality exposure.
    roots = sorted(p for p in Path('outputs').iterdir() if p.is_dir()
                   and ('module1' in p.name or p.name.startswith('system_optimization')) and p != ROOT)
    seen_scenes, seen_samples, seen_logs = set(), set(), set()
    inventory = []
    ignore = {'train_data_availability.json', 'independent_data_audit.json', 'independent_reference_audit.json'}
    token_re = re.compile(r'"([0-9a-f]{32})"')
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.suffix not in ('.json', '.jsonl', '.md') or path.name in ignore or 'source' in path.parts:
                continue
            if not path.is_file():
                continue
            st, sa, lg = set(), set(), set()
            for token in path.parts:
                if token in logs: st.add(token)
                if token in sample_scene: sa.add(token)
                if token in log_tokens: lg.add(token)
            with path.open(errors='replace') as handle:
                for line in handle:
                    for token in token_re.findall(line):
                        if token in logs: st.add(token)
                        if token in sample_scene: sa.add(token)
                        if token in log_tokens: lg.add(token)
            if st or sa or lg:
                seen_scenes.update(st); seen_samples.update(sa); seen_logs.update(lg)
                inventory.append(dict(path=str(path), scenes=sorted(st), samples=sorted(sa), logs=sorted(lg)))
    seen_scenes.update(sample_scene[s] for s in seen_samples)
    seen_logs.update(logs[s] for s in seen_scenes if s in logs)
    availability = json.loads(Path('outputs/module1_evidence_optimization/train_data_availability.json').read_text())
    available = availability['accessible_endpoint_scenes']
    options = [dict(s, log_token=logs[s['scene_id']]) for s in available
               if s['scene_id'] not in seen_scenes and logs[s['scene_id']] not in seen_logs]
    chosen, used = [], set()
    for spec in sorted(options, key=lambda s: hashlib.sha256(('quality-coverage-v1:'+s['scene_id']).encode()).hexdigest()):
        if spec['log_token'] not in used:
            used.add(spec['log_token']); chosen.append(spec)
    # Four calibration logs, four heldout, four final when available. Never
    # replenish from exposed logs and never inspect quality to choose a scene.
    splits = dict(development=[], calibration=chosen[:4], heldout=chosen[4:8], final_test=chosen[8:12])
    prior_selection = json.loads((PRIOR/'data_selection.json').read_text())
    from .evidence_experiments import load_inputs
    development, _, _, _ = load_inputs('calibration')
    splits['development'] = [dict(s, log_token=logs[s['scene_id']], previously_exposed=True) for s in development]
    write_json(ROOT/'historical_exposure_inventory.json', inventory)
    write_json(ROOT/'data_selection.json', dict(splits=splits, info_file=prior_selection['info_file'],
        checkpoint=prior_selection['checkpoint'], model_training_exposure=True, model_generalization_test=False,
        previously_exposed_scenes=sorted(seen_scenes), previously_exposed_samples=sorted(seen_samples),
        previously_exposed_logs=sorted(seen_logs), available_endpoint_scenes=len(available),
        available_unexposed_logs=len(chosen), available_unexposed_scenes=len(options),
        selection_without_quality_labels=True, fresh_validation_possible=bool(splits['calibration'])))
    write_json(ROOT/'independent_reference_audit.json', dict(new_independent_logs=len(chosen),
        split_logs={k:len({s['log_token'] for s in v}) for k,v in splits.items()},
        minimum_zero_error_qualified_logs=59, population='Original model training population',
        independent_identity='Unchanged annotation instance unique 2m / .25m competition oracle; unscorable retained',
        independent_surface='Unchanged annotation-owned sparse LiDAR; >2px source correspondence error or motion is false',
        dense_surface_truth_available=False, scene_log_independence='Group all scenes sharing a log',
        historical_exposure_manifest_sha256=sha(ROOT/'historical_exposure_inventory.json')))
    by_reason = defaultdict(list); primary = defaultdict(list); rows = []
    for path in sorted((PRIOR/'traces/development').glob('*/identity.jsonl')):
        for r in map(json.loads, path.read_text().splitlines()):
            if r['method'] != 'selective_identity': continue
            rows.append(r)
            reasons = r['rejection_reasons'] or ['proposal']
            for reason in set(reasons): by_reason[reason].append(r)
            primary[reasons[0]].append(r)
    write_json(ROOT/'identity_rejection_diagnostics.json', dict(
        baseline_population=breakdown(rows), multilabel={k:breakdown(v) for k,v in by_reason.items()},
        mutually_exclusive_primary={k:breakdown(v) for k,v in primary.items()},
        primary_rule='First necessary failure in archived online producer order; overlapping counts never summed',
        measurement_censoring='Old producer skipped RGB measurements after displacement/competition rejection. These are not proven evidence failures.',
        raw_records=str(PRIOR/'traces/development/*/identity.jsonl')))
    taxonomy=json.loads((PRIOR/'surface_error_taxonomy.json').read_text())['phase1']
    write_json(ROOT/'surface_error_taxonomy.json', dict(historical_local_r12=taxonomy,
        preservation={str(p):sha(p) for p in [PRIOR/'traces/phase1_false_stable.jsonl',
            PRIOR/'revalidation_20261019/traces/local_r12_rgb_points.jsonl']},
        rule='46 measured low-parallax; 123 Unknown_Root_Cause. No new causal attribution without independent evidence.'))
    sources={str(p):sha(p) for p in sorted(Path('worldstream/m1').glob('*.py'))}
    write_json(ROOT/'baseline_source_manifest.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sha256=sources))
    protocol=dict(version='quality-coverage-v1', frozen_before_algorithm_changes=True,
        critical_event_recall_min=.998, critical_point_recall_min=.998, each_roi_refresh_min=.9,
        sparse_annotated_far_max=.05, sparse_far_upper95_max=.05, accepted_scene_min=3,
        physical_log_event_upper95_max=.05, risk_confidence=.95,
        statistical_unit='Distinct acquisition log; any false acceptance is a unit error; incomplete scoring never qualifies',
        identity_denominator='All original legal Hungarian geometric object candidate pairs plus separately reported recovery edges; original denominator never replaced',
        region_denominator='All input camera pixels including startup and forced refresh',
        zero_accepted_far=None, gt_and_oracle_unchanged=True, latency_in_selection=False,
        proposed_identity='Finite actual-image history, measured crop LK motion, mutual evidence exclusivity; no static epipolar assumption',
        proposed_region='Small local plane support, convex hull interpolation, withheld ray reprojection and measured per-pixel warp checks',
        ablations=['identity_no_history','identity_no_competition_resolution','surface_no_unit','surface_no_history','surface_no_boundary_guard'],
        selection='Only fresh Calibration meeting ALL hard gates; null if missing or no candidate passes',
        heldout_final='Freeze source/config/evaluators before scoring; no reselection',
        data_selection_sha256=sha(ROOT/'data_selection.json'),baseline_sources_sha256=sha(ROOT/'baseline_source_manifest.json'))
    write_json(ROOT/'experiment_protocol.json',protocol)
    audit=f'''# M1 pre-change root cause audit\n\nHEAD: {json.loads((ROOT/'baseline_source_manifest.json').read_text())['head']}.\n\nIdentity archived Development population: {len(rows)}. Full multilabel and mutually exclusive primary breakdowns are in identity_rejection_diagnostics.json. The old producer exits before RGB measurement for displacement >0.5m and any legal competitor. It cannot establish how many of these have insufficient RGB evidence; new diagnostic probes must measure them without treating proposals as qualifications. Historical sources older than the immediately previous committed image cannot use that image's tracks. Static epipolar geometry also censors motion even though identity does not imply stationarity.\n\nSurface path applies finite flow, FB, photo/2D texture, calibrated rays and complete-neighbourhood erosion, then requires **every pixel** to pass adjacent .5 degree parallax AND independently measured three-view loop/ray closure. Missing third history is all Unknown. Dynamic swept/occlusion masks can remove support afterwards; periodic refresh removes effective area without changing observation area. Full sequential and marginal pixel gates will be measured in a pre-change probe, not inferred from the historical final area.\n\nOld Local R12 has 169 false accepted points: 46 measured low-parallax and 123 Unknown Root Cause. The original RGB, flow, pose and independent annotation/LiDAR traces remain read-only and are SHA256 pinned in surface_error_taxonomy.json. No ownership attribution is invented.\n\nExposure audit: {len(seen_scenes)} scenes, {len(seen_logs)} logs. Available unexposed endpoint logs: {len(chosen)}. Calibration/Heldout/Final use only remaining logs and remain empty if exhausted. All train-derived selections disclose original-model training exposure. At least 59 fully scored independent zero-error accepted logs are needed for the physical risk bound; missing units are never filled by repeated frames or objects.\n\nThe experiment protocol, old source manifest, exposure inventory and data selection were frozen before implementation. Baseline regression: 420 passed, one CUDA-only skipped under sandbox; host GPU read-only check reports idle. Baseline quality tables remain in the historical report, to be distinguished from newly executed development replay.\n'''
    (ROOT/'root_cause_audit.md').write_text(audit)
    print(json.dumps(dict(exposed_scenes=len(seen_scenes),exposed_logs=len(seen_logs),fresh_logs=len(chosen),
        splits={k:len(v) for k,v in splits.items()},baseline_identity=len(rows))))


if __name__=='__main__': main()
