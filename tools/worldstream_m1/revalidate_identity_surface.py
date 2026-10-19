"""Replay archived, already exposed M1 data without overwriting or reselecting.

This is engineering reproduction, not a new independent quality qualification.
All actual RGB, detector outputs, LiDAR and annotations use the original runner;
only the output directory changes. Oracle data stays in the offline runner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import cv2
import numpy as np

from . import identity_surface_experiments as experiment
from .io_utils import write_json


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_scene(archive, replay):
    """Compare measured outputs; elapsed times and code fingerprints may differ."""
    result = {}
    for filename, excluded in (
        ('identity.jsonl', {'producer_cpu_ms'}),
        ('pixels.jsonl', {'camera_diagnostics'}),
    ):
        old, new = read_rows(archive / filename), read_rows(replay / filename)
        left = [{k: v for k, v in row.items() if k not in excluded} for row in old]
        right = [{k: v for k, v in row.items() if k not in excluded} for row in new]
        result[filename] = dict(archived_rows=len(old), replayed_rows=len(new), equal=left == right)
    old, new = read_rows(archive / 'state.jsonl'), read_rows(replay / 'state.jsonl')
    mask_keys = ('observation_stable_mask', 'observed_changed_mask', 'unknown_mask', 'refresh_required_mask')
    def decisions(rows):
        return [dict(frame=row['precheck']['frame_id'],
            masks={cid: {k: value[k] for k in mask_keys}
                for cid, value in row['precheck']['invalidation_regions'].items()},
            association_pairs=row['commit']['association_pairs'],
            precheck_certificates=row['precheck']['physically_certified_anchor_ids'],
            committed_certificates=row['post_commit']['certified_anchor_ids']) for row in rows]
    result['conservative_masks_associations_certificates'] = dict(
        archived_frames=len(old), replayed_frames=len(new), equal=decisions(old) == decisions(new))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing runs are never overwritten')
    parser.add_argument('--stage', choices=('development', 'calibration'), default='development',
        help='Archived groups; both are already exposed and only used for reproduction')
    args = parser.parse_args()
    if {4, 5} & os.sched_getaffinity(0):
        raise RuntimeError('Run with taskset -c 0-3,6-31 to exclude faulty CPUs')
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    archived_root = experiment.ROOT
    specs, infos, outputs, oracle = experiment.inputs(args.stage)
    code = [*Path('worldstream/m1').glob('*.py'), Path(experiment.__file__), Path(__file__),
        Path('tools/worldstream_m1/evidence_oracle.py'), Path('tools/worldstream_m1/coverage_oracle.py')]
    manifest = dict(command=['taskset', '-c', '0-3,6-31', sys.executable, '-m',
        'tools.worldstream_m1.revalidate_identity_surface', *sys.argv[1:]],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        archive=str(archived_root), stage=args.stage, scenes=specs,
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        affinity=sorted(os.sched_getaffinity(0)), numpy=np.__version__, opencv=cv2.__version__,
        methods=experiment.METHODS, new_independent_quality_data=False,
        algorithm_reselection=False, online_qualification=None,
        model_outputs='Archived original GPU detection outputs; actual RGB/pose replay, no new model inference',
        oracle_scope='Offline evaluation only; never passed to precheck or commit')
    write_json(args.output / 'execution_manifest.json', manifest)
    experiment.ROOT = args.output
    comparisons = {}
    for spec in specs:
        experiment.run_scene(args.stage, spec, infos, outputs, oracle)
        relative = Path('traces') / args.stage / spec['scene_id']
        comparisons[spec['scene_id']] = compare_scene(archived_root / relative, args.output / relative)
        write_json(args.output / 'output_equivalence.json', comparisons)
    result = experiment.stage_summary(args.stage)
    equal = all(value['equal'] for scene in comparisons.values() for value in scene.values())
    write_json(args.output / 'reproduction_result.json', dict(
        all_compared_outputs_equal=equal, scenes=len(specs), frames=sum(len(s['samples']) for s in specs),
        scope='Identity decisions/measurements, pixel quality metrics, conservative masks/association/certificates',
        strict_foreground_preserved=True, new_deployment_qualification=False, identity=result['identity']))
    print(json.dumps(dict(all_compared_outputs_equal=equal, scenes=len(specs)), indent=2), flush=True)
    if not equal:
        raise RuntimeError('Archived output mismatch; inspect output_equivalence.json before claiming parity')


if __name__ == '__main__':
    main()
