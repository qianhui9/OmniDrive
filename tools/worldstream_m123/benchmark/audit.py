"""Offline reconciliation of actual rows, model calls and protected resources."""
import argparse
from collections import Counter
from pathlib import Path

from tools.worldstream_m123.stateful_opportunity.common import sha256, write
from .report import read_json, read_rows


def audit(root, hash_checkpoint=False):
    root = Path(root)
    manifest = read_json(root/'run_manifest.json', {})
    workers, failures = {}, []
    for dest in sorted((root/'experiments').glob('*')):
        if not dest.is_dir():
            continue
        status = read_json(dest/'status.json', {})
        rows = read_rows(dest/'per_frame_metrics.jsonl')
        questions = read_rows(dest/'per_question_metrics.jsonl')
        sequence = read_json(dest/'sequence.json', {})
        validation = read_json(dest/'checkpoint_validation.json', {})
        checks = dict(completed=status.get('status') == 'COMPLETED',
            strict_weights=validation.get('status') == 'PASS' and not validation.get('missing_keys')
                and not validation.get('unexpected_keys') and not validation.get('shape_mismatches'),
            unique_frames=len({r['frame_id'] for r in rows}) == len(rows),
            sequence_exact=[r['frame_id'] for r in rows] == sequence.get('tokens'),
            six_camera_accounting=all((r.get('m2') or {}).get('actual_reused_cameras', 0)+r['native_eva_cameras'] == 6
                for r in rows if (r.get('m2') or {}).get('cache_boundary') != 'window'),
            window_execution_accounting=all(sum(r.get('observed_window_attention_batches', {}).values()) ==
                sum(s.get('executed_windows',0) for s in r['m2'].get('stages',[]))
                for r in rows if (r.get('m2') or {}).get('cache_boundary') == 'window'),
            matched_question_frames=all(q['frame_id'] in {r['frame_id'] for r in rows} for q in questions),
            one_real_question_per_frame=len(questions) == len(rows),
            gt_coverage=all(q['quality'] is not None for q in questions if q['workload'] == 'language'),
            zero_approval_native_fallback=all(q['native_generation_executed'] for q in questions))
        if (dest/'m1_scene_integrity.json').is_file():
            scenes = read_json(dest/'m1_scene_integrity.json')
            checks['m1_scene_ledger'] = all(x.get('ledger', {}).get('status') == 'PASS' and 'metrics' in x for x in scenes.values())
        for name, passed in checks.items():
            if not passed:
                failures.append(dest.name+':'+name)
        workers[dest.name] = dict(checks=checks, frames=len(rows), questions=len(questions),
            scenes=len({r['scene_id'] for r in rows}),
            native_generations=sum(q['native_generation_executed'] for q in questions),
            m3_reasons=dict(Counter(q.get('m3_reason') for q in questions)))
    protected = {p: Path(p).is_file() and sha256(p) == expected
                 for p, expected in manifest.get('preflight', {}).get('protected_hashes', {}).items()}
    failures += ['protected_changed:'+p for p, passed in protected.items() if not passed]
    checkpoint = None
    if hash_checkpoint:
        argv = manifest.get('preflight', {}).get('resources', {})
        checkpoint_path = next((p for p in argv if p.endswith('.pth') and 'iter_' in p), None)
        if checkpoint_path:
            expected = manifest['preflight']['checkpoint']['sha256']
            actual = sha256(checkpoint_path)
            checkpoint = dict(path=checkpoint_path, expected=expected, actual=actual, unchanged=expected == actual)
            if expected != actual:
                failures.append('checkpoint_changed')
    value = dict(status='PASS' if workers and not failures else 'FAIL', failures=failures,
        experiments=workers, protected_unchanged=protected, checkpoint=checkpoint,
        official_full_validation='NOT_RUN' if not read_json(root/'status.json', {}).get('official_detection_available') else 'See experiment statuses',
        scope='PASS reconciles the experiments actually run; it never means blocked full validation completed')
    write(root/'audit.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--hash-checkpoint', action='store_true')
    args = parser.parse_args()
    print(audit(args.run_dir, args.hash_checkpoint)['status'])
