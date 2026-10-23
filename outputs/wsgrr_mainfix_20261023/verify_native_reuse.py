"""Read-only provenance verification before reusing the completed Native evaluation."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'wsgrrr_20261022_followup'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()

manifest = json.loads((OLD / 'run_manifest.json').read_text())['preflight']
checks = {}
def check(path, expected):
    actual = digest(path)
    checks[str(path)] = dict(expected=expected, actual=actual, equal=expected == actual)
    print(path, expected == actual, flush=True)

for path, evidence in manifest['resources'].items():
    check(path, evidence['sha256'])
for path, expected in manifest['protected_hashes'].items():
    check(path, expected)
for path, expected in manifest['source_hashes'].items():
    # Native does not execute the M1/M2 region/cache implementation being fixed.
    if not path.startswith(('worldstream/m1/', 'worldstream/m2/')):
        check(path, expected)
for path, evidence in manifest['dataset']['tables'].items():
    if isinstance(evidence, dict) and evidence.get('sha256'):
        check(Path('data/nuscenes/v1.0-trainval') / (path + '.json'), evidence['sha256'])
for path, expected in json.loads((OLD/'Native_full/benchmark_source_hashes.json').read_text()).items():
    check(path, expected)
artifacts = {str(OLD/'Native_full'/name): digest(OLD/'Native_full'/name)
             for name in ('predictions.pkl', 'sequence.json', 'effective_config.json',
                          'invocation.json', 'official_eval/summary.json')}
summary = json.loads((OLD/'Native_full/official_eval/summary.json').read_text())
result = dict(verified=all(c['equal'] for c in checks.values()), checks=checks,
              native_artifact_hashes=artifacts,
              native_frames=summary['frames'], native_scenes=summary['scenes'],
              native_status=summary['status'],
              excluded_changes='Only candidate M1/M2 implementation; Native bypasses both.',
              image_bytes_rehashed=False,
              image_scope='Configured annotation resources and preprocessing/model source hashes checked; raw image bytes have no historical per-image digest to compare.')
(OUT/'native_reuse_check.json').write_text(json.dumps(result, indent=2))
assert result['verified'], 'Native provenance mismatch; investigate before reuse'
