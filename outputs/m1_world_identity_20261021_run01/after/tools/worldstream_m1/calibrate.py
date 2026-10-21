"""Fit a single threshold on train-only physical oracle trials with a binomial risk bound."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from .io_utils import write_json


def calibrate(rows, risk=0.05, confidence=0.95, min_accepted=20):
    if not 0 < risk < 1 or not 0 < confidence < 1 or min_accepted < 1:
        raise ValueError('Invalid risk/confidence/support')
    labeled = [r for r in rows if r.get('oracle_valid') is not None and r.get('residual') is not None
               and np.isfinite(r['residual']) and r.get('geometry_only')]
    best = None
    # Bonferroni correction accounts for selecting among thresholds on the same trials.
    thresholds = sorted({r['residual'] for r in labeled})
    alpha = (1-confidence)/max(1, len(thresholds))
    for threshold in thresholds:
        accepted = [r for r in labeled if r['residual'] <= threshold]
        n = len(accepted); errors = sum(not r['oracle_valid'] for r in accepted)
        upper = float(beta.ppf(1-alpha, errors+1, n-errors)) if errors < n else 1.
        if n >= min_accepted and upper <= risk:
            best = {'threshold': threshold, 'accepted': n, 'errors': errors, 'risk_upper_bound': upper}
    if best is None:
        return {'threshold': None, 'status': 'insufficient_evidence', 'labeled_eligible': len(labeled)}
    return dict(best, status='calibrated_physical_oracle_only', labeled_eligible=len(labeled))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--risk', type=float, default=.05)
    p.add_argument('--confidence', type=float, default=.95)
    p.add_argument('--min-accepted', type=int, default=20)
    args = p.parse_args()
    root = Path(args.run_dir)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['args'].get('split') != 'calibration':
        raise SystemExit('Refusing to calibrate on an evaluation/test run')
    rows = [json.loads(r) for r in (root/'oracle_records.jsonl').read_text().splitlines()]
    # Cluster to one trial per scene/anchor to avoid pretending dependent adjacent frames
    # are independent Bernoulli samples. Scene correlation still limits this pilot estimate.
    unique = {}
    for r in rows: unique.setdefault((r['scene_id'], r['anchor_id']), r)
    result = calibrate(list(unique.values()), args.risk, args.confidence, args.min_accepted)
    result.update(split='calibration', scene_ids=[manifest['scene_id']], risk=args.risk,
                  observation_protocol={'coordinate_space': 'resized_raw_image', 'max_width': manifest['args']['max_width']},
                  confidence=args.confidence, descriptor=manifest['config']['descriptor'],
                  descriptor_size=manifest['config']['descriptor_size'],
                  oracle='physical anchor oracle; not exact feature reuse',
                  limitation='Pilot bound assumes independent retained anchors; scene correlations remain. '
                             'Use many disjoint scenes before deployment.')
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
