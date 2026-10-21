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
