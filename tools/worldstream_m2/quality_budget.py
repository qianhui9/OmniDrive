"""Offline task acceptance; these budgets never participate in M2 inference."""
import math

DEFAULT_QUALITY_BUDGET = 0.10
METRICS = ('mAP', 'NDS', 'DET_l', 'Token_F1')


def quality_budgets(budget=DEFAULT_QUALITY_BUDGET, vlm_budget=None):
    values = {key: float(budget) for key in METRICS}
    if vlm_budget is not None:
        values['Token_F1'] = float(vlm_budget)
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError('Absolute quality budgets must be finite and in [0, 1]')
    return values


def assess_quality(metrics, budgets):
    """Inclusive absolute loss, allowing only floating point subtraction noise."""
    checks = {}
    for key in METRICS:
        loss, budget = float(metrics[key]['loss']), float(budgets[key])
        checks[key] = math.isfinite(loss) and (
            loss <= budget or math.isclose(loss, budget, rel_tol=0., abs_tol=1e-12))
    return dict(quality_budgets=budgets, metric_budget_pass=checks,
                within_quality_budgets=all(checks.values()),
                budget_units='absolute score loss: native minus candidate; not relative percent')
