"""Simple measured necessary-validation budget; no learned state scores."""
import math


def validation_budget(context,result=None,state=None):
    try:
        explicit=context.get('necessary_validation_ms')
        if explicit is not None:parts=[float(explicit)]
        else:
            timings=(getattr(result,'diagnostics',{}) or {}).get('timings_ms',{}) or {}
            commit=getattr(state,'last_commit_diagnostics',{}) or {}
            parts=[float(timings.get(k,0.)) for k in ('precheck_total','input_adapter_including_transfers')]
            if result is not None:
                parts.extend(float(commit.get(k,0.)) for k in ('commit_total_ms','output_adapter_including_transfers_ms'))
        return sum(parts) if all(math.isfinite(v) and v>=0 for v in parts) else float('inf')
    except (ValueError,TypeError,AttributeError,OverflowError):
        return float('inf')


def optimistic_window_saving(costs,prefix):
    """Rejection only: upper bound across approved measurements, not permission.

    Different keys cannot approve execution. Scheduler still requires the eventual
    actual block key/count. An optimistic envelope can only suppress a rejection.
    """
    return sum(max([0.]+[r['full_wall_ms']-r['incremental_wall_ms'] for r in costs.records
        if r.get('block')==i and r.get('quality_certified') is True]) for i in prefix)
