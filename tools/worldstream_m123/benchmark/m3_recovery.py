"""Benchmark-only zero-reuse control; production always retains its one executor."""
from dataclasses import replace
from functools import wraps


class ZeroReuseControl:
    def __init__(self,runtime):
        original=runtime.planner.plan
        @wraps(original)
        def plan(*args,**kwargs):
            decision,details=original(*args,**kwargs)
            details=dict(details,benchmark_planned_reuse_length=decision.reuse_length,
                benchmark_zero_reuse=True)
            return replace(decision,reuse_length=0,reuse_type='none',proof=None,
                estimated_quality_risk=0.,estimated_saved_ms=0.,cache_source_frame='',
                cache_age_frames=0,approx_reuse_depth=0,root_frame_index=decision.frame_index,
                decision_reason='benchmark_fixed_zero_reuse'),details
        runtime.planner.plan=plan
