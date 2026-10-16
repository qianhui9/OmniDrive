"""Per-block checkpoint/stride-specific costs; old schema tables fail closed."""
_base_ = ['./worldstream_m1_coverage_observe.py']
model = dict(m2_enable=True,m2_config=dict(mode='auto',memory_budget_mb=512,
    cache_capture='adaptive',cost_table='outputs/module2_refinement/break_even_results.json'))
