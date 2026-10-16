"""Measured-cost policy. Unprofiled shapes conservatively use the full block."""
_base_ = ['./worldstream_m1_coverage_observe.py']
model = dict(m2_enable=True, m2_config=dict(mode='auto', memory_budget_mb=512,
    cost_table='outputs/module2/break_even_results.json'))
