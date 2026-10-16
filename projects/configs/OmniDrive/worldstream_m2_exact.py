"""Experimental strict path: all downstream branches still execute normally."""
_base_ = ['./worldstream_m1_coverage_observe.py']
model = dict(m2_enable=True, m2_config=dict(mode='exact', memory_budget_mb=512))
