"""Independent EVA-window prefix execution with native global-attention suffix."""
_base_ = ['./worldstream_m1_coverage_observe.py']
model = dict(
    m2_enable=True,
    m2_config=dict(mode='auto', reuse_boundary='window', cheap_gate=True,
                   m1_service='maintain', memory_budget_mb=512,
                   cost_table='outputs/final_route/window_costs.json'))
