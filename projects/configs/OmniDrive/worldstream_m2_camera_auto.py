"""Opt-in camera boundary; all M1 history maintained; unapproved subsets fall back."""
_base_ = ['./worldstream_m1_coverage_observe.py']
model = dict(m2_enable=True,m2_config=dict(mode='auto',reuse_boundary='camera',
    cheap_gate=True,m1_service='maintain',memory_budget_mb=512,
    camera_cost_table='outputs/module2_rearchitecture/camera_costs.json'))
