"""Full M1 state continuity + existing M2 camera auto + calibrated M3 auto."""
_base_ = ['./worldstream_m2_camera_auto.py']
model = dict(m3_enable=True, m3_config=dict(mode='auto', memory_budget_mb=512))
