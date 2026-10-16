"""Opt-in M3. Uncalibrated compact/exact policies fail closed to full generation."""
_base_ = ['./mask_eva_lane_det_vlm.py']
model = dict(m3_enable=True, m3_config=dict(mode='auto', memory_budget_mb=512))
