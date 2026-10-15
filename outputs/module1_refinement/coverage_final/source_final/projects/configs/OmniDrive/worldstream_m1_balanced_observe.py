"""Calibrated-choice M1 CPU optimization; retain full original GPU inference."""
_base_ = ['./worldstream_m1_observe.py']
model = dict(m1_config='configs/worldstream_m1_balanced.yaml')
