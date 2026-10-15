"""Opt-in round-two observation evidence; no inference computation is skipped."""
_base_ = ['./worldstream_m1_observe.py']
model = dict(m1_config='configs/worldstream_m1_refinement.yaml')
