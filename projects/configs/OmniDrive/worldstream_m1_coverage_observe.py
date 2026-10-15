"""Dense measured image support, full GPU forward retained (M1 only)."""
_base_ = ['./worldstream_m1_observe.py']
model = dict(m1_config='configs/worldstream_m1_coverage.yaml')
