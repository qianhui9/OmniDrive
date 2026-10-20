"""Full original Planning inference with M1 measurements and no M2/M3 reuse."""
_base_ = ['./worldstream_m1_observe.py']
model = dict(m1_enable=True, m1_execution_mode='observe_only',
             m1_config='configs/worldstream_m1_effective.yaml',
             m2_enable=False, m3_enable=False)
