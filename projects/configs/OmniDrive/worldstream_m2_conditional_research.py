"""Explicit research opt-in; mode exact here forces unqualified experiments.

This does not grant Exact equivalence to displaced contextual outputs. No
conditional records qualify Auto; the production configuration stays disabled.
"""
_base_ = ['./worldstream_m2_state_aware.py']
model = dict(m2_config=dict(mode='exact',cache_capture='always',
    experimental_conditional_reuse=True,conditional_max_patch_error=0.0,
    conditional_packing_optimization=True))
