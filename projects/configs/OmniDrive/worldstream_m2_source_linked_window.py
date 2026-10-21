"""Strict local reuse study; production Auto cost/quality gates remain enabled."""
_base_ = ['./worldstream_m2_source_linked.py']
model = dict(m2_config=dict(reuse_boundary='window',state_interface='source_linked',
    cheap_gate=False,m1_service='maintain',mode='auto',cache_capture='adaptive',
    cost_table='outputs/module2_refinement/break_even_results.json',
    experimental_conditional_reuse=False,experimental_exact_lease_separation=False))
