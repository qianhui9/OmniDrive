"""Frozen M1 with existing qualified Exact Auto. Conditional path is disabled."""
_base_ = ['./worldstream_m1_coverage_observe.py']
model = dict(m1_config='configs/worldstream_m1_engineering_frozen.yaml',m2_enable=True,
    m2_config=dict(mode='auto',reuse_boundary='window',state_interface='source_linked',
        memory_budget_mb=512,cache_capture='adaptive',m1_service='maintain',
        cost_table='outputs/module2_refinement/break_even_results.json',
        experimental_conditional_reuse=False,conditional_packing_optimization=False))
