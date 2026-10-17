"""Refined common-prefix Auto: empty evidence tables retain native generation."""
_base_ = ['./worldstream_m3_auto.py']
model = dict(m3_config=dict(mode='auto',common_prefix_seed=True,borrow_prefix=True,
    cost_table='outputs/module3_refinement/cost_table.json',
    quality_certificate='outputs/module3_refinement/quality_certificate.json'))
