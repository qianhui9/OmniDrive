"""M1.1 real measurements and scene provenance, original Planning task."""
_base_ = ['./worldstream_m1_effective.py']
model = dict(m1_config='configs/worldstream_m11_diagnostics.yaml',m2_enable=False,m3_enable=False)
