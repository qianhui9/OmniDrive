"""Discard M1 history on rejection; reinitialize when validation is needed again."""
_base_ = ['./worldstream_m2_camera_auto.py']
model = dict(m2_config=dict(m1_service='output_only'))
