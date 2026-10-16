"""Current-output service without WorldStateManager; exact pixels still certify EVA."""
_base_ = ['./worldstream_m2_camera_auto.py']
model = dict(m1_enable=False,m2_config=dict(m1_service='output_only'))
