"""Offline exact-prefix experiment. Original generation parameters are preserved."""
_base_ = ['./worldstream_m3_auto.py']
model = dict(m3_config=dict(mode='exact', offline_experiment=True))
