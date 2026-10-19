_base_ = ['./worldstream_m2_source_linked.py']
model = dict(m1_config='configs/worldstream_m1_source_state_refined.yaml',
    m2_config=dict(camera_selective_invalidation=True,experimental_exact_lease_separation=False))
