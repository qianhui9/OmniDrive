_base_ = ["./worldstream_m2_camera_auto.py"]
model = dict(m1_enable=True, m1_config="configs/worldstream_m1_source_linked_state.yaml",
    m2_config=dict(state_interface="source_linked", m1_service="maintain", cheap_gate=True,
        # Fresh current-signature zero-update record, freshly rechecked
        # against full Head/Memory/LLM outputs by this development experiment.
        # Missing/incompatible tables and nonzero updates still fall back.
        camera_cost_table="outputs/module1_state_interface_rearchitecture/gpu_validation_zero_cost/repeat/camera_costs.json"))
