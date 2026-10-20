"""Research only. No independent native task certificate has been established."""
_base_ = ['./worldstream_m2_state_aware.py']
model = dict(m3_enable=True, m3_config=dict(mode='exact', offline_experiment=True,
    task_evidence_enabled=True, shadow_evaluation=True, experimental_direct_query=False,
    direct_query_prefix_fallback=True, direct_task_certificate=''))
