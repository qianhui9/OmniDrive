from copy import deepcopy
from worldstream.runtime import UnifiedRuntime


def evidence():
    key=dict(model_version='weights-sha',lora_version='lora-sha',execution_key='native-backend',generation_key='native-sampling',
        input_shape=[1,6,3,320,640],question_lengths=[32],service='persistent_state',strategy='full-eva-overlap',
        cache_state='warm',memory_budget_bytes=1024,refresh_interval=10,frame_gap_s=.5)
    common=dict(key=deepcopy(key),cost_frame_gap_s_range=[.49,.51],range_calibrated=True)
    costs=dict(common,includes_all_incremental_costs=True,paired_increment_upper_ms=-30.,paired_samples=40,
        peak_cache_bytes=100,independent_scene_bound_validated=True,evidence_sha256='raw-paired-results')
    quality=dict(common,exact_targets_pass=True,heldout_samples=20,calibration_scenes=['cal'],heldout_scenes=['test'],
        evidence_sha256='full-token-state-outputs',protocol_sha256='frozen')
    return key,costs,quality


def test_calibrated_gap_interval_only_generalizes_performance_gap():
    key,costs,quality=evidence()
    assert UnifiedRuntime.admit(dict(key,frame_gap_s=.500019),costs,quality,overhead_ms=10)[0]
    for field,value in [('frame_gap_s',.52),('question_lengths',[33]),('service','current_output'),
            ('model_version','changed'),('generation_key','greedy')]:
        assert not UnifiedRuntime.admit(dict(key,**{field:value}),costs,quality)[0]
    assert not UnifiedRuntime.admit(key,dict(costs,range_calibrated=False),quality)[0]


def test_paired_bound_requires_real_evidence_scope_memory_and_net_benefit():
    key,costs,quality=evidence()
    assert not UnifiedRuntime.admit(key,costs,quality,overhead_ms=31)[0]
    for field,value in [('independent_scene_bound_validated',False),('evidence_sha256',''),
            ('paired_samples',0),('peak_cache_bytes',1025),('paired_increment_upper_ms',float('nan'))]:
        assert not UnifiedRuntime.admit(key,dict(costs,**{field:value}),quality)[0]
    assert not UnifiedRuntime.admit(key,costs,dict(quality,heldout_scenes=['cal']))[0]


def test_logging_paths_and_admission_mode_do_not_change_algorithm_key(tmp_path,llm):
    from types import SimpleNamespace
    import torch
    from worldstream.m1 import M1Config,WorldStateManager
    from dataclasses import replace
    (tmp_path/'a').mkdir();(tmp_path/'b').mkdir()
    model=SimpleNamespace(lm_head=llm,m1_manager=WorldStateManager(M1Config(log_path=str(tmp_path/'a/state.jsonl'))),
        simple_test=lambda:None)
    first=UnifiedRuntime(dict(model_version='weights-sha',lora_version='lora-sha',m1_overlap='experimental',reuse_m1_signature=True))
    second=UnifiedRuntime(dict(model_version='weights-sha',lora_version='lora-sha',m1_overlap='auto',reuse_m1_signature=True))
    data=dict(img=torch.zeros(1,6,3,8,8),input_ids=[[torch.tensor([1,2])]],timestamp=torch.tensor([1.]))
    key=first.workload_key(model,data)
    model.m1_manager.config=replace(model.m1_manager.config,log_path=str(tmp_path/'b/state.jsonl'))
    assert key==second.workload_key(model,data)
    assert len(key['strategy']['runtime']['implementation_sha256'])==64
