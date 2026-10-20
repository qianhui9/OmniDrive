from dataclasses import replace
from types import SimpleNamespace
import torch
from worldstream.runtime import UnifiedRuntime
from worldstream.m1 import M1Config,WorldStateManager


def test_measured_policy_cannot_certify_different_timeout_profiling_or_logging(tmp_path,llm):
    model=SimpleNamespace(lm_head=llm,m1_manager=WorldStateManager(M1Config()),simple_test=lambda:None)
    config=dict(model_version='weights-sha',lora_version='lora-sha',m1_overlap='experimental',profile_stages=True,
        capture_cost_key=True,trace_path=str(tmp_path/'trace.jsonl'))
    runtime=UnifiedRuntime(config)
    data=dict(img=torch.zeros(1,6,3,8,8),input_ids=[[torch.tensor([1,2])]],timestamp=torch.tensor([1.]))
    key=runtime.workload_key(model,data)
    quality=dict(key=key,exact_targets_pass=True,heldout_samples=2,calibration_scenes=['a'],heldout_scenes=['b'],
        evidence_sha256='raw',protocol_sha256='frozen')
    costs=dict(key=key,includes_all_incremental_costs=True,full_wall_ms=1000.,optimized_wall_ms=800.,peak_cache_bytes=0)
    assert runtime.admit(key,costs,quality)[0]
    for updates in (dict(profile_stages=False),dict(m1_timeout_s=.01),dict(trace_path=''),dict(capture_cost_key=False)):
        changed=UnifiedRuntime(replace(runtime.config,**updates)).workload_key(model,data)
        assert not runtime.admit(changed,costs,quality)[0]


def test_cpu_affinity_is_part_of_exact_cost_key(monkeypatch,llm):
    import os
    model=SimpleNamespace(lm_head=llm,m1_manager=WorldStateManager(M1Config()),simple_test=lambda:None)
    runtime=UnifiedRuntime(dict(model_version='weights',lora_version='lora'))
    data=dict(img=torch.zeros(1,6,3,8,8),input_ids=[[torch.tensor([1,2])]],timestamp=torch.tensor([1.]))
    monkeypatch.setattr(os,'sched_getaffinity',lambda _: {0,1,2,3,6,7})
    old=runtime.workload_key(model,data)
    monkeypatch.setattr(os,'sched_getaffinity',lambda _: {0,1,2,3,6})
    new=runtime.workload_key(model,data)
    assert old['strategy']['runtime']['cpu_resources']['cpu_affinity']==[0,1,2,3,6,7]
    assert new!=old
