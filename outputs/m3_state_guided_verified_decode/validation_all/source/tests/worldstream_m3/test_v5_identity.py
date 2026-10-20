from types import SimpleNamespace
import numpy as np
import torch

from worldstream.runtime import UnifiedRuntime


def test_identity_hash_reuse_still_observes_tensor_and_buffer_mutations():
    model=torch.nn.Linear(2,2).eval()
    model.register_buffer('witness',torch.ones(1))
    runtime=UnifiedRuntime(dict(model_version='m',lora_version='l'))
    meta=[dict(scene_token='s',sample_idx='f',filename=['CAM_FRONT/a.jpg'])]
    data=dict(img=torch.zeros(1,1,3,2,2),timestamp=torch.tensor([1.]))
    first=runtime.identity(model,meta,data).execution_signature
    assert runtime.identity(model,meta,data).execution_signature==first
    with torch.no_grad():model.weight.add_(1)
    second=runtime.identity(model,meta,data).execution_signature
    assert second!=first
    model.witness.add_(1)
    assert runtime.identity(model,meta,data).execution_signature!=second


def test_overlap_evidence_key_changes_with_admission_code_and_resources(llm,monkeypatch):
    from functools import wraps
    import os
    model=SimpleNamespace(lm_head=llm,m1_manager=None,m2_runtime=None,m3_runtime=None,simple_test=lambda:None)
    runtime=UnifiedRuntime(dict(model_version='m',lora_version='l',m1_overlap='experimental'))
    data=dict(img=torch.zeros(1,1,3,2,2),input_ids=[[torch.tensor([1,2])]],timestamp=torch.tensor([1.]))
    first=runtime.workload_key(model,data)
    original=UnifiedRuntime.optimization_admission
    @wraps(original)
    def changed(*args,**kwargs):return original(*args,**kwargs)
    monkeypatch.setattr(UnifiedRuntime,'optimization_admission',changed)
    assert runtime.workload_key(model,data)!=first
    monkeypatch.setattr(UnifiedRuntime,'optimization_admission',original)
    allowed=set(os.sched_getaffinity(0))
    monkeypatch.setattr(os,'sched_getaffinity',lambda pid:allowed-{max(allowed)})
    assert runtime.workload_key(model,data)!=first
