from copy import deepcopy
from types import SimpleNamespace

from worldstream.runtime import UnifiedRuntime
from .test_cost_intervals import evidence


def test_interval_cache_does_not_poison_later_in_domain_request():
    key,cost,quality=evidence()
    runtime=UnifiedRuntime(dict(model_version='weights-sha',lora_version='lora-sha'))
    runtime.costs={'records':[cost]};runtime.quality={'records':[quality]}
    model=SimpleNamespace(m3_runtime=None)
    runtime._request_key=dict(key,frame_gap_s=.6)
    assert not runtime.optimization_admission(model,{})[0]
    runtime._request_key=dict(key,frame_gap_s=.5)
    assert runtime.optimization_admission(model,{})[0]
    assert len(runtime._admission_matches)==1
    runtime._request_key=dict(key,frame_gap_s=.48)
    assert not runtime.optimization_admission(model,{})[0]
    runtime._request_key=key
    assert not runtime.optimization_admission(model,{},prior_overhead_ms=31)[0]


def test_certificate_inplace_mutation_invalidates_cached_match():
    key,cost,quality=evidence()
    runtime=UnifiedRuntime(dict(model_version='weights-sha',lora_version='lora-sha'))
    runtime.costs={'records':[deepcopy(cost)]};runtime.quality={'records':[deepcopy(quality)]}
    runtime._request_key=key
    model=SimpleNamespace(m3_runtime=None)
    assert runtime.optimization_admission(model,{})[0]
    runtime.quality['records'][0]['exact_targets_pass']=False
    assert not runtime.optimization_admission(model,{})[0]


def test_deployment_hash_checks_changed_forward_defaults_each_request(llm):
    from worldstream.m3.visual_state import deployment_key
    base=llm.get_base_model() if hasattr(llm,'get_base_model') else llm
    original=base.forward
    def wrapper(*args, marker=1, **kwargs):return original(*args,**kwargs)
    base.forward=wrapper
    first=deployment_key(llm)
    wrapper.__kwdefaults__['marker']=2
    assert deployment_key(llm)!=first
    base.forward=original
