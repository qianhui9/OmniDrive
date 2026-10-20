from types import SimpleNamespace
import torch
import pytest

from tools.system_optimization.compact_context import probe_inputs
from worldstream.runtime import UnifiedRuntime


def tokenizer(text,add_special_tokens=False):
    return SimpleNamespace(input_ids=[12,13,14])


@pytest.mark.parametrize('mode',['current_compact','compact_full'])
def test_native_multimodal_attention_positions_and_original_suffix(llm,carriers,mode):
    original=torch.tensor([[1,7,-200,8,9]])
    ids,images=probe_inputs(original,carriers,tokenizer,'predictions',mode)
    assert torch.equal(ids[:,-2:],original[:,-2:])
    assert int((ids==-200).sum())==(mode=='compact_full')
    if images is not None:
        _,positions,mask,_,embeds,_=llm.prepare_inputs_labels_for_multimodal(ids,
            torch.arange(ids.shape[1])[None],torch.ones_like(ids),None,None,images,None)
        assert mask.shape==positions.shape==embeds.shape[:2]
        assert torch.equal(positions[0],torch.arange(embeds.shape[1]))
        assert bool(mask.all())
    else:
        assert ids.min()>=0
    with torch.no_grad():
        answer=llm.generate(inputs=ids,images=images,do_sample=False,max_new_tokens=3)
    assert answer.shape[0]==1


def test_full_carrier_passthrough_preserves_native_output(llm,carriers):
    original=torch.tensor([[1,7,-200,8,9]])
    ids,images=probe_inputs(original,carriers,None,None,'full')
    assert ids is original and images is carriers
    with torch.no_grad():
        native=llm.generate(inputs=original,images=carriers,do_sample=False,max_new_tokens=3)
        probed=llm.generate(inputs=ids,images=images,do_sample=False,max_new_tokens=3)
    assert torch.equal(native,probed)


def test_offline_input_probe_is_not_a_runtime_strategy():
    # No implicit admission via the offline function or an Oracle label.
    from .test_cost_intervals import evidence
    key,cost,quality=evidence()
    runtime=UnifiedRuntime(dict(model_version='m',lora_version='l'))
    runtime.costs={'records':[cost]};runtime.quality={'records':[quality]};runtime._request_key=key
    model=SimpleNamespace(m3_runtime=SimpleNamespace(config=SimpleNamespace(state_context='state_only',offline_experiment=True)))
    admitted,reason=runtime.optimization_admission(model,{})
    assert not admitted and reason=='offline_strategy_cannot_inherit_production_certificate'


def test_uncertified_service_and_implementation_keys_fallback():
    from .test_cost_intervals import evidence
    key,cost,quality=evidence()
    for field,value in [('service','current_output'),('strategy','other-code-resource-contract'),('execution_key','oracle')]:
        assert not UnifiedRuntime.admit(dict(key,**{field:value}),cost,quality)[0]
