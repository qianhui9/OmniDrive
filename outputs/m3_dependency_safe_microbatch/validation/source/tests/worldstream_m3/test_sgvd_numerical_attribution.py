import torch
from tools.worldstream_m3.microbatch.attribution import capture_layers
from worldstream.m3.visual_state import base_model


@torch.no_grad()
def test_layer_observer_preserves_native_logits_and_restores_operations(llm):
    import transformers.models.llama.modeling_llama as llama
    base=base_model(llm);matmul=torch.matmul;softmax=torch.nn.functional.softmax;rope=llama.apply_rotary_pos_emb
    seed=base(input_ids=torch.tensor([[1,3,4]]),use_cache=True)
    kw=dict(input_ids=torch.tensor([[5,6]]),past_key_values=seed.past_key_values,
        attention_mask=torch.ones(1,5,dtype=torch.long),position_ids=torch.tensor([[3,4]]),use_cache=True)
    ref=base(**kw)
    with capture_layers(base,'parallel',3) as (captured,_):actual=base(**kw)
    assert torch.equal(ref.logits,actual.logits)
    assert torch.matmul is matmul and torch.nn.functional.softmax is softmax and llama.apply_rotary_pos_emb is rope
    assert ('00/q_proj',0) in captured and ('00/rope_q',1) in captured
    assert captured[('00/attention_softmax',1)].shape[-1]==5


@torch.no_grad()
def test_layer_observer_restores_after_exception(llm):
    import pytest
    from tools.worldstream_m3.microbatch.attribution import capture_layers
    original=torch.matmul
    with pytest.raises(RuntimeError):
        with capture_layers(base_model(llm),'parallel',1):raise RuntimeError('injected')
    assert torch.matmul is original
