"""Forced selection uses real tiny Llama attention/KV, never a mock forward."""
from types import SimpleNamespace as NS
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from worldstream.m3.prefill_executor import GenerationFacade
from tools.worldstream_m123.benchmark.fixed_tokens import forced_generation


@pytest.mark.parametrize('facade',[False,True])
@torch.no_grad()
def test_forced_history_runs_equal_real_forwards_even_with_early_eos(facade):
    torch.manual_seed(17)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,max_position_embeddings=64,
        bos_token_id=1,eos_token_id=2,pad_token_id=0)).eval()
    ids=torch.tensor([[4,5,6]]);embeddings=model.get_input_embeddings()(ids)
    mask=torch.ones_like(ids);positions=torch.arange(3)[None,:]
    expanded=NS(embeddings=embeddings,attention_mask=mask,position_ids=positions,length=3)
    calls=[];original=model.forward
    def forward(*args,**kwargs):
        calls.append(dict(input_ids=kwargs.get('input_ids'),past=kwargs.get('past_key_values')))
        return original(*args,**kwargs)
    model.forward=forward
    # EOS deliberately occurs first and again last: it must not stop early.
    tokens=[2,8,9,2];processor,overrides=forced_generation(tokens)
    if facade:
        owner=GenerationFacade(model,expanded,lambda:model(inputs_embeds=embeddings,
            attention_mask=mask,position_ids=positions,use_cache=True,return_dict=True))
    else:owner=model
    out=owner.generate(inputs_embeds=embeddings,attention_mask=mask,do_sample=True,**overrides)
    assert out.tolist()==[[1]+tokens]
    assert processor.calls==4 and len(calls)==4
    assert [c['input_ids'].item() for c in calls[1:]]==tokens[:-1]
    assert all(c['past'] is not None for c in calls[1:])


def test_invalid_fixed_reference_fails():
    with pytest.raises(ValueError):forced_generation([])
    with pytest.raises(ValueError):forced_generation([-1])


def test_forced_answers_cannot_be_used_for_task_quality(tmp_path):
    from tools.worldstream_m2.task_quality import evaluate_run
    (tmp_path/'invocation.json').write_text('{"fixed_token_reference":"native_reference.json"}')
    output=tmp_path/'official'
    with pytest.raises(ValueError,match='Forced-token'):evaluate_run(tmp_path,output)
    assert not output.exists()
