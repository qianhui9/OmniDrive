import torch
from worldstream.m3 import expand_input
from worldstream.m3.batched_executor import BatchedExecutor
from worldstream.m3.prefix_kv_cache import PrefixKVCache,minimum_native_prefix_bytes


@torch.no_grad()
def test_impossible_budget_rejected_before_any_native_seed_forward(llm,carriers,monkeypatch):
    from worldstream.m3.visual_state import base_model
    inputs=[expand_input(llm,torch.tensor([1,3,-200,4]),carriers),expand_input(llm,torch.tensor([1,3,-200,5]),carriers)]
    base=base_model(llm)
    calls=[];original=base.forward
    def count(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(base,'forward',count)
    output,diag=BatchedExecutor(PrefixKVCache(0)).try_generate(base,inputs,do_sample=False,max_new_tokens=5)
    assert output is None and diag['reason']=='persistent_prefix_budget_precheck' and not calls


@torch.no_grad()
def test_prefix_lower_bound_never_exceeds_actual_native_entry(llm,carriers):
    from worldstream.m3.visual_state import base_model
    base=base_model(llm);x=expand_input(llm,torch.tensor([1,3,-200,4]),carriers)
    result=base(inputs_embeds=x.embeddings,attention_mask=x.attention_mask,position_ids=x.position_ids,use_cache=True)
    cache=PrefixKVCache(10**8);assert cache.put(x,result.past_key_values,result.logits)
    assert minimum_native_prefix_bytes(base,x,x.length)<=cache.nbytes
