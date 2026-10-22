"""Prefix misses and budget rejection remain on the same prefill executor."""
import pytest
import torch
from worldstream.m3 import M3Runtime
from worldstream.m3.config import M3Config
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from worldstream.m3.prefill_executor import PrefillExecutor
from worldstream.m3.visual_state import expand_input


def test_uncalibrated_miss_never_calls_native_generate(llm,carriers,context,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('runtime must use PrefillExecutor')
    monkeypatch.setattr(llm,'generate',forbidden)
    runtime=M3Runtime()
    answers,d=runtime.generate(llm,[torch.tensor([1,-200,4])],carriers,context=context,max_new_tokens=2)
    assert answers[0].shape[0]==1 and d['requests'][0]['effective_prefill_tokens']==11
    assert d['requests'][0]['reused_tokens']==0 and d['requests'][0]['executor']=='PrefillExecutor'


def test_rejected_budget_uses_full_prefill_twice(llm,carriers,context):
    runtime=M3Runtime(dict(memory_budget_mb=0))
    ids=[torch.tensor([1,-200,4])]*2
    answers,d=runtime.generate(llm,ids,carriers,context=context,max_new_tokens=2)
    assert len(answers)==2 and runtime.cache.entry is None
    assert [x['reused_tokens'] for x in d['requests']]==[0,0]
    assert all(x['forward_prefill_lengths']==[11] for x in d['requests'])


def test_removed_parallel_strategies_rejected():
    for kw in ({'selector':'random'},{'batched_decode':True},{'common_prefix_seed':True},{'borrow_prefix':True}):
        with pytest.raises(TypeError):M3Config(**kw)
    for kw in ({'mode':'compact'},{'state_context':'only'}):
        with pytest.raises(ValueError):M3Config(**kw)


@pytest.mark.parametrize('target',['embeddings','kv','logits'])
def test_external_cache_mutation_becomes_a_full_prefill_miss(llm,carriers,target):
    cache=PrefixKVCache(1024**2);executor=PrefillExecutor(cache)
    x=expand_input(llm,torch.tensor([[1,-200,4]]),carriers,scope=('s',),signature=('weights',))
    executor.generate(llm,x,max_new_tokens=2)
    with torch.no_grad():
        if target=='embeddings':cache.entry.expanded.embeddings.add_(.1)
        elif target=='kv':cache.entry.kv[0][0].zero_()
        else:cache.entry.last_logits.zero_()
    assert cache.match(x)==0 and cache.entry is None
    _,diag=executor.generate(llm,x,reuse_length=0,max_new_tokens=2)
    assert diag['forward_prefill_lengths']==[11]
