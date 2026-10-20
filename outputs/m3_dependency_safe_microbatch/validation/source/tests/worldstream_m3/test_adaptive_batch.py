import torch
from worldstream.m3 import M3Runtime
from worldstream.m3.batched_executor import length_groups


def test_groups_use_lengths_and_preserve_all_indices():
    assert length_groups([30,10,20,10,40],2)==[[1,3],[2,0],[4]]


def test_adaptive_tiny_native_model_outputs_and_order(llm,carriers,context):
    ids=[torch.tensor([1,3,-200,4,5,6]),torch.tensor([1,3,-200,7]),torch.tensor([1,3,-200,8,9])]
    kw=dict(do_sample=False,max_new_tokens=5)
    reference=[llm.generate(inputs=x[None],images=carriers,**kw) for x in ids]
    runtime=M3Runtime(dict(mode='exact',batched_decode=True,adaptive_batch=True,
        max_batch_size=2,offline_experiment=True))
    actual,diag=runtime.generate(llm,ids,carriers,context=context,**kw)
    assert all(torch.equal(a,b) for a,b in zip(actual,reference))
    assert diag['batch']['groups'][0]['indices']==[1,2]
    assert diag['batch']['full_sequence_equality_guaranteed'] is False


def test_adaptive_sampling_preserves_native_rng(llm,carriers,context):
    ids=[torch.tensor([1,3,-200,4,5]),torch.tensor([1,3,-200,7])]
    kw=dict(do_sample=True,temperature=.8,max_new_tokens=5)
    torch.manual_seed(91)
    expected=[llm.generate(inputs=x[None],images=carriers,**kw) for x in ids]
    rng=torch.get_rng_state()
    runtime=M3Runtime(dict(mode='exact',batched_decode=True,adaptive_batch=True,
        offline_experiment=True))
    torch.manual_seed(91)
    actual,diag=runtime.generate(llm,ids,carriers,context=context,**kw)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    assert torch.equal(rng,torch.get_rng_state())
    assert diag['mode']=='full' and 'sampling' in diag['batch_fallback']['reason']


def test_adaptive_hooks_fall_back_before_any_reordering(llm,carriers):
    from worldstream.m3.batched_executor import AdaptiveBatchExecutor
    from worldstream.m3.prefix_kv_cache import PrefixKVCache
    from worldstream.m3 import expand_input
    ids=[torch.tensor([1,3,-200,4,5]),torch.tensor([1,3,-200,7])]
    inputs=[expand_input(llm,x,carriers) for x in ids]
    executor=AdaptiveBatchExecutor(PrefixKVCache(1024**2),1024**2,2)
    output,diag=executor.try_generate(llm,inputs,do_sample=False,stopping_criteria=[])
    assert output is None and 'hook_requires_sequential' in diag['reason']
