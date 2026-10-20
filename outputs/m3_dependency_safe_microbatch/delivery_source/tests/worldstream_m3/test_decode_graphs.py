"""Native generation/KV/adapter ownership contracts, including real CUDA replay."""
import inspect
import pytest
import torch
from transformers import StoppingCriteria,StoppingCriteriaList
from worldstream.m3 import M3Runtime,M3Config
from worldstream.m3.native_decode import StaticDecodeGraphs
from worldstream.m3.visual_state import base_model,execution_signature


def test_existing_positional_evidence_config_never_enables_decode():
    config=M3Config(True,False)
    assert config.task_evidence_enabled is True and config.shadow_evaluation is False
    assert config.decode_cuda_graph is False and config.identical_prefill_only is False


@torch.no_grad()
def test_default_off_and_cpu_native_fallback(llm,carriers,context):
    assert not M3Config().decode_cuda_graph
    ids=torch.tensor([1,3,-200,4,5]);kw=dict(do_sample=False,max_new_tokens=6,use_cache=True)
    reference=llm.generate(inputs=ids[None],images=carriers,**kw)
    runtime=M3Runtime(dict(mode='full',decode_cuda_graph=True))
    actual,diag=runtime.generate(llm,[ids],carriers,context=context,**kw)
    assert torch.equal(reference,actual[0])
    assert diag['decode_execution']['replayed']==0
    assert not runtime.cache.entry


@torch.no_grad()
def test_guarded_prefix_rejects_partial_but_keeps_complete_prefill(llm,carriers,context):
    a=torch.tensor([1,3,-200,4,5]);b=torch.tensor([1,3,-200,4,6])
    runtime=M3Runtime(dict(mode='exact',offline_experiment=True,identical_prefill_only=True,min_prefix_tokens=1))
    kw=dict(do_sample=False,max_new_tokens=7,use_cache=True)
    out,diag=runtime.generate(llm,[a,b,b],carriers,context=context,**kw)
    assert [r['reused_tokens'] for r in diag['requests']]==[0,0,13]
    assert diag['requests'][1]['prefix_admission']=='partial_prefix_has_no_numerical_parity_certificate'
    for x,y in zip((a,b,b),out):assert torch.equal(y,llm.generate(inputs=x[None],images=carriers,**kw))
    changed=carriers.clone();changed[:,0]+=1
    context=dict(context,frame_id='changed',timestamp=2.)
    _,diag=runtime.generate(llm,[b,b],changed,context=context,**kw)
    assert diag['requests'][0]['reused_tokens']==0


def test_profile_preserves_forward_signature(llm):
    from tools.worldstream_m3.decode_optimization.profile import ForwardTimer
    sig=inspect.signature(llm.forward)
    timer=ForwardTimer(llm)
    with timer:assert inspect.signature(llm.forward)==sig
    with timer:assert inspect.signature(llm.forward)==sig


@pytest.mark.parametrize('error',[RuntimeError,torch.cuda.OutOfMemoryError])
@torch.no_grad()
def test_public_runtime_recovers_after_interrupted_generation(llm,carriers,context,monkeypatch,error):
    runtime=M3Runtime(dict(mode='full',decode_cuda_graph=True))
    ids=torch.tensor([1,3,-200,4,5]);kw=dict(do_sample=False,max_new_tokens=4,use_cache=True)
    expected=llm.generate(inputs=ids[None],images=carriers,**kw)
    original=llm.generate
    def fail(*a,**k):raise error('injected request interruption')
    monkeypatch.setattr(llm,'generate',fail)
    with pytest.raises(error):runtime.generate(llm,[ids],carriers,context=context,**kw)
    assert runtime.cache.entry is None and not runtime._decode_graphs.graphs
    assert not hasattr(llm,'_worldstream_graph_owner')
    monkeypatch.setattr(llm,'generate',original)
    actual,_=runtime.generate(llm,[ids],carriers,context=context,**kw)
    assert torch.equal(expected,actual[0])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires real CUDA')
@torch.no_grad()
def test_graph_sampling_stopping_dynamic_kv_and_replay_ownership(llm,carriers):
    from peft import get_peft_model,LoraConfig
    lm=get_peft_model(llm.cuda(),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','k_proj','v_proj','o_proj']))
    lm.eval();base=base_model(lm);graphs=StaticDecodeGraphs();carriers=carriers.cuda()
    ids=torch.tensor([[1,3,-200,4,5]],device='cuda')
    before=execution_signature(lm)
    class Five(StoppingCriteria):
        def __call__(self,input_ids,scores,**kw):return input_ids.shape[1]>=5
    for length in (2,7,15):
        kw=dict(inputs=ids,images=carriers,do_sample=True,temperature=.1,top_p=.75,max_new_tokens=length,use_cache=True,
                stopping_criteria=StoppingCriteriaList([Five()]) if length==15 else None)
        torch.manual_seed(91);torch.cuda.manual_seed_all(91)
        reference=base.generate(**kw);rng=torch.cuda.get_rng_state()
        with graphs.activate(lm) as stats:
            torch.manual_seed(91);torch.cuda.manual_seed_all(91)
            actual=base.generate(**kw)
            assert torch.equal(reference,actual) and torch.equal(rng,torch.cuda.get_rng_state())
        assert stats['replayed']>0 and not stats['capture_failures']
        assert execution_signature(lm)==before
    # Public outputs/KV must remain owned after the next native request. The
    # captured child workspace is private to the pinned decoder, never retained.
    seed=base(input_ids=torch.tensor([[1,3,4]],device='cuda'),use_cache=True)
    with graphs.activate(lm):
        first=base(input_ids=torch.tensor([[5]],device='cuda'),past_key_values=seed.past_key_values,use_cache=True)
        owned=first.logits.clone();kv=tuple(tuple(t.clone() for t in layer) for layer in first.past_key_values)
        base(input_ids=torch.tensor([[6]],device='cuda'),past_key_values=seed.past_key_values,use_cache=True)
        assert torch.equal(first.logits,owned)
        assert all(torch.equal(a,b) for layer,other in zip(first.past_key_values,kv) for a,b in zip(layer,other))
    graphs.clear();assert not graphs.graphs


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires real CUDA')
@torch.no_grad()
def test_graph_oom_fallback_exception_cleanup_and_parameter_invalidation(llm,carriers,monkeypatch):
    llm=llm.cuda().eval();carriers=carriers.cuda();ids=torch.tensor([[1,3,-200,4,5]],device='cuda')
    graphs=StaticDecodeGraphs()
    def oom(*a):raise torch.cuda.OutOfMemoryError('injected capture OOM; no allocation')
    monkeypatch.setattr(graphs,'_capture',oom)
    kw=dict(inputs=ids,images=carriers,do_sample=False,max_new_tokens=5,use_cache=True)
    reference=llm.generate(**kw)
    with graphs.activate(llm) as stats:actual=llm.generate(**kw)
    assert torch.equal(reference,actual) and stats['capture_failures']
    assert not graphs.graphs
    monkeypatch.undo()
    with pytest.raises(RuntimeError,match='request failure'):
        with graphs.activate(llm):
            llm.generate(**kw)
            raise RuntimeError('request failure')
    assert not graphs.graphs and graphs.signature is None
    assert not hasattr(llm,'_worldstream_graph_owner')
    with graphs.activate(llm):llm.generate(**kw)
    old=graphs.signature
    llm.model.layers[0].mlp.gate_proj.weight.add_(.01)
    with graphs.activate(llm) as stats:changed=llm.generate(**kw)
    assert old!=graphs.signature and stats['captured']>0
    assert torch.equal(changed,llm.generate(**kw))
