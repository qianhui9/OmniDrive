from worldstream.m2.state_cache import CachePublicationError
import pytest
import torch
from worldstream.m2 import IncrementalVisionRuntime,M2Config
from worldstream.m2.packing import partition
from worldstream.m2.window_executor import WindowExecutor
from .test_source_linked_admission import cold,bundle_for


@torch.no_grad()
@pytest.mark.parametrize('pixel',[(0,0),(39,39),(17,17)])
def test_selected_windows_match_original_eva(exact_runtime,context,pixel):
    rt=exact_runtime;x=torch.randn(2,3,40,40)
    first,d=cold(rt,x,context);assert torch.equal(first[0],rt.model(x)[0])
    x[0,:,pixel[0],pixel[1]]+=.1
    ctx=dict(context,frame_id='1',timestamp=1.1)
    out,d=rt.forward(x,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert d['actual_skipped_windows']==34 and d['recomputed_windows']==2
    assert d['actual_skipped_block_computations']==0  # projections/MLP still execute
    assert all(s['pointwise_window_work']==1 for s in d['stages'][:2])
    assert torch.allclose(out[0],rt.model(x)[0],atol=3e-6,rtol=3e-6)


@torch.no_grad()
def test_identical_windows_skip_attention_and_pointwise(exact_runtime,context):
    rt=exact_runtime;x=torch.randn(2,3,40,40);cold(rt,x,context)
    ctx=dict(context,frame_id='1',timestamp=1.1)
    out,d=rt.forward(x,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert d['actual_skipped_windows']==d['skipped_pointwise_window_work']==36
    assert d['actual_skipped_block_computations']==2
    assert all(s['pointwise_window_work']==0 for s in d['stages'][:2])
    # An identical H2 may use residual arithmetic; cancellation is rounded.
    assert d['approx_global_residual_reuse']
    assert torch.allclose(out[0],rt.model(x)[0],atol=3e-6,rtol=3e-6)


@torch.no_grad()
@pytest.mark.parametrize('cause',['missing','full_refresh','no_windows','no_quality'])
def test_full_refresh_and_direct_partial_use_one_executor(eva,context,monkeypatch,cause):
    rt=IncrementalVisionRuntime(eva,dict())
    x=torch.randn(2,3,40,40);cold(rt,x,context)
    ctx=dict(context,frame_id='1',timestamp=1.1);bundle=bundle_for(rt,ctx)
    if cause=='full_refresh':bundle.diagnostics['full_refresh_hint']=True
    if cause=='no_windows':x=x+1
    calls=[];original=rt.executor.execute
    def execute(*a,**kw):calls.append(len(a[3]));return original(*a,**kw)
    monkeypatch.setattr(rt.executor,'execute',execute)
    if cause=='missing':
        with pytest.raises(ValueError,match='missing_computation_state'):
            rt.forward(x,execution_context=ctx)
        assert calls==[]
    else:
        out,d=rt.forward(x,execution_context=ctx,computation_state=bundle)
        assert calls==[18,18] if cause=='full_refresh' else calls==[6,6]
        assert d['planned_windows']==d['executed_windows']
        assert d['cache_saved']


@torch.no_grad()
def test_cache_publication_failure_raises_without_retry(eva,context,monkeypatch):
    rt=IncrementalVisionRuntime(eva,dict());calls=[]
    handle=eva.blocks[-1].register_forward_hook(lambda *a:calls.append(1))
    def fail(*a,**k):raise CachePublicationError('injected allocation failure')
    monkeypatch.setattr(rt.cache,'commit',fail)
    with pytest.raises(CachePublicationError,match='injected'):
        cold(rt,torch.ones(2,3,40,40),context)
    handle.remove()
    assert len(calls)==1 and not rt.cache.bytes
    assert not rt.last_diagnostics['execution_completed']


def test_retired_algorithms_and_exact_bypass_are_rejected():
    for kw in ({'mode':'exact'},{'reuse_boundary':'camera'},{'state_interface':'legacy'},
               {'m1_service':'output_only'}):
        with pytest.raises(ValueError):M2Config(**kw)
    with pytest.raises(TypeError):M2Config(experimental_conditional_reuse=True)


@torch.no_grad()
def test_zero_cache_budget_raises(eva,context):
    runtime=IncrementalVisionRuntime(eva,dict(memory_budget_mb=0))
    image=torch.randn(2,3,40,40)
    with pytest.raises(CachePublicationError,match='budget'):
        cold(runtime,image,context)
    assert runtime.cache.bytes==0


@torch.no_grad()
def test_bounded_window_lifetime_and_computation_age(exact_runtime,context):
    rt=exact_runtime;image=torch.randn(2,3,40,40);cold(rt,image,context)
    size=rt.cache.bytes
    for i in range(1,20):
        ctx=dict(context,frame_id=str(i),timestamp=1.+.01*i)
        _,d=rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
        assert rt.cache.bytes==size and rt.cache.version==i+1
        assert d['actual_skipped_windows']==36
        assert all(s['computation_timestamp']==1. for units in rt.cache.sources.values() for s in units)
    rt.reset();assert not rt.cache.bytes


@torch.no_grad()
def test_execution_oom_is_not_retried_by_another_algorithm(exact_runtime,context,monkeypatch):
    rt=exact_runtime;image=torch.randn(2,3,40,40);cold(rt,image,context);calls=[]
    def oom(*a,**kw):calls.append(1);raise torch.cuda.OutOfMemoryError('injected')
    monkeypatch.setattr(rt.executor,'execute',oom)
    ctx=dict(context,frame_id='1',timestamp=1.1)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert calls==[1] and rt.cache.context['frame_id']=='0'


@torch.no_grad()
@pytest.mark.parametrize('mutation',['weight','rotary_buffer','numeric_policy','camera_order'])
def test_changed_execution_never_reuses_prior_window_outputs(exact_runtime,context,monkeypatch,mutation):
    rt=exact_runtime;image=torch.randn(2,3,40,40);cold(rt,image,context)
    ctx=dict(context,frame_id='1',timestamp=1.1)
    if mutation=='weight':rt.model.blocks[0].attn.q_proj.weight.add_(.01)
    if mutation=='rotary_buffer':rt.model.rope_win.freqs_cos.add_(.01)
    if mutation=='numeric_policy':
        flag=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        monkeypatch.setattr(torch.backends.cuda.matmul,'allow_fp16_reduced_precision_reduction',not flag)
    if mutation=='camera_order':ctx['camera_ids']=list(reversed(ctx['camera_ids']))
    with pytest.raises(ValueError,match='execution_signature_mismatch'):
        rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@torch.no_grad()
def test_cuda_native_shapes_match_eva(eva):
    eva=eva.cuda().half();x=torch.randn(2,10,10,32,device='cuda',dtype=torch.float16)
    block=eva.blocks[0];cache=partition(block(x),4)[0];x[0,0,0]+=.1
    out,_=WindowExecutor().execute(block,x,cache,torch.tensor([0],device='cuda'))
    assert torch.allclose(out,block(x),atol=1e-3,rtol=1e-3)
