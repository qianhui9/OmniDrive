from worldstream.m2.state_cache import CachePublicationError
import pytest
import torch
from worldstream.m2 import IncrementalVisionRuntime,M2Config
from worldstream.m2.packing import partition
from worldstream.m2.window_executor import WindowExecutor
from .test_source_linked_admission import cold,bundle_for






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








@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@torch.no_grad()
def test_cuda_native_shapes_match_eva(eva):
    eva=eva.cuda().half();x=torch.randn(2,10,10,32,device='cuda',dtype=torch.float16)
    block=eva.blocks[0];cache=partition(block(x),4)[0];x[0,0,0]+=.1
    out,_=WindowExecutor().execute(block,x,cache,torch.tensor([0],device='cuda'))
    assert torch.allclose(out,block(x),atol=1e-3,rtol=1e-3)
