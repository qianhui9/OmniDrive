"""Window execution owns canonical activations and rejects stale calibration."""
import json
import pytest
import torch

from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.packing import partition
from worldstream.m2.state_cache import StateCache
from worldstream.m2.window_executor import WindowExecutor


@torch.no_grad()
def test_partial_window_cache_padding_is_canonical(eva):
    block=eva.blocks[0]
    block.norm1.bias.fill_(.3)
    block.mlp.w3.bias.fill_(.7)
    x=torch.randn(2,10,10,32)
    cache=partition(block(x),4)[0]
    x[0,9,9]+=.1
    out,packed=WindowExecutor().execute(block,x,cache,torch.tensor([8]))
    reference=partition(block(x),4)[0]
    assert torch.allclose(out,block(x),atol=3e-6,rtol=3e-6)
    assert torch.allclose(packed,reference,atol=3e-6,rtol=3e-6)
    assert torch.count_nonzero(packed[8,2:])==0
    assert torch.count_nonzero(packed[8,:,2:])==0


@torch.no_grad()
def test_single_window_zero_update_does_not_alias_source(eva):
    block=eva.blocks[0]
    x=torch.randn(1,4,4,32)
    cache=partition(block(x),4)[0]
    saved=cache.clone()
    out,_=WindowExecutor().execute(block,x,cache,torch.empty(0,dtype=torch.long))
    out.zero_()
    assert torch.equal(cache,saved)


@torch.no_grad()
def test_window_executor_checks_contract_before_zero_update(eva):
    block=eva.blocks[0];x=torch.randn(1,4,4,32)
    cache=partition(block(x),4)[0]
    block.train()
    with pytest.raises(ValueError,match='inference-only'):
        WindowExecutor().execute(block,x,cache,torch.empty(0,dtype=torch.long))
    block.eval()
    with pytest.raises(ValueError,match='feature grid'):
        WindowExecutor().execute(block,x,cache[:0],torch.empty(0,dtype=torch.long))


def test_window_cache_owns_outputs_and_rejects_over_budget_atomically():
    cache=StateCache(2000);images=torch.ones(1,3,4,4);out={0:torch.ones(1,4,4,8)}
    ctx=dict(frame_id='0',camera_ids=['front'])
    cache.commit(images,out,ctx,'signature',block0_input=torch.ones(1,1,1,8))
    out[0].zero_();out.clear()
    assert cache.outputs[0].count_nonzero()==128
    old=(cache.images,cache.outputs,cache.context,cache.version)
    with pytest.raises(RuntimeError,match='budget'):
        cache.commit(images,{0:torch.ones(1000)},ctx,'signature',block0_input=torch.ones(1,1,1,8))
    assert (cache.images,cache.outputs,cache.context,cache.version)==old


@torch.no_grad()
@pytest.mark.parametrize('target',['images','outputs'])
def test_window_cache_mutation_raises(admitted_runtime,context,target):
    from .test_source_linked_admission import cold,bundle_for
    runtime=admitted_runtime
    x=torch.randn(2,3,40,40)
    cold(runtime,x,context)
    if target=='images':runtime.cache.images.add_(1)
    else:runtime.cache.outputs[0].zero_()
    ctx=dict(context,frame_id='1',timestamp=1.1)
    with pytest.raises(ValueError,match='cache_storage_modified'):
        runtime.forward(x,execution_context=ctx,computation_state=bundle_for(runtime,ctx))








@torch.no_grad()
@pytest.mark.parametrize('fault', ['cost', 'miss', 'source', 'numerical', 'publication'])
def test_invalid_state_and_publication_do_not_retry(eva, context, monkeypatch, fault):
    from .test_source_linked_admission import cold, bundle_for
    from worldstream.m2.state_cache import CachePublicationError
    runtime = IncrementalVisionRuntime(eva, dict())
    image = torch.randn(2, 3, 40, 40)
    cold(runtime, image, context)
    ctx = dict(context, frame_id='1', timestamp=1.1)
    bundle = bundle_for(runtime, ctx)
    if fault == 'miss':runtime.cache.clear()
    if fault == 'source':bundle.execution_signature = 'invalid'
    if fault == 'publication':
        def fail(*a, **kw):raise CachePublicationError('injected publication failure')
        monkeypatch.setattr(runtime.cache, 'commit', fail)
    calls = []
    original = runtime.executor.execute
    def observe(block, x, cached, ids):
        calls.append((cached, len(ids)))
        return original(block, x, cached, ids)
    monkeypatch.setattr(runtime.executor, 'execute', observe)
    if fault=='source':
        with pytest.raises(ValueError,match='m1_context_mismatch'):
            runtime.forward(image,execution_context=ctx,computation_state=bundle)
        assert calls==[]
    elif fault=='publication':
        with pytest.raises(CachePublicationError,match='injected'):
            runtime.forward(image,execution_context=ctx,computation_state=bundle)
        assert [n for _,n in calls]==[6,6]
        assert runtime.cache.version==1
    else:
        out,d=runtime.forward(image,execution_context=ctx,computation_state=bundle)
        assert [n for _,n in calls]==([18,18] if fault=='miss' else [6,6])
        assert d['planned_windows']==d['executed_windows']


@torch.no_grad()
def test_benchmark_observer_does_not_change_eva_topology(eva,context,monkeypatch):
    from types import SimpleNamespace as NS
    from tools.worldstream_m123.benchmark.observe import Observer
    from worldstream.m3 import visual_state
    from .test_source_linked_admission import cold,bundle_for
    class Event:
        def __init__(self,**kw):pass
        def record(self):pass
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(visual_state,'base_model',lambda m:m)
    runtime=IncrementalVisionRuntime(eva,dict())
    noop=lambda *a,**k:None
    model=NS(img_backbone=eva,pts_bbox_head=NS(forward=noop),map_head=NS(forward=noop),
        lm_head=NS(forward=noop,generate=noop),m1_manager=None,m2_runtime=runtime,m3_runtime=None)
    observer=Observer(model);assert runtime.planner.cacheable_prefix()==[0,1]
    x=torch.randn(2,3,40,40);cold(runtime,x,context)
    assert sum(observer.window_attention_batches.values())==36
    observer.clear();x[0,:,0,0]+=.1;ctx=dict(context,frame_id='1',timestamp=1.1)
    _,d=runtime.forward(x,execution_context=ctx,computation_state=bundle_for(runtime,ctx))
    assert d['actual_skipped_windows']==24 and sum(observer.window_attention_batches.values())==12
