"""Exact Block 0 equality, cache ownership and adaptive bootstrap contracts."""
import copy
import json
import numpy as np
import pytest
import torch
from worldstream.m1.mask_codec import encode_mask
from worldstream.m2 import IncrementalVisionRuntime,M2Config
from dataclasses import replace
from worldstream.m2.dependency_planner import block0_input, exact_token_changes, DependencyPlanner, patch_updates
from worldstream.m2.state_cache import StateCache, NonFiniteCacheError, CachePublicationError
from .test_source_linked_admission import cold, bundle_for


def test_exact_tokens_and_nonfinite_dependency(eva):
    old=torch.ones(2,10,10,32);new=old.clone()
    assert not exact_token_changes(new,old).any()
    new[0,4,4,31]+=torch.finfo(new.dtype).eps
    mask=exact_token_changes(new,old)
    assert mask.sum()==1 and mask[0,4,4]
    assert DependencyPlanner(eva).plan(mask)[0].updated.sum()==1
    old[1,0,0,0]=new[1,0,0,0]=float('inf')
    assert exact_token_changes(new,old)[1,0,0]
    with pytest.raises(ValueError):exact_token_changes(new,old[0])


@torch.no_grad()
def test_embedding_once_and_no_pixel_route(admitted_runtime,context,monkeypatch):
    runtime=admitted_runtime;image=torch.randn(2,3,40,40);calls=[]
    handle=runtime.model.patch_embed.register_forward_hook(lambda *a:calls.append(1))
    cold(runtime,image,context)
    def fail(*a):raise AssertionError('Production must not compare pixels or exact tokens')
    monkeypatch.setattr('worldstream.m2.dependency_planner.exact_pixel_changes',fail)
    monkeypatch.setattr('worldstream.m2.dependency_planner.exact_token_changes',fail)
    ctx=dict(context,frame_id='1',timestamp=1.1)
    _,diag=runtime.forward(image,execution_context=ctx,computation_state=bundle_for(runtime,ctx))
    handle.remove()
    assert len(calls)==2 and diag['actual_skipped_windows']==24
    assert torch.equal(runtime.cache.block0_input,block0_input(runtime.model,image))






def test_feature_cache_budget_ownership_and_atomicity():
    image=torch.ones(1,3,4,4);x=torch.ones(1,1,1,8);outputs={0:torch.ones(1,1,1,8)}
    ctx=dict(frame_id='0',camera_ids=['front'])
    needed=sum(t.numel()*t.element_size() for t in [image,x,*outputs.values()])
    cache=StateCache(needed);cache.commit(image,outputs,ctx,'sig',block0_input=x)
    assert cache.bytes==needed
    x.add_(1);assert cache.block0_input.eq(1).all()
    snapshot=(cache.block0_input,cache.images,cache.outputs,cache.version,cache.storage_revision)
    for fault in ('budget','nonfinite','metadata'):
        bad=x.clone();data=ctx
        if fault=='budget':bad=torch.ones(2,1,1,8)
        if fault=='nonfinite':bad[0,0,0,0]=float('nan')
        if fault=='metadata':
            class Broken:
                def __deepcopy__(self,memo):raise TypeError('broken metadata')
            data=Broken()
        with pytest.raises((CachePublicationError,NonFiniteCacheError,TypeError)):
            cache.commit(image,outputs,data,'next',block0_input=bad)
        assert (cache.block0_input,cache.images,cache.outputs,cache.version,cache.storage_revision)==snapshot


@torch.no_grad()
def test_mutated_feature_cache_raises(admitted_runtime,context):
    rt=admitted_runtime;image=torch.randn(2,3,40,40);cold(rt,image,context)
    rt.cache.block0_input.add_(1)
    ctx=dict(context,frame_id='1',timestamp=1.1)
    with pytest.raises(ValueError,match='cache_storage_modified'):
        rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))


@torch.no_grad()
def test_nonfinite_features_cannot_be_published(admitted_runtime,context):
    rt=admitted_runtime;image=torch.randn(2,3,40,40);cold(rt,image,context)
    image[0,:,0,0]=float('nan');ctx=dict(context,frame_id='1',timestamp=1.1)
    previous=rt.cache.version
    with pytest.raises(ValueError,match='nonfinite_model_input'):
        rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert rt.cache.version==previous and rt.cache.bytes>0


@torch.no_grad()
def test_programming_errors_are_not_suppressed(admitted_runtime,context,monkeypatch):
    def broken(*a,**kw):raise RuntimeError('interface error')
    monkeypatch.setattr(admitted_runtime.cache,'commit',broken)
    with pytest.raises(RuntimeError,match='interface error'):
        cold(admitted_runtime,torch.ones(2,3,40,40),context)




@torch.no_grad()
@pytest.mark.parametrize('target',['context','source_structure'])
def test_corrupt_metadata_is_rejected_before_adapter(admitted_runtime,context,target):
    rt=admitted_runtime;image=torch.randn(2,3,40,40);cold(rt,image,context)
    ctx=dict(context,frame_id='1',timestamp=1.1);bundle=bundle_for(rt,ctx)
    if target=='context':rt.cache.context.clear()
    else:rt.cache.sources[0]=[None]
    with pytest.raises(ValueError,match='cache_metadata_modified'):
        rt.forward(image,execution_context=ctx,computation_state=bundle)




@torch.no_grad()
@pytest.mark.parametrize('error',[TypeError('wrong snapshot API'),RuntimeError('unexpected export failure')])
def test_m1_snapshot_programming_errors_propagate(admitted_runtime,context,error):
    class BrokenManager:
        def export_m2_inputs(self,expected_context,computation_only=False):raise error
    with pytest.raises(type(error),match=str(error)):
        admitted_runtime.forward_from_m1(torch.ones(2,3,40,40),BrokenManager(),context)


@torch.no_grad()
def test_new_camera_batch_does_not_pass_stale_shapes_to_full_executor(admitted_runtime,context):
    rt=admitted_runtime;image=torch.randn(2,3,40,40);cold(rt,image,context)
    image=torch.cat([image,image[:1]],0)
    ctx=dict(context,frame_id='1',timestamp=1.1,camera_ids=['front','back','extra'])
    with pytest.raises(ValueError,match='shape_unsupported'):
        cold(rt,image,ctx)
