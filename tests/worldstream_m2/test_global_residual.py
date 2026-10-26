"""Temporal residual semantics, actual suffix execution and joint publication."""
import copy
from dataclasses import replace
import pytest
import torch
from worldstream.m2 import IncrementalVisionRuntime, M2Config
from worldstream.m2.state_cache import StateCache, CachePublicationError, NonFiniteCacheError
from .test_source_linked_admission import cold, bundle_for


def next_frame(rt, image, context, index, force=False):
    ctx=dict(context,frame_id=str(index),timestamp=context['timestamp']+index*.1)
    bundle=bundle_for(rt,ctx)
    bundle.diagnostics['full_refresh_hint']=force
    return rt.forward(image,execution_context=ctx,computation_state=bundle)


@torch.no_grad()
@pytest.mark.parametrize('interval',[2,3,4])
def test_anchor_lifecycle_and_real_suffix_skipping(eva,context,interval):
    rt=IncrementalVisionRuntime(eva,M2Config(global_anchor_interval=interval,global_drift_threshold=1.))
    image=torch.randn(2,3,40,40);calls=[]
    handles=[block.register_forward_hook(lambda *args,i=i:calls.append(i))
             for i,block in enumerate(eva.blocks) if i not in rt.prefix]
    first,d=cold(rt,image,context)
    assert calls==[2,3] and d['global_suffix_executions']==1
    assert torch.equal(first[0],eva(image)[0])
    anchor=rt.cache.global_anchor
    assert torch.equal(anchor.residual,first[0].permute(0,2,3,1)-anchor.h2)
    for index in range(1,interval):
        calls.clear();image=image.clone();image[0,:,0,0]+=.001
        result,d=next_frame(rt,image,context,index)
        assert calls==[] and d['global_suffix_skipped_blocks']==2
        assert d['global_suffix_reuse_frames']==1 and d['global_anchor_age']==index
        assert rt.cache.global_anchor.residual is anchor.residual
        assert result[0].storage().data_ptr()!=anchor.residual.storage().data_ptr()
        assert result[0].storage().data_ptr()!=anchor.h2.storage().data_ptr()
        assert result[0].stride()==first[0].stride()
    calls.clear();_,d=next_frame(rt,image,context,interval)
    assert calls==[2,3] and d['global_anchor_refresh_reason']=='anchor_expired'
    assert rt.cache.global_anchor.generation==2 and rt.cache.global_anchor.age==0
    for handle in handles:handle.remove()


@torch.no_grad()
def test_current_h2_enters_reused_output(eva,context):
    rt=IncrementalVisionRuntime(eva,M2Config(global_drift_threshold=1.))
    image=torch.randn(2,3,40,40);first,_=cold(rt,image,context);anchor=rt.cache.global_anchor
    captured={};rt.stage_observer=lambda i,x:captured.update({i:x.clone()})
    image[0,:,0,0]+=.1
    result,d=next_frame(rt,image,context,1)
    expected=(captured[1]+anchor.residual).permute(0,3,1,2)
    assert torch.equal(result[0],expected) and not torch.equal(result[0],first[0])
    assert d['approx_global_residual_reuse'] and not d['full_recompute']
    assert d['region_refresh']['selection_consistency']['selected_in_execution']


@torch.no_grad()
@pytest.mark.parametrize('force,threshold,reason',[(True,1.,'m1_full_refresh'),(False,0.,'feature_drift')])
def test_single_policy_m1_and_drift_refresh(eva,context,force,threshold,reason):
    rt=IncrementalVisionRuntime(eva,M2Config(global_drift_threshold=threshold))
    image=torch.randn(2,3,40,40);cold(rt,image,context);image[0,:,0,0]+=.1
    _,d=next_frame(rt,image,context,1,force)
    assert d['global_anchor_refresh_reason']==reason and d['global_suffix_executed_blocks']==2


@torch.no_grad()
def test_scene_reset_and_failed_scene_initialization_are_atomic(eva,context):
    rt=IncrementalVisionRuntime(eva);image=torch.randn(2,3,40,40);cold(rt,image,context)
    source=rt.cache;anchor=source.global_anchor
    ctx=dict(context,scene_id='new',frame_id='new0',timestamp=10.,state_epoch=2)
    rt.cache.budget_bytes=1
    with pytest.raises(CachePublicationError):cold(rt,image,ctx)
    assert rt.cache is source and rt.cache.global_anchor is anchor
    rt.cache.budget_bytes=512*1024**2
    _,d=cold(rt,image,ctx)
    assert d['global_suffix_executed_blocks']==2 and rt.cache.global_anchor.scene_id=='new'
    assert rt.cache.global_anchor.residual is not anchor.residual


@torch.no_grad()
@pytest.mark.parametrize('fault',['h2','residual','metadata','budget','source'])
def test_invalid_state_does_not_publish(eva,context,fault):
    rt=IncrementalVisionRuntime(eva);image=torch.randn(2,3,40,40);cold(rt,image,context)
    old=rt.cache;version=old.version
    if fault in ('h2','residual'):getattr(old.global_anchor,fault).add_(1)
    if fault=='metadata':old.global_anchor=replace(old.global_anchor,frame_id='wrong')
    if fault=='budget':old.budget_bytes=1
    ctx=dict(context,frame_id='1',timestamp=1.1);bundle=bundle_for(rt,ctx)
    if fault=='source':bundle.diagnostics['source_cache_version']=-1
    with pytest.raises((ValueError,CachePublicationError)):
        rt.forward(image,execution_context=ctx,computation_state=bundle)
    assert rt.cache is old and rt.cache.version==version


def test_readonly_computation_tree_is_detached_once_and_shared():
    from worldstream.m1.readonly_snapshot import freeze
    original=[dict(source=dict(epoch=3),box=[1,2,3,4])]
    snapshot=freeze(original);original[0]['box'][0]=99
    assert snapshot[0]['box'][0]==1 and copy.deepcopy(snapshot) is snapshot
    for mutate in (lambda:snapshot.append({}),lambda:snapshot[0].clear(),
                   lambda:snapshot[0]['box'].__setitem__(0,99)):
        with pytest.raises(TypeError,match='read-only'):mutate()


@torch.no_grad()
def test_public_anchor_ownership_budget_and_alias_rejection(eva,context):
    rt=IncrementalVisionRuntime(eva);image=torch.randn(2,3,40,40);cold(rt,image,context)
    source=rt.cache;anchor=source.global_anchor;cache=StateCache(512*1024**2)
    def commit(candidate,**kwargs):
        cache.commit(image,source.outputs,context,source.signature,block0_input=source.block0_input,
                     global_anchor=candidate,**kwargs)
    commit(anchor)
    assert all(a.storage().data_ptr()!=b.storage().data_ptr()
               for a,b in zip(anchor.tensors(),cache.global_anchor.tensors()))
    assert cache.bytes==sum(t.numel()*t.element_size() for t in
                           [cache.images,cache.block0_input,*cache.outputs.values(),*cache.global_anchor.tensors()])
    cache=StateCache(512*1024**2)
    bad=replace(anchor,h2=source.block0_input,stride=tuple(source.block0_input.stride()))
    with pytest.raises(CachePublicationError,match='alias'):commit(bad,_transfer_global=True)
    assert cache.version==0 and cache.global_anchor is None


@torch.no_grad()
def test_nonfinite_anchor_suffix_cannot_replace_previous_state(eva,context):
    rt=IncrementalVisionRuntime(eva);image=torch.randn(2,3,40,40);cold(rt,image,context)
    source=rt.cache;before=source.global_anchor.residual.clone()
    def corrupt(i,x):
        if i==len(eva.blocks)-1:x.fill_(float('nan'))
    rt.stage_observer=corrupt
    with pytest.raises(NonFiniteCacheError):next_frame(rt,image,context,1,force=True)
    assert rt.cache is source and torch.equal(before,rt.cache.global_anchor.residual)
