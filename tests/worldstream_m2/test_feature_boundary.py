"""Exact Block 0 equality, cache ownership and adaptive bootstrap contracts."""
from .conftest import synthetic_approval
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
from worldstream.m2.cost_model import CostModel, cost_key, SCHEMA_VERSION
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
    monkeypatch.setattr('worldstream.m2.runtime.exact_token_changes',fail)
    ctx=dict(context,frame_id='1',timestamp=1.1)
    _,diag=runtime.forward(image,execution_context=ctx,computation_state=bundle_for(runtime,ctx))
    handle.remove()
    assert len(calls)==2 and diag['actual_skipped_windows']==24
    assert torch.equal(runtime.cache.block0_input,block0_input(runtime.model,image))


@torch.no_grad()
def test_pixel_change_in_embedding_nullspace_is_exactly_reusable(exact_runtime,context):
    rt=exact_runtime
    # A legal actual convolution which ignores channel 2, no approximate test.
    rt.model.patch_embed.proj.weight[:,2].zero_()
    image=torch.randn(2,3,40,40);cold(rt,image,context)
    image[:,2]+=1
    ctx=dict(context,frame_id='1',timestamp=1.1)
    out,diag=rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert diag['actual_skipped_windows']==36
    assert torch.equal(out[0],rt.model(image)[0])


@torch.no_grad()
def test_dirty_receptive_field_union_in_token_coordinates(exact_runtime,context):
    rt=exact_runtime;image=torch.randn(2,3,40,40);cold(rt,image,context)
    image[0,:,0,0]+=.2
    ctx=dict(context,frame_id='1',timestamp=1.1);bundle=bundle_for(rt,ctx)
    dirty=np.zeros((40,40),bool);dirty[39,39]=True
    bundle.refresh_contract['back'].update(dirty_hint_mask=encode_mask(dirty),unknown_mask=encode_mask(~dirty))
    masks=[];original=rt.planner.plan
    def plan(changed,**kw):masks.append(changed.clone());return original(changed,**kw)
    rt.planner.plan=plan
    _,diag=rt.forward(image,execution_context=ctx,computation_state=bundle)
    assert masks[0].shape==(2,10,10) and masks[0].sum()==2
    assert masks[0][0,0,0] and masks[0][1,9,9]
    assert diag['recomputed_windows']==4
    # Overlapping, padded and dilated convolution: every observed receptive
    # field tap is authoritative, including edge pixels.
    conv=torch.nn.Conv2d(3,4,3,stride=2,padding=2,dilation=2)
    pixels=torch.zeros(1,9,9,dtype=torch.bool);pixels[0,4,4]=True
    expected=torch.nn.functional.conv2d(pixels[:,None].float(),torch.ones(1,1,3,3),stride=2,padding=2,dilation=2)[:,0]>0
    assert torch.equal(patch_updates(pixels,conv),expected)


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
def test_mutated_feature_cache_forces_full(admitted_runtime,context):
    rt=admitted_runtime;image=torch.randn(2,3,40,40);cold(rt,image,context)
    rt.cache.block0_input.add_(1)
    ctx=dict(context,frame_id='1',timestamp=1.1)
    out,diag=rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert diag['actual_skipped_windows']==0 and diag['fallback_reason']=='cache_storage_modified'
    assert torch.equal(out[0],rt.model(image)[0])


@torch.no_grad()
def test_nonfinite_features_cannot_be_published(admitted_runtime,context):
    rt=admitted_runtime;image=torch.randn(2,3,40,40);cold(rt,image,context)
    image[0,:,0,0]=float('nan');ctx=dict(context,frame_id='1',timestamp=1.1)
    _,diag=rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert diag['cache_publication_error']=='nonfinite_cache_features'
    assert not diag['cache_saved'] and rt.cache.bytes==0 and rt.cache.block0_input is None


@torch.no_grad()
def test_programming_errors_are_not_suppressed(admitted_runtime,context,monkeypatch):
    def broken(*a,**kw):raise RuntimeError('interface error')
    monkeypatch.setattr(admitted_runtime.cache,'commit',broken)
    with pytest.raises(RuntimeError,match='interface error'):
        cold(admitted_runtime,torch.ones(2,3,40,40),context)


@torch.no_grad()
@pytest.mark.parametrize('state',['missing','unapproved','wrong_signature','unbeneficial','no_numerics','approved'])
def test_adaptive_capture_bootstraps_only_approved_compatible_scales(eva,context,tmp_path,state):
    image=torch.randn(2,3,40,40);x=block0_input(eva,image);records=[]
    for i in [0,1]:
        records.append(dict(key=cost_key(eva.blocks[i],x,i,model=eva,refresh_policy=M2Config().refresh_signature()),block=i,updated=6,
            numerical_verified=state!='no_numerics',quality_certified=True,
            incremental_gpu_ms=2. if state=='unbeneficial' else .1,full_gpu_ms=1.,incremental_wall_ms=.1,full_wall_ms=1.))
        x=eva.blocks[i](x)
    for row in records:row['approval']=synthetic_approval(tmp_path,row['key'],row['updated'])
    if state=='wrong_signature':
        for row in records:
            key=json.loads(row['key']);key['backbone_version']='wrong';row['key']=json.dumps(key,sort_keys=True)
    path=tmp_path/'costs.json';path.write_text(json.dumps(dict(schema_version=SCHEMA_VERSION,
        records=[] if state=='missing' else records,production_ready=True,
        formal_quality_policy=dict(approved_counts=[] if state=='unapproved' else [6]))))
    rt=IncrementalVisionRuntime(eva,dict(cost_table=str(path)))
    _,diag=cold(rt,image,context)
    assert diag['cache_saved']==(state=='approved')
    if state=='approved':
        ctx=dict(context,frame_id='1',timestamp=1.1)
        out,diag=rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
        assert diag['actual_skipped_windows']==24 and torch.equal(out[0],eva(image)[0])
        # Different selected scale cannot borrow the six-window approval.
        rt.config=replace(rt.config,ordinary_budget=4);ctx=dict(context,frame_id='2',timestamp=1.2)
        _,diag=rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
        assert diag['actual_skipped_windows']==0


@torch.no_grad()
@pytest.mark.parametrize('target',['context','source_structure'])
def test_corrupt_metadata_is_rejected_before_adapter(admitted_runtime,context,target):
    rt=admitted_runtime;image=torch.randn(2,3,40,40);cold(rt,image,context)
    ctx=dict(context,frame_id='1',timestamp=1.1);bundle=bundle_for(rt,ctx)
    if target=='context':rt.cache.context.clear()
    else:rt.cache.sources[0]=[None]
    out,diag=rt.forward(image,execution_context=ctx,computation_state=bundle)
    assert diag['fallback_reason']=='cache_metadata_modified' and diag['actual_skipped_windows']==0
    assert torch.equal(out[0],rt.model(image)[0])


def test_production_costs_require_retained_evidence(eva,tmp_path):
    from .conftest import synthetic_approval
    key=cost_key(eva.blocks[0],torch.zeros(2,10,10,32),0,model=eva)
    row=dict(key=key,block=0,updated=0,numerical_verified=True,quality_certified=True,
        incremental_gpu_ms=.1,full_gpu_ms=1.,incremental_wall_ms=.1,full_wall_ms=1.,
        approval=synthetic_approval(tmp_path,key,0))
    path=tmp_path/'cost.json'
    path.write_text(json.dumps(dict(schema_version=SCHEMA_VERSION,records=[row],production_ready=True,
        formal_quality_policy=dict(approved_counts=[0]))))
    assert CostModel(str(path)).beneficial(key,0)[0]
    (tmp_path/'synthetic_evidence.txt').write_text('changed')
    assert CostModel(str(path)).beneficial(key,0)==(False,'production_approval_evidence_missing')


@torch.no_grad()
@pytest.mark.parametrize('error',[TypeError('wrong snapshot API'),RuntimeError('unexpected export failure')])
def test_m1_snapshot_programming_errors_propagate(admitted_runtime,context,error):
    class BrokenManager:
        def export_m2_inputs(self,expected_context):raise error
    with pytest.raises(type(error),match=str(error)):
        admitted_runtime.forward_from_m1(torch.ones(2,3,40,40),BrokenManager(),context)


@torch.no_grad()
def test_new_camera_batch_does_not_pass_stale_shapes_to_full_executor(admitted_runtime,context):
    rt=admitted_runtime;image=torch.randn(2,3,40,40);cold(rt,image,context)
    image=torch.cat([image,image[:1]],0)
    ctx=dict(context,frame_id='1',timestamp=1.1,camera_ids=['front','back','extra'])
    out,diag=cold(rt,image,ctx)
    assert diag['actual_skipped_windows']==0
    assert all(s['executed_windows']==27 for s in diag['stages'][:2])
    assert torch.equal(out[0],rt.model(image)[0])
