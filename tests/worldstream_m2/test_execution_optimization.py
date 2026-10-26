"""Ownership, static geometry, certified IDs and numeric policy regressions."""
import copy
import numpy as np
import pytest
import torch
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.state_cache import StateCache,CachePublicationError,NonFiniteCacheError
from worldstream.m2.packing import partition,unpartition,window_geometry
from worldstream.m2.region_refresh import pooled_descriptors
from worldstream.m2.window_executor import WindowExecutor
from .test_source_linked_admission import cold,bundle_for


@pytest.mark.parametrize('shape,size',[((2,10,10,32),4),((1,4,4,8),4),((3,5,7,8),6)])
@pytest.mark.parametrize('permuted',[False,True])
@torch.no_grad()
def test_grid_single_gather_preserves_values_strides_and_ownership(shape,size,permuted):
    x=torch.randn(shape)
    if permuted:x=x.permute(0,3,1,2).contiguous().permute(0,2,3,1)
    windows,padded=partition(x,size)
    rt=WindowExecutor();result=rt._grid(windows,size,padded,x)
    assert torch.equal(result,x) and result.stride()==x.stride()
    assert result.storage().data_ptr()!=windows.storage().data_ptr()
    saved=windows.clone();result.zero_();assert torch.equal(windows,saved)


@torch.no_grad()
def test_public_cache_commit_still_owns_external_tensor_storage():
    cache=StateCache(10000);x=torch.randn(1,1,1,4);images=torch.randn(1,3,4,4);packed=x.clone()
    cache.commit(images,{0:packed},dict(frame_id='0',camera_ids=['front']),'sig',block0_input=x)
    for external,owned in ((images,cache.images),(x,cache.block0_input),(packed,cache.outputs[0])):
        assert external.storage().data_ptr()!=owned.storage().data_ptr()
        before=owned.clone();external.zero_();assert torch.equal(owned,before)


@torch.no_grad()
def test_runtime_transfers_packed_outputs_without_aliasing_returned_grids(eva,context,monkeypatch):
    rt=IncrementalVisionRuntime(eva);image=torch.randn(2,3,40,40)
    packed={};grids={};execute=rt.executor.execute
    def observed(block,x,cache,ids):
        output,windows=execute(block,x,cache,ids)
        packed[rt.executing_block]=windows;grids[rt.executing_block]=output
        return output,windows
    monkeypatch.setattr(rt.executor,'execute',observed)
    result,_=cold(rt,image,context)
    for i in rt.prefix:
        assert rt.cache.outputs[i].storage().data_ptr()==packed[i].storage().data_ptr()
        assert rt.cache.outputs[i].storage().data_ptr()!=grids[i].storage().data_ptr()
    # Packed buffers above have been transferred: the private observer borrows
    # them read-only. The public return/input remain independently writable.
    before={i:t.clone() for i,t in rt.cache.outputs.items()}
    result[0].zero_();image.zero_()
    assert all(torch.equal(before[i],rt.cache.outputs[i]) for i in rt.prefix)


@torch.no_grad()
def test_partial_transfer_does_not_overwrite_old_cache_on_success_or_failure(eva,context):
    rt=IncrementalVisionRuntime(eva);image=torch.randn(2,3,40,40);cold(rt,image,context)
    old=copy.copy(rt.cache);saved={i:t.clone() for i,t in old.outputs.items()}
    ctx=dict(context,frame_id='1',timestamp=1.1)
    rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert all(torch.equal(old.outputs[i],saved[i]) for i in rt.prefix)
    current=copy.copy(rt.cache);current_saved={i:t.clone() for i,t in current.outputs.items()}
    rt.cache.budget_bytes=1;ctx=dict(context,frame_id='2',timestamp=1.2)
    with pytest.raises(CachePublicationError):rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert rt.cache.version==2 and rt.cache.outputs is current.outputs
    assert all(torch.equal(rt.cache.outputs[i],current_saved[i]) for i in rt.prefix)


@torch.no_grad()
def test_final_visual_finiteness_is_checked_before_atomic_publication(eva,context):
    rt=IncrementalVisionRuntime(eva);image=torch.randn(2,3,40,40);cold(rt,image,context)
    outputs=rt.cache.outputs
    def bad(block,x):
        if block==rt.prefix[-1]:x.fill_(float('nan'))
    rt.stage_observer=bad;ctx=dict(context,frame_id='1',timestamp=1.1)
    with pytest.raises(NonFiniteCacheError):rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert rt.cache.outputs is outputs and rt.cache.version==1
    assert not rt.last_diagnostics['execution_completed']


@torch.no_grad()
def test_transfer_refuses_input_or_return_feature_alias():
    cache=StateCache(10000);x=torch.randn(1,1,1,4);images=torch.randn(1,3,4,4)
    with pytest.raises(CachePublicationError,match='alias'):
        cache.commit(images,{0:x},dict(frame_id='0',camera_ids=['front']),'sig',block0_input=x,_transfer_outputs=True)
    assert cache.version==0


@torch.no_grad()
def test_planned_id_proof_rejects_mutation_and_public_ids_still_validate(eva):
    rt=IncrementalVisionRuntime(eva)
    ids,_=rt.planner.execution_ids(None,rt.prefix,(2,10,10),'cpu')
    ids[0][0]=1
    with pytest.raises(ValueError,match='modified'):
        rt.executor.execute(eva.blocks[0],torch.randn(2,10,10,32),None,ids[0])
    cache=partition(torch.randn(2,10,10,32),4)[0]
    for bad in (torch.tensor([-1]),torch.tensor([18]),torch.tensor([0,0])):
        with pytest.raises(ValueError,match='unique'):
            rt.executor.execute(eva.blocks[0],torch.randn(2,10,10,32),cache,bad)


@torch.no_grad()
def test_cached_footprint_metadata_cannot_be_modified_through_public_results(eva,context):
    rt=IncrementalVisionRuntime(eva)
    first=rt.planner.input_footprints(40,40);first[0][0]['input_region'][0]=999
    assert rt.planner.input_footprints(40,40)[0][0]['input_region'][0]==0
    image=torch.randn(2,3,40,40);_,diag=cold(rt,image,context)
    diag['prefix_dependency_footprints'][0][0]['input_region'][0]=999
    assert rt.planner.input_footprints(40,40)[0][0]['input_region'][0]==0


@torch.no_grad()
def test_topology_memoization_detects_buffer_replacement(eva,context):
    rt=IncrementalVisionRuntime(eva);image=torch.randn(2,3,40,40);cold(rt,image,context)
    eva.rope_win.freqs_cos=eva.rope_win.freqs_cos.clone()+.01
    ctx=dict(context,frame_id='1',timestamp=1.1)
    with pytest.raises(ValueError,match='signature'):
        rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))


@pytest.mark.parametrize('shape,size',[((2,10,10,3),4),((3,5,7,1),6),((1,4,4,8),4)])
def test_descriptor_static_counts_are_exactly_the_original_reduction(shape,size):
    x=torch.randn(shape);windows,_=partition(x.float(),size)
    valid,_=partition(torch.ones_like(x[...,:1]),size)
    reference=windows.sum((1,2))/valid.sum((1,2)).clamp_min(1)
    assert torch.equal(pooled_descriptors(x,size),reference)
    assert window_geometry(*shape[:3],size) is window_geometry(*shape[:3],size)
