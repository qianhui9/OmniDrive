"""Source-bound refresh contracts; unknown pixels require independent comparison."""
import copy
import numpy as np
import pytest
import torch
from worldstream.m1.state_types import ComputationStateBundle
from worldstream.m1.mask_codec import encode_mask


def cold(runtime,image,context):
    bundle=ComputationStateBundle(context['scene_id'],context['frame_id'],context['timestamp'],
        context['state_epoch'],context['execution_signature'],diagnostics=dict(full_refresh_hint=True))
    return runtime.forward(image,execution_context=context,computation_state=bundle)


def bundle_for(runtime,context):
    cache=runtime.cache;h,w=cache.images.shape[-2:]
    bundle=ComputationStateBundle(context['scene_id'],context['frame_id'],context['timestamp'],
        context['state_epoch'],context['execution_signature'],diagnostics=dict(
            cache_frame_id=cache.context['frame_id'],source_cache_version=cache.version,
            observed_source_frame_id=cache.context['frame_id'],cache_available=True,transaction_status='pending'))
    for cid in context['camera_ids']:
        bundle.observation_links[cid]=dict(source_frame_id=cache.context['frame_id'],source_camera_id=cid,
            source_timestamp=cache.context['timestamp'],validity_lease=dict(source_valid=True,max_observation_age=1.))
        bundle.refresh_contract[cid]=dict(coordinate_space='preprocessed_image',
            dirty_hint_mask=encode_mask(np.zeros((h,w),bool)),reuse_candidate_mask=encode_mask(np.zeros((h,w),bool)),
            unknown_mask=encode_mask(np.ones((h,w),bool)))
    return bundle


@torch.no_grad()
@pytest.mark.parametrize('fault',['scene','frame','time','epoch','signature','source','cache_version','mask'])
def test_invalid_source_expands_to_full_update(admitted_runtime,context,fault):
    runtime=admitted_runtime;x=torch.randn(2,3,40,40);cold(runtime,x,context)
    ctx=dict(context,frame_id='1',timestamp=1.1);bundle=bundle_for(runtime,ctx)
    if fault=='scene':bundle.scene_id='other'
    if fault=='frame':bundle.frame_id='other'
    if fault=='time':bundle.timestamp=2.
    if fault=='epoch':bundle.state_epoch=9
    if fault=='signature':bundle.execution_signature='other'
    if fault=='source':bundle.diagnostics['observed_source_frame_id']='other'
    if fault=='cache_version':bundle.diagnostics['source_cache_version']+=1
    if fault=='mask':bundle.refresh_contract['front']['unknown_mask']=encode_mask(np.zeros((40,40),bool))
    out,d=runtime.forward(x,execution_context=ctx,computation_state=bundle)
    assert d['actual_skipped_windows']==0 and d['recomputed_windows']==36
    assert torch.equal(out[0],runtime.model(x)[0])


@torch.no_grad()
def test_unknown_change_is_compared_and_dirty_hint_only_expands(admitted_runtime,context):
    runtime=admitted_runtime;x=torch.randn(2,3,40,40);cold(runtime,x,context)
    ctx=dict(context,frame_id='1',timestamp=1.1);bundle=bundle_for(runtime,ctx)
    # Independent image comparison discovers this change despite all-unknown M1.
    x[0,:,0,0]+=.1
    dirty=np.zeros((40,40),bool);dirty[0,20]=True
    bundle.refresh_contract['front'].update(dirty_hint_mask=encode_mask(dirty),unknown_mask=encode_mask(~dirty))
    out,d=runtime.forward(x,execution_context=ctx,computation_state=bundle)
    assert d['actual_skipped_windows']==32 and d['recomputed_windows']==4
    assert torch.allclose(out[0],runtime.model(x)[0],atol=3e-6,rtol=3e-6)
    assert d['m1_transformer_certificate_used'] is False
