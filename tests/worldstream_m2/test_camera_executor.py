import json
import pytest
import torch
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.camera_executor import camera_cost_key


@torch.no_grad()
def test_camera_reuse_matches_full_and_preserves_layout(eva,context):
    r=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera'))
    base=torch.randn(2,3,40,40)
    first,d=r.forward(base,execution_context=context)
    saved=r.camera_executor.cache.features;owned=saved.clone()
    for i in range(1,5):
        x=base.clone()
        if i%2:x[0,:,:4,:4]+=i*.01
        y,d=r.forward(x,execution_context=dict(context,frame_id=str(i),timestamp=i))
        assert d['actual_reused_cameras']>=1
        oracle=eva(x)[0]
        assert y[0].stride()==first[0].stride()==oracle.stride()
        assert torch.allclose(y[0],oracle,atol=5e-6,rtol=5e-6)
    assert torch.equal(saved,owned)
    assert r.camera_executor.cache.sources[1]['computation_frame']=='0'
    assert r.camera_executor.cache.sources[1]['verification_frame']=='4'
    assert r.camera_executor.cache.sources[1]['verification_timestamp']==4
    assert r.camera_executor.cache.context['frame_id']=='4'


@torch.no_grad()
def test_camera_fails_closed_without_quality_table(eva,context):
    r=IncrementalVisionRuntime(eva,dict(mode='auto',reuse_boundary='camera'))
    x=torch.randn(2,3,40,40);r.forward(x,execution_context=context)
    y,d=r.forward(x,execution_context=dict(context,frame_id='1'))
    assert d['actual_reused_cameras']==0 and d['fallback_reason']=='unprofiled_camera_cost'
    assert torch.equal(y[0],eva(x)[0])


@torch.no_grad()
def test_camera_finite_full_footprint_and_context_invalidation(eva,context):
    r=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera'))
    x=torch.zeros(2,3,40,40);r.forward(x,execution_context=context)
    x[0,:,1,1]=1  # unsampled: must be found by full comparison
    _,d=r.forward(x,execution_context=dict(context,frame_id='1'))
    assert d['actual_reused_cameras']==1
    _,d=r.forward(x,execution_context=dict(context,frame_id='2',source_version=-1))
    assert d['fallback_reason']=='cache_version_mismatch'
    _,d=r.forward(x,execution_context=dict(context,frame_id='3',scene_id='other'))
    assert d['fallback_reason']=='scene_change'
    c=dict(context,frame_id='4',scene_id='other');c['camera_ids']=list(reversed(context['camera_ids']))
    _,d=r.forward(x,execution_context=c)
    assert d['fallback_reason']=='execution_signature_mismatch'


@torch.no_grad()
def test_camera_weights_and_budget_invalidate(eva,context):
    r=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera'))
    x=torch.zeros(2,3,40,40);r.forward(x,execution_context=context)
    eva.pos_embed.add_(.1)
    _,d=r.forward(x,execution_context=dict(context,frame_id='1'))
    assert d['fallback_reason']=='execution_signature_mismatch'
    r=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera',memory_budget_mb=0))
    _,d=r.forward(x,execution_context=context)
    assert not d['cache_saved'] and r.camera_executor.cache.bytes==0
