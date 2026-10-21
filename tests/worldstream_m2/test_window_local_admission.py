"""Localized source expiry must not discard healthy cameras or windows."""
import numpy as np
import pytest
import torch
from worldstream.m1.mask_codec import encode_mask
from worldstream.m2 import IncrementalVisionRuntime
from .test_source_linked_admission import bundle_for


@torch.no_grad()
@pytest.mark.parametrize('fault', ['window_age','camera_lease','dirty','unknown_change','record'])
def test_local_admission_and_actual_execution(eva,context,fault,record_property):
    import json
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always',state_interface='source_linked'))
    x=torch.randn(2,3,40,40);ctx=dict(context,timestamp=1.)
    # Use the existing legal full-refresh capture through the source-linked path.
    from worldstream.m1.state_types import ComputationStateBundle
    cold=ComputationStateBundle('scene','0',1.,1,'test',diagnostics=dict(full_refresh_hint=True))
    rt.forward(x,execution_context=ctx,computation_state=cold)
    ctx=dict(ctx,frame_id='1',timestamp=1.5)
    if fault=='window_age':
        for units in rt.cache.sources.values():units[0]['computation_timestamp']=-10.
    b=bundle_for(rt,ctx)
    if fault=='camera_lease':b.observation_links['front']['validity_lease']['source_valid']=False
    if fault=='dirty':
        mask=np.zeros((40,40),bool);mask[0,0]=True
        b.refresh_contract['front'].update(dirty_hint_mask=encode_mask(mask),unknown_mask=encode_mask(~mask))
    if fault=='unknown_change':x=x.clone();x[0,:,0,0]+=.125
    if fault=='record':
        source=rt.cache.sources[0][0]
        b.source_records.append(dict(feature_id='eva_prefix:0:0',source_camera_id='front',
            computation_available=True,computation_version_namespace='m2_window_execution',
            computation_frame_id='forged',computation_timestamp=1.,computation_version=source['computation_version']))
    out,d=rt.forward(x,execution_context=ctx,computation_state=b)
    err=float((out[0]-eva(x)[0]).abs().max())
    assert torch.allclose(out[0],eva(x)[0],atol=3e-6,rtol=3e-6)
    expected=0 if fault=='record' else 18 if fault=='camera_lease' else 34
    assert d['actual_skipped_windows']==expected
    assert d['actual_skipped_block_computations']==expected
    if fault!='record':
        assert d['per_camera_windows']['back']['executed_windows']==0
        assert d['per_camera_windows']['back']['reused_windows']==18
        assert all(s['executed_windows']+s['reused_windows']==s['total_windows'] for s in d['stages'][:2])
    else:assert d['fallback_reason']=='m1_computation_source_mismatch'
    record_property('actual_execution',json.dumps(dict(fault=fault,skipped=d['actual_skipped_block_computations'],max_abs=err)))


@torch.no_grad()
def test_per_camera_horizon_is_not_global_minimum(eva,context):
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    x=torch.randn(2,3,40,40);ctx=dict(context,timestamp=1.)
    rt.forward(x,execution_context=ctx)
    from dataclasses import replace
    rt.config=replace(rt.config,state_interface='source_linked')
    ctx=dict(ctx,frame_id='1',timestamp=1.5);b=bundle_for(rt,ctx)
    b.observation_links['front']['validity_lease']['max_observation_age']=.1
    out,d=rt.forward(x,execution_context=ctx,computation_state=b)
    assert d['actual_skipped_windows']==18
    assert d['per_camera_windows']['back']['reused_windows']==18
    assert torch.allclose(out[0],eva(x)[0],atol=3e-6,rtol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@torch.no_grad()
def test_cuda_tf32_partial_windows_fail_closed(eva,context,monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',True)
    eva=eva.cuda()
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    x=torch.randn(2,3,40,40,device='cuda')
    rt.forward(x,execution_context=context)
    x[0,:,0,0]+=.125
    out,d=rt.forward(x,execution_context=dict(context,frame_id='1'))
    assert d['actual_skipped_windows']==0
    assert all(s['fallback_reason']=='uncertified_tf32_partial_window_execution' for s in d['stages'][:2])
    assert torch.equal(out[0],eva(x)[0])


@torch.no_grad()
@pytest.mark.parametrize('fault',['camera','layer','version'])
def test_unexpired_window_lineage_must_match_position(eva,context,fault):
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    x=torch.randn(2,3,40,40);ctx=dict(context,timestamp=1.)
    rt.forward(x,execution_context=ctx)
    from dataclasses import replace
    rt.config=replace(rt.config,state_interface='source_linked')
    source=rt.cache.sources[0][0]
    if fault=='camera':source['camera_id']='back'
    if fault=='layer':source['producer_layer']=1
    if fault=='version':source['computation_version']=rt.cache.version+1
    ctx=dict(ctx,frame_id='1',timestamp=1.5);bundle=bundle_for(rt,ctx)
    out,d=rt.forward(x,execution_context=ctx,computation_state=bundle)
    assert d['actual_skipped_windows']==0
    assert d['fallback_reason']=='m1_computation_source_mismatch'
    assert torch.equal(out[0],eva(x)[0])


@torch.no_grad()
@pytest.mark.parametrize('capture',['always','never'])
def test_dense_execution_counters_do_not_depend_on_cache_capture(eva,context,capture):
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture=capture))
    out,d=rt.forward(torch.randn(2,3,40,40),execution_context=context)
    assert d['recomputed_windows']==36 and d['actual_skipped_block_computations']==0
    assert d['per_camera_windows']['front']['executed_windows']==18
    assert d['cache_saved']==(capture=='always')


@torch.no_grad()
@pytest.mark.parametrize('failure',[torch.cuda.OutOfMemoryError,RuntimeError])
@pytest.mark.parametrize('cold',[True,False])
def test_window_publication_failure_returns_completed_output_once(eva,context,monkeypatch,failure,cold):
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    x=torch.randn(2,3,40,40)
    if not cold:rt.forward(x,execution_context=context)
    ctx=dict(context,frame_id='1')
    def fail(*a,**kw):raise failure('injected cache publication failure')
    monkeypatch.setattr(rt.cache,'commit',fail)
    calls=[];h=eva.blocks[-1].register_forward_hook(lambda *a:calls.append(1))
    out,d=rt.forward(x,execution_context=ctx)
    h.remove()
    assert len(calls)==1
    assert not d['cache_saved'] and not rt.cache.bytes and rt.cache.context is None
    assert d['cache_publication_error']=='cache_publication_failed'
    assert torch.allclose(out[0],eva(x)[0],atol=3e-6,rtol=3e-6)
    if not cold:
        assert d['actual_skipped_block_computations']==36
        assert all(s['computation_frame_id']=='0' for units in d['output_window_provenance'].values() for s in units)


@torch.no_grad()
def test_forced_partial_numerics_are_not_task_certification(eva,context):
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    x=torch.randn(2,3,40,40)
    rt.forward(x,execution_context=context)
    x[0,:,0,0]+=.1
    _,d=rt.forward(x,execution_context=dict(context,frame_id='1'))
    assert d['actual_skipped_windows']==34 and not d['task_quality_certified']
    _,d=rt.forward(x,execution_context=dict(context,frame_id='2'))
    assert d['actual_skipped_windows']==36 and not d['task_quality_certified']


@torch.no_grad()
@pytest.mark.parametrize('value',['invalid',None,float('nan'),float('inf')])
@pytest.mark.parametrize('where',['context','bundle','source'])
def test_malformed_times_return_full_without_forged_sources(eva,context,value,where):
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    x=torch.randn(2,3,40,40);ctx=dict(context,timestamp=1.)
    rt.forward(x,execution_context=ctx)
    from dataclasses import replace
    rt.config=replace(rt.config,state_interface='source_linked')
    ctx=dict(ctx,frame_id='1',timestamp=1.5);bundle=bundle_for(rt,ctx)
    if where=='context':ctx['timestamp']=value
    if where=='bundle':bundle.timestamp=value
    if where=='source':rt.cache.sources[0][0]['computation_timestamp']=value
    out,d=rt.forward(x,execution_context=ctx,computation_state=bundle)
    assert torch.equal(out[0],eva(x)[0])
    assert d['actual_skipped_windows']==0 and not d['cache_saved']
    assert not rt.cache.bytes
    assert d['source_age'] is None or np.isfinite(d['source_age'])


@torch.no_grad()
@pytest.mark.parametrize('fault',['historical_coordinate','historical_epoch','request_epoch'])
def test_historical_geometry_and_request_epoch_cannot_authorize_reuse(eva,context,fault):
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    x=torch.randn(2,3,40,40);ctx=dict(context,timestamp=1.,state_epoch=1)
    rt.forward(x,execution_context=ctx)
    from dataclasses import replace
    rt.config=replace(rt.config,state_interface='source_linked')
    ctx=dict(ctx,frame_id='1',timestamp=1.5);bundle=bundle_for(rt,ctx)
    if fault=='historical_coordinate':rt.cache.context['coordinate_space']='raw_image'
    if fault=='historical_epoch':rt.cache.context['state_epoch']=2
    if fault=='request_epoch':ctx['state_epoch']=2
    out,d=rt.forward(x,execution_context=ctx,computation_state=bundle)
    assert d['actual_skipped_windows']==0
    assert torch.equal(out[0],eva(x)[0])
    assert d['fallback_reason'] in ('preprocessing_geometry_mismatch','m1_state_epoch_mismatch')


@pytest.mark.parametrize('where',['input','feature'])
def test_nonfinite_cache_publication_is_atomic(where):
    from worldstream.m2.state_cache import StateCache,NonFiniteCacheError
    cache=StateCache(100000);x=torch.ones(1,3,4,4);out={0:torch.ones(1,2)}
    ctx=dict(frame_id='0',camera_ids=['front'])
    cache.commit(x,out,ctx,'signature');old=(cache.images,cache.outputs,cache.context,cache.version)
    if where=='input':x=x.clone();x[0,0,0,0]=float('inf')
    else:out={0:torch.tensor([[float('nan'),0.]])}
    with pytest.raises(NonFiniteCacheError):cache.commit(x,out,dict(ctx,frame_id='1'),'signature')
    assert cache.images is old[0] and cache.outputs is old[1] and cache.context is old[2] and cache.version==old[3]


@torch.no_grad()
def test_nonfinite_runtime_output_is_never_published(eva,context):
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    x=torch.randn(2,3,40,40);x[0,0,0,0]=float('nan')
    calls=[];hook=eva.blocks[-1].register_forward_hook(lambda *a:calls.append(1))
    out,d=rt.forward(x,execution_context=context)
    hook.remove()
    assert len(calls)==1
    assert d['cache_publication_error']=='nonfinite_cache_features'
    assert not d['cache_saved'] and not rt.cache.bytes and not d['task_quality_certified']
    assert torch.allclose(out[0],eva(x)[0],atol=3e-6,rtol=3e-6,equal_nan=True)


@torch.no_grad()
def test_corrupt_compressed_mask_cannot_break_full_fallback_diagnostics(eva,context):
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    x=torch.randn(2,3,40,40);ctx=dict(context,timestamp=1.)
    rt.forward(x,execution_context=ctx)
    from dataclasses import replace
    rt.config=replace(rt.config,state_interface='source_linked')
    ctx=dict(ctx,frame_id='1',timestamp=1.5);bundle=bundle_for(rt,ctx)
    bundle.refresh_contract['front']['reuse_candidate_mask']['data']='eA=='
    out,d=rt.forward(x,execution_context=ctx,computation_state=bundle)
    assert torch.equal(out[0],eva(x)[0])
    assert d['fallback_reason']=='invalid_refresh_contract'
    assert d['candidate_windows'] is None and d['actual_skipped_windows']==0
    assert not d['cache_saved'] and not rt.cache.bytes


@torch.no_grad()
def test_forced_exact_is_never_promoted_by_a_reference_refresh(eva,context):
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    x=torch.randn(2,3,40,40)
    for i in range(2):
        _,d=rt.forward(x,execution_context=dict(context,frame_id=str(i)))
        assert not d['task_quality_certified']


@torch.no_grad()
@pytest.mark.parametrize('where',['norm','attention'])
def test_nonlocal_replacement_with_identical_parameter_objects_fails_closed(eva,context,where):
    from projects.mmdet3d_plugin.models.backbones.eva_vit import Attention
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    x=torch.randn(2,3,40,40);rt.forward(x,execution_context=context)
    before=[p.data_ptr() for p in eva.parameters()]
    if where=='norm':
        class GlobalNorm(torch.nn.LayerNorm):
            def forward(self,x):return super().forward(x-x.mean((1,2),keepdim=True))
        old=eva.blocks[0].norm1
        replacement=GlobalNorm(old.normalized_shape,eps=old.eps).eval()
        replacement.weight=old.weight;replacement.bias=old.bias
        eva.blocks[0].norm1=replacement
    else:
        class GlobalAttention(Attention):
            def forward(self,x):return super().forward(x)+x.mean(0,keepdim=True)
        eva.blocks[0].attn.__class__=GlobalAttention
    assert before==[p.data_ptr() for p in eva.parameters()]
    out,d=rt.forward(x,execution_context=dict(context,frame_id='1'))
    assert d['fallback_reason']=='unsupported_model_topology'
    assert d['actual_skipped_windows']==0 and not rt.cache.bytes
    assert torch.equal(out[0],eva(x)[0])
