import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from worldstream.m1.mask_codec import encode_mask
from worldstream.m1.state_types import ComputationStateBundle
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.camera_executor import camera_cost_key
from worldstream.m2.dependency_planner import DependencyPlanner


def state(rt,ctx,dirty=()):
    cache=rt.camera_executor.cache;z=np.zeros((40,40),bool)
    b=ComputationStateBundle(ctx['scene_id'],ctx['frame_id'],ctx['timestamp'],1,'test')
    b.diagnostics=dict(cache_available=True,cache_frame_id=cache.context['frame_id'],
        source_cache_version=cache.version,observed_source_frame_id=cache.context['frame_id'])
    for i,cid in enumerate(ctx['camera_ids']):
        d=z.copy()
        if i in dirty:d[1,1]=True
        b.refresh_contract[cid]=dict(coordinate_space='preprocessed_image',
            dirty_hint_mask=encode_mask(d),reuse_candidate_mask=encode_mask(z),unknown_mask=encode_mask(~d))
        b.observation_links[cid]=dict(source_frame_id=cache.context['frame_id'],source_camera_id=cid,
            source_timestamp=cache.context['timestamp'],validity_lease=dict(source_valid=True,max_observation_age=1.))
    return b


@torch.no_grad()
@pytest.mark.parametrize('dirty,changed',[((),()),((0,),()),((),(0,)),((0,3),()),((0,),(2,))])
def test_six_camera_independent_dirty_and_actual_inputs(eva,dirty,changed,record_property):
    import json
    x=torch.randn(6,3,40,40)
    ctx=dict(scene_id='s',frame_id='0',timestamp=1.,state_epoch=1,
        camera_ids=['CAM_FRONT','CAM_BACK','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK_RIGHT'],
        coordinate_space='preprocessed_image',execution_signature='test')
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera',state_interface='source_linked'))
    rt.forward(x,execution_context=ctx);old=rt.camera_executor.cache.version
    ctx=dict(ctx,frame_id='1',timestamp=1.5);b=state(rt,ctx,dirty)
    for i in changed:x[i,:,0,0]+=.1
    y,d=rt.forward(x,execution_context=ctx,computation_state=b)
    updates=set(dirty)|set(changed)
    assert d['global_cache_reset_reason'] is None
    assert d['source_version']==old
    assert d['force_recompute_cameras']==[i in dirty for i in range(6)]
    assert d['actual_reused_cameras']==6-len(updates)
    assert d['verified_exact_cameras']==6-len(changed)
    assert torch.allclose(y[0],eva(x)[0],atol=3e-6,rtol=3e-6)
    for i,s in enumerate(rt.camera_executor.cache.sources):
        assert s['computation_frame']==('1' if i in updates else '0')
        assert s['computation_version']==(old+1 if i in updates else old)
        assert s['cache_publication_version']==old+1
    record_property('selective_observations',json.dumps(d))


@torch.no_grad()
def test_auto_keeps_unqualified_partial_updates_closed(eva):
    x=torch.randn(6,3,40,40)
    rt=IncrementalVisionRuntime(eva,dict(mode='auto',reuse_boundary='camera'))
    ctx=dict(scene_id='s',frame_id='0',camera_ids=list(range(6)),timestamp=1.)
    rt.forward(x,execution_context=ctx)
    rt.camera_executor.records=[dict(key=camera_cost_key(eva,x),changed_cameras=1,
        quality_certified=False,reuse_wall_ms=1.,full_wall_ms=2.,reuse_gpu_ms=1.,full_gpu_ms=2.)]
    y,d=rt.camera_executor.forward(x,dict(ctx,frame_id='1',timestamp=1.5),rt.signature(ctx),[True]+[False]*5)
    assert d['actual_reused_cameras']==0 and d['verified_exact_cameras']==6
    assert d['fallback_reason']=='camera_downstream_quality_unverified'
    assert d['recomputed_cameras']==6
    assert torch.equal(y[0],eva(x)[0])


@torch.no_grad()
def test_computation_age_separation_is_opt_in_and_cannot_bypass_full_refresh(eva):
    x=torch.randn(6,3,40,40)
    ctx=dict(scene_id='s',frame_id='0',timestamp=1.,state_epoch=1,camera_ids=list(range(6)),
        coordinate_space='preprocessed_image',execution_signature='test')
    def run(separate):
        rt=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera',state_interface='source_linked',
            experimental_exact_lease_separation=separate))
        rt.forward(x,execution_context=ctx)
        hits=[]
        for i in range(1,5):
            now=dict(ctx,frame_id=str(i),timestamp=1.+i*.5);b=state(rt,now)
            _,d=rt.forward(x,execution_context=now,computation_state=b)
            hits.append(d['actual_reused_cameras'])
        assert hits==([6,6,6,6] if separate else [6,6,0,6])
        now=dict(ctx,frame_id='5',timestamp=3.5);b=state(rt,now);b.diagnostics['full_refresh_hint']=True
        _,d=rt.forward(x,execution_context=now,computation_state=b)
        assert d['actual_reused_cameras']==0 and d['fallback_reason']=='m1_full_refresh'
        return rt
    run(True);run(False)


def test_window_dependency_footprint_includes_prior_layout_and_conv(eva):
    eva.blocks[1].window_size=3
    f=DependencyPlanner(eva).input_footprints(40,40)[1][5]
    assert f['token_region']==[0,0,8,8]
    assert f['input_region']==[0,0,32,32]


@torch.no_grad()
def test_actual_producer_version_mismatch_is_global_invalidation(eva):
    x=torch.randn(6,3,40,40)
    ctx=dict(scene_id='s',frame_id='0',timestamp=1.,state_epoch=1,camera_ids=list(range(6)),
        coordinate_space='preprocessed_image',execution_signature='test')
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera',state_interface='source_linked'))
    rt.forward(x,execution_context=ctx);ctx=dict(ctx,frame_id='1',timestamp=1.5);b=state(rt,ctx)
    b.source_records=[dict(feature_id='backbone',source_camera_id=0,computation_available=True,
        execution_signature='test',computation_frame_id='0',computation_version=999,
        computation_version_namespace='m2_camera_execution',computation_timestamp=1.,source_cache_version=rt.camera_executor.cache.version)]
    _,d=rt.forward(x,execution_context=ctx,computation_state=b)
    assert d['actual_reused_cameras']==0 and d['global_cache_reset_reason']=='m1_computation_source_mismatch'


def test_camera_metadata_publication_failure_keeps_old_owned_batch():
    from worldstream.m2.camera_feature_cache import CameraFeatureCache
    c=CameraFeatureCache(100000);x=torch.ones(6,3,4,4);f=torch.ones(6,2,2,2)
    ctx=dict(scene_id='s',frame_id='0',camera_ids=list(range(6)))
    c.publish(x,f,ctx,'sig',[True]*6)
    old=(c.images,c.features,c.sources,c.context,c.version)
    class BadContext(dict):
        def __getitem__(self,key):
            if key=='camera_ids':raise torch.cuda.OutOfMemoryError('metadata OOM')
            return super().__getitem__(key)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        c.publish(x*2,f*2,BadContext(ctx,frame_id='1'),'sig',[True]+[False]*5)
    assert all(getattr(c,k) is v for k,v in zip(('images','features','sources','context'),old[:4]))
    assert c.version==old[4]


@torch.no_grad()
def test_failed_next_publication_does_not_invent_cache_version(eva,monkeypatch):
    x=torch.randn(6,3,40,40)
    rt=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera'))
    ctx=dict(scene_id='s',frame_id='0',camera_ids=list(range(6)),timestamp=1.)
    rt.forward(x,execution_context=ctx);old=rt.camera_executor.cache.version
    def oom(*a,**kw):raise torch.cuda.OutOfMemoryError('publication OOM')
    monkeypatch.setattr(rt.camera_executor.cache,'publish',oom)
    x[0,:,0,0]+=.1
    y,d=rt.camera_executor.forward(x,dict(ctx,frame_id='1',timestamp=1.5),rt.signature(ctx))
    assert not d['cache_saved'] and d['actual_reused_cameras']==5
    assert d['output_camera_provenance'][0]['cache_publication_version'] is None
    assert d['output_camera_provenance'][0]['source_cache_version'] is None
    assert all(s['cache_publication_version']==old for s in d['output_camera_provenance'][1:])
    assert torch.allclose(y[0],eva(x)[0],atol=3e-6,rtol=3e-6)
