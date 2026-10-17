"""A verified cache value is not newly computed by the current M1 commit."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from worldstream.m1 import M1Config
from worldstream.m1.state_store import StateStore
from worldstream.m1.state_types import ComputedFeature
from worldstream.m1.provenance import feature_source_snapshot,preserve_camera_computation_source


def setup_sources():
    store=StateStore(M1Config())
    def frame(t):
        return SimpleNamespace(frame_id='f'+str(t),timestamp=float(t),execution_signature='weights',
            cameras={cid:SimpleNamespace(coordinate_space='image') for cid in ('a','b')})
    def feature(fid):
        return ComputedFeature(fid,fid,(2,3),'float32','cpu','EVA' if fid=='backbone' else 'current_head',
            source_camera_ids=['a','b'])
    store.put_feature(feature('backbone'),frame(1))
    previous=feature_source_snapshot(store.features['backbone'])
    store.put_feature(feature('backbone'),frame(2))
    store.put_feature(feature('object_carrier'),frame(2))
    store.provenance['object_carrier'].upstream_versions['backbone']=2
    return store,previous,frame(2)


def test_full_camera_reuse_preserves_computation_not_verification_time():
    store,previous,frame=setup_sources()
    sources=[dict(camera_id=c,computation_frame='f1',computation_timestamp=1.,
        verification_frame='f2',verification_timestamp=2.) for c in ('a','b')]
    proof=preserve_camera_computation_source(store,frame,dict(actual_reused_cameras=2,
        output_camera_provenance=sources),previous)
    record=store.features['backbone']
    assert record.computation_version==1 and record.computation_timestamp==1.
    assert record.source_frame_id=='f1' and record.source_state_epoch==previous['source_state_epoch']
    assert record.last_verified_timestamp is None and proof['physical_state_certified'] is False
    assert store.provenance['object_carrier'].upstream_versions['backbone']==1
    assert store.features['object_carrier'].computation_timestamp==2.


def test_mixed_camera_sources_are_explicit_not_all_current():
    store,previous,frame=setup_sources()
    sources=[dict(camera_id='a',computation_frame='f1',computation_timestamp=1.),
             dict(camera_id='b',computation_frame='f2',computation_timestamp=2.)]
    preserve_camera_computation_source(store,frame,dict(actual_reused_cameras=1,
        output_camera_provenance=sources),previous)
    record=store.features['backbone']
    assert record.source_frame_id is None and record.source_state_epoch is None
    assert record.computation_timestamp==1. and record.computation_version==2
    assert record.computation_sources==sources


def test_missing_cached_source_is_unknown_and_cannot_survive_freshness_prune():
    store,previous,frame=setup_sources()
    preserve_camera_computation_source(store,frame,dict(actual_reused_cameras=2),previous)
    record=store.features['backbone']
    assert record.source_frame_id is None and record.computation_timestamp is None
    assert record.computation_version==0
    json.dumps(store.feature_versions(),allow_nan=False)
    store.prune(2.)
    assert 'backbone' not in store.features


@torch.no_grad()
def test_output_lineage_survives_failed_cache_publish(eva,context,monkeypatch):
    from worldstream.m2 import IncrementalVisionRuntime
    runtime=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera'))
    image=torch.randn(2,3,40,40)
    runtime.forward(image,execution_context=dict(context,timestamp=1.))
    cache=runtime.camera_executor.cache
    def oom(*a,**kw):raise torch.cuda.OutOfMemoryError('controlled publish failure')
    monkeypatch.setattr(cache,'publish',oom)
    output,diag=runtime.forward(image,execution_context=dict(context,frame_id='next',timestamp=2.))
    assert diag['actual_reused_cameras']==2 and not diag['cache_saved'] and cache.bytes==0
    assert {s['computation_frame'] for s in diag['output_camera_provenance']}=={context['frame_id']}
    assert all(s['computation_timestamp']==1. and s['verification_timestamp']==2. for s in diag['output_camera_provenance'])
    assert torch.equal(output[0],eva(image)[0])


@torch.no_grad()
def test_numeric_policy_change_invalidates_camera_computation(eva,context,monkeypatch):
    from worldstream.m2 import IncrementalVisionRuntime
    from worldstream.m2.camera_executor import camera_cost_key
    runtime=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera'))
    image=torch.randn(2,3,40,40)
    runtime.forward(image,execution_context=dict(context,timestamp=1.))
    key=camera_cost_key(eva,image)
    flag=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_fp16_reduced_precision_reduction',not flag)
    assert camera_cost_key(eva,image)!=key
    _,diag=runtime.forward(image,execution_context=dict(context,frame_id='next',timestamp=2.))
    assert diag['actual_reused_cameras']==0 and diag['fallback_reason']=='execution_signature_mismatch'
