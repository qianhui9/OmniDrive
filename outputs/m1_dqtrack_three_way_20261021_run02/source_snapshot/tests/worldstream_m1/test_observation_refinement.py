from dataclasses import replace
import numpy as np
from worldstream.m1.observation_refinement import scope_support,measured_links,change_guard
from worldstream.m1.measurement_cache import measurement_cache
from worldstream.m1.geometry import valid_transform
from worldstream.m1 import M1Config
from worldstream.m1.state_types import OnlineVisibilityEvidence,CameraFrame


def test_partial_support_never_claims_complete_dependency():
    candidate=np.zeros((16,16),bool);candidate[3,3]=True;dirty=np.zeros_like(candidate)
    s=scope_support(candidate,dirty)
    assert s['observation_support_count']==1 and s['observation_support_ratio']==1/256
    assert s['partial_observation_only'] and not s['observation_valid']
    f=dict(token_region=[0,0,2,2],kernel=[2,2],stride=[3,3],padding=[0,0],dilation=[2,2])
    candidate[:]=True;candidate[1,1]=False
    s=scope_support(candidate,dirty,f)
    assert s['dependency_pixel_count']==16 and s['observation_valid']
    dirty[0,0]=True
    assert scope_support(candidate,dirty,f)['observed_conflict_count']==1


def test_new_flags_preserve_legacy_positional_config():
    assert M1Config(False).enabled is False
    assert not M1Config(False).scope_validity


def test_sparse_centres_recovered_without_patch_fill_and_conflicts_deduplicate():
    im=np.full((32,32,3),100,np.uint8)
    old=CameraFrame('c',im,np.eye(3),np.eye(4),1.)
    new=replace(old,timestamp=1.5)
    p=np.array([[5,5],[10,10],[15,15]],np.float32)
    e=OnlineVisibilityEvidence('c',1.,1.5,p,p.copy(),np.zeros(3),np.ones(3,bool),
        np.ones(3,bool),np.ones(3,bool),np.zeros(3),np.zeros((32,32),bool),np.zeros((32,32),bool))
    e.dense_source_map=np.indices((32,32))[::-1].transpose(1,2,0).astype(np.float32)
    dense=e.stable_mask.copy();dense[5,5]=True;dense[15,15]=True
    e.dense_source_map[15,15]=[20,20]
    dirty=np.zeros_like(dense)
    c,source,target,residual,origin,d,conflict=measured_links(old,new,e,dense,dirty,M1Config())
    assert c.sum()==2 and c[5,5] and c[10,10] and not c[15,15]
    assert d['sparse_only']==1 and d['both']==1 and d['measurement_conflicts']==1
    assert sorted(origin.tolist())==[1,2]
    assert np.all(residual==0)
    e.observation_exclusion_mask=np.zeros_like(dense);e.observation_exclusion_mask[10,10]=True
    c,*_=measured_links(old,new,e,dense,dirty,M1Config())
    assert not c[10,10]


def test_change_guard_only_dilates_measured_local_seeds():
    from types import SimpleNamespace
    d=np.zeros((32,32),bool);d[15,15]=True
    guarded,diag=change_guard(d,SimpleNamespace(measured_change_risk_mask=None),M1Config())
    assert guarded.sum()==49 and not guarded[0,0] and diag['seed_pixels']==1


def test_transform_measurement_cache_cannot_hide_mutation():
    m=np.eye(4)
    with measurement_cache(True):
        assert valid_transform(m)
        m[0,0]=2
        assert not valid_transform(m)


def test_scope_bundle_keeps_partial_support_separate_from_cache(config,make_frame):
    from worldstream.m1 import WorldStateManager,FullOutputs
    cfg=replace(config,source_linked_state=True,local_support=True,temporal_visibility=True,
        scope_validity=True,recover_sparse_links=True,change_first_guard=True,efficient_state=True)
    m=WorldStateManager(cfg);a=make_frame();r=m.precheck(a);m.commit(a,FullOutputs([]),r)
    b=make_frame(100.5);r=m.precheck(b)
    for s in r.computation_state.source_records:
        assert 'observation_support_count' in s and 'observed_conflict_count' in s
        assert not s['observation_valid'] and not s['computation_available']
        assert not s['transformer_dependency_certified']


def test_dense_link_residual_sampling_handles_more_than_opencv_short_limit():
    im=np.full((200,200,3),100,np.uint8);old=CameraFrame('c',im,np.eye(3),np.eye(4),1.)
    p=np.empty((0,2),np.float32);z=np.empty(0,bool)
    e=OnlineVisibilityEvidence('c',1.,1.5,p,p,np.empty(0),z,z,z,np.empty(0),
        np.ones((200,200),bool),np.zeros((200,200),bool))
    e.dense_source_map=np.indices((200,200))[::-1].transpose(1,2,0).astype(np.float32)
    result=measured_links(old,replace(old,timestamp=1.5),e,e.stable_mask,e.changed_mask,M1Config())
    assert len(result[3])==40000 and np.all(result[3]==0)


def test_committed_cache_source_survives_missing_m1_feature_slot(config,make_frame):
    from worldstream.m1 import WorldStateManager,FullOutputs
    cfg=replace(config,source_linked_state=True,local_support=True,temporal_visibility=True,scope_validity=True)
    m=WorldStateManager(cfg);a=make_frame();r=m.precheck(a)
    sources=[dict(camera_id=cid,computation_frame=a.frame_id,computation_timestamp=a.timestamp,
        computation_version=7) for cid in a.cameras]
    receipt=dict(cache_saved=True,cache_boundary='camera',cache_version=7,
        cache_context=dict(frame_id=a.frame_id,execution_signature=a.execution_signature),
        output_camera_provenance=sources)
    m.commit(a,FullOutputs([],execution_receipt=receipt),r)
    assert not m.store.features
    b=make_frame(100.5);r=m.precheck(b)
    computed=[s for s in r.computation_state.source_records if s['feature_id']=='backbone']
    assert len(computed)==len(a.cameras)
    assert all(s['computation_available'] and s['computation_frame_id']==a.frame_id for s in computed)
    assert all(s['feature_record_version'] is None and s['computation_version']==7 for s in computed)
    assert not m.store.features


def test_descriptor_scope_counts_actual_historical_crop_only():
    from worldstream.m1.observation_refinement import source_centre_integrals,descriptor_scope_support
    dirty=np.zeros((16,16),bool)
    source=np.array([[3,4],[12,12]],np.float32)
    integrals=source_centre_integrals(source,None,dirty)
    s=descriptor_scope_support([2,3,6,7],integrals,dirty.shape)
    assert s['dependency_pixel_count']==16
    assert s['observation_support_count']==1 and s['observation_support_ratio']==1/16
    assert s['source_region']==[[2,3,6,7]]
    assert s['partial_observation_only'] and not s['observation_valid']
    assert descriptor_scope_support([2,3,6,7],integrals,dirty.shape,False)['observation_support_count']==0
