from dataclasses import replace
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from worldstream.m1 import M1Config,WorldStateManager,FullOutputs
from worldstream.m1.observation_refinement import motion_risk_masks
from worldstream.m1.measurement_cache import measurement_cache,gray_measurement


def evidence(shape):
    return SimpleNamespace(changed_mask=np.zeros(shape,bool),measured_change_risk_mask=None,
        dense_source_map=np.indices(shape)[::-1].transpose(1,2,0).astype(np.float32),
        current_points=np.empty((0,2)),fb_error_px=np.empty(0),correspondence_valid=np.empty(0,bool))


def test_measured_dynamic_rgb_self_consistency_still_revokes_input_change(make_frame,make_anchor,config):
    old=make_frame().cameras['front'];current=make_frame(100.5).cameras['front']
    current.image=current.image.copy();current.image[48:53,58:65]=255-old.image[48:53,58:65]
    anchor=make_anchor(semantic='car',velocity=np.array([2.,0,0]))
    anchor.source_frame_id='100';anchor.source_execution_signature='default';anchor.source_state_epoch=1
    anchor.observation_support={'front':SimpleNamespace(region=(52,42,69,59),timestamp=100.)}
    e=evidence(old.image.shape[:2])
    dirty,unknown,details=motion_risk_masks([anchor],old,current,e,config,'scene','default')
    assert dirty[50,60] and dirty.sum()<current.image.shape[0]*current.image.shape[1]
    # The real Store owns grayscale history, current inputs remain RGB.
    from worldstream.m1.temporal_visibility import gray_image
    d2,u2,_=motion_risk_masks([anchor],replace(old,image=gray_image(old)),current,e,config,'scene','default')
    assert np.array_equal(dirty,d2) and np.array_equal(unknown,u2)
    assert len(details['envelopes'])==1 and not details['envelopes'][0]['foreground_ownership']
    # Identical input with motion metadata is not labelled changed.
    d,u,_=motion_risk_masks([anchor],old,replace(current,image=old.image),e,config,'scene','default')
    assert not d.any() and not u.any()


@pytest.mark.parametrize('fault',['expired','signature','scene','missing_velocity','bad_velocity','behind'])
def test_unbound_or_unbounded_motion_does_not_invent_envelope(fault,make_frame,make_anchor,config):
    old=make_frame().cameras['front'];current=make_frame(100.5,change=True).cameras['front']
    a=make_anchor(semantic='car',velocity=np.array([1.,0,0]));a.source_frame_id='100';a.source_execution_signature='default'
    a.observation_support={'front':SimpleNamespace(region=(52,42,69,59),timestamp=100.)}
    if fault=='expired':a.source_timestamp=1.
    if fault=='signature':a.source_execution_signature='other'
    if fault=='scene':a.scene_id='other'
    if fault=='missing_velocity':a.velocity_world=None
    if fault=='bad_velocity':a.velocity_world=np.array([np.nan,0,0])
    if fault=='behind':a.geometry_world.points[:,2]=-1
    d,u,details=motion_risk_masks([a],old,current,evidence(old.image.shape[:2]),config,'scene','default')
    assert not d.any() and not u.any() and not details['envelopes']


def test_no_anchor_keeps_measured_change_and_failed_visibility_unknown(make_frame,config):
    old=make_frame().cameras['front'];current=make_frame(100.5).cameras['front'];e=evidence(old.image.shape[:2])
    e.changed_mask[40,40]=True;e.current_points=np.array([[30,30]],float)
    e.fb_error_px=np.array([2.]);e.correspondence_valid=np.array([False])
    d,u,details=motion_risk_masks([],old,current,e,config,'scene','default')
    assert d.sum()==1 and d[40,40] and u.sum()==1 and u[30,30]
    assert details['anchors_considered']==0


def test_motion_revokes_candidate_preserves_actual_link(make_frame,config):
    from worldstream.m1.computation_state import build_bundle
    from worldstream.m1.state_types import M1Result,OnlineVisibilityEvidence
    cfg=replace(config,source_linked_state=True,scope_validity=True,motion_risk_revocation=True,
        temporal_visibility=True,local_support=True)
    m=WorldStateManager(cfg);old=make_frame();new=make_frame(100.5)
    m.store.last_timestamp=old.timestamp;m._previous_frame_id=old.frame_id;m._previous_cameras=old.cameras
    p=np.array([[50,50]],np.float32);z=np.ones(1,bool);shape=(100,120)
    e=OnlineVisibilityEvidence('front',100.,100.5,p,p,np.zeros(1),z,z,z,np.zeros(1),np.zeros(shape,bool),np.zeros(shape,bool))
    e.stable_mask[50,50]=True;e.dense_source_map=evidence(shape).dense_source_map
    e.motion_risk_dirty_mask=np.zeros(shape,bool);e.motion_risk_dirty_mask[50,50]=True
    e.motion_risk_unknown_mask=np.zeros(shape,bool);m.last_evidence={'front':e}
    result=M1Result(new.scene_id,new.frame_id,new.timestamp)
    from worldstream.m1.mask_codec import encode_mask,decode_mask
    result.invalidation_regions={'front':dict(refresh_required_mask=encode_mask(np.ones(shape,bool)),exact_invalidation_mask=encode_mask(np.ones(shape,bool)))}
    b=build_bundle(m,new,result);link=b.observation_links['front']
    assert len(link['current_points'])==1 and link['observation_link_valid'][0]
    assert not link['reuse_candidate_eligible'][0]
    assert not decode_mask(b.refresh_contract['front']['reuse_candidate_mask']).any()
    assert decode_mask(b.refresh_contract['front']['dirty_hint_mask'])[50,50]


def test_measurement_cache_is_exact_invocation_scoped_and_owned():
    rng=np.random.default_rng(17);g=rng.integers(0,256,(80,96),np.uint8)
    with measurement_cache(True,True):
        for axis in (0,1):
            expected=cv2.Sobel(g.astype(np.float32)/255.,cv2.CV_32F,int(axis==0),int(axis==1),ksize=3)/8
            assert np.array_equal(gray_measurement(g,axis),expected)
            assert gray_measurement(g,axis) is gray_measurement(g,axis)
        before=gray_measurement(g)
    g[:]=0
    with measurement_cache(True,True):assert not gray_measurement(g).any()
    assert before.any()


def test_measurement_optimization_preserves_all_patch_decisions():
    from worldstream.m1.geometry_support import measured_patch_support
    rng=np.random.default_rng(41);a=rng.integers(0,256,(80,96),np.uint8);b=a.copy()
    p=np.array([[20,20],[30,30],[40,40]],np.float32);q=p.copy()
    with measurement_cache(True,False):base=measured_patch_support(a,b,p,q,np.ones(3,bool),M1Config())
    with measurement_cache(True,True):new=measured_patch_support(a,b,p,q,np.ones(3,bool),M1Config())
    assert all(np.array_equal(x,y) for x,y in zip(base,new))


def test_final_flags_keep_full_refresh_scene_change_and_commit(make_frame,config):
    cfg=replace(config,source_linked_state=True,local_support=True,temporal_visibility=True,
        motion_risk_revocation=True,measurement_reuse=True,efficient_state=True)
    m=WorldStateManager(cfg)
    for frame in (make_frame(),make_frame(100.5),make_frame(101.,scene='next')):
        r=m.precheck(frame)
        if frame.frame_id in ('100.0','101.0'):assert r.full_refresh_hint
        m.commit(frame,FullOutputs([]),r)
        assert m.store.last_timestamp==frame.timestamp
