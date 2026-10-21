"""Single-path identity lifecycle, provenance and transaction tests."""
from dataclasses import replace
import copy
import numpy as np
import pytest
from worldstream.m1 import WorldStateManager,FullOutputs
from worldstream.m1.geometry import box_geometry
from worldstream.m1.association_tracks import candidates,predict,prune


def commit(manager,frame,predictions):
    r=manager.precheck(frame);manager.commit(frame,FullOutputs(predictions),r)
    return manager.last_commit_diagnostics['committed_anchor_ids']


def test_normal_continuity_and_bounded_history(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,association_track_prior=True,association_track_frames=3))
    for n in range(6):
        ids=commit(m,make_frame(t=100+n*.25),[make_prediction(n*.25)])
        assert ids==['scene:g2:a1']
    record=m.store.association_tracks[ids[0]]
    assert len(record['samples'])==3
    assert record['samples'][-1]['timestamp']==101.25
    assert not m._identity_history  # ordinary association owns no image history
    assert 'rois' not in record and 'observation_support' not in record['snapshot']
    a=m.store.anchors[ids[0]]
    assert not a.identity_link_qualified and not a.velocity_reliable and a.physical_certification is None


def test_history_prediction_enters_only_allocator(config,make_frame,make_prediction,monkeypatch):
    import scipy.optimize
    count=[];original=scipy.optimize.linear_sum_assignment
    def solve(matrix):count.append(matrix.shape);return original(matrix)
    monkeypatch.setattr(scipy.optimize,'linear_sum_assignment',solve)
    m=WorldStateManager(replace(config,association_track_prior=True))
    for t in (100.,100.1,100.2,100.7):
        ids=commit(m,make_frame(t=t),[make_prediction((t-100)*5)])
        assert ids==['scene:g2:a1']
    # First frame has no source rows; all other frames allocate exactly once.
    assert len(count)==3
    a=m.store.anchors[ids[0]]
    center,u=predict(a,101.,m.config,m.store.association_tracks[a.anchor_id])
    assert np.allclose(center,[5.,0.,10.]) and u<1e-8


def test_unstable_history_rejected_without_fallback(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,association_track_prior=True))
    for n,x in enumerate((0.,.5,-.5)):
        aid=commit(m,make_frame(t=100+n*.1),[make_prediction(x)])[0]
    a=m.store.anchors[aid]
    assert predict(a,101.,m.config,m.store.association_tracks[aid]) is None


@pytest.mark.parametrize('fault',['class','size','distance'])
def test_incompatible_predictions_create_identity(config,make_frame,make_prediction,fault):
    m=WorldStateManager(replace(config,association_track_prior=True))
    old=commit(m,make_frame(),[make_prediction()])
    p=make_prediction()
    if fault=='class':p.semantic_class='car'
    if fault=='size':p.geometry_world=box_geometry([0.,0.,10.],[10.,10.,10.])
    if fault=='distance':p=make_prediction(10.)
    assert commit(m,make_frame(t=100.5),[p])!=old


def test_ambiguous_competitors_create_new_ids(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,association_track_prior=True))
    old=commit(m,make_frame(),[make_prediction(-.5),make_prediction(.5)])
    new=commit(m,make_frame(t=100.5),[make_prediction(0.),make_prediction(0.)])
    assert len(set(new))==2 and not set(old)&set(new)
    assert not m.last_commit_diagnostics['association_pairs']


@pytest.mark.parametrize('fault',['scene','epoch','signature','origin','version'])
def test_incompatible_history_is_never_used(config,make_frame,make_prediction,fault):
    m=WorldStateManager(replace(config,association_track_prior=True))
    for n in range(3):aid=commit(m,make_frame(t=100+n*.1),[make_prediction()])[0]
    r=m.store.association_tracks[aid]
    if fault=='version':r['samples'][-1]['version']='forged'
    else:r[dict(scene='scene_id',epoch='epoch',signature='signature',origin='origin')[fault]]='bad'
    assert candidates(m.store,make_frame(t=100.5),m.config,'actual_perception')[1][0] is None
    assert commit(m,make_frame(t=100.5),[make_prediction()])!=[aid]


def test_dormant_uses_same_gate_and_inherits_no_qualification(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,association_track_prior=True,max_state_age_s=.6))
    for n in range(3):aid=commit(m,make_frame(t=100+n*.25),[make_prediction()])[0]
    old=m.store.anchors[aid];old.last_verified_timestamp=100.5;old.certified_observation_count=7
    commit(m,make_frame(t=101.25),[])
    assert aid not in m.store.anchors
    assert commit(m,make_frame(t=101.5),[make_prediction()])==[aid]
    a=m.store.anchors[aid]
    assert a.association_basis=='uncertified_dormant_identity_recovery'
    assert a.last_verified_timestamp is None and a.certified_observation_count==0
    assert a.physical_certification is None and not a.identity_link_qualified
    assert m.last_commit_diagnostics['association_tracks']['recovered']==[aid]
    assert not m.last_result.valid_anchor_ids
    assert all(f.computation_timestamp==101.5 for f in m.store.features.values())


def test_expired_or_outside_dormant_is_not_forced(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,association_track_prior=True,max_state_age_s=.3,association_track_ttl_s=.9))
    aid=commit(m,make_frame(),[make_prediction()])[0]
    commit(m,make_frame(t=100.5),[])
    assert commit(m,make_frame(t=101.),[make_prediction()])!=[aid]
    assert aid not in m.store.association_tracks


def test_tamper_and_commit_failure_are_atomic(config,make_frame,make_prediction,monkeypatch):
    import worldstream.m1.association_tracks as tracks
    m=WorldStateManager(replace(config,association_track_prior=True))
    aid=commit(m,make_frame(),[make_prediction()])[0];before=m.store.source_digest()
    original=tracks.publish
    def fail(*args,**kwargs):original(*args,**kwargs);raise RuntimeError('injected')
    monkeypatch.setattr(tracks,'publish',fail)
    f=make_frame(t=100.5);r=m.precheck(f)
    with pytest.raises(RuntimeError):m.commit(f,FullOutputs([make_prediction()]),r)
    assert m.store.source_digest()==before
    monkeypatch.setattr(tracks,'publish',original)
    f=make_frame(t=101.);r=m.precheck(f)
    m.store.association_tracks[aid]['samples'][0]['center'][0]+=1
    with pytest.raises(ValueError,match='source state changed'):m.commit(f,FullOutputs([make_prediction()]),r)
    assert not m.store.association_tracks


def test_budget_and_reset(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,association_track_prior=True,association_track_limit=1))
    commit(m,make_frame(),[make_prediction(0.),make_prediction(8.)])
    assert len(m.store.association_tracks)==1
    small=replace(m.config,association_track_bytes=1024)
    assert prune(m.store,100.,small)['accounted_memory_bytes']<=1024
    m.reset();assert not m.store.association_tracks


@pytest.mark.parametrize('option',[dict(association_observation_guard=True),dict(dqtrack_mode='hint'),
    dict(identity_tracking_policy='experimental'),dict(identity_tracking_policy='certified_only'),
    dict(dqtrack_association_policy='calibrated_tiebreak')])
def test_retired_assigners_fail_explicitly(config,option):
    with pytest.raises(ValueError,match='Retired identity-assignment'):WorldStateManager(replace(config,**option))


@pytest.mark.parametrize('samples',[1,3])
def test_same_age_budget_limits_short_and_mature_tracks(config,make_frame,make_prediction,samples):
    m=WorldStateManager(replace(config,association_track_prior=True,velocity_uncertainty_mps=2.))
    for n in range(samples):aid=commit(m,make_frame(t=100+n*.25),[make_prediction()])[0]
    a=m.store.anchors[aid];record=m.store.association_tracks[aid]
    assert predict(a,a.source_timestamp+1.1,m.config,record) is None
    center,budget=predict(a,a.source_timestamp+.5,m.config,record)
    assert np.allclose(center,a.geometry_world.center) and budget==1.


def test_negative_age_budget_rejected(config):
    with pytest.raises(ValueError,match='Velocity uncertainty'):replace(config,velocity_uncertainty_mps=-1.)
