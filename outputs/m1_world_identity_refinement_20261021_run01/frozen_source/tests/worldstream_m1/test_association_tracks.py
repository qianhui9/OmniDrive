from dataclasses import replace
import copy
import numpy as np
import pytest
from worldstream.m1 import WorldStateManager, FullOutputs
from worldstream.m1.association_tracks import predict, candidates, prune


def commit(m,frame,preds):
    r=m.precheck(frame);m.commit(frame,FullOutputs(preds),r)
    return m.last_commit_diagnostics['committed_anchor_ids']


def test_robust_prediction_budget(config):
    h={'samples':[dict(timestamp=t,center=[t*2,0,0],frame_id=str(t)) for t in (0,.5,1)]}
    center,u=predict(h,1.5,config)
    assert np.allclose(center,[3,0,0]) and u==0
    h['samples'][-1]['center']=[20,0,0]
    assert predict(h,1.5,config) is None


def test_committed_history_physical_separation(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,association_track_prior=True))
    for n in range(4):
        ids=commit(m,make_frame(t=100+n*.5),[make_prediction(n*.5)])
    a=m.store.anchors[ids[0]];h=m.store.association_tracks[ids[0]]
    assert len(h['samples'])==4
    assert not a.identity_link_qualified and not a.velocity_reliable
    assert a.physical_certification is None and not a.motion_history
    frame=make_frame(t=102)
    _,priors,_=candidates(m.store,frame,m.config,'actual_perception')
    assert len(priors)==1
    assert not candidates(m.store,frame,m.config,'different_source')[1]
    before=m.store.source_digest()
    m.precheck(frame);m.abort_pending()
    assert m.store.source_digest()==before


def test_fast_prediction_after_reliable_seed(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,association_track_prior=True))
    original=None
    for n,t in enumerate((100,100.1,100.2,100.7)):
        # Real measured texture displacement, not a moving box over unchanged RGB.
        import cv2
        from worldstream.m1.geometry import box_geometry
        frame=make_frame(t=t)
        frame.cameras['front'].intrinsic[0,0]=40.;frame.cameras['front'].intrinsic[1,1]=40.
        frame.cameras['front'].image=cv2.warpAffine(frame.cameras['front'].image,
            np.array([[1.,0.,(t-100)*20],[0.,1.,0.]]),(120,100))
        pred=make_prediction((t-100)*5)
        pred.geometry_world=box_geometry(pred.geometry_world.center,[6.,6.,2.])
        ids=commit(m,frame,[pred])
        original=original or ids[0]
        assert ids==[original]
    assert len(m.store.association_tracks[original]['samples'])==4
    assert not m.store.anchors[original].velocity_reliable


def test_dormant_restore_no_old_feature_or_certification(config,make_frame,make_prediction):
    cfg=replace(config,association_track_prior=True,max_state_age_s=.6,uncertainty_px=0.)
    m=WorldStateManager(cfg)
    from worldstream.m1.geometry import box_geometry
    pred=make_prediction();pred.geometry_world=box_geometry(pred.geometry_world.center,[4.,4.,2.])
    for n in range(3):
        aid=commit(m,make_frame(t=100+n*.25),[pred])[0]
    commit(m,make_frame(t=101.25),[])
    assert aid not in m.store.anchors and aid in m.store.association_tracks
    ids=commit(m,make_frame(t=101.5),[pred])
    assert ids==[aid]
    a=m.store.anchors[aid]
    assert a.association_basis=='uncertified_dormant_identity_recovery'
    assert a.physical_certification is None and not a.identity_link_qualified
    assert a.last_verified_timestamp is None and a.certified_observation_count==0
    assert not m.last_result.valid_anchor_ids


@pytest.mark.parametrize('fault',['scene','epoch','signature','version','origin'])
def test_bound_rejects_incompatible_history(config,make_frame,make_prediction,fault):
    m=WorldStateManager(replace(config,association_track_prior=True))
    for n in range(3): aid=commit(m,make_frame(t=100+n*.25),[make_prediction()])[0]
    h=m.store.association_tracks[aid]
    if fault=='version':h['samples'][-1]['version']='forged'
    else:h[{'scene':'scene_id','signature':'signature','epoch':'epoch','origin':'origin'}[fault]]='bad'
    assert not candidates(m.store,make_frame(t=101),m.config,'actual_perception')[1]


def test_bounded_and_reset(config,make_frame,make_prediction):
    cfg=replace(config,association_track_prior=True,association_track_limit=2,association_track_frames=3)
    m=WorldStateManager(cfg)
    for n in range(6):commit(m,make_frame(t=100+n*.25),[make_prediction(-5),make_prediction(0),make_prediction(5)])
    assert len(m.store.association_tracks)<=2
    assert all(len(r['samples'])<=3 for r in m.store.association_tracks.values())
    assert prune(m.store,110,cfg)['tracks']==0
    m.reset();assert not m.store.association_tracks


def test_unknown_appearance_cannot_recover(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,association_track_prior=True,max_state_age_s=.6))
    for n in range(3):aid=commit(m,make_frame(t=100+n*.25),[make_prediction()])[0]
    commit(m,make_frame(t=101.25),[])
    f=make_frame(t=101.5);f.cameras['front'].image[:]=128
    assert commit(m,f,[make_prediction()])!=[aid]


def test_pending_history_tamper_discards_source(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,association_track_prior=True))
    aid=commit(m,make_frame(),[make_prediction()])[0]
    f=make_frame(t=100.5);r=m.precheck(f)
    m.store.association_tracks[aid]['samples'][0]['center'][0]+=1
    with pytest.raises(ValueError,match='source state changed'):m.commit(f,FullOutputs([make_prediction()]),r)
    assert not m.store.association_tracks


def test_competition_not_written_back(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,association_track_prior=True))
    ids=commit(m,make_frame(),[make_prediction(-.7),make_prediction(.7)])
    # Equal geometry gives an exact tie. No labels/descriptors turn missing
    # evidence into an association, and rejected IDs keep only committed seeds.
    f=make_frame(t=100.5);f.cameras['front'].image[:]=128
    new=commit(m,f,[make_prediction(0),make_prediction(0)])
    assert not set(ids)&set(new)
    assert all(len(m.store.association_tracks[aid]['samples'])==1 for aid in ids)


def test_commit_failure_does_not_publish_history(config,make_frame,make_prediction,monkeypatch):
    import worldstream.m1.association_tracks as tracks
    m=WorldStateManager(replace(config,association_track_prior=True))
    commit(m,make_frame(),[make_prediction()]);before=m.store.source_digest()
    original=tracks.publish
    def fail(*args,**kwargs):
        original(*args,**kwargs)
        raise RuntimeError('injected failure after history write')
    monkeypatch.setattr(tracks,'publish',fail)
    f=make_frame(t=100.5);r=m.precheck(f)
    with pytest.raises(RuntimeError,match='injected'):m.commit(f,FullOutputs([make_prediction()]),r)
    assert m.store.source_digest()==before


def test_view_change_unknown_and_memory_bound(config,make_frame,make_prediction):
    import worldstream.m1.association_tracks as tracks
    m=WorldStateManager(replace(config,association_track_prior=True))
    for n in range(3):aid=commit(m,make_frame(t=100+n*.25),[make_prediction()])[0]
    record=copy.deepcopy(m.store.association_tracks[aid])
    for h in record['samples']:h['rois']['front']['ray']=[0,0,-1]
    target=m.store.anchors[aid]
    assert tracks.appearance(record,target,make_frame(t=101),m.config,m.validator,{},0) is None
    tiny=replace(m.config,association_track_bytes=1024)
    d=prune(m.store,100.5,tiny)
    assert d['accounted_memory_bytes']<=1024 and d['serialized_bytes']<=1024
    assert not m.store.association_tracks
