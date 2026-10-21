"""Small measured-image tests and adversarial policy boundary tests."""
from dataclasses import replace
import copy
import cv2
import numpy as np
import pytest
from worldstream.m1 import WorldStateManager, FullOutputs
from worldstream.m1.geometry import box_geometry
from worldstream.m1.association_alignment import AlignmentSession
from worldstream.m1.association_tracks import refine
from worldstream.m1.association import associate


def seed(config,make_frame,make_prediction):
    cfg=replace(config,association_track_prior=True,association_diagnostics=True,uncertainty_px=0.)
    m=WorldStateManager(cfg);f=make_frame();f.cameras['front'].intrinsic[:2,:2]*=2/3
    f.cameras['front'].image=cv2.GaussianBlur(f.cameras['front'].image,(3,3),.7)
    p=make_prediction();p.geometry_world=box_geometry(p.geometry_world.center,[6.,6.,2.])
    r=m.precheck(f);m.commit(f,FullOutputs([p]),r)
    return m,f,p,next(iter(m.store.anchors.values()))


@pytest.mark.parametrize('scale',[1.,1.03])
def test_measured_camera_motion_and_scale(config,make_frame,make_prediction,scale):
    m,old,p,a=seed(config,make_frame,make_prediction)
    f=copy.deepcopy(old);f.timestamp=100.5;f.frame_id='next';cam=f.cameras['front'];cam.timestamp=f.timestamp
    cam.intrinsic=old.cameras['front'].intrinsic.copy();cam.intrinsic[0,0]*=scale;cam.intrinsic[1,1]*=scale
    cam.world_to_camera=np.eye(4);cam.world_to_camera[0,3]=.5
    dx=cam.intrinsic[0,0]*.5/10
    matrix=np.array([[scale,0,60*(1-scale)+dx],[0,scale,50*(1-scale)]])
    cam.image=cv2.warpAffine(old.cameras['front'].image,matrix,(120,100))
    session=AlignmentSession(m._identity_history,m.store,f,m.config,m.validator,[p])
    d=session.measure(a,0)
    assert d['status']=='supported',d
    view=d['views'][0]
    assert view['aligned'] and view['tracks']>=8 and view['withheld_alignment']['withheld_points']>=4
    assert not a.identity_link_qualified and not a.velocity_reliable


@pytest.mark.parametrize('fault',['low_texture','camera_space','camera_time','missing_history','tamper_camera','tamper_version'])
def test_unknown_measurement_sources(config,make_frame,make_prediction,fault):
    m,old,p,a=seed(config,make_frame,make_prediction)
    f=make_frame(t=100.5);f.cameras['front'].intrinsic=old.cameras['front'].intrinsic.copy()
    f.cameras['front'].image=old.cameras['front'].image.copy()
    history=copy.deepcopy(m._identity_history)
    if fault=='low_texture':f.cameras['front'].image[:]=128
    if fault=='camera_space':f.cameras['front'].coordinate_space='different'
    if fault=='camera_time':f.cameras['front'].timestamp=99
    if fault=='missing_history':history=()
    if fault=='tamper_camera':history[0]['cameras']['front'].image=history[0]['cameras']['front'].image.copy();history[0]['cameras']['front'].image[0,0]^=255
    if fault=='tamper_version':a.source_frame_id='forged'
    value=AlignmentSession(history,m.store,f,m.config,m.validator,[p]).measure(a,0)
    assert value['status']=='unknown'


def test_occlusion_unknown_not_conflict(config,make_frame,make_prediction):
    m,old,p,a=seed(config,make_frame,make_prediction)
    f=make_frame(t=100.5);f.cameras['front'].intrinsic=old.cameras['front'].intrinsic.copy()
    f.cameras['front'].image=old.cameras['front'].image.copy()
    near=copy.deepcopy(p);near.geometry_world=box_geometry([0.,0.,5.],[3.,3.,2.])
    value=AlignmentSession(m._identity_history,m.store,f,m.config,m.validator,[p,near]).measure(a,0)
    assert value['status']=='unknown' and value['reason']=='occluded'


def test_reliable_measured_exclusion_vetoes(config,make_frame,make_prediction):
    m,old,p,a=seed(config,make_frame,make_prediction)
    f=make_frame(t=100.5);f.cameras['front'].intrinsic=old.cameras['front'].intrinsic.copy()
    f.cameras['front'].image=old.cameras['front'].image.copy()
    other=copy.deepcopy(p);other.geometry_world=box_geometry([6.,0.,10.],[2.,2.,2.])
    d=AlignmentSession(m._identity_history,m.store,f,m.config,m.validator,[other]).measure(a,0)
    assert d['status']=='contradicted',d


@pytest.mark.parametrize('state',['unknown','supported'])
def test_short_history_outside_gate_needs_metric_measurement(config,make_frame,make_prediction,monkeypatch,state):
    m,old,p,a=seed(config,make_frame,make_prediction);f=make_frame(t=100.5)
    new=copy.deepcopy(p);new.geometry_world=box_geometry([3.,0.,10.],[6.,6.,2.])
    _,cost=associate([a],[new],f.timestamp,m.config)
    assert not np.isfinite(cost[0,0])
    def measured(self,anchor,j,require_metric=False):
        assert require_metric
        return dict(status=state,reason='synthetic_boundary_witness',views=[],metric_supported=state=='supported',metric_budget_m=.2 if state=='supported' else None)
    monkeypatch.setattr(AlignmentSession,'measure',measured)
    result,_,_=refine([a],[new],cost,[],set(),m.store,f,m.config,m.validator,history=m._identity_history,origin='actual_perception')
    assert result==([(0,0)] if state=='supported' else [])
    assert not a.velocity_reliable and a.physical_certification is None


def test_unknown_competitor_prevents_bootstrap(config,make_frame,make_prediction,monkeypatch):
    m,old,p,a=seed(config,make_frame,make_prediction);f=make_frame(t=100.5)
    preds=[]
    for x in (3.,3.1):
        new=copy.deepcopy(p);new.geometry_world=box_geometry([x,0.,10.],[6.,6.,2.]);preds.append(new)
    def measured(self,anchor,j,require_metric=False):
        return dict(status='supported' if j==0 else 'unknown',reason='synthetic_component',views=[],metric_supported=j==0,metric_budget_m=.2 if j==0 else None)
    monkeypatch.setattr(AlignmentSession,'measure',measured)
    result,_,_=refine([a],preds,np.full((1,2),np.inf),[],set(),m.store,f,m.config,m.validator,history=m._identity_history,origin='actual_perception')
    assert not result


def test_diagnostics_are_read_only_and_bounded(config,make_frame,make_prediction):
    values=[]
    for enabled in (False,True):
        m=WorldStateManager(replace(config,association_track_prior=True,association_diagnostics=enabled,
                                   association_diagnostic_limit=1,association_diagnostic_bytes=1024))
        for t in (100.,100.5):
            f=make_frame(t=t);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction(-.7),make_prediction(.7)]),r)
        d=m.last_commit_diagnostics['association_tracks']
        assert len(d['audit']['samples'])<=1
        assert d['audit']['sample_bytes']+d['measurement_receipt_bytes']<=1024
        values.append((m.last_commit_diagnostics['association_pairs'],m.store.source_digest()))
    assert values[0]==values[1]


def test_image_history_tamper_aborts(config,make_frame,make_prediction):
    m,old,p,a=seed(config,make_frame,make_prediction)
    f=make_frame(t=100.5);r=m.precheck(f)
    m._identity_history[0]['cameras']['front'].image=np.zeros((100,120),np.uint8)
    with pytest.raises(ValueError,match='source state changed'):m.commit(f,FullOutputs([p]),r)
    assert not m.store.association_tracks and not m._identity_history
