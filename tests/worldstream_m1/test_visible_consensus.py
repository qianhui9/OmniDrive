"""Controlled geometry tests; these are not real-population quality evidence."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from worldstream.m1 import M1Config
from worldstream.m1.surface_consensus import visible_surface_support,training_models
from worldstream.m1.identity_object import surface_geometry
from worldstream.m1.geometry import box_geometry
from test_identity_surface import triple
from worldstream.m1.temporal_visibility import build_evidence
from worldstream.m1.novelty_detector import unexplained_regions


def measured_planes():
    _,old,cur,cfg,_=triple()
    h,w=cur.image.shape[:2];yy,xx=np.mgrid[:h,:w]
    rays=np.c_[xx.ravel(),yy.ravel(),np.ones(h*w)]@np.linalg.inv(cur.intrinsic).T
    depth=np.where(xx<w//2,8.,15.).ravel()
    cam=rays*depth[:,None];world=(cam-cur.world_to_camera[:3,3])@cur.world_to_camera[:3,:3]
    projected=world@old.world_to_camera[:3,:3].T+old.world_to_camera[:3,3]
    uvh=projected@old.intrinsic.T;source=(uvh[:,:2]/uvh[:,2:]).reshape(h,w,2).astype(np.float32)
    candidate=(source[...,0]>4)&(source[...,0]<w-4)&(source[...,1]>4)&(source[...,1]<h-4)
    return old,cur,cfg,source,candidate


def test_two_visible_planes_keep_measured_subsets_without_filling_holes():
    old,cur,cfg,source,candidate=measured_planes();h,w=candidate.shape
    candidate[40:44,40:44]=False
    mask,diag=visible_surface_support(old,cur,source,candidate,replace(cfg,dense_region_policy='surface_consensus'))
    assert mask[:,:w//2-3].sum()>50 and mask[:,w//2+3:].sum()>50,diag
    assert not mask[~candidate].any()


def test_one_bad_pixel_does_not_destroy_every_supported_plane():
    old,cur,cfg,source,candidate=measured_planes()
    before,_=visible_surface_support(old,cur,source,candidate,cfg)
    source[42,42]+=[8,5]
    after,_=visible_surface_support(old,cur,source,candidate,cfg)
    assert not after[42,42] and after.sum()>.7*before.sum()


@pytest.mark.parametrize('change',['camera','space','time','calibration'])
def test_surface_contract_invalid_returns_unknown(change):
    old,cur,cfg,source,candidate=measured_planes()
    if change=='camera':old.camera_id='other'
    elif change=='space':old.coordinate_space='other'
    elif change=='time':old.timestamp=cur.timestamp
    else:old.intrinsic[:]=np.nan
    assert not visible_surface_support(old,cur,source,candidate,cfg)[0].any()


def test_low_parallax_is_not_rescued_by_plane_consensus():
    old,cur,cfg,source,candidate=measured_planes();old.world_to_camera=cur.world_to_camera.copy()
    yy,xx=np.mgrid[:candidate.shape[0],:candidate.shape[1]];source=np.stack((xx,yy),-1).astype(np.float32)
    assert not visible_surface_support(old,cur,source,candidate,cfg)[0].any()


def test_collinear_training_cannot_define_a_surface():
    old,cur,cfg,_,_=measured_planes()
    rays=np.c_[np.linspace(-.1,.1,12),np.zeros(12),np.ones(12)]
    assert not training_models(rays,np.full(12,.1),np.zeros((12,2)),cur,old)


def test_surface_needs_disjoint_keys_and_unknown_always_refreshes():
    old,cur,cfg,source,candidate=measured_planes();candidate[:]=False;candidate[30:33,30:33]=True
    assert not visible_surface_support(old,cur,source,candidate,cfg)[0].any()
    cfg=replace(cfg,dense_region_policy='surface_consensus');e=build_evidence(old,cur,cfg)
    _,invalid=unexplained_regions(cur,[],[],cfg,False,e)
    assert invalid[~e.stable_mask].all()
    _,forced=unexplained_regions(cur,[],[],cfg,True,e);assert forced.all()


@pytest.mark.parametrize('scale',[-1.,.25])
def test_folded_or_many_to_one_dense_warp_is_never_surface_support(monkeypatch,scale):
    import cv2
    old,cur,cfg,_,_=measured_planes();h,w=cur.image.shape[:2];yy,xx=np.mgrid[:h,:w]
    calls=[]
    class SyntheticFlow:
        def calc(self,a,b,unused):
            reverse=len(calls)%2==1;calls.append(1)
            factor=scale if reverse else 1/scale
            source=(xx-w/2)*factor+w/2
            return np.stack((source-xx,np.zeros_like(xx)),axis=-1).astype(np.float32)
    monkeypatch.setattr(cv2,'DISOpticalFlow_create',lambda *a:SyntheticFlow())
    cfg=replace(cfg,dense_region_policy='surface_consensus',surface_use_history=False)
    e=build_evidence(old,cur,cfg)
    assert not e.stable_mask.any()


def test_surface_observation_claim_does_not_prove_instance_ownership():
    old,cur,cfg,source,candidate=measured_planes()
    # A coherent erroneous correspondence can explain a plane. RGB consensus
    # deliberately exposes this limit; it does not manufacture identity truth.
    mask,diag=visible_surface_support(old,cur,source,candidate,cfg)
    assert mask.any() and 'not_physical_ownership' in diag['surface_claim']


def test_manager_captures_actual_history_for_consensus_policy():
    from worldstream.m1 import WorldStateManager,Frame,FullOutputs
    old,cur,cfg,_,_=measured_planes();cfg=replace(cfg,dense_region_policy='surface_consensus')
    m=WorldStateManager(cfg)
    def frame(c):return Frame('s',str(c.timestamp),c.timestamp,np.eye(4),{c.camera_id:c})
    a,b=frame(old),frame(cur);m.initialize(a,FullOutputs([]));r=m.precheck(b);m.commit(b,FullOutputs([]),r)
    assert cur.camera_id in m._surface_history
    earlier,stamp,mapping=m._surface_history[cur.camera_id]
    assert earlier.timestamp==old.timestamp and stamp==cur.timestamp
    assert not mapping.flags.writeable


def test_consensus_funnel_counts_actual_pixel_decision():
    from worldstream.m1.dense_support import dense_support
    _,old,cur,cfg,_=triple();cfg=replace(cfg,dense_region_policy='surface_consensus')
    mask,source,diag=dense_support(old,cur,cfg)
    funnel=diag['acceptance_funnel']
    assert funnel['visible_surface_consensus']['retained']==int(mask.sum())
    assert not funnel['complete_neighbourhood_baseline_diagnostic']['applied']


def test_object_motion_geometry_requires_real_background_separation():
    old,cur,cfg,_,_=measured_planes()
    a=SimpleNamespace(geometry_world=box_geometry([0,0,10],[8,6,2]))
    b=SimpleNamespace(geometry_world=box_geometry([.8,0,10],[8,6,2]))
    xyz=np.array([[-2,-2,10],[2,-2,10],[-2,2,10],[2,2,10]],float)
    ring=np.array([[-8,-4,20],[8,-4,20],[-8,4,20],[8,4,20]],float)
    def project(x,c):
        z=x@c.world_to_camera[:3,:3].T+c.world_to_camera[:3,3];uv=z@c.intrinsic.T;return uv[:,:2]/uv[:,2:]
    p=project(xyz,old);q=project(xyz+[.8,0,0],cur);rp=project(ring,old);rq=project(ring,cur)
    foreground,background,gap,_=surface_geometry(a,b,old,cur,p,q,rp,rq,cfg)
    assert foreground.all() and background.all() and (gap>5).all()
    # Identical depth gives no measured ownership separation.
    ring[:,2]=10;rp=project(ring,old);rq=project(ring,cur)
    _,_,gap,_=surface_geometry(a,b,old,cur,p,q,rp,rq,cfg)
    assert gap is not None and (gap<cfg.foreground_depth_gap_m).all()
