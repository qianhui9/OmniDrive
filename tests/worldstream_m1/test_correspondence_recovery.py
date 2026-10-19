"""Controlled geometry and decision tests; empirical quality is scored offline."""
import copy
from dataclasses import replace
import cv2
import numpy as np
import pytest
from worldstream.m1 import CameraFrame, M1Config, WorldStateManager, Frame, FullOutputs
from worldstream.m1.geometry_adaptive_patch import GeometryAdaptivePatch
from worldstream.m1.surface_observability import PatchObservation, measured_world, project
from worldstream.m1.surface_visibility_constraints import joint_exclusions, apply_surface_constraints, candidate_masks
from worldstream.m1.temporal_visibility import build_evidence
from worldstream.m1.novelty_detector import unexplained_regions
from test_identity_surface import triple


def plane(slope=(0.,0.)):
    h,w=160,192; k=np.array([[100.,0.,96.],[0.,100.,80.],[0.,0.,1.]])
    rng=np.random.RandomState(42)
    # Band-limited texture avoids making resampling aliasing the tested cause.
    image=cv2.GaussianBlur(rng.randint(0,256,(h,w)).astype(np.uint8),(7,7),1.2)
    yy,xx=np.mgrid[:h,:w]; rho=.4+slope[0]*(xx-96)+slope[1]*(yy-80)
    source=np.stack((xx+60*rho,yy),axis=2).astype(np.float32)
    homography=np.array([[1+60*slope[0],60*slope[1],60*(.4-96*slope[0]-80*slope[1])],[0,1,0],[0,0,1.]])
    old_image=cv2.warpPerspective(image,homography,(w,h))
    old_pose=np.eye(4);old_pose[0,3]=.6
    old=CameraFrame('cam',old_image,k.copy(),old_pose,0.)
    cur=CameraFrame('cam',image,k.copy(),np.eye(4),.5)
    cfg=M1Config(support_patch_radius_px=3)
    q=np.array([[96.,80.],[88.,76.]])
    p=source[q[:,1].astype(int),q[:,0].astype(int)]
    world=measured_world(old,cur,p,q)['world']
    return old,cur,cfg,source,q,world


@pytest.mark.parametrize('slope',[(0.,0.),(.003,0.),(.001,.002)])
def test_measured_inverse_depth_plane_and_spatial_holdout(slope):
    old,cur,cfg,source,q,world=plane(slope)
    patch=GeometryAdaptivePatch(cur,cfg);other=PatchObservation(old,cfg)
    geometry=patch.fit(other,q,world,source,np.ones(source.shape[:2],bool))
    assert geometry['applicable'].all()
    assert np.allclose(geometry['slopes'],slope,atol=1e-6)
    assert geometry['holdout_residual_px'].max()<1e-3
    assert patch.compare(other,q,world,geometry)['supported'].all()


@pytest.mark.parametrize('failure',['depth_boundary','missing_holdout','center_only'])
def test_center_does_not_validate_mixed_or_unobserved_patch(failure):
    old,cur,cfg,source,q,world=plane((.001,.001))
    mask=np.ones(source.shape[:2],bool)
    if failure=='depth_boundary':source[80:, :,0]+=5
    elif failure=='missing_holdout':mask[80,100]=False
    else:mask[:]=False;mask[80,96]=True
    geometry=GeometryAdaptivePatch(cur,cfg).fit(PatchObservation(old,cfg),q[:1],world[:1],source,mask)
    assert not geometry['applicable'].any()


@pytest.mark.parametrize('failure',['blank','exposure','occlusion','new_surface'])
def test_appearance_failure_remains_unknown_never_physical_conflict(failure):
    old,cur,cfg,source,q,world=plane((.001,0.))
    patch=GeometryAdaptivePatch(cur,cfg);other=PatchObservation(old,cfg)
    geometry=patch.fit(other,q,world,source,np.ones(source.shape[:2],bool))
    if failure=='blank':other.image[:]=0
    elif failure=='exposure':other.image=np.clip(other.image+.4,0,1)
    else:other.image[:]=np.random.RandomState(7).rand(*other.image.shape)
    result=patch.compare(other,q,world,geometry)
    assert not result['supported'].any()
    assert result['patch_visibility_unknown'].all()
    assert not result['patch_conflict_observed'].any()


@pytest.mark.parametrize('case',['contradiction','both_supported','not_visible','no_track','no_geometry','insufficient_separation','no_h0'])
def test_joint_exclusion_requires_actual_geometric_conflict(monkeypatch,case):
    from worldstream.m1 import geometry_adaptive_patch as module
    old,cur,cfg,_,q,world=plane()
    view=copy.deepcopy(old);view.timestamp=-.5
    q=q[:1];world=world[:1];uv,_=project(world,view)
    alt=world[:,None].copy()
    # Move the competitor's predicted pixel 2.5px within the measured patch.
    alt[...,0]+=.0625
    class Observation:
        camera=view
    patch=PatchObservation(cur,cfg)
    def compare(*args,**kwargs):
        return dict(observed=np.array([case!='not_visible']),supported=np.array([case=='both_supported']))
    monkeypatch.setattr(patch,'compare',compare)
    monkeypatch.setattr(module,'measured_third_correspondence',lambda *args:dict(source_uv=uv,
        supported=np.array([case!='no_track']),world=world))
    if case=='insufficient_separation':alt=world[:,None].copy()
    geometry=dict(applicable=np.array([case!='no_geometry']))
    p,_=project(world,old)
    rejected,distinguishing,_=joint_exclusions(q,world,alt,np.ones((1,1),bool),
        [('additional_time',view,Observation(),np.array([case!='no_h0']))],patch,geometry,old,p,cfg)
    assert bool(rejected[0,0])==(case=='contradiction')


def test_repeated_texture_unresolved_competitors_never_ranked():
    shape=(2,3)
    maps={k:np.zeros(shape,bool) for k in ['source_consistent','temporal_supported','camera_supported',
        'ambiguous','order_conflict','observation_base','adaptive_source_consistent','adaptive_temporal_supported',
        'adaptive_camera_supported','adaptive_ambiguous','adaptive_order_conflict','joint_ambiguous',
        'joint_legacy_ambiguous','joint_discriminating']}
    for k in ['observation_base','adaptive_source_consistent','adaptive_temporal_supported','joint_ambiguous']:maps[k][:]=True
    assert not candidate_masks(maps)['correspondence_recovery'].any()


def test_new_manager_unknown_refresh_and_periodic_override():
    first,prev,cur,cfg,hist=triple('surface_constraints')
    cfg=replace(cfg,patch_geometry_model='local_inverse_depth',joint_observation_disambiguation=True)
    evidence=build_evidence(prev,cur,cfg)
    apply_surface_constraints(prev,cur,evidence,{cur.camera_id:cur},None,cfg)
    assert not evidence.stable_mask.any()
    assert evidence.refresh_required_mask.all()
    evidence=build_evidence(prev,cur,cfg)
    apply_surface_constraints(prev,cur,evidence,{cur.camera_id:cur},hist,cfg)
    base=evidence.observation_supported_mask.copy()
    assert not (evidence.stable_mask&~base).any()
    _,refresh=unexplained_regions(cur,[],[],cfg,True,evidence)
    assert refresh.all()
    manager=WorldStateManager(cfg)
    for i,camera in enumerate((first,prev,cur)):
        frame=Frame('s',str(i),camera.timestamp,np.eye(4),{camera.camera_id:camera})
        pre=manager.precheck(frame);manager.commit(frame,FullOutputs([]),pre)
        assert not manager.get_current_certifications().certified_anchor_ids


@pytest.mark.parametrize('field,value',[('patch_geometry_model','unknown'),('joint_observation_disambiguation','yes')])
def test_invalid_new_policy_fails_closed(field,value):
    with pytest.raises(ValueError):replace(M1Config(),**{field:value})
