"""Controlled observation contracts, not empirical physical quality claims."""
from dataclasses import replace
import copy
import numpy as np
import pytest
from worldstream.m1 import WorldStateManager,Frame,FullOutputs
from worldstream.m1.temporal_visibility import build_evidence
from worldstream.m1.novelty_detector import unexplained_regions
from worldstream.m1.surface_observability import corroborating_views
from worldstream.m1.surface_visibility_constraints import apply_surface_constraints,depth_order_conflicts,candidate_masks
from test_identity_surface import triple


def measured(history=True):
    first,prev,cur,cfg,hist=triple('surface_constraints')
    e=build_evidence(prev,cur,cfg)
    apply_surface_constraints(prev,cur,e,{cur.camera_id:cur},hist if history else None,cfg)
    return prev,cur,cfg,e,hist


def test_actual_third_view_keeps_nonzero_observed_support():
    prev,cur,cfg,e,_=measured()
    assert e.observation_supported_mask.any()
    assert e.stable_candidate_mask.sum()>100,e.diagnostics
    assert not (e.stable_mask&~e.observation_supported_mask).any()
    assert not (e.stable_mask&e.ambiguity_rejected_mask).any()
    assert not e.transformer_dependency_certified
    _,refresh=unexplained_regions(cur,[],[],cfg,True,e)
    assert refresh.all() and e.refresh_required_mask.all()
    assert e.stable_candidate_mask.any()  # periodic refresh is a policy override


def test_missing_distinct_acquisition_returns_unknown_even_with_two_view_support():
    _,cur,cfg,e,_=measured(False)
    assert e.observation_supported_mask.any() and not e.stable_mask.any()
    assert candidate_masks(e.constraint_measurements)['constraints_two_view_only'].any()
    _,refresh=unexplained_regions(cur,[],[],cfg,False,e)
    assert refresh.all()


@pytest.mark.parametrize('mutation',['future','same_time','coordinate','intrinsic','timing'])
def test_illegal_history_is_not_extra_observation(mutation):
    first,prev,cur,cfg,hist=triple('surface_constraints');old=copy.deepcopy(first)
    if mutation=='future':old.timestamp=cur.timestamp+.1
    elif mutation=='same_time':old.timestamp=cur.timestamp
    elif mutation=='coordinate':old.coordinate_space='other'
    elif mutation=='intrinsic':old.intrinsic[:]=np.nan
    else:old.timing_reliable=False
    views,_=corroborating_views(cur,{cur.camera_id:cur},(old,prev.timestamp,hist[2]),cfg)
    assert not views


def test_distinct_camera_needs_sync_and_actual_fov():
    _,prev,cur,cfg,hist=triple('surface_constraints')
    other=copy.deepcopy(cur);other.camera_id='different';other.world_to_camera[0,3]=1000
    e=build_evidence(prev,cur,cfg)
    apply_surface_constraints(prev,cur,e,{cur.camera_id:cur,'different':other},None,cfg)
    assert not e.constraint_measurements['camera_supported'].any()
    assert not e.stable_mask.any()
    other.timestamp+=.1
    assert not corroborating_views(cur,{'different':other},None,cfg)[0]


def test_observed_incompatible_explanations_rejected_without_score_ranking(monkeypatch):
    from worldstream.m1 import surface_visibility_constraints as module
    first,prev,cur,cfg,hist=triple('surface_constraints')
    original=module.alternative_rays
    def actual_same_depth_alternative(*args):
        ap,aq,aw,av=original(*args)
        # Inject two mutually incompatible *supported* ray hypotheses to test
        # decision semantics, not the synthetic scene's physical quality.
        ap[:]=args[2][:,None,:]+np.array([3.,0.]);av[:]=True
        return ap,aq,aw,av
    monkeypatch.setattr(module,'alternative_rays',actual_same_depth_alternative)
    e=build_evidence(prev,cur,cfg);apply_surface_constraints(prev,cur,e,{cur.camera_id:cur},hist,cfg)
    assert e.constraint_measurements['source_consistent'].any()
    assert e.constraint_measurements['ambiguous'].any()
    assert not (e.stable_mask&e.constraint_measurements['ambiguous']).any()


def test_depth_order_only_rejects_measured_eligible_far_hypothesis():
    _,_,cur,cfg,_=triple();center=-cur.world_to_camera[:3,:3].T@cur.world_to_camera[:3,3]
    ray=np.array([0.,0.,1.])@cur.world_to_camera[:3,:3]
    world=center+np.array([5.,10.,15.])[:,None]*ray
    assert depth_order_conflicts(world,cur,np.array([True,True,False]),.5).tolist()==[False,True,False]
    assert not depth_order_conflicts(world,cur,np.zeros(3,bool),.5).any()


def test_manager_binds_history_and_never_mints_physical_qualification():
    first,prev,cur,cfg,hist=triple('surface_constraints');m=WorldStateManager(cfg)
    for camera in (first,prev,cur):
        f=Frame('s',str(camera.timestamp),camera.timestamp,np.eye(4),{camera.camera_id:camera})
        r=m.precheck(f);m.commit(f,FullOutputs([]),r)
        assert not m.get_current_certifications().certified_anchor_ids
    assert m.last_evidence[cur.camera_id].stable_candidate_mask.any()
    old,stamp,mapping=m._surface_history[cur.camera_id]
    assert old.timestamp==prev.timestamp and stamp==cur.timestamp and not mapping.flags.writeable
