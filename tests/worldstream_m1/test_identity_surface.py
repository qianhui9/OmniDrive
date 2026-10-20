"""Controlled contracts, never real identity/surface quality evidence."""
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import cv2
import numpy as np
import pytest
from worldstream.m1 import M1Config, WorldStateManager, FullOutputs
from worldstream.m1.geometry import box_geometry
from worldstream.m1.identity_continuity import IDENTITY_CLAIM, IDENTITY_PRODUCER, IDENTITY_POPULATION, identity_protocol_digest
from worldstream.m1.identity_continuity import propose_identity_continuity
from worldstream.m1.physical_certifier import EvidenceIssuer, _canonical, certify_physical_evidence
from worldstream.m1.temporal_visibility import build_evidence
from worldstream.m1.novelty_detector import unexplained_regions
from worldstream.m1.state_store import StateStore
from test_refinement import pair


def identity_manager(config,make_frame,make_prediction,qualified=True):
    cfg=replace(config,identity_continuity=True,temporal_visibility=True,max_state_age_s=1.)
    m=WorldStateManager(cfg);p=make_prediction(semantic='car')
    p.geometry_world=box_geometry([0,0,10],[8,6,2]);f=make_frame(support=False)
    m.initialize(f,FullOutputs([p]));current=make_frame(100.5,support=False)
    current.metadata['log_token']='evaluation-log'
    q=dict(producer_id=IDENTITY_PRODUCER,claim=IDENTITY_CLAIM,population=IDENTITY_POPULATION,
        protocol_digest=identity_protocol_digest(cfg,current),independence_basis='disjoint_new_M1_scenes_and_logs',
        calibration_scenes=['cal'],calibration_logs=['cal-log'],
        validation_scene_units=[dict(scene_id=f'unit-{i}',log_token=f'log-{i}',accepted_entities=1,
            unscorable_accepted=0,false_acceptances=0) for i in range(60)])
    if qualified:m._physical_issuer=EvidenceIssuer(q,hashlib.sha256(_canonical(q).encode()).hexdigest())
    return m,p,current,q


def test_post_commit_real_measured_identity_readout(config,make_frame,make_prediction):
    m,p,f,_=identity_manager(config,make_frame,make_prediction)
    r=m.precheck(f);assert not m.get_current_certifications().certified_anchor_ids
    pre=list(r.physically_certified_anchor_ids)
    assert m.commit(f,FullOutputs([p]),r) is r
    result=m.get_current_certifications()
    assert result.frame_id==f.frame_id and result.timestamp==f.timestamp
    assert len(result.certified_anchor_ids)==1,m.last_commit_diagnostics
    assert r.physically_certified_anchor_ids==pre==[]
    assert result.certifications[0].certified_claim==IDENTITY_CLAIM
    assert not result.certifications[0].evidence_provenance['stationarity_certified']
    with pytest.raises(FrozenInstanceError):result.frame_id='future'
    result.certifications[0].evidence_provenance['identity_only']=False
    assert m.get_current_certifications().certifications[0].evidence_provenance['identity_only']


@pytest.mark.parametrize('mutation',['epoch','geometry','timestamp','frame','scene','reset'])
def test_readout_invalidates_changed_binding(config,make_frame,make_prediction,mutation):
    m,p,f,_=identity_manager(config,make_frame,make_prediction);r=m.precheck(f);m.commit(f,FullOutputs([p]),r)
    assert m.get_current_certifications().certified_anchor_ids
    a=next(iter(m.store.anchors.values()))
    if mutation=='epoch':m.store.generation+=1
    elif mutation=='geometry':a.geometry_world=box_geometry([.1,0,10],[8,6,2])
    elif mutation=='timestamp':m.store.last_timestamp+=.1
    elif mutation=='frame':a.source_frame_id='other'
    elif mutation=='scene':m.store.scene_id='other'
    else:m.reset('other')
    assert not m.get_current_certifications().certified_anchor_ids


def test_new_detection_and_pending_do_not_inherit_certificate(config,make_frame,make_prediction):
    m,p,f,_=identity_manager(config,make_frame,make_prediction);r=m.precheck(f);m.commit(f,FullOutputs([p]),r)
    original=m.get_current_certifications();assert original.certified_anchor_ids
    next_frame=make_frame(101,support=False);r=m.precheck(next_frame)
    assert m.get_current_certifications().frame_id==f.frame_id
    # No qualification for new code/population; reject and clear new geometry.
    m._physical_issuer=EvidenceIssuer();m.commit(next_frame,FullOutputs([p]),r)
    assert not m.get_current_certifications().certified_anchor_ids


def test_commit_failure_cannot_publish_prepared_identity(config,make_frame,make_prediction,monkeypatch):
    m,p,f,_=identity_manager(config,make_frame,make_prediction);r=m.precheck(f)
    def fail(*args):raise RuntimeError('after certificate before publication')
    monkeypatch.setattr(StateStore,'put_feature',fail)
    with pytest.raises(RuntimeError):m.commit(f,FullOutputs([p]),r)
    assert not m.get_current_certifications().certified_anchor_ids
    assert m.get_current_certifications().frame_id!=f.frame_id


def identity_receipt(config,make_frame,make_prediction):
    m,p,f,q=identity_manager(config,make_frame,make_prediction);m.precheck(f)
    old=next(iter(m.store.anchors.values()));current=copy.deepcopy(old)
    current.source_frame_id=f.frame_id;current.source_timestamp=f.timestamp
    proposal=propose_identity_continuity(old,current,f,m.last_evidence,m.config,m.validator,m._physical_issuer,
        m._previous_frame_digest,m._previous_frame_id)
    assert proposal['status']=='candidate'
    return m,old,current,f,q,proposal['evidence']


def test_signed_identity_GT_is_forbidden_and_claim_cannot_expand(config,make_frame,make_prediction):
    m,old,current,f,q,e=identity_receipt(config,make_frame,make_prediction)
    digest=identity_protocol_digest(m.config,f)
    assert certify_physical_evidence(old,current,f,e,m._physical_issuer,m._previous_frame_digest,digest,IDENTITY_CLAIM).certified
    measurements=json.loads(e.measurements_json);measurements['gt_instance_id']='forbidden-oracle'
    forged=m._physical_issuer._mint_measured(replace(e,measurements_json=json.dumps(measurements)))
    assert not certify_physical_evidence(old,current,f,forged,m._physical_issuer,m._previous_frame_digest,digest,IDENTITY_CLAIM).certified
    assert not certify_physical_evidence(old,current,f,e,m._physical_issuer,m._previous_frame_digest,digest,'metric_center').certified


def test_identity_qualification_needs_59_zero_error_independent_units(config,make_frame,make_prediction):
    m,old,current,f,q,e=identity_receipt(config,make_frame,make_prediction)
    for n,passes in ((58,False),(59,True)):
        artifact=copy.deepcopy(q);artifact['validation_scene_units']=artifact['validation_scene_units'][:n]
        issuer=EvidenceIssuer(artifact,hashlib.sha256(_canonical(artifact).encode()).hexdigest())
        # New trusted fixture issuer signs the same actual measured test input.
        receipt=issuer._mint_measured(e)
        result=certify_physical_evidence(old,current,f,receipt,issuer,m._previous_frame_digest,artifact['protocol_digest'],IDENTITY_CLAIM)
        assert result.certified is passes


@pytest.mark.parametrize('mutation',['missing_qualification','no_observation','competing','jump','wrong_source'])
def test_identity_necessary_conditions_never_replaced_by_score(config,make_frame,make_prediction,mutation):
    m,p,f,_=identity_manager(config,make_frame,make_prediction,qualified=mutation!='missing_qualification')
    r=m.precheck(f);p.source_score=1.
    out=[p]
    if mutation=='no_observation':m.last_evidence.clear()
    elif mutation=='competing':out.append(copy.deepcopy(p))
    elif mutation=='jump':p.geometry_world=box_geometry([.8,0,10],[8,6,2])
    elif mutation=='wrong_source':m._previous_frame_id='not-actual-source'
    if mutation=='wrong_source':
        with pytest.raises(ValueError,match='source state'):m.commit(f,FullOutputs(out),r)
        assert r.diagnostics['transaction_status']=='aborted'
    else:
        m.commit(f,FullOutputs(out),r)
    assert not m.get_current_certifications().certified_anchor_ids


def triple(policy='surface_visibility'):
    old,prev=pair(4);_,cur=pair(8);cur.timestamp=101.
    cfg=M1Config(temporal_visibility=True,local_support=True,dense_local_support=True,dense_guard_radius_px=2,dense_region_policy=policy)
    first=build_evidence(old,prev,replace(cfg,dense_region_policy='uniform'))
    history=(old,prev.timestamp,first.dense_source_map)
    return old,prev,cur,cfg,history


def test_surface_closure_has_nonzero_measured_plane_support():
    _,prev,cur,cfg,history=triple();e=build_evidence(prev,cur,cfg,surface_history=history)
    assert e.stable_mask.sum()>500,e.diagnostics
    assert e.diagnostics['surface_reason']=='measured_three_view_closure'
    assert not e.stable_mask[:4].any() and not e.stable_mask[:,:4].any()


@pytest.mark.parametrize('mutation',['missing','wrong_time','wrong_shape','wrong_composed_surface'])
def test_surface_missing_or_conflicting_third_view_unknown(mutation):
    old,prev,cur,cfg,history=triple()
    if mutation=='missing':history=None
    elif mutation=='wrong_time':history=(old,0.,history[2])
    elif mutation=='wrong_shape':history=(old,prev.timestamp,history[2][:-1])
    else:history=(old,prev.timestamp,history[2]+np.array([5,0],np.float32))
    e=build_evidence(prev,cur,cfg,surface_history=history)
    assert not e.stable_mask.any()


def test_surface_low_parallax_unknown_and_force_keeps_observation_partition():
    a,b=pair(0);cfg=M1Config(temporal_visibility=True,local_support=True,dense_local_support=True,
        dense_guard_radius_px=2,dense_region_policy='surface_observable_ablation')
    assert not build_evidence(a,b,cfg).stable_mask.any()
    _,prev,cur,cfg,history=triple();e=build_evidence(prev,cur,cfg,surface_history=history)
    normal,_=unexplained_regions(cur,[],[],cfg,False,e);forced,mask=unexplained_regions(cur,[],[],cfg,True,e)
    assert normal['observation_stable_mask']==forced['observation_stable_mask']
    assert normal['unknown_mask']==forced['unknown_mask'] and mask.all()


def test_surface_occlusion_not_supported_and_history_bounded(config,make_frame):
    _,prev,cur,cfg,history=triple();cur.image[40:120,80:160]=0
    e=build_evidence(prev,cur,cfg,surface_history=history)
    assert not e.stable_mask[45:115,85:155].any()
    m=WorldStateManager(cfg)
    for t in np.arange(100.,103.,.5):
        f=make_frame(float(t),support=False);r=m.precheck(f);m.commit(f,FullOutputs([]),r)
        assert len(m._surface_history)<=1
        for old,stamp,mapping in m._surface_history.values():
            assert old.timestamp<stamp and not mapping.flags.writeable
    m.reset('new');assert not m._surface_history
