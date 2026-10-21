"""Measured synthetic evidence fixtures test contracts, never deployment quality."""
import copy
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
from worldstream.m1 import M1Config,WorldStateManager,FullOutputs,ComputedFeature
from worldstream.m1.state_types import PhysicalEvidence,CameraFrame,Frame,WorldAnchor,AnchorObservation,OnlineVisibilityEvidence
from worldstream.m1.physical_certifier import (EvidenceIssuer,certify_physical_evidence,CONTINUITY_CLAIM,PRODUCER_ID,_canonical)
from worldstream.m1.geometry import box_geometry,state_geometry_version,project_anchor,immutable_array
from worldstream.m1.input_contract import frame_digest
from worldstream.m1.foreground_correspondence import propose_foreground_correspondence,producer_protocol_digest
from worldstream.m1.association import legal_assignment,association_ambiguities,assignment_edge_alternatives
from worldstream.m1.temporal_visibility import build_evidence
from worldstream.m1.mask_codec import decode_mask
from test_refinement import pair


def measured_pair(depth_separation=True):
    config=M1Config(temporal_visibility=True,foreground_correspondence=True,min_region_area_px=16)
    image=np.random.RandomState(44).randint(0,256,(160,240),np.uint8)
    k=np.array([[120.,0,120],[0,120.,80],[0,0,1.]])
    source=CameraFrame('front',image,k,np.eye(4),100.)
    pose=np.eye(4);pose[0,3]=.5
    current=CameraFrame('front',image.copy(),k,pose,100.5)
    frame=Frame('scene','current',100.5,np.eye(4),{'front':current})
    old=WorldAnchor('a','scene','object','traffic_cone',box_geometry([0,0,10],[5,5,.5]),100.,
        source_frame_id='source',source_state_epoch=7,source_execution_signature=frame.execution_signature)
    new=copy.deepcopy(old);new.source_frame_id='current';new.source_timestamp=100.5
    # Actual analytically projected 3D tracks on separate near/far surfaces.
    foreground=np.array([[x,y,10.] for x in (-1.6,-.8,.8,1.6) for y in (-1.6,-.8,.8,1.6)])
    background=np.array([[x,y,30. if depth_separation else 10.] for x in (-12,-10,10,12) for y in (-10,-8,8,10)])
    if not depth_separation:background[:,:2]/=3
    xyz=np.r_[foreground,background]
    def project(points,cam):
        c=points@cam.world_to_camera[:3,:3].T+cam.world_to_camera[:3,3]
        uv=c@cam.intrinsic.T;return (uv[:,:2]/uv[:,2:]).astype(np.float32)
    p=project(xyz,source);q=project(xyz,current);n=len(p)
    projected=project_anchor(old,source,config)
    old.observation_support['front']=AnchorObservation('front',projected.region,np.ones((16,16)),100.,1.)
    e=OnlineVisibilityEvidence('front',100.,100.5,p,q,np.zeros(n),np.ones(n,bool),np.ones(n,bool),
        np.ones(n,bool),np.zeros(n),np.zeros((160,240),bool),np.zeros((160,240),bool))
    issuer=EvidenceIssuer()
    proposal=propose_foreground_correspondence(old,new,frame,{'front':source},{'front':e},config,
        issuer,'actual-source-digest','source')
    return config,old,new,frame,source,e,issuer,proposal


def qualified_fixture():
    config,old,new,frame,source,e,_,_=measured_pair()
    frame.metadata['log_token']='evaluation-log'
    # 60 controlled independent-scene fixtures, not claimed as real experiment.
    q=dict(producer_id=PRODUCER_ID,claim=CONTINUITY_CLAIM,protocol_digest=producer_protocol_digest(config,frame),
        population='measured_box_contrast_candidates',independence_basis='disjoint_new_M1_scenes_and_logs',
        calibration_scenes=['calibration'],calibration_logs=['calibration-log'],validation_scene_units=[dict(scene_id=f'validation-{i}',log_token=f'log-{i}',accepted_entities=1,
            unscorable_accepted=0,false_acceptances=0) for i in range(60)])
    issuer=EvidenceIssuer(q,hashlib.sha256(_canonical(q).encode()).hexdigest())
    proposal=propose_foreground_correspondence(old,new,frame,{'front':source},{'front':e},config,
        issuer,'actual-source-digest','source')
    return old,new,frame,issuer,proposal


def certify_fixture(old,new,frame,issuer,proposal,**kwargs):
    return certify_physical_evidence(old,new,frame,proposal['evidence'],issuer,
                                    'actual-source-digest',proposal['protocol_digest'],**kwargs)


def test_measured_receipt_requires_independent_qualification():
    _,old,new,frame,_,_,issuer,proposal=measured_pair()
    assert proposal['status']=='candidate',proposal
    assert proposal['evidence'] is not None
    result=certify_fixture(old,new,frame,issuer,proposal)
    assert not result.certified and 'independent_producer_qualification_missing' in result.rejection_reasons


def test_qualification_protocol_binds_association_and_observation_settings():
    config,_,_,frame,*_=measured_pair()
    digest=producer_protocol_digest(config,frame)
    assert producer_protocol_digest(replace(config,association_gate_m=4.),frame)!=digest
    assert producer_protocol_digest(replace(config,support_residual_max=.08),frame)!=digest
    assert producer_protocol_digest(replace(config,model_id='other-checkpoint'),frame)!=digest
    assert producer_protocol_digest(replace(config,logging=not config.logging,temporal_workers=2),frame)==digest


def test_validated_measured_continuity_claim_can_pass_contract():
    old,new,frame,issuer,proposal=qualified_fixture()
    result=certify_fixture(old,new,frame,issuer,proposal)
    assert result.certified and result.certified_claim==CONTINUITY_CLAIM
    assert not result.evidence_provenance['geometry_attributes_certified']
    assert not result.evidence_provenance['motion_attributes_certified']


@pytest.mark.parametrize('claim',['metric_center','distance','velocity','full_world_state'])
def test_partial_continuity_never_approves_metric_or_motion(claim):
    old,new,frame,issuer,proposal=qualified_fixture()
    assert not certify_fixture(old,new,frame,issuer,proposal,claim=claim).certified


@pytest.mark.parametrize('mutation',['scene','epoch','source_frame','signature','timestamp','geometry','current_geometry','receipt','gt'])
def test_modified_or_cross_source_evidence_rejects(mutation):
    old,new,frame,issuer,proposal=qualified_fixture()
    if mutation=='scene':frame.scene_id='another'
    if mutation=='epoch':old.source_state_epoch+=1
    if mutation=='source_frame':old.source_frame_id='missing'
    if mutation=='signature':frame.execution_signature='different'
    if mutation=='timestamp':frame.timestamp=102.
    if mutation=='geometry':old.geometry_world.center+=.1
    if mutation=='current_geometry':new.geometry_world.size*=2
    if mutation=='receipt':proposal['evidence']=replace(proposal['evidence'],authentication_tag='forged')
    if mutation=='gt':
        m=json.loads(proposal['evidence'].measurements_json);m['gt_mask']=True
        # Even an internally signed forbidden payload must reject independently.
        proposal['evidence']=issuer._mint_measured(replace(proposal['evidence'],measurements_json=_canonical(m)))
    assert not certify_fixture(old,new,frame,issuer,proposal).certified


def test_booleans_score_or_static_class_are_not_evidence():
    _,old,new,frame,_,_,issuer,_=measured_pair()
    old.source_score=1.
    assert not certify_physical_evidence(old,new,frame,dict(identity_verified=True,foreground_verified=True),issuer).certified
    assert not certify_physical_evidence(old,new,frame).certified


def test_same_depth_static_background_returns_unknown():
    *_,proposal=measured_pair(depth_separation=False)
    assert proposal['status']=='unknown' and proposal['evidence'] is None
    assert any(v['reason']=='target_and_background_indistinguishable' for v in proposal['views'])


@pytest.mark.parametrize('risk',['no_parallax','left_view','wrong_camera_time','moved_static_box','multi_view_conflict'])
def test_foreground_observability_and_conflicts(risk):
    cfg,old,new,frame,source,e,issuer,_=measured_pair()
    cameras={'front':source};evidence={'front':e}
    if risk=='no_parallax':frame.cameras['front'].world_to_camera=np.eye(4);e.current_points=e.source_points.copy()
    if risk=='left_view':new.geometry_world=box_geometry([200,0,10],[5,5,.5])
    if risk=='wrong_camera_time':e.source_timestamp=99.
    if risk=='moved_static_box':new.geometry_world=box_geometry([1.5,0,10],[.5,5,.5])
    if risk=='multi_view_conflict':
        e.geometry_valid[:8]=False
    proposal=propose_foreground_correspondence(old,new,frame,cameras,evidence,cfg,issuer,'source-digest','source')
    assert proposal['status']=='unknown' and proposal['evidence'] is None


def test_global_assignment_alternative_differs_from_local_margin():
    costs=np.array([[1.,2.],[1.1,np.inf]])
    pairs=legal_assignment(costs);local=association_ambiguities(pairs,costs)
    global_proof=assignment_edge_alternatives(pairs,costs)
    assert local[0,1]['ambiguous']
    assert global_proof[0,1]['cardinality_lost']==1
    assert global_proof[0,1]['global_cost_margin_m'] is None
    ties=np.ones((2,2));proof=assignment_edge_alternatives(legal_assignment(ties),ties)
    assert all(p['global_cost_margin_m']==0 for p in proof.values())


def test_external_prediction_and_fork_arrays_are_isolated(config,make_frame,make_prediction):
    m=WorldStateManager(config);prediction=make_prediction();m.initialize(make_frame(),FullOutputs([prediction]))
    anchor=next(iter(m.store.anchors.values()));before=anchor.geometry_world.center.copy()
    prediction.geometry_world.center[:]=88
    assert np.array_equal(anchor.geometry_world.center,before)
    with pytest.raises(ValueError):anchor.geometry_world.center[0]=8
    with pytest.raises(ValueError):anchor.geometry_world.center.setflags(write=True)
    fork=m.store.fork();fork.anchors[anchor.anchor_id].geometry_world.center=np.ones(3)
    assert np.array_equal(anchor.geometry_world.center,before)
    fork.anchors[anchor.anchor_id].observation_support['front'].descriptor=np.zeros((2,2))
    assert anchor.observation_support['front'].descriptor.shape!=(2,2)


def test_source_replacement_during_pending_rejects(config,make_frame,make_prediction):
    m=WorldStateManager(config);m.initialize(make_frame(),FullOutputs([make_prediction()]))
    f=make_frame(100.5);r=m.precheck(f)
    next(iter(m.store.anchors.values())).geometry_world.center=np.zeros(3)
    with pytest.raises(ValueError,match='source state'):m.commit(f,FullOutputs([make_prediction()]),r)
    assert m._pending_store is None


def test_new_detection_drops_old_certificate(config,make_frame,make_prediction):
    m=WorldStateManager(config);m.initialize(make_frame(),FullOutputs([make_prediction()]))
    a=next(iter(m.store.anchors.values()));a.last_certified_timestamp=100.;a.certified_observation_count=1
    version=a.state_version
    f=make_frame(100.5);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction(.1)]),r)
    a=next(iter(m.store.anchors.values()))
    assert a.state_version!=version and a.last_certified_timestamp is None and a.physical_certification is None
    assert a.certified_observation_count==1  # historical count, not current approval


def test_tensor_acquisition_cannot_mutate_committed_source(config,make_frame,make_prediction):
    cfg=replace(config,capture_tensors=True,max_tensor_bytes=1_000_000)
    m=WorldStateManager(cfg);m.initialize(make_frame(),FullOutputs([make_prediction()]))
    r=m.precheck(make_frame(100.5));candidate=r.valid_feature_candidates[0]
    got=m.get_candidate_feature(candidate['feature_id'],candidate['computation_version'])
    original=m.store.features[candidate['feature_id']]
    assert got is not original
    with pytest.raises(ValueError):got.tensor[:]=0
    fork=m.store.fork();fork.features[candidate['feature_id']].source_camera_ids.append('injected')
    assert 'injected' not in original.source_camera_ids


def test_local_boundary_policy_preserves_partition_and_force_refresh():
    from worldstream.m1.novelty_detector import unexplained_regions
    a,b=pair();cfg=M1Config(temporal_visibility=True,local_support=True,dense_local_support=True,
        dense_region_policy='local_visibility',dense_guard_radius_px=2)
    e=build_evidence(a,b,cfg)
    assert e.local_rejection_mask is not None
    assert not (e.stable_mask&e.local_rejection_mask).any()
    meta,refresh=unexplained_regions(b,[],[],cfg,False,e)
    forced,all_refresh=unexplained_regions(b,[],[],cfg,True,e)
    for k in ('observation_stable_mask','observed_changed_mask','unknown_mask'):
        assert np.array_equal(decode_mask(meta[k]),decode_mask(forced[k]))
    assert all_refresh.all() and forced['invalidation_area_ratio']==1.


def test_local_boundary_rejection_is_local_and_keeps_unrelated_surface():
    from worldstream.m1.dense_support import local_visibility_rejection
    a,b=pair();h,w=a.image.shape
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    source=np.stack([xx-4,yy],axis=-1);source[:,w//2:,0]-=12
    zeros=np.zeros((h,w),np.float32);flow=np.zeros((h,w,2),np.float32)
    cfg=M1Config(local_boundary_radius_px=8)
    mask,_=local_visibility_rejection(a,b,source,flow,flow,zeros,np.ones_like(zeros),zeros,np.ones_like(zeros,bool),cfg)
    assert mask[:,w//2-6:w//2+6].all()
    assert not mask[20:-20,20:60].any()
    assert not mask[20:-20,180:220].any()


@pytest.mark.parametrize('change',['new_from_border','occlusion_boundary','timestamp_mismatch'])
def test_new_or_unmatched_pixels_never_gain_stable_support(change):
    a,b=pair()
    cfg=M1Config(temporal_visibility=True,local_support=True,dense_local_support=True,
                 dense_region_policy='local_visibility',dense_guard_radius_px=2)
    if change=='timestamp_mismatch':b.timestamp=103.
    elif change=='new_from_border':b.image[40:120,200:]=np.random.RandomState(91).randint(0,256,(80,40),np.uint8)
    else:b.image[40:120,80:160]=0
    e=build_evidence(a,b,cfg)
    if change=='timestamp_mismatch':assert not e.stable_mask.any()
    elif change=='new_from_border':assert not e.stable_mask[45:115,205:235].any()
    else:assert not e.stable_mask[45:115,85:155].any()
