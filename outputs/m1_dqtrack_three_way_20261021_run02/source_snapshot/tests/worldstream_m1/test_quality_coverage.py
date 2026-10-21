"""New measured paths under controlled inputs, never counted as real qualification."""
import copy
from dataclasses import replace
import json
import cv2
import numpy as np
import pytest

from worldstream.m1 import M1Config, WorldStateManager, FullOutputs
from worldstream.m1.geometry import box_geometry
from worldstream.m1.physical_certifier import certify_physical_evidence
from worldstream.m1.identity_continuity import IDENTITY_CLAIM, identity_protocol_digest
from worldstream.m1.temporal_visibility import build_evidence
from worldstream.m1.novelty_detector import unexplained_regions
from test_identity_surface import triple


def moving_pair(config,make_frame,make_prediction,displacement=1.):
    cfg=replace(config,temporal_visibility=True,identity_continuity=True,identity_method='temporal',max_state_age_s=1.)
    m=WorldStateManager(cfg);old=make_frame(support=False)
    # Band-limited texture avoids interpolation aliasing from shifted white
    # noise; this fixture tests motion identity, not unobservable RGB sampling.
    old.cameras['front'].image=cv2.GaussianBlur(old.cameras['front'].image,(5,5),.8)
    p=make_prediction(semantic='car');p.geometry_world=box_geometry([0,0,10],[8,6,2])
    m.initialize(old,FullOutputs([p]));f=make_frame(100.5,support=False)
    shift=60*displacement/10
    f.cameras['front'].image=cv2.warpAffine(old.cameras['front'].image,np.array([[1,0,shift],[0,1,0]],np.float32),(120,100),borderMode=cv2.BORDER_REFLECT101)
    q=copy.deepcopy(p);q.geometry_world=box_geometry([displacement,0,10],[8,6,2])
    return m,p,q,f


def test_moving_identity_measures_tracks_without_claiming_stationarity(config,make_frame,make_prediction):
    m,_,p,f=moving_pair(config,make_frame,make_prediction)
    r=m.precheck(f);m.commit(f,FullOutputs([p]),r)
    candidates=[v for v in m.last_commit_diagnostics['identity_continuity_proposals'] if v['status']=='candidate']
    assert candidates,m.last_commit_diagnostics
    assert candidates[0]['measurements']['displacement_m']>.5
    assert candidates[0]['measurements']['views'][0]['source_points']
    assert candidates[0]['qualification_reasons']==['independent_producer_qualification_missing']
    assert not m.get_current_certifications().certified_anchor_ids
    assert not r.physically_certified_anchor_ids


def test_identical_competing_detections_remain_unknown(config,make_frame,make_prediction):
    m,_,p,f=moving_pair(config,make_frame,make_prediction)
    r=m.precheck(f);m.commit(f,FullOutputs([p,copy.deepcopy(p)]),r)
    assert not any(v['status']=='candidate' for v in m.last_commit_diagnostics['identity_continuity_proposals'])
    assert not m.get_current_certifications().certified_anchor_ids


def test_short_missing_detection_uses_actual_source_image(config,make_frame,make_prediction):
    m,p,q,f=moving_pair(config,make_frame,make_prediction,displacement=0.)
    r=m.precheck(f);m.commit(f,FullOutputs([]),r)
    f2=make_frame(101.,support=False);f2.cameras['front'].image=f.cameras['front'].image.copy()
    r=m.precheck(f2);m.commit(f2,FullOutputs([q]),r)
    candidates=[v for v in m.last_commit_diagnostics['identity_continuity_proposals'] if v['status']=='candidate']
    assert candidates,m.last_commit_diagnostics
    assert candidates[0]['source_frame_id']=='100.0'
    assert candidates[0]['recovered_short_gap']
    assert not m.get_current_certifications().certified_anchor_ids


@pytest.mark.parametrize('mutation',['wrong_camera','space','nan_time','execution','crop','descriptor'])
def test_source_contract_failures_do_not_mint_temporal_candidates(config,make_frame,make_prediction,mutation):
    m,_,p,f=moving_pair(config,make_frame,make_prediction)
    record=m._identity_history[-1];anchor=next(iter(m.store.anchors.values()))
    if mutation=='wrong_camera':record['cameras']['front'].camera_id='wrong'
    elif mutation=='space':record['cameras']['front'].coordinate_space='other'
    elif mutation=='nan_time':record['cameras']['front'].timestamp=float('nan')
    elif mutation=='execution':record['execution_signature']='different'
    elif mutation=='crop':anchor.observation_support['front'].region=(0,0,10,10)
    else:anchor.observation_support['front'].descriptor=np.zeros_like(anchor.observation_support['front'].descriptor)
    r=m.precheck(f);m.commit(f,FullOutputs([p]),r)
    assert not any(v['status']=='candidate' for v in m.last_commit_diagnostics['identity_continuity_proposals'])


def test_history_reset_and_failed_commit_are_atomic(config,make_frame,make_prediction,monkeypatch):
    from worldstream.m1.state_store import StateStore
    m,_,p,f=moving_pair(config,make_frame,make_prediction)
    before=[v['frame_id'] for v in m._identity_history]
    r=m.precheck(f)
    def fail(*a):raise RuntimeError('publication failed')
    monkeypatch.setattr(StateStore,'put_feature',fail)
    with pytest.raises(RuntimeError):m.commit(f,FullOutputs([p]),r)
    assert [v['frame_id'] for v in m._identity_history]==before
    m.reset('other');assert not m._identity_history


def test_long_gap_does_not_recover_or_accumulate_history(config,make_frame,make_prediction):
    m,_,p,f=moving_pair(config,make_frame,make_prediction)
    f.timestamp=103.;f.cameras['front'].timestamp=103.;f.frame_id='long-gap'
    r=m.precheck(f);m.commit(f,FullOutputs([p]),r)
    assert not m.last_commit_diagnostics['identity_continuity_proposals']
    assert len(m._identity_history)==1


def test_temporal_protocol_binds_history_and_competition_policies(config,make_frame):
    cfg=replace(config,identity_continuity=True,temporal_visibility=True,identity_method='temporal');f=make_frame()
    assert identity_protocol_digest(cfg,f)!=identity_protocol_digest(replace(cfg,identity_use_history=False),f)
    assert identity_protocol_digest(cfg,f)!=identity_protocol_digest(replace(cfg,identity_resolve_competition=False),f)


def test_local_plane_has_measured_support_with_missing_third_view():
    _,previous,current,cfg,_=triple()
    e=build_evidence(previous,current,replace(cfg,dense_region_policy='surface_local_plane'),surface_history=None)
    assert e.stable_mask.sum()>100,e.diagnostics
    assert e.diagnostics['surface_plane_units_accepted']>0
    assert e.diagnostics['surface_two_view_pixels']>0
    assert not e.stable_mask[:4].any()


@pytest.mark.parametrize('mutation',['wrong_camera','wrong_space','nan_time','calibration'])
def test_local_plane_camera_contract_rejects(mutation):
    _,previous,current,cfg,_=triple();cfg=replace(cfg,dense_region_policy='surface_local_plane')
    if mutation=='wrong_camera':previous.camera_id='other'
    elif mutation=='wrong_space':previous.coordinate_space='other'
    elif mutation=='nan_time':previous.timestamp=float('nan')
    else:previous.intrinsic=np.full((3,3),np.nan)
    assert not build_evidence(previous,current,cfg).stable_mask.any()


def test_local_plane_low_parallax_unknown_and_force_refresh():
    from test_refinement import pair
    a,b=pair(0);cfg=M1Config(temporal_visibility=True,local_support=True,dense_local_support=True,dense_region_policy='surface_local_plane')
    assert not build_evidence(a,b,cfg).stable_mask.any()
    _,previous,current,_,history=triple();e=build_evidence(previous,current,cfg,surface_history=history)
    meta,mask=unexplained_regions(current,[],[],cfg,True,e)
    assert mask.all() and meta['effective_stable_area_ratio']==0


def test_surface_dynamic_occlusion_and_unknown_always_refresh():
    _,previous,current,cfg,history=triple();cfg=replace(cfg,dense_region_policy='surface_local_plane')
    current.image[30:100,60:130]=0
    e=build_evidence(previous,current,cfg,surface_history=history)
    assert not e.stable_mask[35:95,65:125].any()
    _,mask=unexplained_regions(current,[],[],cfg,False,e)
    assert mask[~e.stable_mask].all()


@pytest.mark.parametrize('key,value',[('identity_method','bad'),('identity_history_frames',0),('surface_unit_px',1)])
def test_invalid_research_configuration(key,value):
    with pytest.raises(ValueError):M1Config(**{key:value})


def measured_temporal_receipt(config,make_frame,make_prediction):
    from worldstream.m1.association import associate
    from worldstream.m1.identity_temporal import temporal_identity_pairs
    from worldstream.m1.state_types import WorldAnchor
    m,_,pred,frame=moving_pair(config,make_frame,make_prediction);m.precheck(frame)
    old=list(m.store.anchors.values());_,cost=associate(old,[pred],frame.timestamp,m.config)
    p=temporal_identity_pairs(old,[pred],cost,frame,m._identity_history,m.config,m.validator,m._physical_issuer)[0,0]
    assert p['status']=='candidate'
    current=WorldAnchor(old[0].anchor_id,frame.scene_id,pred.anchor_type,pred.semantic_class,pred.geometry_world,frame.timestamp,
        source_frame_id=frame.frame_id,source_state_epoch=old[0].source_state_epoch,source_execution_signature=frame.execution_signature)
    return m,old[0],current,frame,p['evidence']


@pytest.mark.parametrize('mutation',['fb_negative','fb_nan','duplicate_tracks','crop','current_crop','descriptor','protocol',
    'epoch','execution','frame_digest','source_digest','fake_competition','fake_loop','authentication','forbidden_GT'])
def test_temporal_signed_false_measurements_are_rejected(config,make_frame,make_prediction,mutation):
    m,old,current,frame,e=measured_temporal_receipt(config,make_frame,make_prediction)
    protocol=identity_protocol_digest(m.config,frame)
    baseline=certify_physical_evidence(old,current,frame,e,m._physical_issuer,m._previous_frame_digest,protocol,IDENTITY_CLAIM)
    assert baseline.rejection_reasons==('independent_producer_qualification_missing',)
    measurements=json.loads(e.measurements_json);view=measurements['views'][0]
    if mutation=='fb_negative':view['fb_error_px'][0]=-.01
    elif mutation=='fb_nan':view['fb_error_px'][0]=float('nan')
    elif mutation=='duplicate_tracks':view['source_points']=[view['source_points'][0]]*len(view['source_points'])
    elif mutation=='crop':view['source_region'][0]+=1
    elif mutation=='current_crop':view['current_region'][2]+=20
    elif mutation=='descriptor':view['source_descriptor'][0][0]+=1
    elif mutation=='protocol':measurements['producer_config']['identity_use_history']=False
    elif mutation=='epoch':e=replace(e,source_epoch=100)
    elif mutation=='execution':e=replace(e,execution_signature='different')
    elif mutation=='frame_digest':e=replace(e,current_frame_digest='wrong')
    elif mutation=='source_digest':e=replace(e,source_frame_digest='wrong')
    elif mutation=='fake_competition':measurements.update(competing_edges=[dict(measured_status='unknown',positive_exclusion=True)],resolved_competition=True,temporal_cycles=[])
    elif mutation=='fake_loop':measurements.update(competing_edges=[dict(measured_status='unknown',positive_exclusion=True)],resolved_competition=True,temporal_cycles=[{}])
    elif mutation=='forbidden_GT':measurements['gt_instance_id']='forbidden'
    encoded=json.dumps(measurements)
    if mutation=='fb_nan':
        # Noncanonical numeric measurements cannot even be signed.
        with pytest.raises(ValueError):m._physical_issuer._mint_measured(replace(e,measurements_json=encoded,evidence_timestamp=float('nan')))
    e=m._physical_issuer._mint_measured(replace(e,measurements_json=encoded))
    if mutation=='authentication':e=replace(e,authentication_tag='forged')
    result=certify_physical_evidence(old,current,frame,e,m._physical_issuer,m._previous_frame_digest,protocol,IDENTITY_CLAIM)
    assert not result.certified
    assert set(result.rejection_reasons)-{'independent_producer_qualification_missing'}


def test_local_plane_measured_third_view_conflict_cannot_be_ignored():
    old,prev,cur,cfg,history=triple();cfg=replace(cfg,dense_region_policy='surface_local_plane')
    wrong=(old,prev.timestamp,history[2]+np.array([5,0],np.float32))
    e=build_evidence(prev,cur,cfg,surface_history=wrong)
    from worldstream.m1.local_surface import _history_tracks
    yy,xx=np.mgrid[:cur.image.shape[0],:cur.image.shape[1]]
    tracks=_history_tracks(prev,cur,e.dense_source_map,wrong,np.stack((xx,yy),-1).astype(np.float32))
    assert tracks[3].any() and not (e.stable_mask&tracks[3]).any()


def test_source_snapshot_image_replacement_cannot_inherit_digest(config,make_frame,make_prediction):
    m,_,p,f=moving_pair(config,make_frame,make_prediction)
    m._identity_history[-1]['cameras']['front'].image=np.zeros((100,120),np.uint8)
    r=m.precheck(f);m.commit(f,FullOutputs([p]),r)
    assert not any(v['status']=='candidate' for v in m.last_commit_diagnostics['identity_continuity_proposals'])


def test_temporal_identity_mixed_object_and_map_commit(config,make_frame,make_prediction):
    cfg=replace(config,temporal_visibility=True,identity_continuity=True,identity_method='temporal')
    manager=WorldStateManager(cfg);obj=make_prediction(semantic='car');lane=make_prediction(kind='map',semantic='divider')
    manager.initialize(make_frame(support=False),FullOutputs([obj,lane]))
    frame=make_frame(100.5,support=False);result=manager.precheck(frame)
    manager.commit(frame,FullOutputs([obj,lane]),result)
    assert len(manager.store.anchors)==2
    for row in manager.last_commit_diagnostics['identity_continuity_proposals']:
        assert row['measurements']['source_class']=='car'
    assert not manager.get_current_certifications().certified_anchor_ids


def test_temporal_cycle_checks_bound_camera_times_not_frame_reference(config,make_frame,make_prediction):
    from worldstream.m1.identity_temporal import temporal_identity_pairs,_cycle
    from worldstream.m1.association import associate
    cfg=replace(config,identity_continuity=True,temporal_visibility=True,identity_method='temporal',max_state_age_s=1.)
    manager=WorldStateManager(cfg);prediction=make_prediction(semantic='car')
    prediction.geometry_world=box_geometry([0,0,10],[8,6,2])
    for t in (99.5,100.):
        frame=make_frame(t,support=False);frame.cameras['front'].timestamp=t-.03
        result=manager.precheck(frame);manager.commit(frame,FullOutputs([prediction]),result)
    frame=make_frame(100.5,support=False);frame.cameras['front'].timestamp=100.47
    manager.precheck(frame);old=list(manager.store.anchors.values());_,cost=associate(old,[prediction],frame.timestamp,cfg)
    proposal=temporal_identity_pairs(old,[prediction],cost,frame,manager._identity_history,cfg,manager.validator,manager._physical_issuer)[0,0]
    assert proposal['status']=='candidate'
    view=proposal['measurements']['views'][0]
    cycle=_cycle(view,manager._identity_history[-1],manager._identity_history[0],frame,old[0].anchor_id)
    assert cycle and frame.timestamp-cycle['older_timestamp']>1
    # Controlled competitor exclusion, measured points lie outside this alternative
    # crop. This fixture only tests certificate clocks and never grants qualification.
    exclusion=dict(source_region=view['source_region'],current_region=[0,0,10,10],total_tracks=len(view['source_points']),
        exclusion_source_points=view['source_points'],exclusion_current_points=view['current_points'],
        exclusion_fb_error_px=view['fb_error_px'],exclusion_patch_residual=view['patch_residual'])
    measurement=proposal['measurements'];measurement.update(competing_edges=[dict(measured_status='unknown',positive_exclusion=True,exclusion_views=[exclusion])],
        resolved_competition=True,temporal_cycles=[dict(camera='front',**cycle)])
    evidence=manager._physical_issuer._mint_measured(replace(proposal['evidence'],measurements_json=json.dumps(measurement)))
    current=replace(old[0],source_frame_id=frame.frame_id,source_timestamp=frame.timestamp)
    certificate=certify_physical_evidence(old[0],current,frame,evidence,manager._physical_issuer,
        manager._previous_frame_digest,identity_protocol_digest(cfg,frame),IDENTITY_CLAIM)
    assert certificate.rejection_reasons==('independent_producer_qualification_missing',)
