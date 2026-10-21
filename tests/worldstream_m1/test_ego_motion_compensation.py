"""Synthetic geometric truth tests, excluded from qualification populations."""
from dataclasses import replace
import copy
import cv2
import numpy as np
import pytest
from worldstream.m1 import M1Config,CameraFrame,WorldStateManager,FullOutputs,WorldAnchor
from worldstream.m1.geometry import project_points,box_geometry,state_geometry_version
from worldstream.m1.ego_motion import (project_source_depth,point_predictions,static_box_predictions,
    segment_residual,track_with_prediction,measured_depth_history,camera_pair_reason)
from worldstream.m1.identity_temporal import camera_snapshot_digest
from worldstream.m1.temporal_visibility import build_evidence,anchor_visibility


def camera(t=100.,position=(0,0,0),yaw=0.,image=None):
    r=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    w=np.eye(4);w[:3,:3]=r;w[:3,3]=-r@np.array(position)
    k=np.array([[180.,0,160],[0,180,120],[0,0,1.]])
    if image is None:image=cv2.GaussianBlur(np.random.RandomState(17).randint(0,256,(240,320),dtype=np.uint8),(5,5),1.)
    return CameraFrame('front',image,k,w,t)


@pytest.mark.parametrize('position,yaw', [((1,0,0),0),((0,0,0),.15),((1,0,.5),.1),((-.5,.2,0),-.1)])
@pytest.mark.parametrize('depth', [3.,10.,60.])
def test_world_projection_truth(position,yaw,depth):
    a,b=camera(),camera(100.5,position,yaw)
    world=np.array([[-.2,-.3,depth],[.2,.3,depth]])
    p,z,_=project_points(world,a.intrinsic,a.world_to_camera)
    truth,_,_=project_points(world,b.intrinsic,b.world_to_camera)
    q,valid,xyz,_=project_source_depth(a,b,p,z)
    assert valid.all() and np.allclose(q,truth,atol=1e-10)
    assert np.allclose(xyz,world)


def test_different_depth_different_parallax():
    a,b=camera(),camera(100.5,(1,0,0));p=np.array([[160.,120.],[160.,120.]])
    q,_,_,_=project_source_depth(a,b,p,[5,50])
    assert np.allclose(q[:,0],[124,156.4])
    # A single 2D map cannot map the same source ray to both measurements.
    assert abs(q[0,0]-q[1,0])>30


def test_rotation_depth_free_translation_uncertain():
    p=np.array([[100.,80.],[200.,160.]])
    cfg=M1Config(temporal_visibility=True,ego_motion_mode='rotation')
    q,valid,spread,z,_=point_predictions(camera(),camera(100.5,yaw=.1),p,cfg)
    truth,_,_,_=project_source_depth(camera(),camera(100.5,yaw=.1),p,[3,80])
    assert valid.all() and np.allclose(q,truth) and (spread==0).all() and np.isnan(z).all()
    _,_,spread,_,_=point_predictions(camera(),camera(100.5,(1,0,0)),p,cfg)
    assert np.isinf(spread).all()


def test_prior_depth_source_and_uncertainty():
    cfg=M1Config(temporal_visibility=True,ego_motion_mode='depth');a,b=camera(),camera(100.5,(1,0,0))
    p=np.array([[100.,80.],[200.,160.]])
    h=dict(camera_sha256=camera_snapshot_digest(a),timestamp=a.timestamp,pixels=p,depths=np.array([5.,50.]),uncertainty_m=np.array([.25,1.]))
    q,ok,spread,z,info=point_predictions(a,b,p,cfg,h)
    truth,*_=project_source_depth(a,b,p,[5,50]);assert ok.all() and np.allclose(q,truth)
    assert (spread>0).all() and info['depth_points']==2 and not info['metric_geometry_certified']
    h['camera_sha256']='tampered'
    _,_,spread,z,info=point_predictions(a,b,p,cfg,h)
    assert np.isinf(spread).all() and np.isnan(z).all() and info['depth_kind']=='source_depth_binding_mismatch'


@pytest.mark.parametrize('bad', ['pose','time','backward','long_gap','K','mapping','camera','coordinate'])
def test_prediction_contract_rejection(bad):
    cfg=M1Config(temporal_visibility=True,ego_motion_mode='depth');a,b=camera(),camera(100.5)
    if bad=='pose':b.world_to_camera[0,0]=2
    elif bad=='time':b.timing_reliable=False
    elif bad=='backward':b.timestamp=99
    elif bad=='long_gap':b.timestamp=102
    elif bad=='K':b.intrinsic[2,0]=.2
    elif bad=='mapping':b.raw_to_image=np.zeros((3,3))
    elif bad=='camera':b.camera_id='other'
    else:b.coordinate_space='other'
    assert camera_pair_reason(a,b,cfg)
    _,valid,_,_,_=point_predictions(a,b,np.array([[160.,120.]]),cfg)
    assert not valid.any()


def test_preprocessed_rays_use_mapped_intrinsic():
    a,b=camera(),camera(100.5,(1,0,0),.05)
    p=np.array([[130.,90.],[190.,150.]])
    truth,*_=project_source_depth(a,b,p,[8,16])
    mapping=np.array([[.5,0,7],[0,.5,3],[0,0,1.]])
    a.intrinsic=mapping@a.intrinsic;b.intrinsic=mapping@b.intrinsic
    mapped=p*.5+[7,3];q,*_=project_source_depth(a,b,mapped,[8,16])
    assert np.allclose(q,truth*.5+[7,3])


def test_static_ray_box_interval_not_dynamic_geometry():
    a,b=camera(),camera(100.5,(.7,0,0),.05)
    cfg=M1Config(temporal_visibility=True,ego_motion_mode='depth')
    anchor=WorldAnchor('a','s','object','barrier',box_geometry([0,0,10],[3,3,4]),100.)
    xyz=np.array([[-.8,-.8,8],[.8,.8,12]])
    p,*_=project_points(xyz,a.intrinsic,a.world_to_camera);q,*_=project_points(xyz,b.intrinsic,b.world_to_camera)
    near,far,valid=static_box_predictions(anchor,a,b,p,cfg)
    assert valid.all() and (segment_residual(q,near,far)<1e-7).all()
    anchor.semantic_class='car'
    assert not static_box_predictions(anchor,a,b,p,cfg)[2].any()


def test_real_image_rotation_tracking_and_low_texture():
    cfg=M1Config(temporal_visibility=True,ego_motion_mode='rotation',observation_diagnostics=True)
    a=camera();b=camera(100.5,yaw=.035)
    homography=b.intrinsic@b.world_to_camera[:3,:3]@np.linalg.inv(a.intrinsic)
    b.image=cv2.warpPerspective(a.image,homography,(320,240))
    e=build_evidence(a,b,cfg)
    assert e.correspondence_valid.sum()>100
    m=e.motion_measurements;mask=e.correspondence_valid
    assert np.median(m['prediction_residual_px'][mask])<.2
    assert 'raw_points' in e.diagnostics['point_measurements']
    a.image[:]=128;b.image[:]=128
    assert build_evidence(a,b,cfg).diagnostics['reason']=='insufficient_texture'


def test_conflicting_valid_initializations_reject(monkeypatch):
    import worldstream.m1.ego_motion as module
    p=np.array([[100.,100.],[200.,100.]])
    replies=iter([(p.copy(),np.zeros(2),np.ones(2,bool)),(p+10,np.zeros(2),np.ones(2,bool))])
    monkeypatch.setattr(module,'lk_measure',lambda *a,**kw:next(replies))
    _,_,good,m=track_with_prediction(camera(),camera(100.5),p,M1Config(temporal_visibility=True,ego_motion_mode='rotation'))
    assert not good.any() and m['initialization_conflict'].all()


@pytest.mark.parametrize('mode', ['rotation','depth'])
def test_state_transaction_scene_isolation_and_no_qualification(make_frame,make_prediction,mode):
    cfg=M1Config(temporal_visibility=True,local_support=True,ego_motion_mode=mode,identity_continuity=True,
        identity_method='temporal',dynamic_motion_prediction=True,logging=False)
    manager=WorldStateManager(cfg)
    for t in [100.,100.5,101.]:
        f=make_frame(t=t);r=manager.precheck(f);manager.commit(f,FullOutputs([make_prediction()]),r)
        assert not manager.get_current_certifications().certified_anchor_ids
    before=manager._history_digest();f=make_frame(t=101.5);r=manager.precheck(f)
    assert manager._history_digest()==before
    manager.abort_pending();assert manager._history_digest()==before
    f=make_frame(t=200,scene='other');manager.precheck(f)
    assert not manager._depth_history and not manager._identity_history and not manager._previous_cameras
    assert all(not a.motion_prediction.get('reliable') for a in manager.store.anchors.values())


def test_pending_depth_mutation_aborts(make_frame,make_prediction):
    manager=WorldStateManager(M1Config(temporal_visibility=True,ego_motion_mode='depth',logging=False))
    for t in [100,100.5]:
        f=make_frame(t=t);r=manager.precheck(f);manager.commit(f,FullOutputs([make_prediction()]),r)
    f=make_frame(t=101);r=manager.precheck(f);manager._depth_history['front']['depths']=np.array([99.])
    with pytest.raises(ValueError,match='source state changed'):manager.commit(f,FullOutputs([]),r)
    assert not manager._depth_history and not manager.store.anchors


def test_no_reliable_motion_from_nearby_or_raw_velocity(make_anchor):
    from worldstream.m1.world_motion import update_world_motion
    old=make_anchor(semantic='car');new=copy.deepcopy(old);new.source_timestamp=100.5
    new.velocity_world=np.array([20.,0,0]);new.identity_link_qualified=False
    update_world_motion(old,new,M1Config())
    assert not new.motion_prediction['reliable'] and not new.motion_history


def test_raw_mapping_disagreement_rejects():
    a,b=camera(),camera(100.5)
    b.raw_intrinsic=b.intrinsic.copy();b.raw_to_image=np.diag([.5,.5,1.])
    assert camera_pair_reason(a,b,M1Config())=='invalid_calibration_or_preprocessing'


def test_measured_plane_depth_is_prior_only_and_excluded():
    cfg=M1Config(temporal_visibility=True,ego_motion_mode='depth')
    a,b=camera(),camera(100.5,(.5,0,0))
    b.image=cv2.warpAffine(a.image,np.float32([[1,0,-9],[0,1,0]]),(320,240))
    evidence=build_evidence(a,b,cfg)
    binding=dict(scene_id='s',frame_id='b',epoch=2,execution_signature='x',frame_digest='actual',commit_version=2)
    h=measured_depth_history(a,b,evidence,cfg,binding)
    assert len(h['depths'])>100 and abs(np.median(h['depths'])-10)<.2
    assert evidence.motion_measurements['info']['depth_points']==0
    assert not h['metric_geometry_certified']
    c=camera(101,(1,0,0));p=h['pixels'][:20]
    predicted,_,spread,z,info=point_predictions(b,c,p,cfg,h)
    assert info['depth_points']==len(p) and np.isfinite(spread).all()
    truth,*_=project_source_depth(b,c,p,z);assert np.allclose(predicted,truth)
    evidence.observation_exclusion_mask=np.ones((240,320),bool)
    assert not len(measured_depth_history(a,b,evidence,cfg,binding)['depths'])


@pytest.mark.parametrize('bad', ['scene_id','frame_id','epoch','execution_signature','frame_digest','commit_version'])
def test_depth_source_context_mismatch_refuses(make_frame,make_prediction,bad):
    m=WorldStateManager(M1Config(temporal_visibility=True,ego_motion_mode='depth',logging=False))
    for t in [100.,100.5]:
        f=make_frame(t=t);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
    h=m._depth_history['front'];h[bad]='tampered'
    with pytest.raises(ValueError,match='depth source binding'):m.precheck(make_frame(t=101.))


def test_exception_during_depth_commit_rolls_back(make_frame,make_prediction,monkeypatch):
    import worldstream.m1.ego_motion as module
    m=WorldStateManager(M1Config(temporal_visibility=True,ego_motion_mode='depth',logging=False))
    f=make_frame();r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
    digest=m.store.source_digest();history=m._history_digest()
    f=make_frame(t=100.5);r=m.precheck(f)
    def fail(*a,**kw):raise RuntimeError('synthetic write failure')
    monkeypatch.setattr(module,'measured_depth_history',fail)
    with pytest.raises(RuntimeError,match='write failure'):m.commit(f,FullOutputs([make_prediction()]),r)
    assert m.store.source_digest()==digest and m._history_digest()==history


def qualified_link(old,new):
    # Synthetic contract fixture ONLY. No qualification artifact is generated.
    from worldstream.m1.state_types import PhysicalCertificationResult
    from worldstream.m1.identity_continuity import IDENTITY_CLAIM
    new.identity_link_qualified=True
    new.physical_certification=PhysicalCertificationResult(True,IDENTITY_CLAIM,(),
        {'risk_qualification_sha256':'synthetic-test-only'},source_state_version=state_geometry_version(old),
        current_state_version=state_geometry_version(new),source_frame_id=old.source_frame_id,current_frame_id=new.source_frame_id)


@pytest.mark.parametrize('speed', [0.,1.,15.])
def test_motion_requires_qualified_calibrated_history(speed):
    from worldstream.m1.world_motion import update_world_motion,prediction
    from worldstream.m1.association import associate
    from worldstream.m1 import PerceptionAnchor
    cfg=M1Config(temporal_visibility=True,identity_continuity=True,dynamic_motion_prediction=True)
    a=WorldAnchor('a','s','object','car',box_geometry([0,0,10],[2,2,2]),100.,source_frame_id='f0',
        source_state_epoch=2,source_execution_signature='x',geometry_uncertainty=.05,geometry_uncertainty_calibrated=True)
    for i in [1,2]:
        b=copy.deepcopy(a);b.source_timestamp=100+i*.5;b.source_frame_id='f'+str(i)
        b.geometry_world=box_geometry([speed*i*.5,0,10],[2,2,2])
        qualified_link(a,b);update_world_motion(a,b,cfg);a=b
    assert a.motion_prediction['reliable'] and not a.velocity_reliable
    center,cov=prediction(a,101.5)
    assert np.allclose(center,[speed*1.5,0,10]) and (np.linalg.eigvalsh(cov)>0).all()
    p=PerceptionAnchor('object','car',box_geometry(center,[2,2,2]))
    pairs,costs=associate([a],[p],101.5,cfg);assert pairs==[(0,0)]
    assert np.allclose(project_source_depth(camera(),camera(100.5,(1,0,0)),[[160,120]],[10])[0],[[142,120]])


@pytest.mark.parametrize('failure', ['direction_change','no_calibration','certificate_version','time_gap'])
def test_motion_refuses_turning_uncertainty_and_bad_evidence(failure):
    from worldstream.m1.world_motion import update_world_motion
    cfg=M1Config()
    a=WorldAnchor('a','s','object','car',box_geometry([0,0,10],[2,2,2]),100.,source_frame_id='a',
        source_state_epoch=2,source_execution_signature='x',geometry_uncertainty=.05,geometry_uncertainty_calibrated=True)
    for i,x in [(1,5),(2,-5 if failure=='direction_change' else 10)]:
        b=copy.deepcopy(a);b.source_frame_id=str(i);b.source_timestamp=100+i*.5
        b.geometry_world=box_geometry([x,0,10],[2,2,2]);qualified_link(a,b)
        if failure=='no_calibration':b.geometry_uncertainty_calibrated=False
        if failure=='certificate_version':b.geometry_world=box_geometry([x+1,0,10],[2,2,2])
        if failure=='time_gap':b.source_timestamp=105
        update_world_motion(a,b,cfg);a=b
    assert not a.motion_prediction['reliable']


def test_prediction_covariance_does_not_widen_association_gate():
    from worldstream.m1.association import associate,association_ambiguities
    from worldstream.m1 import PerceptionAnchor
    a=WorldAnchor('a','s','object','car',box_geometry([0,0,10],[2,2,2]),100.)
    a.motion_prediction=dict(reliable=True,source_state_version=state_geometry_version(a),velocity_world=[10,0,0],
        velocity_covariance=np.eye(3).tolist(),position_sigma_m=.5)
    p=PerceptionAnchor('object','car',box_geometry([5,0,10],[2,2,2]))
    cfg=M1Config(temporal_visibility=True,identity_continuity=True,dynamic_motion_prediction=True)
    assert not associate([a],[p],100.5,cfg)[0]  # uncertainty consumes >2m fixed budget
    a.motion_prediction['velocity_covariance']=(np.eye(3)*.01).tolist();a.motion_prediction['position_sigma_m']=.01
    pairs,costs=associate([a],[p,copy.deepcopy(p)],100.5,cfg)
    assert not pairs  # ambiguity is rejected before the single allocator
    assert association_ambiguities([(0,0)],costs)[0,0]['ambiguous']


@pytest.mark.parametrize('mode', ['rotation','depth'])
def test_occlusion_retains_state_reentry_does_not_mint_identity(make_frame,make_prediction,mode):
    cfg=M1Config(temporal_visibility=True,local_support=True,ego_motion_mode=mode,logging=False)
    m=WorldStateManager(cfg);f=make_frame();r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
    aid=next(iter(m.store.anchors))
    f=make_frame(t=100.5);f.cameras['front'].image[:]=128
    r=m.precheck(f);assert not r.observation_supported_anchor_ids
    m.commit(f,FullOutputs([]),r)
    assert aid in m.store.anchors and m.store.anchors[aid].source_timestamp==100.
    assert m.store.anchors[aid].last_verified_timestamp is None
    f=make_frame(t=101.);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
    assert not m.get_current_certifications().certified_anchor_ids
    f=make_frame(t=102.5);r=m.precheck(f)
    assert not m._previous_cameras and not m._depth_history and not m.store.anchors


def test_wrong_finite_pose_cannot_be_proved_by_epipolar_alone():
    cfg=M1Config(temporal_visibility=True,ego_motion_mode='depth')
    a,b=camera(),camera(100.5,(1,0,0));p=np.array([[130.,80.],[190.,160.]])
    truth,*_=project_source_depth(a,b,p,[5,20])
    wrong=copy.deepcopy(b);wrong.world_to_camera[0,3]=-2.
    predicted,*_=project_source_depth(a,wrong,p,[5,20])
    assert np.linalg.norm(predicted-truth,axis=1).min()>8
    from worldstream.m1.geometry_support import calibrated_correspondence
    # Collinear baseline scale error is epipolar-invisible: demonstrates the
    # explicit limitation rather than pretending calibration is self-verifying.
    assert calibrated_correspondence(a,wrong,p,truth,cfg.flow_epipolar_max_px).all()
    assert not point_predictions(a,wrong,p,cfg)[4]['metric_geometry_certified']


def test_dynamic_foreground_does_not_triangulate(monkeypatch,make_anchor,make_frame):
    import worldstream.m1.foreground_correspondence as module
    a=make_anchor(semantic='car');a.source_frame_id='source';b=copy.deepcopy(a)
    def fail(*args,**kwargs):raise AssertionError('dynamic static triangulation forbidden')
    monkeypatch.setattr(module,'triangulate_correspondences',fail)
    proposal=module.propose_foreground_correspondence(a,b,make_frame(t=100.5),{}, {},M1Config(),
        source_frame_digest='actual',actual_source_frame_id='source')
    assert proposal['status']=='unknown' and 'dynamic_geometry_requires_independent_motion' in proposal['rejection_reasons']


def test_unchanged_acceptance_thresholds_and_opt_in_configs():
    a=M1Config.load('configs/worldstream_m11_diagnostics.yaml')
    for variant in ['baseline','a','b','c']:
        b=M1Config.load('configs/worldstream_m12_'+variant+'.yaml')
        for key in ['flow_fb_max_px','flow_lk_max_error','flow_epipolar_max_px','support_min_points',
                    'support_point_fraction','support_quadrant_fraction','search_radius_px','residual_threshold',
                    'association_gate_m','association_ambiguity_margin_m','foreground_depth_gap_m']:
            assert getattr(a,key)==getattr(b,key)
        assert b.execution_mode=='observe_only' and not b.capture_tensors and not b.trust_detection_velocity


def test_offline_motion_scoring_zero_denominators_and_ambiguous_reference():
    from tools.worldstream_m1.motion_scoring import counts,score_points
    assert counts([])['correspondence_accuracy'] is None
    ref=dict(scorable=np.array([True,True]),target_uv=np.array([[10.,10.],[10.,10.]]),
        source_uv=np.array([[10.,10.],[20.,20.]]),depth_m=np.array([5.,10.]),
        semantic_class=np.array(['car','barrier']),moving=np.array([False,False]))
    m=dict(source_points=[[10.,10.]],current_points=[[10.,10.]],correspondence_valid=[True],geometry_valid=[True],patch_supported=[True])
    scored=score_points(ref,m)
    assert scored[0]['score']=='unscorable'



def test_visibility_consumes_prior_point_depth_for_map_without_affine():
    from worldstream.m1.state_types import AnchorObservation,OnlineVisibilityEvidence
    from worldstream.m1.geometry import Projection
    cfg=M1Config(temporal_visibility=True,ego_motion_mode='depth')
    p=np.array([[100.,70.],[200.,70.],[100.,170.],[200.,170.]])
    q=p+np.array([[-36.,0],[-3.6,0],[-18.,0],[-1.8,0]])
    valid=np.ones(4,bool)
    e=OnlineVisibilityEvidence('front',100,100.5,p,q,np.zeros(4),valid,valid,valid,np.zeros(4),np.zeros((240,320),bool),np.zeros((240,320),bool))
    e.motion_measurements=dict(source_depth_m=np.array([5,50,10,100]),prediction_valid=valid,
        prediction_spread_px=np.full(4,.2),predicted_points=q.copy())
    old=AnchorObservation('front',(50,50,250,190),np.ones((16,16)),100.,1.)
    projection=Projection((50,50,250,190),q,5.,1.,True,'ok',4)
    accepted,_,diagnostics=anchor_visibility(e,old,projection,cfg,None,camera(),camera(100.5,(1,0,0)))
    assert accepted and diagnostics['supported_points']==4
    e.motion_measurements['prediction_spread_px'][:]=5.
    assert not anchor_visibility(e,old,projection,cfg,None,camera(),camera(100.5,(1,0,0)))[0]


def test_identity_depth_pending_mutation_aborts(make_frame,make_prediction):
    cfg=M1Config(temporal_visibility=True,ego_motion_mode='depth',identity_continuity=True,identity_method='temporal',logging=False)
    m=WorldStateManager(cfg)
    for t in [100.,100.5]:
        f=make_frame(t=t);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
    f=make_frame(t=101.);r=m.precheck(f)
    m._identity_history[-1]['depth_history']['front']['epoch']=-1
    with pytest.raises(ValueError,match='source state changed'):m.commit(f,FullOutputs([]),r)
    assert not m._identity_history and not m._depth_history



def test_simultaneous_ego_and_object_motion_are_separate_world_terms():
    from worldstream.m1.world_motion import prediction
    a=WorldAnchor('a','s','object','car',box_geometry([0,0,10],[2,2,2]),100.)
    a.motion_prediction=dict(reliable=True,source_state_version=state_geometry_version(a),velocity_world=[6.,0,0],
        velocity_covariance=(np.eye(3)*.01).tolist(),position_sigma_m=.01)
    current_camera=camera(100.5,(1,0,0))
    center,_=prediction(a,100.5)
    actual_uv,*_=project_points(center[None],current_camera.intrinsic,current_camera.world_to_camera)
    ego_only_uv,*_=project_source_depth(camera(),current_camera,[[160.,120.]],[10.])
    assert np.allclose(center,[3,0,10]) and np.allclose(actual_uv,[[196,120]])
    assert np.allclose(ego_only_uv,[[142,120]])
    assert np.linalg.norm(actual_uv-ego_only_uv)==54.


def test_crossing_targets_use_predictions_and_still_reject_ties():
    from worldstream.m1.association import associate,association_ambiguities
    from worldstream.m1 import PerceptionAnchor
    anchors=[]
    for name,x,v in [('left',-1.,4.),('right',1.,-4.)]:
        a=WorldAnchor(name,'s','object','car',box_geometry([x,0,10],[2,2,2]),100.)
        a.motion_prediction=dict(reliable=True,source_state_version=state_geometry_version(a),velocity_world=[v,0,0],
            velocity_covariance=(np.eye(3)*.01).tolist(),position_sigma_m=.01)
        anchors.append(a)
    cfg=M1Config(temporal_visibility=True,identity_continuity=True,dynamic_motion_prediction=True)
    predictions=[PerceptionAnchor('object','car',box_geometry([x,0,10],[2,2,2])) for x in [1.,-1.]]
    pairs,costs=associate(anchors,predictions,100.5,cfg)
    assert pairs==[(0,0),(1,1)]
    assert all(not r['ambiguous'] for r in association_ambiguities(pairs,costs).values())
    # At the crossing itself both endpoints compete; no nearest-class identity.
    predictions=[PerceptionAnchor('object','car',box_geometry([0,0,10],[2,2,2])) for _ in range(2)]
    pairs,costs=associate(anchors,predictions,100.25,cfg)
    assert all(r['ambiguous'] for r in association_ambiguities(pairs,costs).values())



def test_diagnostic_panel_never_draws_invalid_raw_placeholder(monkeypatch,tmp_path,make_frame):
    from tools.worldstream_m1.motion_scoring import panel
    markers=[]
    monkeypatch.setattr(cv2,'drawMarker',lambda *args,**kwargs:markers.append(args))
    m=dict(source_points=[[20.,20.]],current_points=[[22.,20.]],correspondence_valid=[True],
        raw_points=[[0.,0.]],raw_valid=[False],predicted_points=[[0.,0.]],prediction_valid=[False])
    scored=[dict(index=0,measured=True,score='unscorable')]
    panel(make_frame(),make_frame(t=100.5),'front',m,scored,tmp_path/'case.png')
    assert not markers and (tmp_path/'case.png').is_file()



@pytest.mark.parametrize('kind', ['near_plane','outside','partial'])
def test_visibility_does_not_call_unknown_depth_out_of_view(make_frame,make_prediction,kind):
    cfg=M1Config(temporal_visibility=True,ego_motion_mode='depth',logging=False)
    m=WorldStateManager(cfg);pred=make_prediction()
    if kind=='near_plane':pred.geometry_world=box_geometry([0,0,0],[2,2,2])
    elif kind=='outside':pred.geometry_world=box_geometry([100,0,10],[2,2,2])
    else:pred.geometry_world=box_geometry([9,0,10],[2,2,2])
    f=make_frame();r=m.precheck(f);m.commit(f,FullOutputs([pred]),r)
    r=m.precheck(make_frame(t=100.5));view=next(iter(r.diagnostics['anchors'].values()))['cameras']['front']
    actual=view['evidence']['visibility_status']
    assert actual=={'near_plane':'unknown_depth_or_near_plane','outside':'outside_field_of_view','partial':'partially_outside_field_of_view'}[kind]
    assert not view['valid'] and not view['evidence']['absence_proves_disappearance']
