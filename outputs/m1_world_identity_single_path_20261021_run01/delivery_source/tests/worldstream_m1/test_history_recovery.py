"""Synthetic diagnostics only; never qualification or online evaluation data."""
from dataclasses import replace
import ast
import copy
from pathlib import Path
import cv2
import numpy as np
import pytest
from worldstream.m1 import M1Config,WorldStateManager,FullOutputs,WorldAnchor
from worldstream.m1.geometry import box_geometry,state_geometry_version
from worldstream.m1.correspondence_recovery import retain_depth_points,select_candidates
from worldstream.m1.ego_motion import point_predictions,track_with_prediction
from worldstream.m1.identity_temporal import camera_snapshot_digest
from test_ego_motion_compensation import camera


def cfg(**kw):
    return M1Config(temporal_visibility=True,ego_motion_mode='depth',persistent_depth_tracks=True,
        multi_hypothesis_tracking=True,logging=False,**kw)


def history(cam,p,z=None,u=None):
    return dict(timestamp=cam.timestamp,camera_sha256=camera_snapshot_digest(cam),pixels=np.asarray(p,np.float32),
        depths=np.full(len(p),10.) if z is None else np.asarray(z),uncertainty_m=np.full(len(p),.3) if u is None else np.asarray(u))


def test_retained_tracks_do_not_require_new_corner_coincidence():
    a=camera();p=np.array([[30.,30.],[80.,80.]])
    h=history(a,[[34.3,30.7],[90.2,80.1]])
    retained,d=retain_depth_points(p,h,a,cfg())
    assert np.array_equal(retained[:len(p)],p) and len(retained)==4 and d['retained_points']==2
    _,_,_,depths,info=point_predictions(a,camera(100.5,(.3,0,0)),retained,cfg(),h)
    assert np.isfinite(depths[2:]).all() and np.isnan(depths[:2]).all()
    assert info['depth_association']['one_to_one']==2


@pytest.mark.parametrize('bad',['camera','time','invalid_depth','large_uncertainty'])
def test_invalid_history_never_seeds_other_surface(bad):
    a=camera();h=history(a,[[50.,50.]])
    if bad=='camera':h['camera_sha256']='other'
    elif bad=='time':h['timestamp']-=.5
    elif bad=='invalid_depth':h['depths'][:]=np.nan
    else:h['uncertainty_m'][:]=20
    retained,d=retain_depth_points([[100.,100.]],h,a,cfg())
    assert len(retained)==1 and d['retained_points']==0


@pytest.mark.parametrize('kind',['many_to_one','two_surfaces'])
def test_depth_matching_uniqueness(kind):
    a,b=camera(),camera(100.5,(.3,0,0))
    p=np.array([[50.,50.],[50.1,50.]]) if kind=='many_to_one' else np.array([[50.,50.]])
    hp=[[50.,50.]] if kind=='many_to_one' else [[49.8,50.],[50.2,50.]]
    h=history(a,hp,z=[5.] if kind=='many_to_one' else [5.,50.])
    _,_,_,z,d=point_predictions(a,b,p,cfg(),h)
    assert np.isnan(z).all()
    assert d['depth_association']['duplicate' if kind=='many_to_one' else 'ambiguous']>0


def candidate(q,valid):
    return dict(points=np.asarray(q,float),fb=np.zeros(len(q)),valid=np.asarray(valid,bool))


def test_complete_gate_failure_can_be_excluded_before_conflict(monkeypatch):
    import worldstream.m1.correspondence_recovery as mod
    p=np.array([[50.,50.],[100.,100.]])
    candidates=dict(raw=candidate(p,[True,True]),rotation=candidate(p+10,[True,True]))
    monkeypatch.setattr(mod,'calibrated_correspondence',lambda old,new,p,q,t:np.ones(len(p),bool))
    def support(a,b,p,q,eligible,config):
        ok=(q[:,0]<55)|((q[:,0]>105)&(q[:,0]<115))
        return None,None,ok,np.where(ok,0.,1.)
    monkeypatch.setattr(mod,'measured_patch_support',support)
    q,_,good,d=select_candidates(camera(),camera(100.5),p,candidates,cfg())
    assert good.all() and np.array_equal(q,[[50,50],[110,110]])
    assert d['selected_candidate']==['raw','rotation']


def test_indistinguishable_valid_candidates_remain_unknown(monkeypatch):
    import worldstream.m1.correspondence_recovery as mod
    p=np.array([[50.,50.],[100.,100.]])
    monkeypatch.setattr(mod,'calibrated_correspondence',lambda *a:np.ones(2,bool))
    monkeypatch.setattr(mod,'measured_patch_support',lambda *a:(None,None,np.ones(2,bool),np.zeros(2)))
    q,_,good,d=select_candidates(camera(),camera(100.5),p,dict(raw=candidate(p,[True,True]),depth=candidate(p+10,[True,True])),cfg())
    assert not good.any() and d['ambiguity_rejected']==2


def test_large_camera_motion_pipeline_requires_real_measurement():
    a=camera();b=camera(100.5,(.7,0,0),.09)
    plane_h=b.intrinsic@(b.world_to_camera[:3,:3]+np.outer(b.world_to_camera[:3,3],[0,0,.1]))@np.linalg.inv(a.intrinsic)
    b.image=cv2.warpPerspective(a.image,plane_h,(320,240))
    p=cv2.goodFeaturesToTrack(a.image,maxCorners=200,qualityLevel=.01,minDistance=5)[:,0]
    h=history(a,p)
    q,fb,valid,m=track_with_prediction(a,b,p,cfg(),h)
    assert valid.sum()>20 and m['candidate_funnel']['depth']['attempted']>20
    a.image[:]=128;b.image[:]=128
    assert not track_with_prediction(a,b,p,cfg(),h)[2].any()


def tentative_history(speed=1.,turn=False):
    records=[]
    for i in range(3):
        x=i*.5*speed if not turn or i<2 else -i*.5*speed
        a=WorldAnchor('a','scene','object','car',box_geometry([x,0,50],[2,2,2]),100+i*.5,
            source_frame_id=str(i),source_state_epoch=2,source_execution_signature='default')
        a.association_history=[dict(status='unqualified_geometric_track',source_state_version=state_geometry_version(records[-1]['anchors']['a']) if records else None)]
        records.append(dict(scene_id='scene',frame_id=str(i),timestamp=a.source_timestamp,epoch=2,execution_signature='default',frame_digest='real'+str(i),anchors={'a':a}))
    return records






def test_pending_depth_history_and_search_reset(make_frame,make_prediction,monkeypatch):
    c=cfg(identity_continuity=True,identity_method='temporal')
    m=WorldStateManager(c)
    for t in [100,100.5,101]:
        f=make_frame(t=t);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction(semantic='car')]),r)
    before=m.store.source_digest();f=make_frame(t=101.5);r=m.precheck(f)
    import worldstream.m1.ego_motion as module
    def fail(*a,**kw):raise RuntimeError('actual commit failure')
    monkeypatch.setattr(module,'measured_depth_history',fail)
    with pytest.raises(RuntimeError):m.commit(f,FullOutputs([make_prediction()]),r)
    assert m.store.source_digest()==before
    f=make_frame(t=200,scene='other');m.precheck(f)
    assert not m._identity_history and not m._depth_history


def test_oracle_dependency_is_offline_only(make_frame):
    for file in Path('worldstream/m1').glob('*.py'):
        tree=ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('tools.')
            if isinstance(node,ast.Import):assert not any(a.name.startswith('tools.') for a in node.names)
    from worldstream.m1.input_contract import frame_digest
    f=make_frame();f.metadata['oracle']={'depth':10.}
    with pytest.raises(ValueError,match='forbidden'):frame_digest(f)


def test_search_config_cannot_enable_experimental_tracking():
    with pytest.raises(ValueError,match='shadow'):
        cfg(identity_continuity=True,identity_method='temporal',search_dynamic_motion=True,identity_tracking_policy='experimental')


def test_empty_multi_candidate_shapes():
    q,fb,good,m=track_with_prediction(camera(),camera(100.5),np.empty((0,2)),cfg())
    assert q.shape==(0,2) and good.shape==(0,) and m['initialization_conflict'].shape==(0,)


@pytest.mark.parametrize('bad',['none','heldout','descriptor','timestamp','spatial','low_texture'])
def test_alignment_requires_measured_heldout_quality(bad):
    from types import SimpleNamespace
    from worldstream.m1.correspondence_recovery import measured_aligned_descriptor
    from worldstream.m1.observation_validator import ObservationValidator
    a,b=camera(),camera(100.5)
    b.image=cv2.warpAffine(a.image,np.float32([[1,0,5],[0,1,2]]),(320,240))
    validator=ObservationValidator(cfg())
    x,y=np.meshgrid([70.,90.,110.,130.],[60.,80.,100.,120.]);p=np.c_[x.ravel(),y.ravel()];q=p+[5,2]
    r=(60,50,140,130);s=(65,52,145,132)
    old=SimpleNamespace(region=r,timestamp=a.timestamp,descriptor=validator.descriptor(a.image,r))
    e=SimpleNamespace(source_points=p,current_points=q,supported=np.ones(len(p),bool),source_timestamp=a.timestamp,timestamp=b.timestamp)
    if bad=='heldout':e.current_points[1]+=[3,0]
    elif bad=='descriptor':old.descriptor=old.descriptor+1
    elif bad=='timestamp':e.source_timestamp-=.1
    elif bad=='spatial':e.source_points=np.tile([70.,60.],(16,1))
    elif bad=='low_texture':b.image[:]=128
    descriptor,d=measured_aligned_descriptor(old,a,b,SimpleNamespace(region=s),e,validator)
    if bad=='none':
        assert d['status']=='measured' and validator.residual(descriptor,old.descriptor)<1e-5
    elif bad=='low_texture':assert descriptor is None
    else:assert descriptor is None and d['status']=='unknown'


def test_tentative_velocity_never_becomes_qualified(make_frame):
    from worldstream.m1.world_motion import update_world_motion
    records=tentative_history();old=records[-1]['anchors']['a'];h=dict(status='search_only',center_world=[4.5,0,50])
    new=copy.deepcopy(old);new.source_timestamp=101.5;new.geometry_world.center[:]=h['center_world']
    update_world_motion(old,new,cfg())
    assert h['status']=='search_only' and not new.motion_history and not new.velocity_reliable
    assert not new.motion_prediction['reliable'] and new.physical_certification is None


def test_low_parallax_cannot_create_depth():
    from worldstream.m1.ego_motion import measured_depth_history
    from worldstream.m1.temporal_visibility import build_evidence
    a,b=camera(),camera(100.5,yaw=.05)
    b.image=cv2.warpPerspective(a.image,b.intrinsic@b.world_to_camera[:3,:3]@np.linalg.inv(a.intrinsic),(320,240))
    e=build_evidence(a,b,cfg())
    assert not len(measured_depth_history(a,b,e,cfg(),{})['depths'])


def test_oracle_requires_finished_commits_before_sensor_read(tmp_path,monkeypatch):
    import json
    import tools.worldstream_m1.depth_oracle_diagnosis as module
    folder=tmp_path/'online';(folder/'scenes'/'s').mkdir(parents=True)
    (folder/'protocol.json').write_text(json.dumps({'config':'configs/worldstream_m121_b.yaml'}))
    (folder/'scenes'/'s'/'repeat_0.jsonl').write_text(json.dumps({'precheck':{'diagnostics':{'transaction_status':'pending'}}})+'\n')
    monkeypatch.setattr(module,'world_lidar',lambda *a:pytest.fail('sensor read before commit'))
    with pytest.raises(ValueError,match='completed online commits'):module.run(folder,tmp_path/'offline')


def test_additional_scorer_shared_source_rejects_wrong_and_ambiguous():
    from tools.worldstream_m1.source_anchored_scoring import score_source_anchored
    ref=dict(scorable=np.array([True,True]),source_uv=np.array([[10.,10.],[30.,30.]]),target_uv=np.array([[20.,10.],[40.,30.]]))
    m=dict(source_points=[[10.,10.],[30.,30.],[100.,100.]],current_points=[[20.,10.],[30.,30.],[100.,100.]])
    assert [r['score'] for r in score_source_anchored(ref,m)]==['correct','wrong','unscorable']
    ref['source_uv'][1]=[10.1,10.1]
    assert score_source_anchored(ref,m)[0]['score']=='unscorable'


@pytest.mark.parametrize('field',['scene_id','epoch','execution_signature','frame_id','frame_digest','commit_version'])
def test_new_depth_history_still_requires_manager_source_binding(make_frame,make_prediction,field):
    m=WorldStateManager(cfg());f=make_frame();r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
    m._depth_history['front']=dict(scene_id=f.scene_id,epoch=m.store.generation,execution_signature=f.execution_signature,
        frame_id=m._previous_frame_id,frame_digest=m._previous_frame_digest,commit_version=m.store.commit_count)
    m._depth_history['front'][field]='tampered'
    with pytest.raises(ValueError,match='binding mismatch'):m._bound_depth_history('front',make_frame(t=100.5))


def test_manager_rejects_retired_identity_search():
    with pytest.raises(ValueError,match='Retired'):
        WorldStateManager(cfg(identity_continuity=True,identity_method='temporal',search_dynamic_motion=True))


def test_new_producer_configuration_invalidates_old_protocol(make_frame):
    from worldstream.m1.identity_continuity import identity_protocol_digest
    f=make_frame();a=M1Config.load('configs/worldstream_m121_old_b.yaml');b=M1Config.load('configs/worldstream_m121_b.yaml')
    assert identity_protocol_digest(a,f)!=identity_protocol_digest(b,f)
