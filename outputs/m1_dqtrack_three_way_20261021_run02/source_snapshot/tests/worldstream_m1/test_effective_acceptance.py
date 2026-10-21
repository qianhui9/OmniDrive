"""Adversarial M1 lifecycle contracts. Synthetic tests are not qualification data."""
import copy
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
from worldstream.m1 import WorldStateManager, M1Config, FullOutputs, ComputedFeature
from worldstream.m1.physical_certifier import EvidenceIssuer, _canonical
from worldstream.m1.foreground_correspondence import propose_foreground_correspondence, producer_protocol_digest
from test_evidence_contract import qualified_fixture, certify_fixture, measured_pair
from test_integration import make_model, inputs


@pytest.mark.parametrize('mutation', ['duplicate', 'missing_cameras', 'invalid_pose', 'invalid_K', 'invalid_time', 'unknown_signature'])
def test_invalid_input_never_seeds_actual_history(config,make_frame,make_prediction,mutation):
    m=WorldStateManager(replace(config,temporal_visibility=True))
    m.initialize(make_frame(),FullOutputs([make_prediction()]))
    frame=make_frame(100.5)
    if mutation=='duplicate':frame.frame_id='100.0'
    elif mutation=='missing_cameras':frame.cameras={}
    elif mutation=='invalid_pose':frame.ego_to_world=None
    elif mutation=='invalid_K':frame.cameras['front'].intrinsic[0,0]=np.nan
    elif mutation=='invalid_time':frame.cameras['front'].timestamp=np.nan
    else:frame.execution_signature=''
    r=m.precheck(frame)
    assert r.full_refresh_hint and not r.valid_anchor_ids
    m.commit(frame,FullOutputs([make_prediction()]),r)
    assert r.diagnostics['transaction_status']=='rejected'
    assert not m.store.anchors and not m._previous_cameras
    assert m.export_state_audit()['transaction_status']=='empty'
    assert not m.get_current_certifications().certified_anchor_ids


@pytest.mark.parametrize('mutation', ['signature','scene','feature_signature','feature_source','feature_dependency','provenance','score','camera_list'])
def test_complete_source_metadata_bound_during_transaction(config,make_frame,make_prediction,mutation):
    m=WorldStateManager(config);m.initialize(make_frame(),FullOutputs([make_prediction()]))
    f=make_frame(100.5);r=m.precheck(f)
    a=next(iter(m.store.anchors.values()));feature=next(iter(m.store.features.values()))
    if mutation=='signature':m.store.execution_signature='tampered'
    elif mutation=='scene':m.store.scene_id='tampered'
    elif mutation=='feature_signature':feature.execution_signature='tampered'
    elif mutation=='feature_source':feature.source_frame_id='tampered'
    elif mutation=='feature_dependency':feature.dependency_scope='tampered'
    elif mutation=='provenance':next(iter(m.store.provenance.values())).computation_unit='tampered'
    elif mutation=='score':a.source_score=.999
    else:feature.source_camera_ids.append('missing')
    with pytest.raises(ValueError,match='source state'):m.commit(f,FullOutputs([make_prediction()]),r)
    assert r.diagnostics['transaction_status']=='aborted' and m._pending is None
    assert not m.store.anchors and r.diagnostics['source_state_discarded']


@pytest.mark.parametrize('mutation', ['missing_calibration_logs','validation_log_reuse','missing_evaluation_log','scene_leak','log_leak','too_few','errors','unscorable','protocol'])
def test_qualification_independence_and_risk_fail_closed(mutation):
    old,new,frame,issuer,proposal=qualified_fixture()
    q=json.loads(issuer._qualification_json)
    if mutation=='missing_calibration_logs':q.pop('calibration_logs')
    elif mutation=='validation_log_reuse':frame.metadata['log_token']='log-1'
    elif mutation=='missing_evaluation_log':frame.metadata.pop('log_token')
    elif mutation=='scene_leak':q['calibration_scenes']=[q['validation_scene_units'][0]['scene_id']]
    elif mutation=='log_leak':q['calibration_logs']=['log-0']
    elif mutation=='too_few':q['validation_scene_units']=q['validation_scene_units'][:58]
    elif mutation=='errors':q['validation_scene_units'][0]['false_acceptances']=1
    elif mutation=='unscorable':q['validation_scene_units'][0]['unscorable_accepted']=1
    else:q['protocol_digest']='stale'
    issuer=EvidenceIssuer(q,hashlib.sha256(_canonical(q).encode()).hexdigest())
    proposal['evidence']=issuer._mint_measured(proposal['evidence'])
    assert not certify_fixture(old,new,frame,issuer,proposal).certified


def test_digest_pin_rejects_modified_artifact():
    with pytest.raises(ValueError,match='audited digest'):EvidenceIssuer({'modified':True},'wrong')


@pytest.mark.parametrize('mutation',['source_points','current_points'])
def test_foreground_duplicate_tracks_cannot_mint_evidence(mutation):
    cfg,old,new,frame,source,e,issuer,_=measured_pair()
    pts=getattr(e,mutation);pts[1]=pts[0]
    proposal=propose_foreground_correspondence(old,new,frame,{'front':source},{'front':e},cfg,issuer,'digest','source')
    assert proposal['status']=='unknown' and proposal['evidence'] is None
    assert any(v['reason']=='nonunique_or_invalid_measured_tracks' for v in proposal['views'])


def test_foreground_protocol_binds_transitive_code(monkeypatch):
    from pathlib import Path
    cfg,_,_,frame,*_=measured_pair();before=producer_protocol_digest(cfg,frame)
    read=Path.read_bytes
    def changed(path):return read(path)+(b'\n# changed helper' if path.name=='geometry.py' else b'')
    monkeypatch.setattr(Path,'read_bytes',changed)
    assert producer_protocol_digest(cfg,frame)!=before


@pytest.mark.parametrize('phase',['backbone','detection','adapter'])
def test_real_inference_failure_aborts_pending_immediately(tmp_path,config,monkeypatch,phase):
    m=WorldStateManager(config);model=make_model(tmp_path,m)
    meta,data=inputs();model.simple_test(meta,**data)
    digest=m.store.source_digest();last_frame=m._previous_frame_id
    def fail(*a,**k):raise RuntimeError('injected runtime failure')
    if phase=='backbone':model.img_backbone=fail
    elif phase=='detection':model.pts_bbox_head=fail
    else:
        import worldstream.m1.omnidrive_adapter as adapter
        monkeypatch.setattr(adapter,'outputs_from_model',fail)
    meta,data=inputs(100.5)
    with pytest.raises(RuntimeError,match='injected'):model.simple_test(meta,**data)
    assert m._pending is None and m._pending_store is None
    assert m.last_result.diagnostics['transaction_status']=='aborted'
    assert m.store.source_digest()==digest and m._previous_frame_id==last_frame


def test_export_distinguishes_metadata_and_claims(config,make_frame,make_prediction):
    m=WorldStateManager(config)
    feature=ComputedFeature('backbone','backbone',(1,4),'float32','cuda','EVA')
    m.initialize(make_frame(),FullOutputs([make_prediction()],[feature]))
    audit=m.export_state_audit();row=audit['anchors'][0]
    assert row['observed'] and not row['identity_certified'] and not row['physical_or_geometry_certified']
    assert audit['features']['backbone']['tensor_availability']=='unavailable'
    assert not audit['features']['backbone']['network_reuse_authorized']
    audit['anchors'][0]['cameras'].clear()
    assert m.export_state_audit()['anchors'][0]['cameras']


def test_effective_config_is_observation_only():
    from mmcv import Config
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_effective.py')
    m1=M1Config.load(cfg.model.m1_config)
    assert cfg.model.m1_enable and not cfg.model.m2_enable and not cfg.model.m3_enable
    assert m1.temporal_visibility and m1.identity_continuity and m1.foreground_correspondence
    assert m1.execution_mode=='observe_only' and not m1.capture_tensors
    assert m1.physical_qualification_path is None


@pytest.mark.parametrize('mutation',['current_log','source_image','source_timestamp','source_frame'])
def test_pending_camera_and_metadata_provenance_cannot_change(config,make_frame,make_prediction,mutation):
    m=WorldStateManager(replace(config,temporal_visibility=True))
    m.initialize(make_frame(),FullOutputs([make_prediction()]))
    f=make_frame(100.5);f.metadata['log_token']='actual-log';r=m.precheck(f)
    if mutation=='current_log':f.metadata['log_token']='forged-log'
    elif mutation=='source_image':m._previous_cameras['front'].image=np.zeros_like(m._previous_cameras['front'].image)
    elif mutation=='source_timestamp':m._previous_cameras['front'].timestamp-=.1
    else:m._previous_frame_id='forged-source'
    with pytest.raises(ValueError):m.commit(f,FullOutputs([make_prediction()]),r)
    assert m._pending is None and r.diagnostics['transaction_status']=='aborted'
    if mutation!='current_log':
        assert not m.store.anchors and not m._previous_cameras
        assert r.diagnostics['source_state_discarded']


def test_acceptance_verifier_rejects_future_sources(tmp_path,config,make_frame,make_prediction):
    from worldstream.m1.omnidrive_adapter import append_result
    from tools.worldstream_m1.effective_acceptance import verify
    m=WorldStateManager(config);out=tmp_path/'trace.jsonl'
    for t in (100.,100.5):
        f=make_frame(t);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
        append_result(out,r,m.last_commit_diagnostics,m.export_state_audit())
    assert verify(out)['counts']['frames']==2
    rows=[json.loads(x) for x in out.read_text().splitlines()]
    next(iter(rows[1]['precheck']['anchor_provenance'].values()))['source_timestamp']=101.
    out.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    with pytest.raises(AssertionError):verify(out)


def test_nonadjacent_frame_id_repeat_is_rejected(config,make_frame,make_prediction):
    m=WorldStateManager(config)
    for t in (100.,100.5,101.):
        f=make_frame(t);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
    f=make_frame(101.5,frame_id='100.0');r=m.precheck(f)
    assert r.diagnostics['reset_reason']=='duplicate_frame_id'
    m.commit(f,FullOutputs([make_prediction()]),r)
    assert r.diagnostics['transaction_status']=='rejected' and not m.store.anchors
