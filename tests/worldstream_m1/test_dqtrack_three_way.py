import copy
from dataclasses import replace
import numpy as np
import pytest
from worldstream.m1 import WorldStateManager, FullOutputs
from tools.worldstream_m1.dqtrack_three_way_policy import OfflineDQManager, choose, fresh
from test_dqtrack_bridge import payload, write


def hint(frame,tid):
    return dict(namespace=['run',frame.scene_id],tracking_id=tid,timestamp=frame.timestamp,
        frame_id=frame.frame_id,status='aligned',tracking_score=.8)


def historical(anchor,frame,tid):
    anchor.source_frame_id='old';anchor.source_timestamp=frame.timestamp-.5
    anchor.source_state_epoch=0;anchor.source_execution_signature=frame.execution_signature
    anchor.evidence_origin='actual_perception';anchor.state_version='version'
    anchor.external_tracking=dict(hint(frame,tid),frame_id='old',timestamp=frame.timestamp-.5,
        count=3,state_version='version',epoch=0,signature=frame.execution_signature,origin='actual_perception')
    return anchor


def test_conflict_missing_qualified(config,make_frame,make_anchor):
    f=make_frame();a=historical(make_anchor(),f,'1');cost=np.array([[.1]])
    def run(h):return choose([a],[None],[(0,0)],cost,h,f,0,'actual_perception','B')[0]
    assert run({0:hint(f,'2')})==[]
    assert run({})==[(0,0)]
    a.identity_link_qualified=True
    assert run({0:hint(f,'2')})==[(0,0)]


@pytest.mark.parametrize('invalid',['missing','duplicate','expired','scene','version','epoch','signature','origin','immature','illegal'])
def test_component_fail_closed(invalid,make_frame,make_anchor):
    f=make_frame();aa=[historical(make_anchor(),f,str(i)) for i in range(2)]
    h={i:hint(f,str(i)) for i in range(2)};cost=np.ones((2,2))*.2
    if invalid=='missing':h.pop(1)
    if invalid=='duplicate':h[1]=h[0]
    if invalid=='expired':aa[0].external_tracking['timestamp']-=2
    if invalid=='scene':h[0]['namespace'][1]='wrong'
    if invalid=='version':aa[0].state_version='changed'
    if invalid=='epoch':aa[0].external_tracking['epoch']=9
    if invalid=='signature':aa[0].external_tracking['signature']='wrong'
    if invalid=='origin':aa[0].external_tracking['origin']='wrong'
    if invalid=='immature':aa[0].external_tracking['count']=2
    if invalid=='illegal':cost[0,0]=np.inf
    assert choose(aa,[None,None],[],cost,h,f,0,'actual_perception','C')[0]==[]


def test_complete_component(make_frame,make_anchor):
    f=make_frame();aa=[historical(make_anchor(),f,str(i)) for i in range(2)]
    h={i:hint(f,str(1-i)) for i in range(2)};cost=np.ones((2,2))*.2
    assert choose(aa,[None,None],[],cost,h,f,0,'actual_perception','B')[0]==[]
    pairs,d=choose(aa,[None,None],[],cost,h,f,0,'actual_perception','C')
    assert pairs==[(0,1),(1,0)] and len(d['resolved'])==2


def test_missing_parity_and_off_no_bridge(config,make_frame,make_prediction,tmp_path,monkeypatch):
    import tools.worldstream_m1.dqtrack_three_way_policy as policy
    a=WorldStateManager(config);b=OfflineDQManager(config,'B',tmp_path/'missing')
    monkeypatch.setattr(policy,'align',lambda *args:({},dict(elapsed_ms=0,alignments={})))
    for t in (100.,100.5):
        f=make_frame(t=t)
        for m in (a,b):m.commit(f,FullOutputs([make_prediction()]),m.precheck(f))
        assert a.store.source_digest()==b.store.source_digest()
        assert a.last_commit_diagnostics['association_pairs']==b.last_commit_diagnostics['association_pairs']
    assert 'dqtrack' not in a.last_commit_diagnostics
    with pytest.raises(ValueError):WorldStateManager(replace(config,dqtrack_mode='hint'))


def test_metadata_atomicity_no_qualification(config,make_frame,make_prediction,tmp_path,monkeypatch):
    cfg=write(tmp_path,payload(('100.0','100.5','101.0','101.5')),config,'off')
    m=OfflineDQManager(cfg,'C',cfg.dqtrack_results_path)
    for t in (100.,100.5,101.):
        f=make_frame(t=t);m.commit(f,FullOutputs([make_prediction()]),m.precheck(f))
    a=next(iter(m.store.anchors.values()));assert a.external_tracking['count']==3
    assert not a.identity_link_qualified and not a.velocity_reliable and not a.physical_certification
    assert a.certified_observation_count==0
    digest=m.store.source_digest();f=make_frame(t=101.5);r=m.precheck(f)
    def fail(*args,**kwargs):raise RuntimeError('injected failure after research policy')
    monkeypatch.setattr(m.validator,'measure',fail,raising=False)
    # Force an exception after private _commit has updated geometry and metadata.
    original=OfflineDQManager._commit
    def late_failure(worker,*args):original(worker,*args);raise RuntimeError('late rollback')
    monkeypatch.setattr(OfflineDQManager,'_commit',late_failure)
    with pytest.raises(RuntimeError):m.commit(f,FullOutputs([make_prediction()]),r)
    assert m.store.source_digest()==digest


def test_metadata_tamper_is_transaction_source(config,make_frame,make_prediction,tmp_path):
    cfg=write(tmp_path,payload(),config,'off');m=OfflineDQManager(cfg,'B',cfg.dqtrack_results_path)
    f=make_frame();m.commit(f,FullOutputs([make_prediction()]),m.precheck(f))
    f=make_frame(t=100.5);r=m.precheck(f)
    next(iter(m.store.anchors.values())).external_tracking['tracking_id']='tampered'
    with pytest.raises(ValueError):m.commit(f,FullOutputs([make_prediction()]),r)
    assert not m.store.anchors
