import copy
import json
from dataclasses import replace
import pytest
from worldstream.m1 import WorldStateManager, FullOutputs
from worldstream.m1.dqtrack_bridge import align


def payload(frames=('100.0',)):
    return dict(m1_manifest=dict(coordinate_system='world', size_order='wlh',
        timestamp_unit='seconds', run_id='run', model_config='verified_config',
        frames={f:dict(scene_id='scene', timestamp=float(f),sample_token='token'+f) for f in frames}),
        results={'token'+f:[dict(sample_token='token'+f, tracking_id='1', tracking_name='barrier',
            tracking_score=.8,translation=[0,0,10],size=[2,2,2],rotation=[1,0,0,0],velocity=[0,0])]
            for f in frames})


def write(tmp_path, data, config, mode='hint'):
    p=tmp_path/'tracks.json';p.write_text(json.dumps(data))
    return replace(config,dqtrack_mode=mode,dqtrack_results_path=str(p))


def test_read_mapping_and_namespace(tmp_path,config,make_frame,make_prediction):
    data=payload();cfg=write(tmp_path,data,config)
    hints,d=align(make_frame(),[make_prediction()],cfg)
    assert hints[0]['namespace']==['run','scene'] and d['aligned']==1
    assert not align(make_frame(scene='other'),[make_prediction()],cfg)[0]
    assert not align(make_frame(t=100.1),[make_prediction()],cfg)[0]
    data['m1_manifest']['frames']['alias']=data['m1_manifest']['frames']['100.0']
    assert not align(make_frame(),[make_prediction()],write(tmp_path,data,config))[0]


@pytest.mark.parametrize('field,value', [('translation',[100,0,10]),('size',[100,2,2]),
    ('rotation',[0,0,0,0]),('tracking_name','car'),('tracking_score',float('nan')),
    ('velocity',[float('inf'),0]),('sample_token','wrong'),('tracking_id',None)])
def test_reject_invalid(tmp_path,config,make_frame,make_prediction,field,value):
    data=payload();data['results']['token100.0'][0][field]=value
    assert not align(make_frame(),[make_prediction()],write(tmp_path,data,config))[0]


def test_competition_and_coordinates(tmp_path,config,make_frame,make_prediction):
    data=payload();cfg=write(tmp_path,data,config)
    assert not align(make_frame(),[make_prediction(),make_prediction(.2)],cfg)[0]
    row=copy.deepcopy(data['results']['token100.0'][0]);row['tracking_id']='2'
    data['results']['token100.0'].append(row)
    assert not align(make_frame(),[make_prediction()],write(tmp_path,data,config))[0]
    data['m1_manifest']['coordinate_system']='ego'
    assert not align(make_frame(),[make_prediction()],write(tmp_path,data,config))[0]


def test_modes_transactions_and_certification(tmp_path,config,make_frame,make_prediction):
    data=payload(('100.0','100.5'))
    managers=[]
    for mode in ('off','shadow','hint'):
        cfg=write(tmp_path,data,config,mode);m=WorldStateManager(cfg)
        for t in (100.,100.5):
            f=make_frame(t=t);r=m.precheck(f)
            assert 'dqtrack' not in r.diagnostics
            m.commit(f,FullOutputs([make_prediction()]),r)
            with pytest.raises(ValueError):m.commit(f,FullOutputs([make_prediction()]),r)
        managers.append(m)
    baseline,shadow,hint=managers
    assert baseline.store.source_digest()==shadow.store.source_digest()
    assert baseline.last_commit_diagnostics['association_pairs']==shadow.last_commit_diagnostics['association_pairs']
    a=next(iter(hint.store.anchors.values()));b=next(iter(baseline.store.anchors.values()))
    assert a.anchor_id==b.anchor_id
    assert a.association_basis=='uncertified_dqtrack_corroborated_geometry'
    assert a.certified_observation_count==b.certified_observation_count
    assert not a.identity_link_qualified
    assert not next(iter(shadow.store.anchors.values())).external_tracking


@pytest.mark.parametrize('t,scene,x',[(100.5,'scene',100),(103.,'scene',0),(100.5,'other',0)])
def test_hint_hard_gates(tmp_path,config,make_frame,make_prediction,t,scene,x):
    data=payload(('100.0',str(t)));cfg=write(tmp_path,data,config)
    m=WorldStateManager(cfg)
    f=make_frame();r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
    old=next(iter(m.store.anchors))
    f=make_frame(t=t,scene=scene);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction(x)]),r)
    assert all(a.anchor_id != old for a in m.store.anchors.values() if a.source_timestamp == t)


def test_unavailable(tmp_path,config,make_frame,make_prediction):
    cfg=replace(config,dqtrack_mode='hint',dqtrack_results_path=str(tmp_path/'missing'))
    assert not align(make_frame(),[make_prediction()],cfg)[0]
    for data in ([],{}, {'m1_manifest':None}):
        assert not align(make_frame(),[make_prediction()],write(tmp_path,data,config))[0]


def test_hint_rejects_conflicting_continuation(tmp_path,config,make_frame,make_prediction):
    data=payload(('100.0','100.5'));data['results']['token100.5'][0]['tracking_id']='2'
    for mode in ('off','shadow','hint'):
        m=WorldStateManager(write(tmp_path,data,config,mode))
        for t in (100.,100.5):
            f=make_frame(t=t);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
        assert bool(m.last_commit_diagnostics['association_pairs']) == (mode!='hint')
        assert all(not a.identity_link_qualified for a in m.store.anchors.values())


def test_prepare_and_evaluate(tmp_path,config,make_frame,make_prediction):
    from tools.worldstream_m1.prepare_dqtrack_results import prepare
    from tools.worldstream_m1.evaluate_dqtrack import evaluate
    data=payload(('100.0','100.5'))
    samples=[dict(token='token'+str(t),scene_token='scene',timestamp=int(t*1e6)) for t in (100.,100.5)]
    normalized=prepare(data,samples,'run','verified_config','a'*64)
    assert normalized['m1_manifest']['frames']['token100.0']['timestamp']==100.
    cfg=write(tmp_path,data,config)
    batches=[(make_frame(t=t),FullOutputs([make_prediction()])) for t in (100.,100.5)]
    result=evaluate(batches,config,cfg.dqtrack_results_path)
    assert result['methods']['hint']['consistent_associations']==1
    assert result['methods']['shadow']['pairs_equal_baseline']
    assert result['methods']['shadow']['certifications_equal_baseline']
    assert result['methods']['hint']['wrong_identity_continuations'] is None


def test_size_order_and_map_exclusion(tmp_path,config,make_frame,make_prediction):
    from worldstream.m1.geometry import box_geometry
    data=payload();data['results']['token100.0'][0]['size']=[1,4,2]
    pred=make_prediction();pred.geometry_world=box_geometry([0,0,10],[4,1,2])
    hints,_=align(make_frame(),[pred,make_prediction(kind='map')],write(tmp_path,data,config))
    assert list(hints)==[0]


def test_external_metadata_transaction_isolation(tmp_path,config,make_frame,make_prediction):
    m=WorldStateManager(write(tmp_path,payload(),config))
    f=make_frame();r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
    aid=next(iter(m.store.anchors));staged=m.store.fork()
    staged.anchors[aid].external_tracking['namespace'][0]='mutated'
    assert m.store.anchors[aid].external_tracking['namespace'][0]=='run'
    audit=m.export_state_audit();audit['anchors'][0]['external_tracking']['tracking_id']='mutated'
    assert m.store.anchors[aid].external_tracking['tracking_id']=='1'
    f=make_frame(t=100.5);r=m.precheck(f)
    m.store.anchors[aid].external_tracking['tracking_id']='tampered'
    with pytest.raises(ValueError):m.commit(f,FullOutputs([make_prediction()]),r)


def test_precheck_never_reads_tracker(tmp_path,config,make_frame,make_prediction,monkeypatch):
    import worldstream.m1.dqtrack_bridge as bridge
    m=WorldStateManager(write(tmp_path,payload(),config))
    def forbidden(*args):raise AssertionError('Tracker read before actual perception')
    monkeypatch.setattr(bridge,'align',forbidden)
    m.precheck(make_frame())


def test_ambiguous_geometry_is_not_rescued(tmp_path,config,make_frame,make_prediction):
    data=payload(('100.0','100.5'));m=WorldStateManager(write(tmp_path,data,config))
    f=make_frame();r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
    f=make_frame(t=100.5);r=m.precheck(f)
    m.commit(f,FullOutputs([make_prediction(),make_prediction()]),r)
    assert not m.last_commit_diagnostics['association_pairs']
    assert not m.last_commit_diagnostics['dqtrack']['aligned']
    assert all(not a.identity_link_qualified for a in m.store.anchors.values())


def test_run_namespace_is_unknown_not_conflict(tmp_path,config,make_frame,make_prediction):
    from worldstream.m1.dqtrack_bridge import identity_conflicts,corroborated_edges
    m=WorldStateManager(write(tmp_path,payload(),config))
    f=make_frame();r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
    old=next(iter(m.store.anchors.values()))
    other=dict(old.external_tracking,namespace=['another_run','scene'],tracking_id='2')
    assert not identity_conflicts([old],{0:other},[(0,0)])
    assert not corroborated_edges([old],{0:other},[(0,0)],__import__('numpy').zeros((1,1)))
