"""Scene isolation, trusted non-label context, denominators and read-only funnels."""
from dataclasses import replace
import copy
import numpy as np
import pytest
from worldstream.m1 import WorldStateManager, FullOutputs
from worldstream.m1.source_context import LogSourceContext,metadata_log_source
from worldstream.m1.omnidrive_adapter import frame_from_model
from tools.worldstream_m1.scene_protocol import select_scenes,aggregate,ratio
from tools.worldstream_m1.scene_metrics import summarize_scene
from test_integration import inputs


def context():
    return LogSourceContext.from_rows([dict(token='scene',log_token='real-log'),dict(token='other',log_token='other-log')],
        [dict(token='100.0',scene_token='scene'),dict(token='100.5',scene_token='scene'),dict(token='b0',scene_token='other')])


def test_same_log_multiple_scenes_is_legal_but_not_independent_units():
    ctx=LogSourceContext.from_rows([dict(token='a',log_token='one-log'),dict(token='b',log_token='one-log')])
    assert len(set(ctx.scene_logs.values()))==1


@pytest.mark.parametrize('change',['duplicate_scene','duplicate_frame','missing_log','unknown_sample_scene'])
def test_trusted_context_rejects_inconsistent_tables(change):
    scenes=[dict(token='scene',log_token='real-log')];samples=[dict(token='f',scene_token='scene')]
    if change=='duplicate_scene':scenes*=2
    elif change=='duplicate_frame':samples*=2
    elif change=='missing_log':scenes[0]['log_token']=''
    else:samples[0]['scene_token']='absent'
    with pytest.raises(ValueError):LogSourceContext.from_rows(scenes,samples)


def test_raw_and_model_metadata_use_same_real_log(config):
    meta,data=inputs();ctx=context()
    f=frame_from_model(meta,data,config,ctx)
    assert f.metadata['log_token']=='real-log' and f.metadata['log_source']['context_sha256']==ctx.digest
    assert f.metadata['log_source']==metadata_log_source({},'scene','100.0',ctx)['log_source']


@pytest.mark.parametrize('change',['wrong_log','wrong_scene','missing_frame','blank_log'])
def test_model_source_mismatch_is_rejected(config,change):
    meta,data=inputs();ctx=context()
    if change=='wrong_log':meta[0]['log_token']='forged'
    elif change=='wrong_scene':meta[0]['scene_token']='other'
    elif change=='missing_frame':meta[0]['sample_idx']='absent'
    else:meta[0]['log_token']=''
    with pytest.raises(ValueError):frame_from_model(meta,data,config,ctx)


def test_missing_context_does_not_invent_log(config):
    meta,data=inputs();f=frame_from_model(meta,data,config)
    assert 'log_token' not in f.metadata and f.metadata['log_provenance_status']=='missing'


@pytest.mark.parametrize('change',['missing','changed','binding'])
def test_same_scene_log_change_clears_history(config,make_frame,make_prediction,change):
    m=WorldStateManager(replace(config,temporal_visibility=True))
    f=make_frame();f.metadata.update(context().metadata_for('scene',f.frame_id))
    m.initialize(f,FullOutputs([make_prediction()]))
    g=make_frame(100.5);g.metadata.update(context().metadata_for('scene',g.frame_id))
    if change=='missing':g.metadata={}
    elif change=='changed':g.metadata['log_token']='different'
    else:g.metadata['log_source']['scene_id']='other'
    r=m.precheck(g)
    assert r.diagnostics['input_rejection'] in ('log_source_changed_or_missing','log_scene_binding_mismatch')
    m.commit(g,FullOutputs([make_prediction()]),r)
    assert not m.store.anchors and not m._previous_cameras and not m._identity_history


def test_scene_boundary_has_no_anchor_image_identity_or_digest_reuse(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,temporal_visibility=True,identity_continuity=True,identity_method='temporal'))
    for t in (100.,100.5):
        f=make_frame(t);f.metadata.update(context().metadata_for('scene',f.frame_id))
        r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
    old_ids=set(m.store.anchors);old_digest=m.store.source_digest();old_frame_digest=m._previous_frame_digest
    assert m._previous_cameras and m._identity_history
    g=make_frame(101,scene='other',frame_id='b0');g.metadata.update(context().metadata_for('other','b0'))
    r=m.precheck(g)
    assert r.diagnostics['reset_reason']=='scene_change' and not r.anchor_provenance
    assert not m.store.anchors and not m._previous_cameras and not m._identity_history and m._previous_frame_digest is None
    m.commit(g,FullOutputs([make_prediction()]),r)
    assert old_ids.isdisjoint(m.store.anchors)
    assert m.store.source_digest()!=old_digest and m._previous_frame_digest!=old_frame_digest
    assert all(x['scene_id']=='other' for x in m._identity_history)
    assert m.export_state_audit()['log_token']=='other-log'


def test_selector_does_not_interleave_and_records_missing_keyframes():
    rows=[dict(token=f'{s}{t}',scene_token=s,timestamp=t*500000) for t in (3,1,2) for s in ('b','a')]
    selected,manifest=select_scenes(rows,{'a1','a3','b1','b2','b3'},scene_limit=1)
    assert list(selected)==['a'] and [r['token'] for r in selected['a']]==['a1','a3']
    spec=manifest['scenes'][0]
    assert spec['missing_prediction_frames']==['a2']
    assert [r['dataset_sequence_index'] for r in spec['selected_frames']]==[0,2]
    assert manifest['unexecuted_scene_ids']==['b']
    all_selected,_=select_scenes(rows,scene_limit=0,frame_limit=0)
    assert len(all_selected)==2 and all(len(v)==3 for v in all_selected.values())


@pytest.mark.parametrize('change',['token','time'])
def test_selector_rejects_duplicate_dataset_identity(change):
    a=dict(token='f',scene_token='s',timestamp=10)
    b=dict(token='f' if change=='token' else 'g',scene_token='s',timestamp=20 if change=='token' else 10)
    with pytest.raises(ValueError):select_scenes([a,b])


def test_micro_and_macro_use_distinct_denominators_and_null():
    def scene(name,n,d):
        return dict(scene_id=name,counts=dict(observation_supported_anchor_events=n,historical_projection_candidates=d),
            observation_anchor_coverage=ratio(n,d),observed_pixel_coverage=None,certification_coverage=None,
            identity_candidate_coverage=None,identity_unknown_fraction=None)
    result=aggregate([scene('large',0,100),scene('small',1,1),scene('empty',0,0)])
    assert result['micro']['observation_anchor_coverage']==1/101
    assert result['macro']['observation_anchor_coverage']['mean']==.5
    assert result['macro']['observation_anchor_coverage']['n']==2
    assert result['macro']['certification_coverage']['mean'] is None
    assert result['micro']['identity_candidate_coverage'] is None


def test_diagnostics_do_not_change_acceptance(config,make_frame,make_prediction):
    results=[]
    for enabled in (False,True):
        m=WorldStateManager(replace(config,temporal_visibility=True,observation_diagnostics=enabled))
        m.initialize(make_frame(),FullOutputs([make_prediction()]))
        f=make_frame(100.5);r=m.precheck(f)
        results.append((r.valid_anchor_ids,m.last_masks['front'].copy()))
        if enabled:
            v=next(iter(r.diagnostics['anchors'].values()))['cameras']['front']
            assert v['evidence']['observation_funnel']['source_descriptor_reconstruction']['passed']
            assert v['evidence']['finite_claims']['foreground_ownership']=='unknown'
    assert results[0][0]==results[1][0] and np.array_equal(results[0][1],results[1][1])


def test_offline_wrong_unknown_is_not_false_accept_and_logs_not_scenes():
    from tools.worldstream_m1.offline_identity_scoring import score_links,summarize,log_risk
    identities={'f0':{'a':'instance-a'},'f1':{'b':'instance-b'}}
    base=dict(source_frame_id='f0',current_frame_id='f1',anchor_id='a',target_anchor_id='b',status='unknown',certified=False)
    rows=score_links([base],identities,'scene','one-log')
    result=summarize(rows)
    assert result['counts']['wrong']==1 and result['counts']['formal_wrong']==0
    assert result['candidate_accuracy'] is None and result['formal_false_acceptance_risk'] is None
    # Two scenes sharing a log never become two independent qualification units.
    rows+=score_links([dict(base,status='candidate',certified=True)],identities,'second-scene','one-log')
    risk=log_risk(rows)
    assert len(risk['calibration_logs'])==1 and not risk['validation_logs']
    assert risk['status']=='BLOCKED' and risk['confidence_upper_95'] is None


def test_offline_risk_does_not_count_unscorable_acceptance_as_success():
    from tools.worldstream_m1.offline_identity_scoring import log_risk
    row=dict(log_token='validation',scene_id='s',formal_certified=True,score='unscorable')
    risk=log_risk([row],['calibration'],['validation'])
    assert risk['unscorable_formal_accepts']==1 and risk['fully_scorable_log_units']==0
    assert risk['confidence_upper_95'] is None and risk['status']=='BLOCKED'


def test_offline_split_rejects_log_leakage():
    from tools.worldstream_m1.offline_identity_scoring import log_risk
    with pytest.raises(ValueError,match='overlap'):
        log_risk([],['shared-log'],['shared-log'])


def ledger_fixture(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,temporal_visibility=True,observation_diagnostics=True));rows=[]
    for t in (100.,100.5):
        f=make_frame(t);f.metadata.update(context().metadata_for('scene',f.frame_id))
        r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
        rows.append(dict(precheck=r.to_dict(),commit=copy.deepcopy(m.last_commit_diagnostics),committed_state=m.export_state_audit()))
    return rows


def test_independent_ledger_matches_prior_actual_versions(config,make_frame,make_prediction):
    from tools.worldstream_m1.scene_revalidate import validate_rows
    result=validate_rows(ledger_fixture(config,make_frame,make_prediction),context())
    assert result['checks']['historical_versions_verified']==1


@pytest.mark.parametrize('mutation',['historical_version','camera_version','current_without_commit','retained_version','log_context','future_source'])
def test_independent_ledger_detects_tampered_sources(config,make_frame,make_prediction,mutation):
    from tools.worldstream_m1.scene_revalidate import validate_rows
    rows=ledger_fixture(config,make_frame,make_prediction)
    p=rows[1]['precheck'];s=rows[1]['committed_state']
    if mutation=='historical_version':next(iter(p['anchor_provenance'].values()))['state_version']='tampered'
    elif mutation=='camera_version':next(iter(next(iter(p['diagnostics']['anchors'].values()))['cameras'].values()))['evidence']['source_state_version']='tampered'
    elif mutation=='current_without_commit':rows[1]['commit']['committed_anchor_ids']=[]
    elif mutation=='retained_version':
        a=copy.deepcopy(rows[0]['committed_state']['anchors'][0]);a['state_version']='tampered';s['anchors'].append(dict(a,anchor_id='uncomputed-retained'))
    elif mutation=='log_context':p['diagnostics']['execution_context']['log_source']['context_sha256']='tampered'
    else:next(iter(p['anchor_provenance'].values()))['source_frame_id']=p['frame_id']
    with pytest.raises(AssertionError):validate_rows(rows,context())


def test_visualization_does_not_draw_failed_lk_placeholders():
    from tools.worldstream_m1.identity_cases import measured_tracks
    record=dict(source_points=[[1,2],[3,4],[5,6]],current_points=[[1,2],[np.nan,9],[7,8]],correspondence_valid=[False,True,True])
    p,q=measured_tracks(record)
    assert p.tolist()==[[5.,6.]] and q.tolist()==[[7.,8.]]
