from dataclasses import replace
import copy
import json
import numpy as np
import pytest
from worldstream.m1 import WorldStateManager,FullOutputs
from worldstream.m1.dqtrack_policy import propose,recent_metadata,qualified_profile
from tools.worldstream_m1.dqtrack_policy_eval import strict_labels,scene_bootstrap


PROFILE=dict(min_score=.5,min_observations=3,max_hint_age_s=1.)


def hint(identity,t=100.):
    return dict(namespace=['run','scene'],tracking_id=identity,timestamp=t,status='aligned',
                tracking_score=.9,aligned_observations=3,expires_at=t+1)


def setup(make_anchor,make_prediction):
    anchors=[make_anchor(x=-.5),make_anchor(x=.5)]
    for i,a in enumerate(anchors):a.anchor_id=str(i);a.external_tracking=hint(str(i))
    predictions=[make_prediction(x=-1.5),make_prediction(x=1.5)]
    hints={0:hint('1',100.5),1:hint('0',100.5)}
    return anchors,predictions,hints,np.array([[1.,2.],[2.,1.]])


def test_identity_breaks_legal_tie_without_changing_costs(config,make_anchor,make_prediction,make_frame):
    a,p,h,c=setup(make_anchor,make_prediction);original=c.copy()
    pairs,changed,d=propose(a,p,[(0,0),(1,1)],c,h,make_frame(t=100.5),100.,config,PROFILE)
    assert set(pairs)=={(0,1),(1,0)} and len(changed)==2
    assert np.array_equal(original,c)
    assert not any(old.identity_link_qualified for old in a)


@pytest.mark.parametrize('failure',['illegal','newborn','low','stale','scene','missing','duplicate','unknown_competitor','reappearance'])
def test_unreliable_or_illegal_components_not_rescued(failure,config,make_anchor,make_prediction,make_frame):
    a,p,h,c=setup(make_anchor,make_prediction);frame=make_frame(t=100.5)
    if failure=='illegal':c[0,1]=np.inf
    if failure=='newborn':a[0].external_tracking['aligned_observations']=1
    if failure=='low':h[0]['tracking_score']=.1
    if failure=='stale':a[0].external_tracking['expires_at']=100.1
    if failure=='scene':frame.scene_id='other'
    if failure=='missing':h.pop(0)
    if failure=='duplicate':a[1].external_tracking=copy.deepcopy(a[0].external_tracking)
    if failure=='unknown_competitor':a[1].external_tracking={}
    if failure=='reappearance':a[0].source_timestamp=99.
    pairs,changed,d=propose(a,p,[],c,h,frame,100.,config,PROFILE)
    assert not changed and not pairs


def test_recent_metadata_not_current_evidence(make_anchor):
    a=make_anchor();a.external_tracking=hint('x')
    recent=recent_metadata(a,None,100.5,PROFILE)
    assert recent['status']=='recent_only' and recent['timestamp']==100.
    assert not recent_metadata(a,None,101.1,PROFILE)
    a.external_tracking=recent
    assert recent_metadata(a,hint('x',100.6),100.6,PROFILE)['aligned_observations']==1


def test_qualification_missing_or_forged_fails_closed(tmp_path,config):
    assert qualified_profile(config)[0] is None
    p=tmp_path/'false.json';p.write_text('{}')
    c=replace(config,dqtrack_policy_qualification_path=str(p),dqtrack_policy_qualification_sha256='bad')
    assert qualified_profile(c)[0] is None


def test_unsupported_policy_and_pin_rejected(config):
    with pytest.raises(ValueError):replace(config,dqtrack_association_policy='relax')
    with pytest.raises(ValueError):replace(config,dqtrack_policy_qualification_path='unbound')
    assert config.dqtrack_association_policy=='conservative'


def test_no_qualification_matches_conservative(config,make_frame,make_prediction):
    # Missing tracks and missing qualification both preserve ordinary inference.
    managers=[WorldStateManager(replace(config,dqtrack_mode='hint',dqtrack_association_policy=policy))
              for policy in ('conservative','calibrated_tiebreak')]
    for m in managers:
        for t in (100.,100.5):
            f=make_frame(t=t);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction()]),r)
    assert managers[0].store.source_digest()==managers[1].store.source_digest()
    assert managers[0].last_commit_diagnostics['association_pairs']==managers[1].last_commit_diagnostics['association_pairs']


def test_physical_policy_never_changed(config,make_anchor,make_prediction,make_frame):
    a,p,h,c=setup(make_anchor,make_prediction)
    cfg=replace(config,foreground_correspondence=True,temporal_visibility=True)
    pairs,changed,d=propose(a,p,[(0,0),(1,1)],c,h,make_frame(t=100.5),100.,cfg,PROFILE)
    assert pairs==[(0,0),(1,1)] and not changed and d['status']=='inactive'


def test_strict_oracle_unknown_competitors_excluded(make_prediction):
    p=make_prediction()
    ann=dict(instance_id='GT',**{'class':p.semantic_class},geometry=p.geometry_world,has_sensor_support=True)
    assert strict_labels([p],[ann])=={0:'GT'}
    assert strict_labels([p,p],[ann])=={}


def test_scene_bootstrap_is_scene_clustered():
    r=scene_bootstrap([dict(metrics=dict(scored=10,wrong=0)),dict(metrics=dict(scored=1,wrong=1))])
    assert r['unit']=='scene' and r['n_scenes']==2 and not r['independent_validation']


def test_transaction_tiebreak_is_uncertified_and_commit_only(tmp_path,config,make_frame,make_prediction):
    from tools.worldstream_m1.dqtrack_policy_eval import ResearchManager
    frames={};results={}
    for step,t in enumerate((100.,100.5,101.,101.5)):
        token=str(t);frames[token]=dict(scene_id='scene',sample_token=token,timestamp=t)
        centers=[[-1.4,0,10],[1.4,0,10]] if step<3 else [[0,1.4,10],[0,-1.4,10]]
        results[token]=[dict(sample_token=token,tracking_id=str(i),tracking_name='barrier',
            tracking_score=.9,translation=center,size=[2,2,2],rotation=[1,0,0,0]) for i,center in enumerate(centers)]
    path=tmp_path/'tracks.json';path.write_text(json.dumps(dict(results=results,m1_manifest=dict(
        coordinate_system='world',size_order='wlh',timestamp_unit='seconds',run_id='run',model_config='synthetic-test',frames=frames))))
    cfg=replace(config,dqtrack_mode='hint',dqtrack_results_path=str(path),dqtrack_association_policy='calibrated_tiebreak')
    research=ResearchManager(cfg);research.research_profile=PROFILE
    baseline=WorldStateManager(replace(cfg,dqtrack_association_policy='conservative'))
    for manager in (research,baseline):
        for step,t in enumerate((100.,100.5,101.,101.5)):
            f=make_frame(t=t);r=manager.precheck(f)
            predictions=[make_prediction(x=-1.4),make_prediction(x=1.4)]
            if step==3:
                from worldstream.m1.geometry import box_geometry
                predictions[0].geometry_world=box_geometry([0,1.4,10],[2,2,2])
                predictions[1].geometry_world=box_geometry([0,-1.4,10],[2,2,2])
            manager.commit(f,FullOutputs(predictions),r)
    assert len(research.last_commit_diagnostics['association_pairs'])==2
    assert not baseline.last_commit_diagnostics['association_pairs']
    current=[a for a in research.store.anchors.values() if a.source_timestamp==101.5]
    assert all(a.association_basis=='uncertified_dqtrack_identity_tiebreak' for a in current)
    assert all(not a.identity_link_qualified and a.certified_observation_count==0 and not a.velocity_reliable for a in current)
    assert not research.get_current_certifications().certified_anchor_ids


def test_recent_record_cannot_corroborate_or_veto(make_anchor):
    from worldstream.m1.dqtrack_bridge import identity_conflicts,corroborated_edges
    a=make_anchor();a.external_tracking=dict(hint('old'),status='recent_only')
    assert not identity_conflicts([a],{0:hint('different',100.5)},[(0,0)])
    assert not corroborated_edges([a],{0:hint('old',100.5)},[(0,0)],np.zeros((1,1)))


def test_oriented_3d_iou_is_diagnostic_not_gate():
    from worldstream.m1.dqtrack_geometry import geometry_features
    from worldstream.m1.geometry import box_geometry
    a=box_geometry([0,0,0],[2,2,2]);b=box_geometry([1,0,0],[2,2,2])
    assert geometry_features(a,b)['oriented_iou_3d']==pytest.approx(1/3)
    assert geometry_features(a,box_geometry([5,0,0],[2,2,2]))['oriented_iou_3d']==0
    rotated=box_geometry([0,0,0],[2,2,2],yaw=np.pi/4)
    assert 0<geometry_features(a,rotated)['oriented_iou_3d']<1


def test_scene_prefix_validation():
    from tools.worldstream_m1.dqtrack_scene_protocol import validate_sequence
    infos=[dict(token='a',scene_token='s',timestamp=1),dict(token='b',scene_token='s',timestamp=2)]
    samples=[dict(infos[0],prev=''),dict(infos[1],prev='a')]
    spec=dict(scene_id='s',samples=['a','b'])
    assert validate_sequence(infos,spec,samples)==infos
    with pytest.raises(ValueError):validate_sequence(infos,dict(scene_id='s',samples=['b']),samples)
    with pytest.raises(ValueError):validate_sequence(infos,spec,[samples[0],dict(samples[1],timestamp=3)])
    with pytest.raises(ValueError):validate_sequence(infos,spec,[samples[0],dict(samples[1],prev='other')])
    with pytest.raises(ValueError):validate_sequence(infos,dict(scene_id='s',samples=['a','a']),samples)


def test_subset_merge_namespaces_and_completeness(tmp_path):
    from tools.worldstream_m1.dqtrack_official_subset import merge_subset
    protocol=dict(development=[dict(scene_id='s',samples=['a'])])
    (tmp_path/'s').mkdir();path=tmp_path/'s/normalized_tracking.json'
    path.write_text(json.dumps(dict(results={'a':[dict(tracking_id='1')]})))
    assert merge_subset(protocol,tmp_path)['results']['a'][0]['tracking_id']=='s:1'
    path.write_text(json.dumps(dict(results={})))
    with pytest.raises(ValueError):merge_subset(protocol,tmp_path)


def test_recent_metadata_owns_nested_external_lists(make_anchor):
    a=make_anchor();a.external_tracking=hint('x')
    current=hint('x',100.5)
    new=recent_metadata(a,current,100.5,PROFILE)
    new['namespace'][0]='mutated'
    assert current['namespace'][0]=='run' and a.external_tracking['namespace'][0]=='run'
    retained=recent_metadata(a,None,100.5,PROFILE)
    retained['namespace'][0]='mutated'
    assert a.external_tracking['namespace'][0]=='run'
