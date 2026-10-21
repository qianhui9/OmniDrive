"""Legacy offline scoring/source utilities; identity tiebreak is retired."""
from dataclasses import replace
import json
import numpy as np
import pytest
from worldstream.m1 import WorldStateManager
from tools.worldstream_m1.dqtrack_policy_eval import strict_labels,scene_bootstrap

def test_strict_oracle_unknown_competitors_excluded(make_prediction):
    p=make_prediction()
    ann=dict(instance_id='GT',**{'class':p.semantic_class},geometry=p.geometry_world,has_sensor_support=True)
    assert strict_labels([p],[ann])=={0:'GT'}
    assert strict_labels([p,p],[ann])=={}

def test_scene_bootstrap_is_scene_clustered():
    r=scene_bootstrap([dict(metrics=dict(scored=10,wrong=0)),dict(metrics=dict(scored=1,wrong=1))])
    assert r['unit']=='scene' and r['n_scenes']==2 and not r['independent_validation']

def test_transaction_tiebreak_retired(config):
    with pytest.raises(ValueError,match='Retired'):
        WorldStateManager(replace(config,dqtrack_mode='hint',dqtrack_association_policy='calibrated_tiebreak'))

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
