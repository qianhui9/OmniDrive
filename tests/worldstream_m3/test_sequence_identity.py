import pytest
from tools.worldstream_m123.benchmark.sequence import token_index,resolve_sequence,validate_sequence


def info(token,t,scene='s',position=None):
    return dict(token=token,timestamp=int(t*1e6),scene_token=scene,
        ego2global_translation=position or [t,0,0],gt_names=[])


def test_shuffled_annotation_arrays_join_by_token_and_validate_actual_execution():
    raw=[info('b',2),info('a',1)]
    spec=dict(scene='s',indices=[0,1],tokens=['a','b'])
    executed={t:[dict(scene_id='s',timestamp=float(k))] for t,k in [('a',1),('b',2)]}
    selected,coverage=resolve_sequence(spec,token_index(raw),executed)
    assert [x['token'] for x in selected]==['a','b'] and coverage['duration_s']==1
    executed['a'][0]['scene_id']='other'
    with pytest.raises(ValueError,match='execution scene'):resolve_sequence(spec,token_index(raw),executed)


@pytest.mark.parametrize('rows',[[info('a',1),info('a',2)],[info('a',2),info('b',1)],
    [info('a',1),info('b',2,'other')],[info('a',1),info('b',2,position=[1000,0,0])]])
def test_corrupted_sequences_are_rejected(rows):
    with pytest.raises(ValueError):validate_sequence(rows,'s')


def test_no_integer_index_fallback_or_duplicate_token_join():
    with pytest.raises(ValueError):resolve_sequence(dict(scene='s',indices=[0]),token_index([info('a',1)]))
    with pytest.raises(ValueError):token_index([info('a',1),info('a',2)])


def test_rotation_sign_equivalence_and_angular_pose_jump():
    a=dict(info('a',1),ego2global_rotation=[1,0,0,0])
    b=dict(info('b',1.5),ego2global_rotation=[-1,0,0,0])
    assert validate_sequence([a,b],'s')['ego_angular_steps_rad']==[0.]
    b['ego2global_rotation']=[0,0,0,1]
    with pytest.raises(ValueError,match='rotation jump'):validate_sequence([a,b],'s')
    b['ego2global_rotation']=[0,0,0,0]
    with pytest.raises(ValueError,match='quaternion'):validate_sequence([a,b],'s')
