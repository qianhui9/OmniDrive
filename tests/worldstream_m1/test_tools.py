import json
import numpy as np
import pytest
from tools.worldstream_m1.calibrate import calibrate
from tools.worldstream_m1.io_utils import iter_json_array, read_prediction_subset
from worldstream.m1 import M1Config


def test_streaming_json_readers(tmp_path):
    path = tmp_path/'annotations.json'
    values = [{'id': i, 'text': '驾驶'*10000} for i in range(60)]
    path.write_text(json.dumps(values))
    assert list(iter_json_array(path)) == values
    path.write_text(json.dumps({'meta': {'camera': True}, 'results': {'a': values, 'b': [], 'c': values[:2]}}))
    assert read_prediction_subset(path, {'c'}) == {'c': values[:2]}


def test_calibration_risk_and_insufficient_support():
    rows = [{'residual': .1, 'oracle_valid': True, 'geometry_only': True} for _ in range(200)]
    rows += [{'residual': .9, 'oracle_valid': False, 'geometry_only': True} for _ in range(200)]
    result = calibrate(rows)
    assert result['threshold'] == .1 and result['risk_upper_bound'] <= .05
    assert calibrate(rows[:2])['threshold'] is None


def test_configuration_rejects_unknown_or_unsafe_modes():
    with pytest.raises(ValueError): M1Config.load({'skip_backbone': True})
    with pytest.raises(ValueError): M1Config(execution_mode='reuse')
    with pytest.raises(ValueError): M1Config(min_coverage=0.)


def test_oracle_uses_annotation_reverse_links_instead_of_filtered_sample(tmp_path):
    from tools.worldstream_m1.dataset import load_oracle
    table = tmp_path/'v1.0-trainval'; table.mkdir()
    values = {
        'sample': [{'token': 's', 'anns': ['a']}],
        'category': [{'token': 'c', 'name': 'vehicle.car'}],
        'instance': [{'token': 'i', 'category_token': 'c'}, {'token': 'j', 'category_token': 'c'}],
        'sample_annotation': [dict(token=a, sample_token='s', instance_token=i, translation=[0, 0, 10],
             size=[2, 4, 1.5], rotation=[1, 0, 0, 0], num_lidar_pts=3, num_radar_pts=0, visibility_token='4')
             for a, i in [('a', 'i'), ('b', 'j')]],
    }
    for name, data in values.items(): (table/(name+'.json')).write_text(json.dumps(data))
    oracle = load_oracle([{'token': 's'}], tmp_path)
    assert {a['instance_id'] for a in oracle['s']} == {'i', 'j'}


def test_incompatible_calibration_protocol_cannot_certify(tmp_path, make_frame, make_prediction):
    from worldstream.m1 import WorldStateManager, FullOutputs
    artifact = tmp_path/'calibration.json'
    artifact.write_text(json.dumps(dict(split='calibration', threshold=.18, descriptor='normalized_gray',
        descriptor_size=16, observation_protocol={'coordinate_space': 'resized_raw_image', 'max_width': 640})))
    manager = WorldStateManager({'calibration_path': str(artifact)})
    manager.initialize(make_frame(), FullOutputs([make_prediction()]))
    result = manager.precheck(make_frame(100.5))
    assert result.diagnostics['reset_reason'] == 'calibration_protocol_incompatible'
    assert result.full_refresh_hint and not result.valid_anchor_ids
