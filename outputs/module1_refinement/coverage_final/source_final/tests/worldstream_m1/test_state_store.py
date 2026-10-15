from dataclasses import replace
import numpy as np
import pytest
from worldstream.m1 import ComputedFeature, FullOutputs, WorldStateManager


def test_versions_only_advance_after_actual_compute(config, make_frame, make_prediction):
    manager = WorldStateManager(config)
    f0 = make_frame()
    manager.initialize(f0, FullOutputs([make_prediction()]))
    versions = manager.store.feature_versions()
    result = manager.precheck(make_frame(100.5))
    assert result.valid_feature_candidates
    for key, value in manager.store.feature_versions().items():
        assert value['computation_version'] == versions[key]['computation_version']
        assert value['computation_timestamp'] == f0.timestamp
        assert value['last_verified_timestamp'] == 100.5
    manager.commit(make_frame(100.5), FullOutputs([make_prediction()]), result)
    for key, value in manager.store.feature_versions().items():
        assert value['computation_version'] == 2
        assert value['computation_timestamp'] == 100.5
    assert next(iter(result.feature_versions.values()))['computation_timestamp'] == 100.


@pytest.mark.parametrize('change,reason', [('scene', 'scene_change'), ('backward', 'timestamp_reversal_or_repeat'),
                                         ('gap', 'long_frame_gap'), ('signature', 'execution_conditions_changed'),
                                         ('pose', 'invalid_timestamp_or_pose')])
def test_resets(config, make_frame, make_prediction, change, reason):
    manager = WorldStateManager(config); manager.initialize(make_frame(), FullOutputs([make_prediction()]))
    frame = make_frame(100.5)
    if change == 'scene': frame.scene_id = 'another'
    if change == 'backward': frame.timestamp = 99.
    if change == 'gap': frame.timestamp = 110.
    if change == 'signature': frame.execution_signature = 'new_weights'
    if change == 'pose': frame.ego_to_world[0, 0] = np.nan
    result = manager.precheck(frame)
    assert result.full_refresh_hint
    assert result.diagnostics['reset_reason'] == reason
    assert not manager.store.anchors and not manager.store.features


def test_same_anchor_id_and_branch_dependencies_unknown(config, make_frame, make_prediction):
    manager = WorldStateManager(config)
    feature = ComputedFeature('object_carrier', 'object_carrier', (1, 23, 37), 'float32', 'cpu', 'branch')
    manager.initialize(make_frame(), FullOutputs([make_prediction()], [feature]))
    aid = next(iter(manager.store.anchors))
    result = manager.precheck(make_frame(100.5))
    assert not any(r['feature_id'] == 'object_carrier' for r in result.valid_feature_candidates)
    manager.commit(make_frame(100.5), FullOutputs([make_prediction(.1)], [feature]), result)
    assert list(manager.store.anchors) == [aid]
    assert manager.store.provenance['object_carrier'].dependency_scope == 'unknown/conservative'


def test_owned_tensors_detach_budget_and_reset(config, make_frame):
    import torch
    from worldstream.m1.state_store import StateStore
    cfg = replace(config, capture_tensors=True, max_tensor_bytes=64)
    store = StateStore(cfg); tensor = torch.ones(8, requires_grad=True)
    feature = ComputedFeature('x', 'carrier', (8,), 'float32', 'cpu', 'branch', tensor)
    rec = store.put_feature(feature, make_frame())
    assert rec.tensor.grad_fn is None and not rec.tensor.requires_grad
    assert rec.tensor.data_ptr() != tensor.data_ptr()
    feature.feature_id = 'y'; store.put_feature(feature, make_frame())
    feature.feature_id = 'z'; assert store.put_feature(feature, make_frame()).tensor is None
    assert store.tensor_bytes == 64
    store.reset(); assert store.tensor_bytes == 0 and not store.features


def test_bounded_anchors_and_expiration(config, make_frame, make_prediction):
    cfg = replace(config, max_anchors=2)
    manager = WorldStateManager(cfg)
    manager.initialize(make_frame(), FullOutputs([make_prediction(x) for x in range(5)]))
    assert len(manager.store.anchors) <= 2
    manager.precheck(make_frame(101.9))
    manager.store.prune(105.)
    assert not manager.store.anchors and not manager.store.features


def test_stale_or_duplicate_commit_rejected(config, make_frame):
    manager = WorldStateManager(config); frame = make_frame()
    result = manager.precheck(frame); manager.commit(frame, FullOutputs([]), result)
    with pytest.raises(ValueError): manager.commit(frame, FullOutputs([]), result)


def test_export_isolated(config, make_frame):
    manager = WorldStateManager(config); manager.precheck(make_frame())
    exported = manager.export_m2_inputs(); exported.invalidation_regions.clear()
    assert manager.last_result.invalidation_regions


def test_precheck_timestamp_rollback_without_commit(config, make_frame, make_prediction):
    manager = WorldStateManager(config); manager.initialize(make_frame(), FullOutputs([make_prediction()]))
    manager.precheck(make_frame(101.))
    result = manager.precheck(make_frame(100.5))
    assert result.diagnostics['reset_reason'] == 'timestamp_reversal_or_repeat'


def test_old_epoch_ids_cannot_collide_after_same_scene_reset(config, make_frame, make_prediction):
    manager = WorldStateManager(config); manager.initialize(make_frame(), FullOutputs([make_prediction()]))
    before = set(manager.store.anchors)
    manager.initialize(make_frame(101.), FullOutputs([make_prediction()]))
    assert before.isdisjoint(manager.store.anchors)


def test_candidate_tensor_must_be_acquired_before_commit(config, make_frame, make_prediction):
    manager = WorldStateManager(config); manager.initialize(make_frame(), FullOutputs([make_prediction()]))
    result = manager.precheck(make_frame(100.5))
    candidate = result.valid_feature_candidates[0]
    record = manager.get_candidate_feature(candidate['feature_id'], candidate['computation_version'])
    assert record.computation_timestamp == 100.
    manager.commit(make_frame(100.5), FullOutputs([make_prediction()]), result)
    with pytest.raises(KeyError): manager.get_candidate_feature(candidate['feature_id'], candidate['computation_version'])


def test_feature_from_unobserved_camera_not_certified(config, make_frame, make_prediction):
    import copy
    frame = make_frame(); frame.cameras['side'] = copy.deepcopy(frame.cameras['front'])
    manager = WorldStateManager(config); manager.initialize(frame, FullOutputs([make_prediction()]))
    next_frame = make_frame(100.5)
    result = manager.precheck(next_frame)
    assert result.valid_anchor_ids
    assert all('side' not in c['feature_id'] for c in result.valid_feature_candidates)


def test_invalid_velocity_not_treated_as_motion_evidence(config, make_frame, make_prediction):
    p = make_prediction(semantic='car', velocity=np.full(3, np.nan))
    manager = WorldStateManager(config); manager.initialize(make_frame(), FullOutputs([p]))
    result = manager.precheck(make_frame(100.5))
    assert result.invalid_anchor_ids and not result.valid_anchor_ids


def test_replaced_branch_provenance_drops_unmatched_anchor_link(config, make_frame, make_prediction):
    manager = WorldStateManager(config)
    feature = ComputedFeature('object_carrier', 'object_carrier', (1, 4, 8), 'float32', 'cpu', 'branch')
    manager.initialize(make_frame(), FullOutputs([make_prediction(), make_prediction(8.)], [feature]))
    unmatched = list(manager.store.anchors)[1]
    assert 'object_carrier' in manager.store.anchors[unmatched].associated_feature_ids
    result = manager.precheck(make_frame(100.5))
    manager.commit(make_frame(100.5), FullOutputs([make_prediction()], [feature]), result)
    assert unmatched in manager.store.anchors
    assert 'object_carrier' not in manager.store.anchors[unmatched].associated_feature_ids
