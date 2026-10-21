"""Fault injection and adversarial state/observation boundary checks."""
import builtins
import copy
from dataclasses import replace
import pickle
import numpy as np
import pytest
from worldstream.m1 import FullOutputs, WorldStateManager, ComputedFeature
from worldstream.m1.association import legal_assignment, association_ambiguities
from worldstream.m1.geometry import box_geometry, oriented_box_membership, region_slices
from worldstream.m1.mask_codec import decode_mask
from worldstream.m1.novelty_detector import unexplained_regions
from worldstream.m1.temporal_visibility import build_evidence
from test_refinement import pair


def snapshot(manager):
    return pickle.dumps(manager.store.__dict__)


def test_pending_verification_and_expiry_are_private(config, make_frame, make_prediction):
    m = WorldStateManager(replace(config, max_state_age_s=.7))
    m.initialize(make_frame(), FullOutputs([make_prediction()]))
    before = snapshot(m)
    r = m.precheck(make_frame(100.5))
    assert r.valid_anchor_ids == r.observation_supported_anchor_ids
    assert r.association_candidate_ids and not r.physically_certified_anchor_ids
    assert snapshot(m) == before
    m.abort_pending()
    assert snapshot(m) == before
    r = m.precheck(make_frame(100.9))
    assert r.invalid_anchor_ids and r.diagnostics['expired_anchor_ids']
    assert snapshot(m) == before
    m.commit(make_frame(100.9), FullOutputs([]), r)
    assert not m.store.anchors and not m.store.features


def test_commit_late_failure_rolls_back_and_consumes_pending(config, make_frame, make_prediction, monkeypatch):
    from worldstream.m1.state_store import StateStore
    m = WorldStateManager(replace(config, temporal_visibility=True))
    m.initialize(make_frame(), FullOutputs([make_prediction()]))
    before = snapshot(m); source = m._previous_cameras['front'].image.copy()
    r = m.precheck(make_frame(100.5)); original = StateStore.put_feature
    calls = []
    def fail_second(store, feature, frame):
        calls.append(feature.feature_id)
        if len(calls) == 2:
            raise RuntimeError('injected after first feature write')
        return original(store, feature, frame)
    monkeypatch.setattr(StateStore, 'put_feature', fail_second)
    feature = ComputedFeature('late', 'carrier', (1,), 'float32', 'cpu', 'branch')
    with pytest.raises(RuntimeError):
        m.commit(make_frame(100.5), FullOutputs([make_prediction()], [feature]), r)
    assert len(calls) == 2 and snapshot(m) == before
    assert np.array_equal(source, m._previous_cameras['front'].image)
    with pytest.raises(ValueError):
        m.commit(make_frame(100.5), FullOutputs([]), r)
    monkeypatch.setattr(StateStore, 'put_feature', original)
    r = m.precheck(make_frame(101.))
    assert 'aborted_frame_recovery' in r.diagnostics['full_refresh_reasons']
    assert m.last_masks['front'].all()
    m.commit(make_frame(101.), FullOutputs([make_prediction()]), r)
    assert m.store.last_timestamp == 101.


def test_precheck_failure_preserves_state(config, make_frame, make_prediction, monkeypatch):
    m = WorldStateManager(config); m.initialize(make_frame(), FullOutputs([make_prediction()]))
    before = snapshot(m)
    monkeypatch.setattr(m.validator, 'validate', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('fault')))
    with pytest.raises(RuntimeError): m.precheck(make_frame(100.5))
    assert snapshot(m) == before and m.last_result is None and m._pending_store is None


def test_partial_writes_are_never_public(config, make_frame, make_prediction, monkeypatch):
    from worldstream.m1.state_store import StateStore
    m = WorldStateManager(config); m.initialize(make_frame(), FullOutputs([make_prediction()]))
    original_store = m.store; before = snapshot(m)
    validate = m.validator.validate
    def inspect_validation(*a, **kw):
        assert m.store is original_store and snapshot(m) == before
        return validate(*a, **kw)
    monkeypatch.setattr(m.validator, 'validate', inspect_validation)
    f = make_frame(100.5); r = m.precheck(f)
    put = StateStore.put_feature
    def inspect_write(staged, feature, frame):
        result = put(staged, feature, frame)
        assert staged is not m.store and m.store is original_store and snapshot(m) == before
        return result
    monkeypatch.setattr(StateStore, 'put_feature', inspect_write)
    m.commit(f, FullOutputs([make_prediction()]), r)
    assert m.store is not original_store and m.store.last_timestamp == 100.5


@pytest.mark.parametrize('mutation', ['image', 'pose', 'intrinsic', 'camera_time', 'timeout', 'epoch'])
def test_commit_is_bound_to_exact_verified_frame(config, make_frame, mutation):
    m = WorldStateManager(config); f = make_frame(); r = m.precheck(f)
    if mutation == 'image': f.cameras['front'].image[0, 0] = 0
    if mutation == 'pose': f.ego_to_world[0, 3] = 1
    if mutation == 'intrinsic': f.cameras['front'].intrinsic[0, 0] += 1
    if mutation == 'camera_time': f.cameras['front'].timestamp += .01
    if mutation == 'timeout': m._pending_started -= config.pending_timeout_s + 1
    if mutation == 'epoch': m.store.generation += 1
    before = snapshot(m)
    with pytest.raises(ValueError): m.commit(f, FullOutputs([]), r)
    if mutation == 'epoch':
        assert not m.store.anchors and r.diagnostics['source_state_discarded']
    else:
        assert snapshot(m) == before
    assert m._pending_store is None


def test_camera_reorder_and_missing_reset(config, make_frame, make_prediction):
    f = make_frame(); side = copy.deepcopy(f.cameras['front']); side.camera_id = 'side'
    f.cameras['side'] = side
    m = WorldStateManager(config); m.initialize(f, FullOutputs([make_prediction()]))
    g = copy.deepcopy(f); g.timestamp += .5; g.frame_id = 'next'
    g.cameras = dict(reversed(list(g.cameras.items())))
    r = m.precheck(g)
    assert r.diagnostics['reset_reason'] == 'camera_layout_changed'
    assert not m.store.anchors and r.full_refresh_hint
    m.commit(g, FullOutputs([]), r)
    h = make_frame(101.)
    assert m.precheck(h).diagnostics['reset_reason'] == 'camera_layout_changed'


def test_same_rgb_background_cannot_certify_object(config, make_frame, make_prediction):
    m = WorldStateManager(config)
    m.initialize(make_frame(), FullOutputs([make_prediction()]))
    r = m.precheck(make_frame(100.5))
    assert r.valid_anchor_ids
    assert not r.physically_certified_anchor_ids
    proof = next(iter(r.diagnostics['anchors'].values()))['physical_certification']
    assert 'independent_foreground_ownership_missing' in proof['rejection_reasons']
    assert not proof['foreground_ownership_certified']


def test_ambiguity_crossing_and_missing_frames_break_candidate_track(config, make_frame, make_prediction):
    m = WorldStateManager(config)
    m.initialize(make_frame(), FullOutputs([make_prediction(-1), make_prediction(1)]))
    original = set(m.store.anchors)
    f = make_frame(100.5); r = m.precheck(f)
    m.commit(f, FullOutputs([make_prediction(0), make_prediction(0)]), r)
    assert m.last_commit_diagnostics['association_tracks']['ambiguous_edges'] == 4
    assert not m.last_commit_diagnostics['association_pairs']
    assert original.isdisjoint(m.last_commit_diagnostics['committed_anchor_ids'])
    f = make_frame(101.); r = m.precheck(f); m.commit(f, FullOutputs([]), r)
    f = make_frame(101.5); r = m.precheck(f)
    m.commit(f, FullOutputs([make_prediction(0)]), r)
    assert not m.last_commit_diagnostics['association_pairs']


def test_capacity_selection_before_assignment_and_order_invariance(config, make_frame, make_prediction):
    states = []
    for reverse in (False, True):
        m = WorldStateManager(replace(config, max_anchors=1))
        m.initialize(make_frame(), FullOutputs([make_prediction()]))
        aid = next(iter(m.store.anchors))
        low, high = make_prediction(.01), make_prediction(.4)
        low.source_score, high.source_score = .4, .9
        preds = [low, high][::(-1 if reverse else 1)]
        f = make_frame(100.5); r = m.precheck(f); m.commit(f, FullOutputs(preds), r)
        assert m.last_commit_diagnostics['excluded_prediction_count'] == 1
        assert m.last_commit_diagnostics['association_pairs'][0][0] == aid
        assert m.store.anchors[aid].association_count == 1
        assert m.store.anchors[aid].certified_observation_count == 0
        states.append(m.store.anchors[aid].geometry_world.center.copy())
    assert np.array_equal(*states) and states[0][0] == .4


def test_scipy_missing_is_explicit_failure(monkeypatch):
    original = builtins.__import__
    def without_scipy(name, *args, **kwargs):
        if name == 'scipy.optimize': raise ImportError('SciPy unavailable')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_scipy)
    with pytest.raises(ImportError): legal_assignment([[1., 1.9], [1.5, np.inf]])


def test_margin_gate_is_bidirectional():
    costs = np.array([[.1, .15], [.11, 1.]])
    checks = association_ambiguities([(0, 0)], costs, .02)
    assert checks[0, 0]['ambiguous']


def test_oriented_membership_rejects_rotated_envelope_background():
    geom = box_geometry([0, 0, 10], [4, 1, 2], np.pi / 4)
    xyz = np.array([[0, 0, 10], [1.6, -1.6, 10], [1, 1, 10]])
    assert ((xyz >= geom.points.min(0)) & (xyz <= geom.points.max(0))).all()
    assert oriented_box_membership(xyz, geom).tolist() == [True, False, True]
    geom.points = geom.points[[7, 5, 3, 1, 0, 2, 4, 6]]
    assert oriented_box_membership(xyz, geom).tolist() == [True, False, True]
    geom.points[0] += .5
    assert not oriented_box_membership(xyz, geom).any()


@pytest.mark.parametrize('force', [False, True])
def test_four_masks_partition_and_tile_envelope(force, config):
    a, b = pair()
    cfg = replace(config, temporal_visibility=True, local_support=True)
    e = build_evidence(a, b, cfg)
    # A localized residual at an otherwise supported correspondence is a change.
    e.changed_mask[30, 30] = True
    meta, invalid = unexplained_regions(b, [], [], cfg, force, e)
    masks = [decode_mask(meta[key]) for key in ('observation_stable_mask', 'observed_changed_mask', 'unknown_mask')]
    assert np.all(sum(mask.astype(int) for mask in masks) == 1)
    assert np.array_equal(decode_mask(meta['refresh_required_mask']), invalid)
    assert meta['invalidation_area_ratio'] == invalid.mean()
    assert meta['verified_stable_area_ratio'] == masks[0].mean() > 0
    assert np.array_equal(invalid, np.ones_like(invalid) if force else masks[1] | masks[2])
    envelope = np.zeros_like(invalid)
    for r in meta['regions']: envelope[region_slices(r, invalid.shape)] = True
    assert (envelope | ~invalid).all()


@pytest.mark.parametrize('key', ['gt_mask', 'oracle', 'instance_token', 'annotations'])
def test_online_metadata_rejects_offline_labels(config, make_frame, key):
    m = WorldStateManager(config); f = make_frame(); f.metadata = {'nested': {key: 'forbidden'}}
    with pytest.raises(ValueError, match='oracle'): m.precheck(f)
    assert not m.store.anchors and m._pending_store is None


def test_commit_rejects_oracle_origin(config, make_frame):
    m = WorldStateManager(config); f = make_frame(); r = m.precheck(f)
    with pytest.raises(ValueError): m.commit(f, FullOutputs([], source='offline_gt'), r)


def test_history_length_and_tensor_budget(config, make_frame, make_prediction):
    cfg = replace(config, association_history_limit=3, max_tensor_bytes=32, capture_tensors=True)
    m = WorldStateManager(cfg)
    for i in range(15):
        f = make_frame(100 + .1 * i); r = m.precheck(f)
        m.commit(f, FullOutputs([make_prediction()]), r)
    a = next(iter(m.store.anchors.values()))
    assert len(a.association_history) == 3 and a.association_count == 14
    assert a.track_birth_timestamp == 100 and a.last_actual_observation_timestamp == 101.4
    assert a.certified_observation_count == 0 and a.last_certified_timestamp is None
    assert m.store.tensor_bytes <= 32


def test_resize_crop_pad_and_multi_camera_mask_spaces(config, make_frame):
    from worldstream.m1.geometry import project_points
    f = make_frame(); raw = f.cameras['front']
    cam = copy.deepcopy(raw)
    # Crop x=12,y=8, resize by 2, then pad left=3,top=5.
    affine = np.array([[2., 0, -21], [0, 2., -11], [0, 0, 1.]])
    cam.camera_id = 'processed'; cam.image = np.zeros((184, 219, 3), np.uint8)
    cam.coordinate_space = 'processed_crop_resize_pad'; cam.raw_to_image = affine
    cam.intrinsic = affine @ raw.intrinsic
    cam.visibility_mask = None; cam.geometry_support_mask = None
    point = np.array([[1., 1., 10.]])
    uv, _, _ = project_points(point, raw.intrinsic, raw.world_to_camera)
    processed, _, _ = project_points(point, cam.intrinsic, cam.world_to_camera)
    assert np.allclose(processed, (np.c_[uv, np.ones(1)] @ affine.T)[:, :2])
    f.cameras['processed'] = cam
    m = WorldStateManager(config); r = m.precheck(f)
    for cid, camera in f.cameras.items():
        meta = r.invalidation_regions[cid]
        assert decode_mask(meta['unknown_mask']).shape == camera.image.shape[:2]
        assert decode_mask(meta['unknown_mask']).all()
        assert not decode_mask(meta['observed_changed_mask']).any()
        assert meta['coordinate_space'] == camera.coordinate_space
    assert r.invalidation_regions['processed']['raw_to_image'] == affine.tolist()
