from copy import deepcopy
import numpy as np
import pytest
import torch

from worldstream.m1 import CameraFrame, Frame, WorldAnchor
from worldstream.m1.config import M1Config
from worldstream.m1.geometry import box_geometry, project_anchor
from worldstream.m1.state_types import AnchorObservation
from worldstream.m3.visual_feature_bridge import inside_convex, projected_region_features


def fixture():
    cam = CameraFrame('front', np.zeros((96, 128, 3), np.uint8),
        np.array([[80., 0, 64], [0, 80., 48], [0, 0, 1]]), np.eye(4), 1.)
    frame = Frame('s', 'f', 1., np.eye(4), {'front': cam})
    anchor = WorldAnchor('a', 's', 'object', 'car', box_geometry([0, 0, 10], [6, 6, 4]), 1., None)
    anchor.source_frame_id = 'f'; anchor.state_version = 'v1'
    cfg = M1Config(uncertainty_px=0, min_region_area_px=1)
    p = project_anchor(anchor, cam, cfg)
    assert p.valid
    anchor.observation_support['front'] = AnchorObservation('front', p.region, np.ones(4), 1., 1.)
    features = torch.arange(64.).reshape(1, 1, 1, 8, 8)
    return frame, anchor, features, cfg


def test_native_grid_geometry_padding_and_owned_gather():
    frame, anchor, features, cfg = fixture()
    regions, rejected = projected_region_features(frame, [anchor], features, cfg, camera_ids=['front'])
    assert not rejected and len(regions) == 1
    r = regions[0]
    assert not r.foreground_certified and r.source_state_version == 'v1'
    # Image is unpadded height 96, features cover padded 128. Never stretch y.
    idx = np.asarray(r.grid_indices)
    assert ((idx // 8+.5)*16 < 96).all()
    assert torch.equal(r.features[:, 0], torch.as_tensor(idx).float())
    old = r.features.clone(); features.fill_(-1)
    assert torch.equal(old, r.features)


def test_current_competing_projection_cannot_become_object_feature():
    frame, anchor, features, cfg = fixture()
    other = deepcopy(anchor); other.anchor_id = 'b'
    regions, rejected = projected_region_features(frame, [anchor, other], features, cfg,
        camera_ids=['front'], selected_ids=['a'])
    assert not regions and rejected['a'] == 'insufficient_unambiguous_projected_grid_support'


def test_stale_or_asynchronous_observation_rejected():
    frame, anchor, features, cfg = fixture()
    anchor.source_frame_id = 'old'
    regions, rejected = projected_region_features(frame, [anchor], features, cfg, camera_ids=['front'])
    assert not regions and rejected['a'] == 'not_current_committed_object'
    anchor.source_frame_id = 'f'; frame.cameras['front'].timing_reliable = False
    assert not projected_region_features(frame, [anchor], features, cfg, camera_ids=['front'])[0]


def test_camera_order_and_patch_coverage_fail_closed():
    frame, anchor, features, cfg = fixture()
    with pytest.raises(ValueError, match='camera order'):
        projected_region_features(frame, [anchor], features, cfg, camera_ids=['rear'])
    with pytest.raises(ValueError, match='cover'):
        projected_region_features(frame, [anchor], features[..., :3, :3], cfg, camera_ids=['front'])


def test_hull_uses_projected_polygon_not_axis_aligned_box():
    polygon = np.array([[0, 2], [2, 0], [4, 2], [2, 4]])
    points = np.array([[2, 2], [0, 0], [2, 0]])
    assert inside_convex(points, polygon).tolist() == [True, False, True]
    assert np.array_equal(inside_convex(points, polygon), inside_convex(points, polygon[::-1]))
