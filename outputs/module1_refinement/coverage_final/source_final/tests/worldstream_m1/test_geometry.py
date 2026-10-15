from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from worldstream.m1.geometry import (rigid_transform, inverse_transform, transform_points, box_geometry,
    predict_geometry, project_points, project_anchor, clip_region, valid_transform)


def test_roundtrip_and_full_box_roll():
    r = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    t = rigid_transform(r, [1, 2, 3])
    p = np.random.RandomState(0).randn(40, 3)
    np.testing.assert_allclose(transform_points(transform_points(p, t), inverse_transform(t)), p, atol=1e-12)
    g = box_geometry([2, 3, 4], [2, 4, 6], .4, t)
    assert g.points.shape == (8, 3)
    np.testing.assert_allclose(g.points.mean(0), g.center)


@pytest.mark.parametrize('bad', [None, np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan),
                               np.diag([1, 1, -1, 1]), np.full((4, 4), np.inf)])
def test_invalid_transform(bad):
    assert not valid_transform(bad)
    with pytest.raises(ValueError): inverse_transform(bad)


def test_depth_and_boundary(make_anchor, make_frame, config):
    uv, depth, good = project_points(np.array([[0, 0, 1], [0, 0, -1], [20, 0, 1]]), np.eye(3), np.eye(4))
    assert good.tolist() == [True, False, True]
    assert np.isnan(uv[1]).all()
    assert clip_region((-20, -3, 150, 101), (100, 120)) == (0., 0., 120., 100.)
    anchor = make_anchor(x=200.)
    p = project_anchor(anchor, make_frame().cameras['front'], config)
    assert not p.valid and p.coverage == 0
    anchor.geometry_world = box_geometry([0, 0, 0], [2, 2, 2])
    assert project_anchor(anchor, make_frame().cameras['front'], config).reason == 'incomplete_depth'


def test_motion_and_camera_time(make_anchor, make_frame, config):
    static = make_anchor()
    np.testing.assert_allclose(predict_geometry(static, 101).center, [0, 0, 10])
    dynamic = make_anchor(velocity=np.array([2., 0., 0.]))
    np.testing.assert_allclose(predict_geometry(dynamic, 100.5).center, [1, 0, 10])
    frame = make_frame(100.5)
    a = project_anchor(dynamic, frame.cameras['front'], config)
    frame.cameras['front'].timestamp = 101.
    b = project_anchor(dynamic, frame.cameras['front'], config)
    assert b.region[0] > a.region[0]
    frame.cameras['front'].timing_reliable = False
    assert not project_anchor(dynamic, frame.cameras['front'], config).valid


def test_uncertainty_is_not_silently_clamped(make_anchor, make_frame, config):
    anchor = make_anchor(); anchor.geometry_uncertainty = 10
    assert project_anchor(anchor, make_frame().cameras['front'], config).reason == 'uncertainty_exceeds_search'


def test_local_real_lidar2img_crosscheck():
    path = Path('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    if not path.is_file(): pytest.skip('local nuScenes data unavailable')
    from tools.worldstream_m1.dataset import select_infos
    from tools.worldstream_m1.inspect_pipeline import projection_crosscheck
    errors = projection_crosscheck(select_infos(path, limit=1)[0])
    assert len(errors) == 6
    assert max(e for e in errors.values() if e is not None) < 1e-5


def test_preprocessing_projection_uses_actual_transform():
    # Invoke the actual project's image transform method, then numerically check K.
    import ast
    import torch
    from PIL import Image
    source = Path('projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py')
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'ResizeCropFlipRotImage')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('_img_transform', '_get_rot')]
    cls.bases = []; cls.decorator_list = []; cls.body = methods
    ns = {'torch': torch, 'np': np, 'Image': Image}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), ns)
    obj = ns['ResizeCropFlipRotImage']()
    _, ida = obj._img_transform(Image.new('RGB', (160, 90)), .5, (80, 45), (4, 5, 74, 40), True, 8.)
    k = np.array([[100., 0, 80], [0, 100, 45], [0, 0, 1]])
    points = np.array([[1., 2., 10.], [-2., 1., 8.]])
    original, _, _ = project_points(points, k, np.eye(4))
    transformed, _, _ = project_points(points, ida.numpy() @ k, np.eye(4))
    expected = np.c_[original, np.ones(2)] @ ida.numpy().T
    np.testing.assert_allclose(transformed, expected[:, :2], atol=1e-5)
