from dataclasses import replace
import numpy as np
import pytest
from worldstream.m1 import M1Config, CameraFrame, Frame, FullOutputs, PerceptionAnchor, WorldAnchor
from worldstream.m1.geometry import box_geometry


@pytest.fixture
def config():
    return M1Config(max_state_age_s=2., max_frame_gap_s=2., min_region_area_px=16,
                    velocity_uncertainty_mps=0., full_refresh_interval=10)


@pytest.fixture
def make_frame():
    def factory(t=100., scene='scene', frame_id=None, support=True, change=False):
        rng = np.random.RandomState(9 if not change else 12)
        image = rng.randint(0, 256, (100, 120, 3), dtype=np.uint8)
        k = np.array([[60., 0, 60], [0, 60, 50], [0, 0, 1]])
        mask = np.ones((100, 120), bool) if support else None
        camera = CameraFrame('front', image, k, np.eye(4), t, visibility_mask=mask,
                             geometry_support_mask=mask)
        return Frame(scene, str(t) if frame_id is None else frame_id, t, np.eye(4), {'front': camera})
    return factory


@pytest.fixture
def make_prediction():
    def factory(x=0., semantic='barrier', velocity=None, kind='object'):
        geom = box_geometry([x, 0, 10.], [2., 2., 2.])
        return PerceptionAnchor(kind, semantic, geom, velocity, velocity is not None)
    return factory


@pytest.fixture
def make_anchor(make_prediction):
    def factory(x=0., semantic='barrier', velocity=None, kind='object'):
        p = make_prediction(x, semantic, velocity, kind)
        return WorldAnchor('a', 'scene', kind, semantic, p.geometry_world, 100.,
                           velocity_world=velocity, velocity_reliable=velocity is not None)
    return factory
