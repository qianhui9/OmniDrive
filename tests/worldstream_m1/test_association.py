import numpy as np
from worldstream.m1 import Geometry
from worldstream.m1.association import associate, legal_assignment


def test_class_space_and_one_to_one(make_anchor, make_prediction, config):
    anchors = [make_anchor(0), make_anchor(8)]
    predictions = [make_prediction(8.1), make_prediction(.1), make_prediction(0, 'car')]
    matches, _ = associate(anchors, predictions, 100.5, config)
    assert matches == [(0, 1), (1, 0)]
    assert associate(anchors, [make_prediction(50)], 100.5, config)[0] == []


def test_legal_assignment_masks_before_optimization():
    result = legal_assignment([[1., 1.9], [1.5, np.inf]])
    assert result == [(0, 1), (1, 0)]
    assert legal_assignment(np.empty((0, 4))) == []


def test_map_reversal_and_direction(make_anchor, make_prediction, config):
    a = make_anchor(kind='map'); p = make_prediction(kind='map')
    points = np.array([[0., 0, 1], [1, 0, 1], [2, 0, 1]])
    a.geometry_world = Geometry(points, points.mean(0))
    p.geometry_world = Geometry(points[::-1]+.01, points.mean(0)+.01)
    assert associate([a], [p], 100.5, config)[0] == [(0, 0)]
    perpendicular = points[:, [1, 0, 2]]
    p.geometry_world = Geometry(perpendicular, perpendicular.mean(0))
    assert associate([a], [p], 100.5, config)[0] == []


def test_ego_motion_compensation_preserves_world_identity(make_anchor, make_prediction, config):
    from worldstream.m1.geometry import box_geometry
    previous = make_anchor(x=0.)
    current = make_prediction()
    ego_to_world = np.eye(4); ego_to_world[0, 3] = 2.
    current.geometry_world = box_geometry([-2., 0, 10], [2., 2., 2.], to_world=ego_to_world)
    assert associate([previous], [current], 100.5, config)[0] == [(0, 0)]


def test_unmatched_option_has_no_cardinality_bonus():
    from worldstream.m1.association import unmatched_assignment
    costs=np.array([[.01,.9],[.9,np.inf]])
    assert unmatched_assignment(costs,[1.,1.])==[(0,0)]
