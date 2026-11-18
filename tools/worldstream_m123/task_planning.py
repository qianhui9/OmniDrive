"""Existing vehicle-collision kernel plus explicit drivable-footprint checks."""
import numpy as np


def footprint_pixels(point, heading, metric):
    """Raster XY ego rectangle using the original collision kernel's dimensions."""
    import cv2
    offset = .5+.985793
    corners = np.asarray([[-metric.H/2+offset, metric.W/2],
                          [metric.H/2+offset, metric.W/2],
                          [metric.H/2+offset, -metric.W/2],
                          [-metric.H/2+offset, -metric.W/2]])
    c, s = np.cos(heading), np.sin(heading)
    xy = corners @ np.asarray([[c, s], [-s, c]]) + np.asarray(point)
    poly = np.round((xy-metric.bev_start_position[:2]+metric.bev_resolution[:2]/2)
                    / metric.bev_resolution[:2]).astype(np.int32)
    width, height = map(int, metric.bev_dimension[:2])
    outside = bool((poly < 0).any() or (poly[:, 0] >= width).any() or (poly[:, 1] >= height).any())
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return mask.astype(bool), outside


def drivable_violations(pred, drivable, metric):
    initial, initial_outside = footprint_pixels((0., 0.), 0., metric)
    initial_offroad = initial_outside or bool(np.any(initial & ~np.asarray(drivable, dtype=bool)))
    boundaries, outside_grid = [], []
    previous = np.zeros(2)
    for point in pred:
        delta = point-previous
        heading = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) >= 1 else 0.
        footprint, outside = footprint_pixels(point, heading, metric)
        boundaries.append(outside or bool(np.any(footprint & ~np.asarray(drivable, dtype=bool))))
        outside_grid.append(outside)
        previous = point
    return dict(road_boundary=boundaries, initially_offroad=initial_offroad, out_of_grid=outside_grid)


class PlanningBackend:
    def __init__(self, data_root):
        from evaluation.planning_utils import PlanningMetric
        self.metric = PlanningMetric(str(data_root))

    def __call__(self, info, pred, gt):
        import torch
        from pyquaternion import Quaternion
        metric = self.metric
        boxes = np.concatenate([info['gt_boxes'], info['gt_velocity']], axis=-1)
        features = np.concatenate([
            info['gt_fut_traj'][:, :6].reshape(-1, 12), info['gt_fut_traj_mask'][:, :6],
            info['gt_fut_yaw'][:, :6], info['gt_fut_idx']], axis=-1)
        bev = metric.get_birds_eye_view_label(boxes, features, add_rec=True)
        n = len(pred)
        occupancy = torch.from_numpy(bev[1:n+1]).unsqueeze(0)
        prediction = torch.as_tensor(pred.copy(), dtype=torch.float32).unsqueeze(0)
        reference = torch.as_tensor(gt.copy(), dtype=torch.float32).unsqueeze(0)
        _, collision = metric.evaluate_coll(prediction, reference, occupancy)
        raw = metric.evaluate_single_coll(prediction[0], occupancy[0])
        drivable = metric.get_drivable_area(info['ego2global_translation'],
                    Quaternion(info['ego2global_rotation']).rotation_matrix, info)
        road = drivable_violations(pred, drivable, metric)
        return dict(collision=collision.bool().tolist(), raw_collision=raw.bool().tolist(), **road)
