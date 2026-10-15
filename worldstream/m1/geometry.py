"""Column-vector SE(3), metres; projections use each camera's own timestamp."""
from dataclasses import dataclass
from itertools import product
from typing import Optional
import numpy as np
from .state_types import Geometry, Region


def valid_transform(matrix):
    if matrix is None:
        return False
    a = np.asarray(matrix)
    return (a.shape == (4, 4) and np.isfinite(a).all()
            and np.allclose(a[3], [0, 0, 0, 1], atol=1e-4)
            and np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-3)
            and abs(np.linalg.det(a[:3, :3]) - 1) < 1e-3)


def rigid_transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation)
    result[:3, 3] = np.asarray(translation)
    if not valid_transform(result):
        raise ValueError("Invalid rigid transform")
    return result


def inverse_transform(matrix):
    if not valid_transform(matrix):
        raise ValueError("Invalid rigid transform")
    r = np.asarray(matrix)[:3, :3]
    return rigid_transform(r.T, -r.T @ np.asarray(matrix)[:3, 3])


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3 or not valid_transform(matrix) or not np.isfinite(points).all():
        raise ValueError("Invalid points/transform")
    return points @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def box_geometry(center, size, yaw=0.0, to_world=None):
    """Gravity centre, dx/dy/dz; keep full 3D transformed corners (including roll/pitch)."""
    center, size = np.asarray(center), np.asarray(size)
    if (center.shape != (3,) or size.shape != (3,) or not np.isfinite(size).all()
            or np.any(size <= 0) or not np.isfinite(yaw)):
        raise ValueError("Invalid box")
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    points = np.array(list(product([-0.5, 0.5], repeat=3))) * size
    points = points @ rotation.T + center
    matrix = np.eye(4) if to_world is None else to_world
    return Geometry(transform_points(points, matrix), transform_points(center, matrix), size.copy())


def predict_geometry(anchor, timestamp):
    dt = timestamp - anchor.source_timestamp
    delta = np.zeros(3)
    if anchor.velocity_reliable and anchor.velocity_world is None:
        raise ValueError("Missing purported reliable velocity")
    if anchor.velocity_reliable and anchor.velocity_world is not None:
        velocity = np.asarray(anchor.velocity_world)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("Invalid velocity")
        delta = velocity * dt
    g = anchor.geometry_world
    return Geometry(g.points + delta, g.center + delta, g.size)


def clip_region(region, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = region
    # Scalar clipping avoids four NumPy dispatches for every crop in the search loop.
    return (float(min(max(x1, 0), w)), float(min(max(y1, 0), h)),
            float(min(max(x2, 0), w)), float(min(max(y2, 0), h)))


def region_area(region):
    return max(0., region[2] - region[0]) * max(0., region[3] - region[1])


def intersect_area(a, b):
    return region_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def region_slices(region, shape):
    r = clip_region(region, shape)
    return (slice(int(np.floor(r[1])), int(np.ceil(r[3]))),
            slice(int(np.floor(r[0])), int(np.ceil(r[2]))))


def valid_projection_camera(camera):
    k = np.asarray(camera.intrinsic)
    return (valid_transform(camera.world_to_camera) and k.shape == (3, 3)
            and np.isfinite(k).all() and abs(np.linalg.det(k)) >= 1e-9
            and np.allclose(k[2], [0, 0, 1], atol=1e-5))


def project_points(points_world, intrinsic, world_to_camera, min_depth=0.1, _camera_validated=False):
    k = np.asarray(intrinsic)
    if not _camera_validated and (k.shape != (3, 3) or not np.isfinite(k).all() or abs(np.linalg.det(k)) < 1e-9
            or not np.allclose(k[2], [0, 0, 1], atol=1e-5)):
        raise ValueError("Invalid intrinsic")
    if _camera_validated:
        points_world = np.asarray(points_world, dtype=np.float64)
        if points_world.shape[-1] != 3 or not np.isfinite(points_world).all():
            raise ValueError("Invalid points")
        matrix = np.asarray(world_to_camera)
        points = points_world @ matrix[:3, :3].T + matrix[:3, 3]
    else:
        points = transform_points(points_world, world_to_camera)
    depths = points[..., 2]
    good = depths > min_depth
    uvh = points @ k.T
    uv = np.full(uvh.shape[:-1] + (2,), np.nan)
    uv[good] = uvh[good, :2] / uvh[good, 2:3]
    return uv, depths, good


@dataclass
class Projection:
    region: Optional[Region]
    uv: np.ndarray
    depth: float
    coverage: float
    valid: bool
    reason: str
    search_radius_px: int = 0


def project_anchor(anchor, camera, config, _camera_validated=False):
    empty = Projection(None, np.empty((0, 2)), float("inf"), 0., False, "invalid_calibration")
    if not camera.timing_reliable:
        empty.reason = "asynchronous_pose_unknown"
        return empty
    dt = camera.timestamp - anchor.source_timestamp
    if not np.isfinite(dt) or abs(dt) > config.max_state_age_s:
        empty.reason = "outside_camera_horizon"
        return empty
    try:
        geom = predict_geometry(anchor, camera.timestamp)
        uv, depths, good = project_points(geom.points, camera.intrinsic, camera.world_to_camera,
                                         _camera_validated=_camera_validated)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return empty
    # Near-plane crossings are not certified from the remaining subset of corners.
    if not good.all() or len(uv) < 2:
        return Projection(None, uv, float(np.min(depths)), 0., False, "incomplete_depth")
    region = (float(uv[:, 0].min()), float(uv[:, 1].min()),
              float(uv[:, 0].max()), float(uv[:, 1].max()))
    clipped = clip_region(region, camera.image.shape)
    area = region_area(region)
    coverage = region_area(clipped) / max(area, 1e-12)
    uncertainty_m = anchor.geometry_uncertainty
    if anchor.anchor_type == "object" and anchor.semantic_class not in ("barrier", "traffic_cone"):
        uncertainty_m += config.velocity_uncertainty_mps * max(0., dt)
    radius = int(np.ceil(config.uncertainty_px + abs(camera.intrinsic[0, 0])
                         * uncertainty_m / max(float(depths.min()), 0.1)))
    valid = coverage >= config.min_coverage and region_area(clipped) >= config.min_region_area_px
    reason = "ok" if valid else ("insufficient_coverage" if coverage < config.min_coverage else "tiny_region")
    if radius > config.search_radius_px:
        valid, reason = False, "uncertainty_exceeds_search"
    return Projection(clipped, uv, float(depths.min()), coverage, valid, reason,
                      min(radius, config.search_radius_px))


def sample_polyline(points, count=32):
    points = np.asarray(points)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if len(points) < 2 or lengths[-1] < 1e-8:
        raise ValueError("Degenerate polyline")
    return np.stack([np.interp(np.linspace(0, lengths[-1], count), lengths, points[:, i])
                     for i in range(3)], axis=1)
