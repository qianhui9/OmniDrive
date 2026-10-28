"""Read-only, causal projection into the existing EVA output grid.

These are contextual image-region features, never detection-query embeddings or
certified foreground pixels. No GT, track ID encoding, interpolation, or encoder
is used. This diagnostic interface does not enable an M3 deployment policy.
"""
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
import numpy as np
import torch

from worldstream.m1.geometry import project_anchor


@dataclass
class RegionFeature:
    anchor_id: str
    semantic_class: str
    camera_id: str
    frame_id: str
    source_state_version: str
    region: tuple
    grid_indices: tuple
    features: torch.Tensor
    candidate_cells: int
    excluded_cells: int
    source_kind: str = 'EVA_output_contextual_projected_region'
    foreground_certified: bool = False
    # Lazy views are current-frame only; the manager owns detached history.
    feature_grid: torch.Tensor = None
    grid_width: int = 40

    def read_features(self):
        if self.features is not None:
            return self.features
        if self.feature_grid is None:
            raise ValueError('Missing real EVA feature source')
        indices = torch.as_tensor(self.grid_indices, device=self.feature_grid.device)
        return self.feature_grid.flatten(1).index_select(1, indices).T.contiguous()


@lru_cache(maxsize=8)
def grid_centres(height, width, stride):
    y, x = np.meshgrid((np.arange(height)+.5)*stride,
                       (np.arange(width)+.5)*stride, indexing='ij')
    points = np.stack((x.ravel(), y.ravel()), -1)
    points.setflags(write=False)
    return points


def inside_convex(points, polygon):
    """Half-plane test, including edges, independent of polygon winding."""
    import cv2
    hull = cv2.convexHull(np.asarray(polygon, np.float32)).reshape(-1, 2)
    if len(hull) < 3 or abs(cv2.contourArea(hull)) < 1e-6:
        return np.zeros(len(points), dtype=bool)
    edges = np.roll(hull, -1, axis=0) - hull
    delta = points[:, None, :] - hull[None, :, :]
    cross = edges[None, :, 0] * delta[:, :, 1] - edges[None, :, 1] * delta[:, :, 0]
    return (cross >= -1e-5).all(1) | (cross <= 1e-5).all(1)


@torch.no_grad()
def projected_region_features(frame, anchors, feature_map, config, *,
                              camera_ids, patch_stride=16, minimum_cells=4,
                              selected_ids=None, materialize=True, timings=None):
    """Select the largest supported camera region after geometric exclusions.

    Camera order must be supplied from the actual model input, and the stride
    from the existing patch embed. Scaling by the unpadded image size is wrong.
    Other projected boxes are conservative exclusions regardless of depth;
    unobserved occluders/background remain unknown. Call after current commit.
    """
    if (feature_map.ndim != 5 or feature_map.shape[0] != 1
            or feature_map.shape[1] != len(camera_ids)
            or tuple(camera_ids) != tuple(frame.cameras)
            or len(set(camera_ids)) != len(camera_ids)):
        raise ValueError('Feature grid/camera order contract mismatch')
    if type(patch_stride) is not int or patch_stride <= 0 or minimum_cells < 1:
        raise ValueError('Invalid patch geometry')
    _, _, channels, height, width = feature_map.shape
    anchors = list(anchors)
    selected_ids = set(selected_ids) if selected_ids is not None else {a.anchor_id for a in anchors}
    if not selected_ids:
        return [], {}
    points = grid_centres(height, width, patch_stride)
    tick = perf_counter()
    regions, rejections = [], {}
    projections, exclusions = {}, {}
    for cid, camera in frame.cameras.items():
        if camera.image.shape[0] > height*patch_stride or camera.image.shape[1] > width*patch_stride:
            raise ValueError('Feature grid does not cover the observation image')
        projections[cid] = {a.anchor_id: project_anchor(a, camera, config)
                            for a in anchors if a.anchor_type == 'object' and a.anchor_id in selected_ids}
        # Reuse current M1 support boxes for other detections. Do not reproject
        # the entire state or infer per-pixel foreground from those boxes.
        exclusions[cid] = [(a.anchor_id, a.observation_support[cid].region)
            for a in anchors if a.anchor_type == 'object' and cid in a.observation_support
            and a.source_frame_id == frame.frame_id
            and a.observation_support[cid].timestamp == camera.timestamp]
    if timings is not None:
        timings['geometry_projection_ms'] = (perf_counter()-tick)*1000
    selection_tick = perf_counter()
    gather_ms = 0.
    for anchor in anchors:
        if anchor.anchor_id not in selected_ids or anchor.anchor_type != 'object':
            continue
        if anchor.scene_id != frame.scene_id or anchor.source_frame_id != frame.frame_id:
            rejections[anchor.anchor_id] = 'not_current_committed_object'
            continue
        choices = []
        for ci, (cid, camera) in enumerate(frame.cameras.items()):
            support = anchor.observation_support.get(cid)
            projection = projections[cid][anchor.anchor_id]
            if support is None or support.timestamp != camera.timestamp or not projection.valid:
                continue
            valid = ((points[:, 0] < camera.image.shape[1])
                     & (points[:, 1] < camera.image.shape[0]))
            mask = inside_convex(points, projection.uv) & valid
            count = int(mask.sum())
            exclusion = np.zeros(len(points), bool)
            boxes = np.asarray([box for aid, box in exclusions[cid] if aid != anchor.anchor_id])
            if len(boxes):
                exclusion = ((points[:, None, 0] >= boxes[None, :, 0])
                    & (points[:, None, 0] < boxes[None, :, 2])
                    & (points[:, None, 1] >= boxes[None, :, 1])
                    & (points[:, None, 1] < boxes[None, :, 3])).any(1)
            clean = mask & ~exclusion
            indices = np.flatnonzero(clean)
            if len(indices) >= minimum_cells:
                choices.append((len(indices), cid, ci, indices, count, projection.region))
        if not choices:
            rejections[anchor.anchor_id] = 'insufficient_unambiguous_projected_grid_support'
            continue
        _, cid, ci, indices, count, region = sorted(choices, key=lambda c: (-c[0], c[1]))[0]
        gather_tick = perf_counter()
        values = None
        if materialize:
            device_indices = torch.as_tensor(indices, device=feature_map.device)
            values = feature_map[0, ci].reshape(channels, -1).index_select(1, device_indices).T.contiguous()
        gather_ms += (perf_counter()-gather_tick)*1000
        regions.append(RegionFeature(anchor.anchor_id, anchor.semantic_class, cid,
            frame.frame_id, anchor.state_version, tuple(region), tuple(map(int, indices)),
            values, count, count-len(indices), feature_grid=feature_map[0, ci] if not materialize else None,
            grid_width=width))
    if timings is not None:
        timings['region_grid_selection_ms'] = (perf_counter()-selection_tick)*1000-gather_ms
        timings['eva_gather_host_ms'] = gather_ms
    return regions, rejections


def update_visual_memory(manager, frame, anchors, feature_map, config, context,
                         qualified_ids, qualification_sha256):
    """Current geometry every frame; EVA reads only on actual creation."""
    tick = perf_counter()
    timings = dict(geometry_projection_ms=0., region_grid_selection_ms=0.,
                   eva_gather_host_ms=0., memory_create_ms=0., memory_keep_ms=0.,
                   memory_invalidation_ms=0., hash_version_ms=0., gpu_sync_d2h_ms=0.)
    regions, rejected = projected_region_features(frame, anchors, feature_map, config,
        camera_ids=list(frame.cameras), minimum_cells=manager.tokens,
        selected_ids=qualified_ids, materialize=False, timings=timings)
    memory, event = manager.update(context, regions, qualified_ids, qualification_sha256)
    timings.update(manager.last_timings)
    timings['bridge_total_host_ms'] = (perf_counter()-tick)*1000
    return memory, event, rejected, timings
