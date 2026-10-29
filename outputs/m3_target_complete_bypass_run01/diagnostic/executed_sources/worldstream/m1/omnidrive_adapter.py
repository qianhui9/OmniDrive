"""Read-only bridge for the inspected Ego-coordinate OmniDrive pipeline."""
import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from pathlib import Path
import numpy as np
from .geometry import rigid_transform, inverse_transform, transform_points, valid_transform
from .state_types import CameraFrame, ComputedFeature, Frame, FullOutputs, Geometry, PerceptionAnchor
CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
_transfer_stats = ContextVar('m1_transfer_stats', default=None)

@contextmanager
def measure_transfers():
    stats = {'gpu_to_cpu_transfer_ms': 0.0, 'gpu_to_cpu_transfer_calls': 0, 'gpu_to_cpu_transfer_bytes': 0}
    token = _transfer_stats.set(stats)
    try:
        yield stats
    finally:
        _transfer_stats.reset(token)

def numpy_copy(value):
    if hasattr(value, 'detach'):
        start = perf_counter()
        cuda = getattr(value, 'is_cuda', False)
        host = value.detach().cpu().numpy()
        result = host if cuda else host.copy()
        stats = _transfer_stats.get()
        if stats is not None and getattr(value, 'is_cuda', False):
            stats['gpu_to_cpu_transfer_ms'] += (perf_counter() - start) * 1000
            stats['gpu_to_cpu_transfer_calls'] += 1
            stats['gpu_to_cpu_transfer_bytes'] += value.numel() * value.element_size()
        return result
    return np.array(value, copy=True)

def quaternion_transform(rotation, translation):
    from pyquaternion import Quaternion
    return rigid_transform(Quaternion(rotation).rotation_matrix, translation)

def frame_from_model(img_metas, data, config, source_context=None):
    if len(img_metas) != 1 or data['img'].shape[0] != 1:
        raise ValueError('M1 streaming supports batch size 1 only')
    meta = img_metas[0]
    images = numpy_copy(data['img'][0])
    if images.ndim != 4:
        raise ValueError('M1 expects B,N,C,H,W images and a single test augmentation')
    timestamp = float(numpy_copy(data['timestamp']).reshape(-1)[0])
    pose = numpy_copy(data['ego_pose'][0])
    intrinsics = numpy_copy(data['intrinsics'][0])
    extrinsics = numpy_copy(data['extrinsics'][0])
    image_times = numpy_copy(data['img_timestamp'][0]).reshape(-1)
    norm = meta.get('img_norm_cfg', {})
    mean = np.asarray(norm.get('mean', [0, 0, 0]))
    std = np.asarray(norm.get('std', [1, 1, 1]))
    cam_infos = meta.get('cam_infos', {})
    filenames = meta.get('filename', [])
    ids = list(cam_infos) if cam_infos else [Path(f).parent.name for f in filenames]
    if len(ids) != len(images) or len(set(ids)) != len(ids):
        raise ValueError('Cannot establish the actual unique camera order from metadata')
    if cam_infos and filenames and ([Path(f).parent.name for f in filenames] != ids):
        raise ValueError('Camera metadata order disagrees with image filenames')
    if not len(intrinsics) == len(extrinsics) == len(image_times) == len(ids):
        raise ValueError('Camera/timestamp/calibration counts do not agree')
    cameras = {}
    for (i, cid) in enumerate(ids):
        (h, w) = meta['img_shape'][i][:2]
        image = images[i, :, :h, :w].transpose(1, 2, 0) * std + mean
        if not norm.get('to_rgb', False):
            image = image[..., ::-1]
        image = np.clip(image, 0, 255).astype(np.uint8)
        k = intrinsics[i][:3, :3]
        (mapping, exact_pose, raw_intrinsic) = (None, False, None)
        try:
            if cid in cam_infos:
                info = cam_infos[cid]
                camera_world = quaternion_transform(info['ego2global_rotation'], info['ego2global_translation']) @ quaternion_transform(info['sensor2ego_rotation'], info['sensor2ego_translation'])
                world_to_camera = inverse_transform(camera_world)
                raw_intrinsic = np.asarray(info['cam_intrinsic']).copy()
                mapping = k @ np.linalg.inv(raw_intrinsic)
                exact_pose = abs(info['timestamp'] / 1000000.0 - image_times[i]) < 1e-05
            else:
                world_to_camera = extrinsics[i] @ inverse_transform(pose)
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError):
            world_to_camera = None
        timing = bool(exact_pose or abs(image_times[i] - timestamp) <= config.max_camera_skew_s)
        cameras[cid] = CameraFrame(cid, image, k, world_to_camera, float(image_times[i]), 'preprocessed_image', mapping, timing, raw_intrinsic=raw_intrinsic)
    signature_info = {'model': config.model_id, 'normalization': {k: numpy_copy(v).tolist() for (k, v) in norm.items()}, 'camera_order': ids, 'intrinsics': intrinsics.tolist(), 'extrinsics': extrinsics.tolist(), 'shapes': {k: list(c.image.shape) for (k, c) in cameras.items()}, 'coordinates': 'Ego outputs; world SE3; preprocessed K; camera-time poses', 'input_dtype': str(data['img'].dtype), 'input_device': str(data['img'].device)}
    signature = hashlib.sha256(json.dumps(signature_info, sort_keys=True).encode()).hexdigest()
    signature_info['observation_protocol'] = {'coordinate_space': 'preprocessed_image', 'normalization': signature_info['normalization'], 'shapes': signature_info['shapes']}
    from .source_context import metadata_log_source
    signature_info.update(metadata_log_source(meta, str(meta['scene_token']), str(meta['sample_idx']), source_context))
    return Frame(str(meta['scene_token']), str(meta['sample_idx']), timestamp, pose, cameras, signature, signature_info)

def outputs_from_model(frame, bbox_results, lane_results, features, config):
    anchors = []
    pose = frame.ego_to_world
    if not valid_transform(pose):
        return FullOutputs([], source='invalid_pose_no_state_committed')
    if bbox_results:
        result = bbox_results[0]
        boxes = result['boxes_3d']
        corners = numpy_copy(boxes.corners)
        centers = numpy_copy(boxes.gravity_center)
        sizes = numpy_copy(boxes.dims)
        labels = numpy_copy(result['labels_3d'])
        values = numpy_copy(boxes.tensor)
        scores = numpy_copy(result['scores_3d'])
        for i in range(len(centers)):
            if scores[i] < config.min_detection_score:
                continue
            if not np.isfinite(corners[i]).all() or not np.isfinite(centers[i]).all():
                continue
            g = Geometry(transform_points(corners[i], pose), transform_points(centers[i], pose), sizes[i])
            velocity = None
            if values.shape[1] >= 9 and np.isfinite(values[i, 7:9]).all():
                velocity = pose[:3, :3] @ np.r_[values[i, 7:9], 0.0]
            anchors.append(PerceptionAnchor('object', CLASS_NAMES[int(labels[i])], g, velocity, False, source_score=float(scores[i])))
    if lane_results:
        (lines, scores) = lane_results[0]
        for (line, score) in zip(lines, scores):
            if np.max(score) < config.min_map_score or not np.isfinite(line).all():
                continue
            points = transform_points(numpy_copy(line), pose)
            anchors.append(PerceptionAnchor('map', 'lane_control_points', Geometry(points, points.mean(axis=0))))
    records = []
    for (name, tensor) in features.items():
        kind = {'backbone': 'backbone', 'object_carrier': 'object_carrier', 'map_carrier': 'map_carrier'}[name]
        regions = {cid: [(0.0, 0.0, float(cam.image.shape[1]), float(cam.image.shape[0]))] for (cid, cam) in frame.cameras.items()}
        records.append(ComputedFeature(name, kind, tuple(tensor.shape), str(tensor.dtype), str(tensor.device), {'backbone': 'EVA/selected_output', 'object_carrier': 'StreamPETR/branch_output', 'map_carrier': 'PETRHeadM/branch_output'}[name], tensor, list(frame.cameras), regions, 'unknown/conservative'))
    return FullOutputs(anchors, records)

def append_result(path, result, commit_diagnostics, committed_state=None, experiment=None):
    from .state_types import json_value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(json_value({'precheck': result.to_dict(), 'commit': commit_diagnostics, 'committed_state': committed_state, 'experiment': experiment}), allow_nan=False) + '\n')
