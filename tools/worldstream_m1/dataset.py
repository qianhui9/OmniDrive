"""Current raw input is separated from commit outputs and offline-only annotations."""
import pickle
import json
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from worldstream.m1 import CameraFrame, Frame, FullOutputs, Geometry, PerceptionAnchor
from worldstream.m1.geometry import box_geometry, inverse_transform, transform_points
from worldstream.m1.omnidrive_adapter import quaternion_transform
from .io_utils import iter_json_array, read_prediction_subset


@lru_cache(maxsize=4)
def log_context_from_root(root):
    """Read only scene/sample provenance tables, never annotation/instance tables."""
    from worldstream.m1.source_context import LogSourceContext
    tables=Path(root)/'v1.0-trainval'
    return LogSourceContext.from_rows(json.loads((tables/'scene.json').read_text()),
                                     json.loads((tables/'sample.json').read_text()))


def select_infos(path, scene=None, limit=12, stride=1, scene_index=0):
    if stride < 1 or limit < 1:
        raise ValueError('limit and stride must be positive')
    with open(path, 'rb') as handle:
        infos = pickle.load(handle)['infos']
    scenes = sorted({i['scene_token'] for i in infos})
    scene = scene or scenes[scene_index]
    selected = sorted((i for i in infos if i['scene_token'] == scene), key=lambda i: i['timestamp'])
    selected = selected[::stride][:limit]
    if not selected:
        raise ValueError('No matching scene')
    return selected


def resolve_image(path, root):
    path = Path(path)
    if path.is_file(): return path
    parts = path.parts
    for category in ('samples', 'sweeps'):
        if category in parts:
            return Path(root).joinpath(*parts[parts.index(category):])
    raise FileNotFoundError(str(path))


def frame_from_info(info, root, max_width=640, signature='raw-replay',source_context=None):
    pose = quaternion_transform(info['ego2global_rotation'], info['ego2global_translation'])
    cameras = {}
    for cid, cam in info['cams'].items():
        path = resolve_image(cam['data_path'], root)
        bgr = cv2.imread(str(path))
        if bgr is None: raise FileNotFoundError(str(path))
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        scale = min(1., max_width / w)
        neww, newh = int(round(w * scale)), int(round(h * scale))
        image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)
        affine = np.diag([neww / w, newh / h, 1.])
        c2w = (quaternion_transform(cam['ego2global_rotation'], cam['ego2global_translation'])
               @ quaternion_transform(cam['sensor2ego_rotation'], cam['sensor2ego_translation']))
        cameras[cid] = CameraFrame(cid, image, affine @ cam['cam_intrinsic'], inverse_transform(c2w),
                                   cam['timestamp'] / 1e6, 'resized_raw_image', affine, raw_intrinsic=np.array(cam['cam_intrinsic'],copy=True))
    from worldstream.m1.source_context import metadata_log_source
    context=source_context
    if context is None and (Path(root)/'v1.0-trainval/scene.json').is_file():
        context=log_context_from_root(str(root))
    provenance=metadata_log_source(info,info['scene_token'],info['token'],context)
    return Frame(info['scene_token'], info['token'], info['timestamp'] / 1e6, pose, cameras,
                 signature, {**provenance,'sampling': 'nuScenes keyframes, not high-frequency video',
                             'observation_protocol': {'coordinate_space': 'resized_raw_image', 'max_width': max_width},
                             'camera_order': list(cameras)})


def prediction_outputs(predictions, config):
    from pyquaternion import Quaternion
    anchors = []
    for p in predictions:
        if p['detection_score'] < config.min_detection_score: continue
        rotation = Quaternion(p['rotation']).rotation_matrix
        matrix = np.eye(4); matrix[:3, :3] = rotation; matrix[:3, 3] = p['translation']
        size = np.asarray(p['size'])[[1, 0, 2]]
        g = box_geometry(np.zeros(3), size, to_world=matrix)
        velocity = np.r_[p.get('velocity', [np.nan, np.nan]), 0.]
        reliable = False  # finite detector velocity is not independently verified motion
        anchors.append(PerceptionAnchor('object', p['detection_name'], g,
                                        velocity if np.isfinite(velocity).all() else None, reliable,
                                        source_score=float(p['detection_score'])))
    return FullOutputs(anchors, source='historical_full_model_detection_replay')


def load_oracle(infos, root):
    """Only evaluation/annotation-assisted commit calls this, never precheck."""
    root = Path(root)
    table = root / 'v1.0-trainval'
    wanted = {i['token'] for i in infos}
    samples = {s['token']: s for s in iter_json_array(table / 'sample.json') if s['token'] in wanted}
    # Local sample.json may have filtered `anns` lists from earlier projects. The
    # annotation table's explicit sample_token is the authoritative reverse link.
    anns = {token: [] for token in samples}
    for a in iter_json_array(table / 'sample_annotation.json'):
        if a['sample_token'] in anns:
            anns[a['sample_token']].append(a)
    categories = {c['token']: c['name'] for c in iter_json_array(table / 'category.json')}
    # annotation records have instance_token; categories are resolved through instance.json.
    instances = {i['token']: categories[i['category_token']] for i in iter_json_array(table / 'instance.json')}
    from mmdet3d.datasets import NuScenesDataset
    mapping = NuScenesDataset.NameMapping
    result = {}
    for token, sample in samples.items():
        current = []
        for a in anns[token]:
            name = mapping.get(instances[a['instance_token']])
            if name is None: continue
            matrix = quaternion_transform(a['rotation'], a['translation'])
            g = box_geometry(np.zeros(3), np.asarray(a['size'])[[1, 0, 2]], to_world=matrix)
            current.append({'instance_id': a['instance_token'], 'class': name, 'geometry': g,
                            'visibility': a.get('visibility_token'),
                            'has_sensor_support': a['num_lidar_pts'] + a['num_radar_pts'] > 0})
        result[token] = current
    return result


import ast
from worldstream.m1.geometry import project_points
def local_dataset_method():
    """Execute actual get_data_info body without constructing a training dataset/LLM."""
    from pyquaternion import Quaternion
    path = Path('projects/mmdet3d_plugin/datasets/nuscenes_dataset.py')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CustomNuScenesDataset')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_data_info')
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
               ('convert_egopose_to_matrix_numpy', 'invert_matrix_egopose_numpy')]
    namespace = {'np': np, 'Quaternion': Quaternion}
    exec(compile(ast.Module(body=helpers + [method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['get_data_info']

def projection_crosscheck(info, prepared=None):
    from types import SimpleNamespace
    local = prepared if prepared is not None else local_dataset_method()(
        SimpleNamespace(data_infos=[info], modality={'use_camera': True}, test_mode=True), 0)
    pose = quaternion_transform(info['ego2global_rotation'], info['ego2global_translation'])
    from worldstream.m1.geometry import transform_points, inverse_transform
    errors = {}
    for i, cid in enumerate(info['cams']):
        # Well-conditioned rays in every actual camera, including the rear cameras.
        rays = np.array([[x, y, z] for x in (-2, 0, 2) for y in (-1, 1) for z in (5, 15, 30)])
        camera_to_ego = np.linalg.inv(local['extrinsics'][i].astype(float))
        points = rays @ camera_to_ego[:3, :3].T + camera_to_ego[:3, 3]
        world = transform_points(points, pose)
        k = local['intrinsics'][i][:3, :3]
        uv, _, good = project_points(world, k, local['extrinsics'][i] @ inverse_transform(pose))
        direct = np.c_[points, np.ones(len(points))] @ local['lidar2img'][i].T
        expected = direct[good, :2] / direct[good, 2:3]
        errors[cid] = float(np.max(np.abs(uv[good] - expected))) if good.any() else None
    return errors
