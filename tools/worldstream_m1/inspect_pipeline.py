"""Record installed versions, data availability, and numerical local projection checks."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from .dataset import select_infos, frame_from_info
from .io_utils import write_json
from worldstream.m1.geometry import project_points
from worldstream.m1.omnidrive_adapter import quaternion_transform


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
