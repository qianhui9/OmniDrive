"""Execute actual Petr3D inference methods with CPU deterministic test doubles.

This checks the integration boundary and numerical non-interference, not real EVA/LLM accuracy.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import os
import json
import time
import numpy as np
import pytest
import torch
from worldstream.m1 import WorldStateManager


def inference_class():
    import mmcv
    from mmdet3d.core import bbox3d2result
    source = Path('projects/mmdet3d_plugin/models/detectors/petr3d.py')
    original = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Petr3D')
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
               and n.name in ('simple_test', 'simple_test_pts', 'extract_img_feat')]
    cls = ast.ClassDef(name='InferenceMethods', bases=[], keywords=[], body=methods, decorator_list=[])
    ns = dict(torch=torch, os=os, json=json, time=time, mmcv=mmcv, bbox3d2result=bbox3d2result)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), 'exec'), ns)
    return ns['InferenceMethods']


class Head:
    def __init__(self, map_head=False):
        self.calls = 0; self.memory_timestamp = torch.tensor([0.]); self.map_head = map_head

    def __call__(self, metas, pos, **data):
        self.calls += 1; self.memory_timestamp -= data['timestamp']
        return {}, data['img_feats'].mean((-1, -2)).reshape(1, -1, 3)

    def get_bboxes(self, outs, metas):
        if self.map_head:
            return [[np.array([[[0., 0, 10.], [1., 0, 10.], [2., 0, 10.]]]), np.array([[.9]])]]
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        return [[LiDARInstance3DBoxes(torch.tensor([[0., 0., 9., 2., 2., 2., 0., 0., 0.]]), box_dim=9),
                 torch.tensor([.9]), torch.tensor([5])]]


def make_model(tmp_path, manager):
    model = inference_class()()
    model.m1_manager = manager
    model.named_parameters = lambda: []
    model.with_pts_bbox = model.with_map_head = model.with_lm_head = True
    model.with_img_neck = model.use_grid_mask = False
    model.position_level = 0
    model.img_backbone = lambda tensor: [tensor * 2 + 1]
    model.pts_bbox_head = Head(); model.map_head = Head(True)
    model.prepare_location = lambda *args, **kwargs: None
    model.forward_roi_head = lambda *args, **kwargs: {}
    model.position_embeding = lambda *args, **kwargs: None
    model.save_path = str(tmp_path) + '/'
    model.lm_inputs = []
    def generate(**kwargs):
        model.lm_inputs.append(kwargs['images'].clone())
        return torch.randint(0, 10, (1, 3))
    model.lm_head = SimpleNamespace(generate=generate)
    model.tokenizer = SimpleNamespace(batch_decode=lambda ids, **kwargs: ids.tolist())
    return model


def inputs(timestamp=100.):
    img = torch.arange(2*3*100*120).reshape(1, 2, 3, 100, 120).float() % 255
    k = torch.tensor([[60., 0, 60, 0], [0, 60, 50, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    data = {'img': img, 'timestamp': torch.tensor([timestamp], dtype=torch.float64),
            'img_timestamp': torch.tensor([[timestamp, timestamp]], dtype=torch.float64),
            'ego_pose': torch.eye(4)[None], 'intrinsics': k[None, None].repeat(1, 2, 1, 1),
            'extrinsics': torch.eye(4)[None, None].repeat(1, 2, 1, 1),
            'input_ids': [[torch.tensor([1, 2])]]}
    meta = {'scene_token': 'scene', 'sample_idx': str(timestamp),
            'filename': ['samples/front/a.jpg', 'samples/side/a.jpg'],
            'img_shape': [(100, 120, 3), (80, 100, 3)],
            'img_norm_cfg': dict(mean=np.zeros(3), std=np.ones(3), to_rgb=True),
            'vlm_labels': SimpleNamespace(data=['Q'])}
    return [meta], data


def test_disabled_and_observe_only_equal_at_real_entry(tmp_path, config):
    from tools.worldstream_m1.model_smoke import digest
    baseline = make_model(tmp_path/'baseline', None)
    enabled = make_model(tmp_path/'observe', WorldStateManager(config))
    for timestamp in (100., 100.5):
        meta, data = inputs(timestamp)
        torch.manual_seed(3); before = baseline.simple_test(meta, **data)
        meta, data = inputs(timestamp)
        torch.manual_seed(3); after = enabled.simple_test(meta, **data)
        assert digest(before) == digest(after)
        assert before[0]['text_out'] == after[0]['text_out']
        assert set(before[0]) == {'pts_bbox', 'text_out', 'lane_results'}
        assert torch.equal(baseline.lm_inputs[-1], enabled.lm_inputs[-1])
        for attr in ('pts_bbox_head', 'map_head'):
            assert torch.equal(getattr(baseline, attr).memory_timestamp, getattr(enabled, attr).memory_timestamp)
    assert enabled.pts_bbox_head.calls == enabled.map_head.calls == 2
    assert enabled.m1_manager.last_result.timestamp == 100.5


def test_adapter_is_readonly_batch_and_actual_shapes(config):
    from worldstream.m1.omnidrive_adapter import frame_from_model
    meta, data = inputs()
    img_copy = data['img'].clone()
    frame = frame_from_model(meta, data, config)
    assert frame.cameras['side'].image.shape == (80, 100, 3)
    assert list(frame.cameras) == ['front', 'side']
    assert torch.equal(data['img'], img_copy)
    assert data['img'].shape == (1, 2, 3, 100, 120)
    with pytest.raises(ValueError, match='batch size 1'):
        frame_from_model(meta*2, data, config)


@pytest.mark.parametrize('invalid', [None, np.nan, np.inf])
def test_bad_calibration_is_conservative(config, make_frame, make_prediction, invalid):
    from worldstream.m1 import FullOutputs
    manager = WorldStateManager(config); manager.initialize(make_frame(), FullOutputs([make_prediction()]))
    frame = make_frame(100.5)
    if invalid is None: frame.cameras['front'].world_to_camera = None
    else: frame.cameras['front'].intrinsic[0, 0] = invalid
    result = manager.precheck(frame)
    assert result.invalid_anchor_ids and not result.valid_anchor_ids
    assert manager.last_masks['front'].all()


def test_live_config_preserves_exact_base_test_processing():
    from mmcv import Config
    base = Config.fromfile('projects/configs/OmniDrive/mask_eva_lane_det_vlm.py')
    enabled = Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py')
    a, b = base.data.test.pipeline, enabled.data.test.pipeline
    copied = b.copy()
    copied[-1]['transforms'][-1]['meta_keys'] = a[-1]['transforms'][-1]['meta_keys']
    assert a == copied
