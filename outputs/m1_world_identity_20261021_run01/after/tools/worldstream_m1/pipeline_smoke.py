"""Actual CPU image preprocessing and M1 adapter on a real six-camera sample."""
import argparse
from types import SimpleNamespace
import numpy as np
from .dataset import select_infos
from .inspect_pipeline import local_dataset_method, projection_crosscheck
from .io_utils import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='outputs/module1/real_pipeline_smoke.json')
    args = p.parse_args()
    import torch
    from mmcv import Config
    from mmdet3d.datasets.pipelines import Compose
    import projects.mmdet3d_plugin
    from worldstream.m1 import M1Config, FullOutputs, WorldStateManager
    from worldstream.m1.omnidrive_adapter import frame_from_model
    cfg = Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py')
    info = select_infos(cfg.data.test.ann_file, limit=1)[0]
    prepared = local_dataset_method()(SimpleNamespace(data_infos=[info], modality={'use_camera': True}, test_mode=True), 0)
    prepared['img_fields'] = []; prepared['bbox3d_fields'] = []; prepared['bbox_fields'] = []
    prepared = Compose(cfg.data.test.pipeline[:5])(prepared)
    meta = {k: prepared[k] for k in ('scene_token', 'sample_idx', 'filename', 'img_shape', 'img_norm_cfg', 'cam_infos')}
    data = {'img': torch.from_numpy(np.stack([i.transpose(2, 0, 1) for i in prepared['img']]))[None],
            'ego_pose': torch.tensor(prepared['ego_pose'])[None],
            'timestamp': torch.tensor([prepared['timestamp']], dtype=torch.float64),
            'img_timestamp': torch.tensor([prepared['img_timestamp']], dtype=torch.float64),
            'intrinsics': torch.tensor(np.stack(prepared['intrinsics']))[None],
            'extrinsics': torch.tensor(np.stack(prepared['extrinsics']))[None]}
    original_shape = tuple(data['img'].shape)
    before = data['img'].clone()
    frame = frame_from_model([meta], data, M1Config())
    assert torch.equal(data['img'], before) and tuple(data['img'].shape) == original_shape
    manager = WorldStateManager(); result = manager.precheck(frame)
    manager.commit(frame, FullOutputs([]), result)
    errors = projection_crosscheck(info, prepared)
    assert max(errors.values()) < 1e-5
    write_json(args.output, {'status': 'passed_actual_CPU_image_pipeline_and_adapter',
        'scope': 'Actual Load/ResizeCropFlip/Resize/Normalize/Pad; no EVA/Head/LLM inference',
        'sample_id': info['token'], 'image_tensor_shape': list(original_shape),
        'camera_order': list(frame.cameras),
        'all_camera_time_poses_available': all(c.timing_reliable for c in frame.cameras.values()),
        'processed_lidar2img_projection_error_px': errors,
        'raw_to_preprocessed': {cid: c.raw_to_image.tolist() for cid, c in frame.cameras.items()},
        'input_unchanged': True, 'empty_state_refresh_hint': result.full_refresh_hint})
    print(args.output)


if __name__ == '__main__': main()
