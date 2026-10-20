"""Bounded real PyTorch inference; optional disabled/enabled tensor-digest comparison."""
import argparse
import hashlib
from pathlib import Path
import random
from time import perf_counter,process_time
import numpy as np
from .io_utils import write_json

_status_path = None


def digest(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    if hasattr(value, 'tensor'): return digest(value.tensor)
    if isinstance(value, dict): return {k: digest(v) for k, v in value.items() if k != 'text_out'}
    if isinstance(value, (list, tuple)): return [digest(v) for v in value]
    return value


def main():
    global _status_path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='projects/configs/OmniDrive/worldstream_m1_observe.py')
    p.add_argument('--checkpoint', default='ckpts/iter_10548.pth')
    p.add_argument('--m1-config', default='configs/worldstream_m1.yaml')
    p.add_argument('--frames', type=int, default=3)
    p.add_argument('--scene')
    p.add_argument('--compare', action='store_true')
    p.add_argument('--perception-only', action='store_true',
                   help='Validate original EVA/detection/map outputs without LLM execution or M2/M3')
    p.add_argument('--output', default='outputs/module1/model_smoke')
    args = p.parse_args()
    import torch
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    if ((root/'results.json').exists() or (root/'m1_state.jsonl').exists()
            or (root/'baseline/language').exists() or (root/'observe_only/language').exists()):
        raise SystemExit('Output contains an earlier run; choose a fresh --output to preserve results and generation')
    _status_path = root/'status.json'
    if not torch.cuda.is_available():
        write_json(root/'status.json', {'status': 'blocked', 'reason': 'CUDA is not visible to this process; '
                   'this does not establish a host driver failure', 'args': vars(args),
                   'cuda_available': False,
                   'nvidia_kernel_driver_present': Path('/proc/driver/nvidia/version').exists(),
                   'visible_device_nodes': [str(p) for p in Path('/dev').glob('nvidia*')]})
        raise SystemExit('CUDA is not visible in this execution context; check sandbox/device access')
    run_metadata = {'args': vars(args), 'cuda_available': True, 'torch_version': torch.__version__,
                    'torch_cuda_version': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
                    'device_total_bytes': torch.cuda.get_device_properties(0).total_memory}
    write_json(root/'status.json', dict(run_metadata, status='running'))
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.models import build_model
    import projects.mmdet3d_plugin
    from mmdet3d.datasets import build_dataset
    from worldstream.m1 import M1Config, WorldStateManager
    from dataclasses import replace
    cfg = Config.fromfile(args.config)
    settings = M1Config.load(args.m1_config)
    stat = Path(args.checkpoint).stat()
    identity = (str(Path(args.checkpoint).resolve())+':'+str(stat.st_size)+':'+str(stat.st_mtime_ns)
                +':config='+hashlib.sha256(cfg.pretty_text.encode()).hexdigest())
    settings = replace(settings, model_id=identity, logging=True, log_path=str(root/'m1_state.jsonl'))
    cfg.model.m1_enable = False
    if args.perception_only:
        cfg.model.lm_head = None
        cfg.model.m2_enable = False
        cfg.model.m3_enable = False
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    for step in cfg.data.test.pipeline:
        for transform in step.get('transforms', []):
            if transform['type'] == 'Collect3D' and 'cam_infos' not in transform['meta_keys']:
                transform['meta_keys'] = tuple(transform['meta_keys']) + ('cam_infos',)
    dataset = build_dataset(cfg.data.test)
    from .dataset import log_context_from_root
    from .scene_protocol import select_scenes
    context=log_context_from_root('data/nuscenes')
    chosen_scene=args.scene or sorted({x['scene_token'] for x in dataset.data_infos})[0]
    by_token={info['token']:i for i,info in enumerate(dataset.data_infos)}
    sequence=sorted((x for x in dataset.data_infos if x['scene_token']==chosen_scene),key=lambda x:x['timestamp'])
    sequence=sequence[:args.frames] if args.frames else sequence
    indices=[by_token[x['token']] for x in sequence]
    write_json(root/'selection.json',dict(scene_id=chosen_scene,log_token=context.log_for_scene(chosen_scene),
        source_context_sha256=context.digest,sequence_type='nuScenes keyframes, not high-frequency video',
        frame_limit=args.frames,selected=[dict(frame_id=i['token'],timestamp=i['timestamp']/1e6) for i in sequence]))
    if not indices: raise ValueError('No selected samples')
    run_metadata['sample_ids'] = [dataset.data_infos[i]['token'] for i in indices]
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'): wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda().eval()
    model.m1_source_context=context
    run_metadata['perception_only'] = args.perception_only
    all_results = {}
    for mode in (['baseline', 'observe_only'] if args.compare else ['observe_only']):
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        model.test_flag = False
        model.m1_manager = WorldStateManager(settings) if mode == 'observe_only' else None
        model.save_path = str(root/mode/'language')+'/'
        outputs, memory, backbone = [], [], []
        frame_measurements=[]
        hook = model.img_backbone.register_forward_hook(lambda module, inputs, out: backbone.append(digest(out)))
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for index in indices:
                batch = scatter(collate([dataset[index]], samples_per_gpu=1), [torch.cuda.current_device()])[0]
                torch.cuda.synchronize();wall=perf_counter();cpu=process_time()
                result = model(return_loss=False, rescale=True, **batch)
                torch.cuda.synchronize()
                frame_measurements.append(dict(frame_id=dataset.data_infos[index]['token'],
                    end_to_end_wall_ms=(perf_counter()-wall)*1000,
                    measurement_cpu_process_ms=(process_time()-cpu)*1000,
                    end_to_end_scope='native full model call including M1 and Planning; excludes batch IO and output hashing'))
                outputs.append(digest(result))
                memory.append({branch: {key: digest(value) for key, value in vars(head).items()
                              if key.startswith('memory_') or key == 'sample_time'}
                              for branch, head in [('det', model.pts_bbox_head), ('map', model.map_head)]})
        end.record(); torch.cuda.synchronize(); hook.remove()
        all_results[mode] = {'outputs': outputs, 'memory': memory, 'backbone': backbone,
                             'frame_measurements':frame_measurements,
                             'gpu_elapsed_ms_including_host_gaps': start.elapsed_time(end),
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    if args.compare:
        all_results['comparisons'] = {key: all_results['baseline'][key] == all_results['observe_only'][key]
                                     for key in ('outputs', 'memory', 'backbone')}
        all_results['observe_only_equal'] = all(all_results['comparisons'].values())
        if not all_results['observe_only_equal']:
            write_json(root/'results.json', all_results)
            write_json(root/'status.json', dict(run_metadata, status='failed', reason='Numerical digest mismatch'))
            raise SystemExit('Baseline/observe-only numerical digest mismatch; inspect saved results')
    all_results['run_metadata'] = run_metadata
    write_json(root/'results.json', all_results)
    write_json(root/'status.json', dict(run_metadata, status='passed',
               observe_only_equal=all_results.get('observe_only_equal')))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if _status_path is not None and _status_path.is_file():
            import json
            status = json.loads(_status_path.read_text())
            status.update(status='failed', error_type=type(error).__name__, reason=str(error))
            write_json(_status_path, status)
        raise
