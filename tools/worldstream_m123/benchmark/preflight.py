"""Inventory real resources without downloading or modifying any of them."""
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

from .common import sha256, write

BASELINE = 'projects/configs/OmniDrive/mask_eva_lane_det_vlm.py'
FINAL = 'projects/configs/OmniDrive/worldstream_m123_auto.py'
PROTECTED = [BASELINE, FINAL, 'configs/worldstream_m1_final.yaml',
             'outputs/m123_final_quality_validation_run01/resources/final_route/window_costs.json',
             'outputs/m123_final_quality_validation_run01/resources/final_route/m3_prefill_costs.json',
             'outputs/m123_final_quality_validation_run01/resources/final_route/m3_prefill_quality.json',
             'outputs/m123_final_quality_validation_run01/resources/final_route/m3_state_quality.json']


def command(args):
    r = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return dict(command=args, returncode=r.returncode, output=r.stdout)


def inventory(dest, checkpoint, seed):
    import torch
    from mmcv import Config
    from .metrics import validate_sequence
    dest = Path(dest)
    cfg = Config.fromfile(FINAL)
    root = Path('data/nuscenes')
    ann = Path(cfg.data.test.ann_file)
    blockers = []
    resources = {}
    for path in [checkpoint, str(ann), 'data/nuscenes/data_dict_sample.pkl',
                 'data/nuscenes/data_dict_subset_B_val.pkl']:
        p = Path(path)
        resources[path] = dict(exists=p.is_file(), resolved=str(p.resolve()),
                               bytes=p.stat().st_size if p.is_file() else None,
                               sha256=sha256(p) if p.is_file() else None)
        if not p.is_file():
            blockers.append('Missing resource: '+path)
    infos = pickle.load(ann.open('rb'))['infos'] if ann.is_file() else []
    ordered = sorted(infos, key=lambda i: (i['scene_token'], i['timestamp']))
    validate_sequence(ordered)
    missing_images, missing_questions, gt_counts = [], Counter(), Counter()
    for info in ordered:
        if len(info['cams']) != 6:
            blockers.append('Frame does not have six cameras: '+info['token'])
        for cam in info['cams'].values():
            if not Path(cam['data_path']).is_file():
                missing_images.append(cam['data_path'])
        for folder in ('conv/val', 'vqa/val'):
            p = root / folder / (info['token']+'.json')
            if not p.is_file():
                missing_questions[folder] += 1
            else:
                pairs = json.loads(p.read_text())
                gt_counts[folder] += len(pairs)
    if missing_images:
        blockers.append(f'{len(missing_images)} validation camera images missing')
    tables = {}
    for name in ('scene', 'sample', 'sample_annotation', 'instance', 'category'):
        p = root/'v1.0-trainval'/(name+'.json')
        tables[name] = dict(exists=p.is_file(), sha256=sha256(p) if p.is_file() else None)
        if not p.is_file():
            blockers.append('Missing nuScenes table: '+str(p))
    sample_path = root/'v1.0-trainval/sample.json'
    if sample_path.is_file():
        samples = json.loads(sample_path.read_text())
        tables['sample'].update(rows=len(samples), annotation_links=sum(len(x['anns']) for x in samples))
        from tools.worldstream_m1.io_utils import iter_json_array
        wanted = {i['token'] for i in infos}
        linked = {s['token']: set(s['anns']) for s in samples if s['token'] in wanted}
        reverse = {t: set() for t in wanted}
        for a in iter_json_array(root/'v1.0-trainval/sample_annotation.json'):
            if a['sample_token'] in reverse:
                reverse[a['sample_token']].add(a['token'])
        bad = [t for t in wanted if linked.get(t, set()) != reverse[t]]
        tables['raw_annotation_link_audit'] = dict(
            status='PASS' if not bad else 'FAIL', affected_frames=len(bad),
            linked_annotations=sum(len(v) for v in linked.values()),
            reverse_annotations=sum(len(v) for v in reverse.values()), examples=sorted(bad)[:10],
            consequence='Raw-file audit only. The official NuScenes constructor rebuilds anns; validate its actual runtime object before deciding evaluation availability.')
        try:
            from nuscenes import NuScenes
            from nuscenes.utils.splits import create_splits_scenes
            from nuscenes.eval.common.loaders import load_gt
            from nuscenes.eval.detection.data_classes import DetectionBox
            nusc = NuScenes(version='v1.0-trainval', dataroot=str(root), verbose=False)
            val_names = set(create_splits_scenes()['val'])
            official_tokens = {s['token'] for s in nusc.sample if nusc.get('scene', s['scene_token'])['name'] in val_names}
            actual_bad = [t for t in wanted if set(nusc.get('sample', t)['anns']) != reverse[t]]
            gt = load_gt(nusc, 'val', DetectionBox, verbose=False)
            tables['validation_annotation_integrity'] = dict(
                status='PASS' if not actual_bad and official_tokens == wanted else 'FAIL',
                sdk_runtime_validation=True, affected_frames=len(actual_bad),
                linked_annotations=sum(len(nusc.get('sample', t)['anns']) for t in wanted),
                reverse_annotations=sum(len(v) for v in reverse.values()),
                official_frames=len(official_tokens), dataset_frames=len(wanted),
                official_detection_gt_boxes=len(gt.all),
                missing_tokens=sorted(official_tokens-wanted), extra_tokens=sorted(wanted-official_tokens),
                consequence='Official SDK reconstructs annotation links from unchanged raw tables in memory.')
            del nusc, gt
        except Exception:
            import traceback
            tables['validation_annotation_integrity'] = dict(status='FAIL', sdk_runtime_validation=True,
                error=traceback.format_exc())
    nvidia = command(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,driver_version', '--format=csv,noheader'])
    gpu_available = torch.cuda.is_available()
    witness = None
    if gpu_available:
        torch.manual_seed(seed)
        x = torch.arange(64, device='cuda', dtype=torch.float32).reshape(8, 8)
        result = (x @ x.T).sum().item()
        witness = dict(value=result, expected=510720., passed=result == 510720.)
        if not witness['passed']:
            blockers.append('Seeded CUDA kernel witness failed')
    else:
        blockers.append('CUDA unavailable in this process; GPU execution requires device access')
    evidence = {p: dict(sha256=sha256(p), content=json.loads(Path(p).read_text()))
                for p in PROTECTED if p.endswith('.json') and 'summary' not in p and 'verification' not in p}
    packages = {}
    for package in ('torch', 'mmcv-full', 'mmdet', 'mmdet3d', 'transformers', 'numpy', 'nuscenes-devkit', 'flash-attn'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    source = sorted(set([str(p) for folder in ('worldstream', 'projects/mmdet3d_plugin',
                                             'tools/worldstream_m123/benchmark')
                         for p in Path(folder).rglob('*.py')]
                        + [str(p) for p in Path('projects/configs/OmniDrive').glob('*.py')]
                        + [str(p) for p in Path('configs').glob('worldstream*.yaml')]))
    result = dict(checkpoint=resources[str(checkpoint)], resources=resources, dataset=dict(
        split='val', version='v1.0-trainval', frames=len(infos), scenes=len({i['scene_token'] for i in infos}),
        camera_requests=sum(len(i['cams']) for i in infos), missing_images=len(missing_images),
        missing_image_examples=missing_images[:20], missing_question_files=dict(missing_questions),
        question_pairs=dict(gt_counts), tables=tables,
        ordered_token_sha256=__import__('hashlib').sha256('\n'.join(i['token'] for i in ordered).encode()).hexdigest()),
        cuda_available=gpu_available, cuda_kernel_witness=witness, nvidia_smi=nvidia,
        environment=dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, cuda_build=torch.version.cuda, packages=packages,
            affinity=sorted(os.sched_getaffinity(0)),
            gpu=torch.cuda.get_device_name(0) if gpu_available else None),
        calibration=evidence, protected_hashes={p: sha256(p) for p in PROTECTED},
        source_hashes={p: sha256(p) for p in source}, blockers=blockers)
    write(dest/'preflight.json', result)
    manifest = dict(commit=command(['git', 'rev-parse', 'HEAD'])['output'].strip(),
        initial_git_status=command(['git', 'status', '--short'])['output'], seed=seed,
        config_files=dict(B0=BASELINE, B3=FINAL), preflight=result,
        commands=[dict(argv=sys.argv, cwd=str(Path.cwd()))],
        precision='Unchanged configuration; no forced autocast, TF32 or fp16 policy',
        generation=dict(do_sample=True, temperature=.1, top_p=.75, num_beams=1,
                        max_new_tokens=320, use_cache=True))
    write(dest/'run_manifest.json', manifest)
    return result
