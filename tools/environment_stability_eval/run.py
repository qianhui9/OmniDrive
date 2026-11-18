"""CPU preflight/GT grouping, fresh-process paired inference, and offline reports."""
import argparse
from datetime import datetime
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import traceback

from .common import (METHODS, PYTHON, ROOT, csv_write, dump_jsonl, gpu_lease, read,
                     require_affinity, write)


def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    if config['scope'] != 'main-scenes':
        raise ValueError('This paired supplement uses the complete frozen main-scenes evaluation set')
    if config['calibration'] not in ('remaining-val-scenes', 'evaluation-GT'):
        raise ValueError('Unsupported calibration policy')
    if config['threshold_quantiles'] != [.25, .5, .75]:
        raise ValueError('Four levels require quartiles [.25,.5,.75]')
    if config['repeats'] < 1 or config['warmup_frames'] < 1 or config['smoke_frames'] < 1:
        raise ValueError('Repeat/warmup/smoke counts must be positive')
    if config['bootstrap_samples'] < 0 or config['response_horizon_seconds'] <= 0 or config['other_questions'] < 0:
        raise ValueError('Invalid evaluation settings')
    if Path(config['r0_config']) != Path('configs/worldstream_r0.json'):
        raise ValueError('Use the canonical frozen complete R0 configuration')
    if (ROOT / config['data_root']).resolve() != (ROOT / 'data/nuscenes').resolve():
        raise ValueError('The existing QA evaluator requires the data/nuscenes root')
    return config


def prepare(args, config):
    from .data_adapter import (build_query_records, coverage_report, load_infos,
                               load_objects, select_scenes, validate_frames)
    from .stability_classifier import (assign_levels, build_windows, distribution, fit_thresholds)
    from tools.worldstream_m123.r0_task_evaluation import CHECKPOINT_SHA, sha
    out = args.output
    required = [config['infos'], config['checkpoint'], config['r0_config'], config['main_manifest'],
        'ckpts/pretrain_qformer/config.json', 'ckpts/pretrain_qformer/pytorch_model-00001-of-00002.bin',
        'ckpts/pretrain_qformer/pytorch_model-00002-of-00002.bin',
        'outputs/m123_final_quality_validation_run01/reference_policy.json']
    required += [str(Path(config['data_root']) / 'v1.0-trainval' / p) for p in
                 ('sample.json', 'sample_annotation.json', 'category.json', 'instance.json', 'attribute.json')]
    missing = [p for p in required if not (ROOT / p).exists()]
    modules = ('numpy', 'yaml', 'matplotlib', 'torch', 'mmcv', 'mmdet', 'mmdet3d',
               'transformers', 'peft', 'nuscenes', 'pyquaternion', 'cv2', 'scipy')
    missing += ['module:' + p for p in modules if importlib.util.find_spec(p) is None]
    if missing:
        raise FileNotFoundError(repr(missing))
    print('LOAD_GT_METADATA', flush=True)
    infos, by = load_infos(config)
    evaluation, calibration = select_scenes(config, infos)
    if args.smoke:
        evaluation = evaluation[:1]
    validate_frames(evaluation, by)
    validate_frames(calibration, by, check_images=False)
    tokens = {t for s in evaluation + calibration for t in s['tokens']}
    print('LOAD_AUTHORITATIVE_ANNOTATIONS', len(tokens), 'frames', flush=True)
    objects = load_objects(config, tokens)
    print('BUILD_FIXED_GT_WINDOWS', flush=True)
    kwargs = dict(seconds=config['window_seconds'], max_gap=config['max_transition_gap_seconds'])
    cal_windows = build_windows(calibration, by, objects, **kwargs)
    windows = build_windows(evaluation, by, objects, **kwargs)
    thresholds = fit_thresholds(cal_windows, config['threshold_quantiles'], config['thresholds'])
    assign_levels(windows, thresholds['thresholds'])
    thresholds.update(calibration_scene_ids=[s['scene_id'] for s in calibration],
                      evaluation_scene_ids=[s['scene_id'] for s in evaluation],
                      disjoint_scenes=not bool({s['scene_id'] for s in calibration} & {s['scene_id'] for s in evaluation}),
                      independent_TEST=False, dataset_role='existing nuScenes val; no independent test split')
    write(out / 'stability_thresholds.json', thresholds)
    dump_jsonl(out / 'calibration_windows.jsonl', cal_windows)
    dump_jsonl(out / 'test_windows.jsonl', windows)
    csv_write(out / 'stability_distribution.csv', distribution(windows, thresholds['thresholds']))
    # Classify against complete scenes before restricting the inference smoke prefix.
    measured = [dict(s, tokens=s['tokens'][:config['smoke_frames']]) for s in evaluation] if args.smoke else evaluation
    measured_tokens = {t for s in measured for t in s['tokens']}
    query_rows = build_query_records(config, measured, by, objects)
    dump_jsonl(out / 'query_ground_truth.jsonl', query_rows)
    dump_jsonl(out / 'evaluation_objects.jsonl', [dict(frame_id=t, objects=objects[t]) for t in sorted(measured_tokens)])
    coverage = coverage_report(query_rows, {t: objects[t] for t in measured_tokens})
    coverage.update(complete_GT_windows=len(windows), partial_windows=sum(not w['complete'] for w in windows),
                    calibration_frames=sum(len(s['tokens']) for s in calibration),
                    evaluation_full_scene_frames=sum(len(s['tokens']) for s in evaluation),
                    measured_frames=len(measured_tokens), smoke=args.smoke)
    write(out / 'data_coverage.json', coverage)
    warmup_scene = calibration[0]
    if len(warmup_scene['tokens']) < config['warmup_frames']:
        raise ValueError('Insufficient warmup frames')
    warmup = dict(warmup_scene, tokens=warmup_scene['tokens'][:config['warmup_frames']])
    validate_frames([warmup], by)
    if set(warmup['tokens']) & measured_tokens:
        raise ValueError('Warmup must be disjoint from measured frames; use disjoint calibration scenes')
    data_files = {ROOT / cam['data_path'] for t in measured_tokens | set(warmup['tokens'])
                  for cam in by[t]['cams'].values()}
    for token in measured_tokens | set(warmup['tokens']):
        for folder in ('conv', 'vqa'):
            path = ROOT / config['data_root'] / folder / 'val' / (token + '.json')
            if not path.is_file():
                raise FileNotFoundError(path)
            data_files.add(path)
    print('VERIFY_FROZEN_WEIGHTS_AND_INPUTS', flush=True)
    checkpoint_sha = sha(ROOT / config['checkpoint'])
    if checkpoint_sha != CHECKPOINT_SHA:
        raise ValueError('Checkpoint differs from the frozen R0 checkpoint')
    sources = {p for folder in ('worldstream', 'projects', 'tools/environment_stability_eval',
                              'tools/worldstream_m123', 'tools/worldstream_ablation')
               for p in (ROOT / folder).rglob('*.py') if 'outputs' not in p.parts}
    sources.update(ROOT / p for p in required if p != config['checkpoint'])
    sources.add(args.config.resolve())
    if getattr(args, 'driver_environment', {}).get('manifest'):
        sources.add(Path(args.driver_environment['manifest']))
    manifest = dict(status='PREPARED_NO_INFERENCE', config=config, smoke=args.smoke,
        scenes=measured, classification_scenes=evaluation, warmup_scene=warmup,
        frames=len(measured_tokens), methods=list(METHODS), checkpoint_sha256=checkpoint_sha,
        thresholds=thresholds, cpu_affinity=sorted(os.sched_getaffinity(0)),
        driver_environment=getattr(args, 'driver_environment', {}),
        source_and_GT_hashes={str(p): sha(p) for p in sorted(sources)},
        input_hashes={str(p): sha(p) for p in sorted(data_files)},
        frozen_artifact_hashes={p.name: sha(p) for p in out.iterdir() if p.is_file()},
        timing_definition='CUDA-synchronized wall time: image read/preprocessing + prompt encoding + H2D + full model + export; excludes model loading, warmup, offline scoring, input auditing and receipt serialization.',
        fps_definition='Frames / measured processing-segment wall time, including receipt/instrumentation gaps; independent of mean E2E.',
        compute_definition='Reused Meter: actually executed Linear/Conv2d and attention MACs; includes EVA functional QKV, LoRA, global suffix. Excludes elementwise/memory/sampling; not hardware FLOPs.',
        query_definition='Same existing three fixed facts + up to two original conv/vqa queries; same order every method.',
        repeat_definition='One chronological paired pass (repeat=0), as frozen main; optional configured repeats use matched request RNG.',
        warmup_definition='Four disjoint calibration keyframes per method by default, following the frozen main 4-frame pilot size; reset at sequence boundary afterward.',
        grouping_definition='Atomic annotation-set presence + class + nonempty attributes + visible ego-relative sectors + all comparable visible object-pair sectors; equal atomic weighting.',
        partial_windows='All frames still inferred; incomplete trailing windows remain Unclassified and are counted separately.',
        independent_TEST=False)
    write(out / 'manifest.json', manifest)
    write(out / 'experiment_summary.json', dict(status='PREPARED_NO_INFERENCE', manifest=manifest,
        data_coverage=coverage, metrics_available=False,
        limitation='GT coverage/grouping only; no model speed or accuracy result has been measured.'))
    print('PREPARED', len(measured_tokens), 'frames/method; thresholds', thresholds['thresholds'], flush=True)
    return manifest


def launch(args):
    # Each method gets a fresh interpreter; a busy GPU is only observed/waited for.
    print('WAIT_PROJECT_GPU_LEASE (existing jobs are not interrupted)', flush=True)
    with gpu_lease() as lease:
        _launch_locked(args, lease)


def _launch_locked(args, lease):
    for method in METHODS:
        directory = args.output / method
        directory.mkdir(exist_ok=False)
        command = ['taskset', '-c', '0-3,6-31', PYTHON, '-B', '-u', '-X', 'faulthandler',
                   '-m', 'tools.environment_stability_eval.run', '--_worker', method,
                   '--output', str(args.output.resolve()), '--_lock-fd', str(lease)]
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1')
        with (directory / 'runtime.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=(lease,))
            write(directory / 'process.json', dict(pid=process.pid, command=command))
            print('STARTED', method, 'PID', process.pid, 'log', directory / 'runtime.log', flush=True)
            code = process.wait()
        if code:
            detail = '\n'.join((directory / 'runtime.log').read_text(errors='replace').splitlines()[-15:])[-4000:]
            raise RuntimeError(f'{method} worker exited {code}; see {directory}/runtime.log\n{detail}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true', help='Real GT preparation/coverage; no model or CUDA')
    mode.add_argument('--smoke', action='store_true', help='Only the paired short GPU smoke prefix')
    mode.add_argument('--run-full', action='store_true', help='Explicit opt-in to the full GPU experiment')
    mode.add_argument('--report-only', action='store_true', help='Score completed receipts into previously absent report files')
    parser.add_argument('--_worker', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--_lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    require_affinity()
    if args.smoke or args.run_full or args._worker:
        from .gpu_environment import ensure_gpu_environment
        args.driver_environment = ensure_gpu_environment(sys.argv[1:])
    if args._worker:
        from .evaluator import worker
        worker(args.output.resolve(), args._worker, args._lock_fd)
        return
    if args.report_only:
        if args.output is None:
            parser.error('--report-only requires --output')
        from .metrics import report
        report(args.output.resolve())
        return
    config = load_config(args.config)
    args.output = args.output or ROOT / config['output_root'] / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        prepare(args, config)
        if args.smoke or args.run_full:
            launch(args)
            from .metrics import report
            report(args.output)
    except Exception:
        write(args.output / 'failure.json', dict(traceback=traceback.format_exc()))
        raise
    print('OUTPUT', args.output, flush=True)


if __name__ == '__main__':
    main()
