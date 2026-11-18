"""Prepare, short GPU smoke, opt-in full run, restart whole attempts, or report."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import traceback

from .protocol import ROOT, PYTHON, METHODS, read, write, require_affinity


def load_config(path):
    import yaml
    config = yaml.safe_load(path.read_text())
    config.setdefault('smoke_scenes', 1)
    config.setdefault('bootstrap_samples', 2000)
    config.setdefault('bootstrap_seed', config['seed'])
    config.setdefault('require_scene_count', None)
    config.setdefault('require_independent_logs', False)
    if config['sequence_pool'] not in ('main-scenes', 'full-val'):
        raise ValueError('Unsupported sequence pool')
    for name in ('repeats', 'sequence_count', 'smoke_frames', 'smoke_scenes'):
        if type(config[name]) is not int or config[name] < (3 if name == 'smoke_frames' else 1):
            raise ValueError('Invalid ' + name)
    if type(config['bootstrap_samples']) is not int or config['bootstrap_samples'] < 100:
        raise ValueError('Use at least 100 scene bootstrap draws')
    if type(config['bootstrap_seed']) is not int:
        raise ValueError('Invalid bootstrap seed')
    if type(config['require_independent_logs']) is not bool:
        raise ValueError('Invalid independent-log requirement')
    if config['require_scene_count'] is not None and (
            type(config['require_scene_count']) is not int or config['require_scene_count'] < 1):
        raise ValueError('Invalid required scene count')
    if config['other_questions'] != 2 or config['r0_config'] != 'configs/worldstream_r0.json':
        raise ValueError('Use frozen R0 and exactly two original QA per frame')
    if (ROOT / config['data_root']).resolve() != (ROOT / 'data/nuscenes').resolve():
        raise ValueError('Existing QA evaluator requires data/nuscenes')
    return config


def gpu_environment():
    """Use only the existing verified local bundle; never install/modify drivers."""
    from tools.environment_stability_eval.gpu_environment import nvml_probe
    result = nvml_probe()
    if result.returncode == 0:
        return dict(os.environ)
    from tools.worldstream_ablation.driver_compat import kernel_version, environment, DEFAULT_ROOT
    bundle = DEFAULT_ROOT / kernel_version()
    if not (bundle / 'manifest.json').is_file():
        raise RuntimeError('GPU driver unavailable; no existing compatible process-local bundle. ' + result.stderr)
    env = environment(bundle)
    probe = nvml_probe(env)
    if probe.returncode:
        raise RuntimeError('GPU remains unavailable with verified local libraries: ' + probe.stderr)
    return env


def launch(out):
    from tools.environment_stability_eval.common import gpu_lease
    from .data import verify_manifest
    from .protocol import read_journal, validate_rows
    from tools.worldstream_m123.r0_task_evaluation import sha
    manifest = verify_manifest(out)
    env = gpu_environment()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1')
    if env.get('OMNIDRIVE_DRIVER_COMPAT_MANIFEST'):
        write(out / 'driver_environment.json', dict(manifest=env['OMNIDRIVE_DRIVER_COMPAT_MANIFEST'],
              scope='Verified existing bundle; child-process environment only'))
    print('WAIT_GPU_LEASE: other jobs will not be interrupted', flush=True)
    # One lease spans every child process and prevents races with other project jobs.
    with gpu_lease() as lease:
        for repeat in range(manifest['effective_repeats']):
            for method in METHODS:
                base = out / 'runs' / f'repeat_{repeat:03d}' / method
                base.mkdir(parents=True, exist_ok=True)
                attempts = sorted(base.glob('attempt_*'))
                completed = []
                for d in attempts:
                    if (d / 'complete.json').exists():
                        status = read(d / 'complete.json')
                        rows = read_journal(d / 'receipts.jsonl')
                        validate_rows(manifest, rows, method, repeat)
                        if sha(d / 'receipts.jsonl') != status['receipts_sha256']:
                            raise ValueError('Modified completed journal')
                        completed.append(d)
                if completed:
                    print('KEEP_COMPLETED', repeat, method, completed[-1], flush=True)
                    continue
                directory = base / f'attempt_{len(attempts):03d}'
                directory.mkdir(exist_ok=False)
                command = ['taskset', '-c', '0-3,6-31', PYTHON, '-B', '-u', '-X', 'faulthandler',
                    '-m', 'tools.experiments.long_run_state_aging.run', '--_worker', method,
                    '--_repeat', str(repeat), '--output', str(out), '--_attempt', str(directory),
                    '--_lock-fd', str(lease)]
                write(directory / 'attempt.json', dict(created_at=datetime.now().astimezone().isoformat(),
                    method=method, repeat=repeat, command=command, full_system_state_restored=False,
                    restart_policy='New independent process from first frame; previous partial records preserved'))
                with (directory / 'runtime.log').open('x') as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, pass_fds=(lease,))
                    write(directory / 'process.json', dict(pid=process.pid, parent=os.getpid()))
                    print('STARTED', method, repeat, process.pid, directory, flush=True)
                    try:
                        code = process.wait()
                    except BaseException:
                        # Terminate only this launcher's own worker; never another job.
                        process.terminate()
                        process.wait()
                        raise
                if code:
                    raise RuntimeError(f'{method} repeat {repeat} failed ({code}); see {directory}/runtime.log')
                # Durable paired summaries after each method; no incomplete
                # attempt contributes to the paired main table.
                if not (repeat == manifest['effective_repeats'] - 1 and method == METHODS[-1]):
                    from .report import report
                    report(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--run-full', action='store_true')
    mode.add_argument('--resume', action='store_true', help='Restart incomplete method/repeat from first frame; keep completed attempts')
    mode.add_argument('--report-only', action='store_true')
    parser.add_argument('--_worker', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--_repeat', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--_attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--_lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    require_affinity()
    if args._worker:
        import faulthandler
        from .worker import worker
        with (args._attempt / 'native_crash.log').open('a') as crash:
            faulthandler.enable(file=crash, all_threads=True)
            worker(args.output, args._attempt, args._worker, args._repeat, args._lock_fd)
        return
    if args.resume or args.report_only:
        if args.output is None:
            parser.error('--resume/--report-only requires --output')
        out = args.output.resolve()
        from .data import verify_manifest
        manifest = verify_manifest(out)
    else:
        config = load_config(args.config)
        if args.repeats is not None:
            if args.repeats < 1:
                parser.error('--repeats must be positive')
            config['repeats'] = args.repeats
        out = (args.output or ROOT / config['output_root'] / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')).resolve()
        out.mkdir(parents=True, exist_ok=False)
        from .data import prepare
        manifest = prepare(out, config, args.smoke, args.config)
        manifest['effective_repeats'] = 1 if args.smoke else config['repeats']
        write(out / 'manifest.json', manifest)
    try:
        if args.smoke or args.run_full or args.resume:
            launch(out)
    except BaseException:
        write(out / 'failure.json', dict(time=datetime.now().astimezone().isoformat(), traceback=traceback.format_exc()))
        raise
    finally:
        from .report import report
        report(out)
    print('OUTPUT', out, flush=True)


if __name__ == '__main__':
    main()
