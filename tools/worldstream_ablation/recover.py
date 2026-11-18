"""Recover missing variants with frozen inference code and preserved failed attempts."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from zoneinfo import ZoneInfo

from . import run


def wait_gpu(out, once=False, own_process=False, interval=30, timeout=0):
    """Wait for NVML to recover and GPU 0 to be idle; never reset/kill a GPU job."""
    started = time.monotonic()
    while True:
        try:
            apps = subprocess.run(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'], capture_output=True,
                                  text=True, check=True, timeout=15)
            usage = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], capture_output=True,
                                   text=True, check=True, timeout=15)
            memory, utilization = [int(x.strip()) for x in usage.stdout.strip().split(',')]
            active = [p.strip() for p in apps.stdout.splitlines() if p.strip()
                      and (not own_process or p.strip() != str(os.getpid()))]
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            detail = getattr(exc, 'stderr', None) or str(exc)
            run.stage(out, 'WAIT_GPU_DRIVER', error=str(detail).strip()[:2000],
                      message='Waiting for driver access; no GPU reset or process termination')
        else:
            if not active and (own_process or (memory < 1000 and utilization < 10)):
                return
            run.stage(out, 'WAIT_GPU_IDLE', active_compute_pids=active,
                      memory_MiB=memory, utilization=utilization)
        if once or (timeout and time.monotonic() - started >= timeout):
            raise RuntimeError('GPU driver unavailable or GPU busy; no inference started')
        delay = interval if not timeout else min(interval, max(0, timeout - (time.monotonic() - started)))
        time.sleep(delay)


def verify_inputs(manifest):
    for label in ('source_hashes', 'data_hashes'):
        for path, digest in manifest[label].items():
            if run.sha(path) != digest:
                raise RuntimeError('Frozen source/input changed: ' + path)
        print('VERIFIED', label, len(manifest[label]), flush=True)
    if run.sha(manifest['checkpoint']) != manifest['checkpoint_sha256']:
        raise RuntimeError('Frozen checkpoint changed')
    print('VERIFIED checkpoint', flush=True)


def plan(out, manifest):
    tokens = [t for scene in manifest['scenes'] for t in scene['tokens']]
    completed, missing = [], []
    for name in manifest['variants']:
        directory = out / name
        marker = directory / 'inference_complete.json'
        if not marker.exists():
            missing.append(name)
            continue
        done = run.read(marker)
        if (done['status'] != 'COMPLETED' or done['smoke'] != manifest['smoke']
                or done['frames'] != len(tokens)
                or done['receipts_sha256'] != run.sha(directory / 'receipts.jsonl')):
            raise RuntimeError('Invalid completed inference; refusing replacement: ' + name)
        # Validate frame identity without loading multiple GB of receipts together.
        with (directory / 'receipts.jsonl').open() as handle:
            seen = [json.loads(line)['frame_id'] for line in handle]
        if seen != tokens:
            raise RuntimeError('Invalid completed frame order: ' + name)
        completed.append(name)
    return dict(completed=completed, missing=missing, frames_per_variant=len(tokens))


@contextmanager
def recovery_lock(out):
    with (out / '.recovery.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another recovery process is already running for this output') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare_attempt(out, name, history):
    directory = out / name
    if (directory / 'inference_complete.json').exists():
        raise RuntimeError('Refusing to archive completed inference: ' + name)
    if directory.exists():
        if not any((directory / n).exists() for n in ('failure.json', 'exception.json')):
            # An unmarked partial directory may belong to a live worker. Do not
            # guess from a PID in a different namespace or overwrite that work.
            raise RuntimeError('Partial directory has no failure marker; inspect its worker before recovery: ' + name)
        history.mkdir(parents=True, exist_ok=True)
        directory.rename(history / name)
    directory.mkdir()
    return directory


def worker(args):
    original_wait = run.wait_gpu
    run.wait_gpu = lambda out, **kw: wait_gpu(out, interval=args.poll_seconds,
                                             timeout=args.gpu_wait_timeout, **kw)
    try:
        run.worker(args)
    except BaseException as exc:
        run.write(args.output / args.variant / 'exception.json',
                  dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        run.wait_gpu = original_wait


def launch(args, name, directory):
    cmd = ['taskset', '-c', '0-3,6-31', run.PYTHON, '-B', '-u', '-X', 'faulthandler',
           '-m', 'tools.worldstream_ablation.recover', '--worker', '--variant', name,
           '--output', str(args.output), '--poll-seconds', str(args.poll_seconds),
           '--gpu-wait-timeout', str(args.gpu_wait_timeout)]
    run.write(directory / 'command.json', cmd)
    with (directory / 'runtime.log').open('x') as handle:
        proc = subprocess.Popen(cmd, cwd=run.ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        run.write(directory / 'worker.json', dict(pid=proc.pid, command=cmd))
        try:
            for line in proc.stdout:
                handle.write(line)
                handle.flush()
                print(line, end='', flush=True)
            code = proc.wait()
        except BaseException:
            # Only stop the child launched by this recovery process.
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            run.write(directory / 'failure.json', dict(error='Recovery launcher interrupted'))
            raise
    if code:
        run.write(directory / 'failure.json', dict(returncode=code))
        raise RuntimeError(f'{name} exited {code}; see {directory}/runtime.log; rerun this recovery command to retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/worldstream_ablation_full_run01'))
    parser.add_argument('--dry-run', action='store_true', help='Verify hashes and list missing variants; no output changes or GPU use')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--gpu-wait-timeout', type=float, default=0, help='0 waits indefinitely for driver/GPU availability')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--variant', choices=run.VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(run.ROOT)
    args.output = args.output.resolve()
    if {4, 5} & os.sched_getaffinity(0):
        parser.error('Use taskset -c 0-3,6-31')
    if args.poll_seconds <= 0 or args.gpu_wait_timeout < 0:
        parser.error('poll-seconds must be positive; timeout must be nonnegative')
    if args.worker:
        worker(args)
        return
    manifest = run.read(args.output / 'manifest.json')
    if manifest['smoke']:
        parser.error('Recovery is for full-run outputs only')
    verify_inputs(manifest)
    if args.dry_run:
        print(json.dumps(plan(args.output, manifest), ensure_ascii=False, indent=2), flush=True)
        return
    with recovery_lock(args.output):
        pending = plan(args.output, manifest)
        print(json.dumps(pending, ensure_ascii=False, indent=2), flush=True)
        attempt = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        history = args.output / 'recovery_history' / attempt
        history.mkdir(parents=True)
        compatibility = os.environ.get('OMNIDRIVE_DRIVER_COMPAT_MANIFEST')
        run.write(history / 'recovery.json', dict(**pending, manifest_sha256=run.sha(args.output / 'manifest.json'),
            recovery_code_sha256=run.sha(__file__),
            recovery_shell_sha256=run.sha(Path(__file__).with_name('recover.sh')),
            summary_code_sha256=run.sha(Path(__file__).with_name('summarize_results.py')),
            driver_compat_manifest=compatibility,
            driver_compat_manifest_sha256=run.sha(compatibility) if compatibility else None,
            LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH', ''),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            note='Only driver waiting and launch/recovery changed; original inference worker and policies retained'))
        for name in pending['missing']:
            directory = prepare_attempt(args.output, name, history)
            launch(args, name, directory)
        # This validates every variant and regenerates the IDE's Chinese report.
        from .summarize_results import evaluate, render
        m, metrics, audit, tags = evaluate(args.output)
        render(args.output, m, metrics, audit, tags)
        if audit['status'] != 'COMPLETE':
            raise RuntimeError('Recovery did not produce all planned variants')
        run.write(args.output / 'completion.json', dict(status='COMPLETED', smoke=False,
            variants=list(metrics), frames_per_variant=manifest['frames'], report='结果.md',
            recovery_history=str(history.relative_to(args.output))))
        print('RECOVERY_COMPLETED', args.output / '结果.md', flush=True)


if __name__ == '__main__':
    main()
