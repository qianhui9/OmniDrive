"""Reproducible B0/B1/B2/B3 runner; never changes production approval evidence.

taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
    -m tools.worldstream_m123.benchmark.run
"""
import argparse
import fcntl
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from .common import write
from .preflight import inventory
from .report import generate, read_json


def wait_gpu(root):
    """Only observe other users' processes; never signal or terminate them."""
    while True:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            raise RuntimeError('GPU visibility failed: '+result.stdout)
        if int(result.stdout.splitlines()[0]) < 500:
            return
        print('WAITING_FOR_GPU', result.stdout.strip(), 'MiB; other processes untouched', flush=True)
        write(root/'driver_status.json', dict(status='WAITING_FOR_GPU', memory_used_mib=result.stdout.strip()))
        time.sleep(20)


def run_execution_audit(args, root):
    """One job through the existing worker; never schedules baselines or full split."""
    from .audit import audit, execution_summary
    dest = root/'experiments/B3_execution_audit'
    if dest.exists() or (root/'preflight.json').exists():
        raise FileExistsError('Execution audit requires a new run directory; existing evidence is preserved')
    test_log = root/'raw_logs/regression_tests.log'
    tests = ['taskset', '-c', '0-3,6-31', sys.executable, '-m', 'pytest', '-q', '-rs',
             'tests/worldstream_m1', 'tests/worldstream_m2', 'tests/worldstream_m3', 'tests/worldstream_m123']
    if args.regression_log:
        shutil.copy2(args.regression_log, test_log)
        passed = ' passed' in test_log.read_text() and ' failed' not in test_log.read_text()
    else:
        with test_log.open('w') as handle:
            passed = subprocess.run(tests, stdout=handle, stderr=subprocess.STDOUT).returncode == 0
    if not passed:
        write(root/'driver_status.json', dict(status='FAILED', stage='unit_tests'))
        return 1
    # Check resources and dispatch a seeded CUDA kernel, not just import torch.
    wait_gpu(root)
    preflight = inventory(root, args.checkpoint, args.seed)
    write(root/'plan.json', dict(mode='execution_audit', config='projects/configs/OmniDrive/worldstream_m123_auto.py',
        conditions=['B3'], requested_frames=args.timing_frames, warmup=args.warmup,
        full_validation=False, production_approvals_added=0,
        task='One labelled Conv/VQA question per consecutive frame; native production generation unchanged',
        timing='analysis-only CUDA Event stream spans and synchronized request wall clock'))
    if preflight['blockers'] or args.preflight_only:
        write(root/'driver_status.json', dict(status='NOT_RUN', blockers=preflight['blockers']))
        return 2 if preflight['blockers'] else 0
    command = ['taskset', '-c', '0-3,6-31', sys.executable, '-m', 'tools.worldstream_m123.benchmark.worker',
        '--output', str(dest), '--run-root', str(root), '--condition', 'B3', '--workload', 'language',
        '--checkpoint', args.checkpoint, '--limit', str(args.timing_frames), '--warmup', str(args.warmup),
        '--seed', str(args.seed), '--measurement-set', 'smoke', '--execution-audit']
    if args.scene:
        command.extend(['--scene', args.scene])
    manifest = read_json(root/'run_manifest.json')
    manifest['commands'].extend([dict(argv=tests, reused_log=args.regression_log), dict(argv=command)])
    write(root/'run_manifest.json', manifest)
    write(root/'driver_status.json', dict(status='RUNNING', command=command))
    with (root/'raw_logs/B3_execution_audit.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            time.sleep(10)
            print('AUDIT_PROGRESS', read_json(dest/'progress.json'), flush=True)
    summary = execution_summary(dest)
    reconciliation = audit(root, hash_checkpoint=True)
    passed = child.returncode == 0 and summary['status'] == reconciliation['status'] == 'PASS'
    write(root/'driver_status.json', dict(status='COMPLETED' if passed else 'FAILED',
        returncode=child.returncode, measured_frames=summary['frames'], audit=reconciliation['status']))
    print('EXECUTION_REPORT', dest/'execution_report.md', flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir')
    parser.add_argument('--checkpoint', default='ckpts/iter_10548.pth')
    parser.add_argument('--subset', type=int, default=320)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timing-frames', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--regression-log', help='Reuse a completed regression log from this session')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--execution-audit', action='store_true', help='Only B3, one short continuous scene, no baselines or full validation')
    parser.add_argument('--scene', help='Optional scene token for execution-audit')
    args = parser.parse_args()
    if args.subset < 1 or args.repeats < 1 or args.warmup < 1 or args.timing_frames < 1:
        parser.error('subset, repeats and warmup must be positive')
    affinity = set(os.sched_getaffinity(0))
    if affinity & {4, 5}:
        parser.error('Launch with taskset -c 0-3,6-31; faulty CPUs 4/5 are forbidden')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    root = Path(args.run_dir or ('outputs/final_route/benchmark/'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('run_%Y%m%d_%H%M%S')))
    root.mkdir(parents=True, exist_ok=True)
    (root/'raw_logs').mkdir(exist_ok=True)
    (root/'experiments').mkdir(exist_ok=True)
    lock = (root/'driver.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    if args.execution_audit:
        return run_execution_audit(args, root)
    # Preserve earlier snapshots; --run-dir resumes, never overwrites workers.
    if not (root/'preflight.json').is_file():
        preflight = inventory(root, args.checkpoint, args.seed)
    else:
        preflight = read_json(root/'preflight.json')
    manifest = read_json(root/'run_manifest.json')
    manifest['commands'].append(dict(argv=sys.argv, cwd=str(Path.cwd())))
    write(root/'run_manifest.json', manifest)
    if args.regression_log:
        log = Path(args.regression_log)
        shutil.copy2(log, root/'raw_logs/regression_tests.log')
        if ' passed' not in log.read_text() or ' failed' in log.read_text():
            raise RuntimeError('Supplied regression log is not a passing test run')
    elif not (root/'raw_logs/regression_tests.log').is_file():
        command = ['taskset', '-c', '0-3,6-31', sys.executable, '-m', 'pytest',
                   'tests/worldstream_m1', 'tests/worldstream_m2', 'tests/worldstream_m3',
                   'tests/worldstream_m123', '-q']
        with (root/'raw_logs/regression_tests.log').open('w') as handle:
            code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode
        if code:
            write(root/'driver_status.json', dict(status='FAILED', reason='Regression tests failed'))
            generate(root)
            return 1
    integrity = preflight['dataset']['tables'].get('validation_annotation_integrity', {})
    main_allowed = integrity.get('status') == 'PASS'
    jobs = []
    if main_allowed:
        jobs.extend(dict(name=f'{c}_full_planning', condition=c, workload='planning', limit=0, repeat=0, measurement_set='main')
                    for c in ('B0', 'B3'))
    jobs.extend(dict(name=f'{c}_attribution_language', condition=c, workload='language',
                     limit=args.subset, repeat=0, measurement_set='attribution')
                for c in ('B0', 'B1', 'B2', 'B3'))
    for repeat in range(args.repeats):
        # All four conditions process exactly the same continuous subset, one
        # real GT question per frame. This executes the actual three-module path.
        jobs.extend(dict(name=f'{c}_subset_language_r{repeat}', condition=c, workload='language',
                         limit=args.timing_frames, repeat=repeat, measurement_set='efficiency') for c in ('B0', 'B1', 'B2', 'B3'))
    plan = dict(full_validation=dict(status='PENDING' if main_allowed else 'NOT_RUN',
        reason=None if main_allowed else 'Official annotation links are inconsistent; valid full-split GT is a prerequisite. Neither data nor evaluator is silently repaired.',
        conditions=['B0', 'B3'], expected_frames=preflight['dataset']['frames']),
        subset_frames=args.subset, timing_repeats=args.repeats, warmup_frames=args.warmup,
        timing_frames=args.timing_frames,
        jobs=jobs, task_definition='Real Conv/VQA question, round-robin question index, native production sampling',
        caveat='Sequential first scenes sorted by token; representative coverage is described, not a random statistical sample')
    write(root/'plan.json', plan)
    generate(root)
    if preflight['blockers'] or args.preflight_only:
        write(root/'driver_status.json', dict(status='NOT_RUN', reasons=preflight['blockers'], preflight_only=args.preflight_only))
        generate(root)
        return 2 if preflight['blockers'] else 0
    failures = []
    for job in jobs:
        dest = root/'experiments'/job['name']
        if dest.exists():
            status = read_json(dest/'status.json', {}).get('status')
            while status == 'RUNNING':
                # A resumed controller can adopt an earlier child without
                # stopping or duplicating the actual GPU experiment.
                print('ADOPTING_RUNNING', job['name'], read_json(dest/'progress.json'), flush=True)
                pid = read_json(dest/'worker_pid.json', {}).get('pid')
                if pid is not None:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        write(dest/'status.json', dict(status='FAILED', reason='Prior worker exited before completing status'))
                        status = 'FAILED'
                        break
                time.sleep(10)
                status = read_json(dest/'status.json', {}).get('status')
            if status != 'COMPLETED':
                failures.append(job['name'])
            print('PRESERVED_EXISTING', job['name'], status, flush=True)
            continue
        wait_gpu(root)
        command = ['taskset', '-c', '0-3,6-31', sys.executable, '-m',
            'tools.worldstream_m123.benchmark.worker', '--output', str(dest), '--run-root', str(root),
            '--checkpoint', args.checkpoint, '--condition', job['condition'], '--workload', job['workload'],
            '--limit', str(job['limit']), '--repeat', str(job['repeat']), '--warmup', str(args.warmup),
            '--measurement-set', job['measurement_set'],
            '--seed', str(args.seed)]
        manifest['commands'].append(dict(argv=command, cwd=str(Path.cwd())))
        write(root/'run_manifest.json', manifest)
        write(root/'driver_status.json', dict(status='RUNNING', current=job['name']))
        print('START', job['name'], flush=True)
        with (root/'raw_logs'/(job['name']+'.log')).open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                time.sleep(10)
                progress = read_json(dest/'progress.json')
                print('PROGRESS', job['name'], json.dumps(progress), flush=True)
            code = child.returncode
        if dest.exists():
            write(dest/'invocation.json', {**job, 'command': command, 'returncode': code})
        if code:
            failures.append(job['name'])
        metrics = read_json(dest/'metrics.json', {})
        if job['measurement_set'] == 'main' and (metrics.get('official_status') != 'MEASURED' or metrics.get('official_map') is None):
            failures.append(job['name']+':official_evaluation_failed')
        if any(v != 'PASS' for v in metrics.get('m1_scene_integrity', {}).values()):
            failures.append(job['name']+':state_integrity_failed')
        print('END', job['name'], 'exit', code, flush=True)
        generate(root)
        # A failed learned-weight validation affects every condition. Preserve
        # the actual failure and do not spend hours producing random outputs.
        validation = read_json(dest/'checkpoint_validation.json', {})
        if validation.get('status') == 'FAIL':
            break
    write(root/'driver_status.json', dict(status='COMPLETED' if not failures and main_allowed else 'PARTIAL',
        failed_experiments=failures, official_full_validation=main_allowed,
        limitation=None if main_allowed else 'Invalid official GT links; full validation NOT_RUN'))
    generate(root)
    index = Path('outputs/final_route/experiment_report.md')
    entry = f'\n\n## 实际评测运行 {root.name}\n\n结果与限制见 [{root.name}](benchmark/{root.name}/report.md)。逐实验状态、真实数值和未完成范围均保存在该目录，不替代此前清理审计记录。\n'
    entry = f'\n\n## 实际评测运行 {root.name}\n\n结果与限制见 [{root.name}]({os.path.relpath(root / "report.md", index.parent)})。原始记录保留在该目录。\n'
    if not index.exists():
        index.write_text('# 正式路线实验报告索引\n\n历史执行契约见 [2026-10-25](../m123_execution_contract_20261025/experiment_report.md)。\n')
    if root.name not in index.read_text():
        with index.open('a') as handle:
            handle.write(entry)
    print('REPORT', root/'report.md', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
