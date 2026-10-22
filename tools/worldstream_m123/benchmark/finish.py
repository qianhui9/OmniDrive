"""Finish audit and regenerate reports after a durable benchmark controller exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.worldstream_m123.stateful_opportunity.common import sha256, write
from .audit import audit
from .report import generate, read_json


def process_alive(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        return stat.rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def recover_finished_workers(root):
    for dest in sorted((root/'experiments').glob('*')):
        status = read_json(dest/'status.json', {})
        recovery = read_json(dest/'recovery_status.json', {})
        if (status.get('status') == 'FAILED' and 'Out of range float values' in status.get('error', '')
                and recovery.get('status') != 'FAILED'):
            command = ['taskset', '-c', '0-3,6-31', sys.executable, '-u', '-m',
                       'tools.worldstream_m123.benchmark.recover', '--experiment', str(dest)]
            print('RECOVER_SAVED_OUTPUTS', dest.name, flush=True)
            with (root/'raw_logs'/(dest.name+'_recovery.log')).open('a') as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            print('RECOVERY_EXIT', dest.name, code, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    pid = read_json(root/'controller_pid.json')['pid']
    while process_alive(pid):
        recover_finished_workers(root)
        status = read_json(root/'driver_status.json', {})
        print('CONTROLLER', pid, status, flush=True)
        time.sleep(30)
    print('CONTROLLER_EXITED', pid, flush=True)
    recover_finished_workers(root)
    controller_status = read_json(root/'driver_status.json', {}).get('status')
    if controller_status not in ('COMPLETED', 'PARTIAL', 'FAILED', 'NOT_RUN'):
        write(root/'driver_status.json', dict(status='FAILED', reason='Controller exited without final status; raw logs retained'))
    check = subprocess.run(['taskset', '-c', '0-3,6-31', 'git', 'diff', '--check'],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (root/'raw_logs/git_diff_check.log').write_text(check.stdout)
    verification = audit(root, hash_checkpoint=True)
    jobs = read_json(root/'plan.json', {}).get('jobs', [])
    all_complete = bool(jobs) and all(read_json(root/'experiments'/job['name']/'status.json', {}).get('status') == 'COMPLETED' for job in jobs)
    main_measured = all(read_json(root/'experiments'/job['name']/'metrics.json', {}).get('official_status') == 'MEASURED'
                        and read_json(root/'experiments'/job['name']/'metrics.json', {}).get('official_map') is not None
                        for job in jobs if job.get('measurement_set') == 'main')
    if all_complete and main_measured and verification['status'] == 'PASS' and check.returncode == 0:
        original_status = read_json(root/'driver_status.json', {})
        write(root/'driver_status_before_final_audit.json', original_status)
        write(root/'driver_status.json', dict(status='COMPLETED', all_planned_experiments_verified=True,
            prior_controller_status=original_status, recovery_history_preserved=True))
        controller_status = 'COMPLETED'
    verification['git_diff_check'] = dict(returncode=check.returncode, output=check.stdout)
    verification['source_hashes_at_finish'] = {str(p): sha256(p) for p in Path('tools/worldstream_m123/benchmark').glob('*.py')}
    write(root/'verification.json', verification)
    generate(root)
    index = Path('outputs/final_route/experiment_report.md')
    if root.name not in index.read_text():
        with index.open('a') as handle:
            handle.write(f'\n\n## 实际评测 {root.name}\n\n结果、完成范围与限制见 [{root.name}](benchmark/{root.name}/report.md)。原清理审计记录保留。\n')
    with (root/'report.md').open('a') as handle:
        handle.write('\n## 完成后的独立审计\n\n'
            f'已运行实验的记录核对：`{verification["status"]}`；'
            f'`git diff --check` 返回 {check.returncode}。权重及保护文件校验见 verification.json。'
            '此审计不会将未完成或失败的全量实验改写为已完成。\n')
    write(root/'finalizer_status.json', dict(status='COMPLETED', audit=verification['status'],
                                            controller_status=controller_status))
    print('FINAL_REPORT', root/'report.md', verification['status'], flush=True)


if __name__ == '__main__':
    main()
