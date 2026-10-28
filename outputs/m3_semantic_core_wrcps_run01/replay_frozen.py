"""Replay declared real inputs in a NEW directory; dry-run by default.

Run from the repository root with the documented taskset / OmniDrive Python.
This reproduces the bounded failed campaign, not a new acceptance study.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--diagnostic',action='store_true')
    args=parser.parse_args()
    if set(os.sched_getaffinity(0)) & {4,5}:parser.error('Use taskset -c 0-3,6-31')
    source=Path(__file__).resolve().parent
    target=Path(args.root)
    if target.exists():parser.error('Output must be a NEW absent directory')
    if sys.executable!='/home/qh/miniconda3/envs/OmniDrive/bin/python':
        parser.error('Use the documented OmniDrive interpreter')
    # Read the successful recorded commands, without rerunning the aborted
    # startup, reselecting scenes, or treating repeated scenes as new evidence.
    commands={r['name']:r['command'] for r in json.loads((source/'commands.json').read_text()) if r.get('returncode')==0}
    old=str(source.relative_to(Path.cwd()))
    steps=['qualification_capture','validation_extension_capture','qualify',
           'adaptation_capture','training-manifest','adapter_training',
           'quality_native','quality_zero','quality_native_official',
           'quality_zero_official','analyze']
    if args.diagnostic:steps.append('failure_diagnostic')
    if args.execute:
        target.mkdir(parents=True)
        for name in ('experiment_plan.json','qualification_sequence.json',
                     'validation_extension_sequence.json','qualification_extensions.json',
                     'calibration_frozen_rules.json','acceptance_reserved_sequence.json',
                     'adaptation_sequence.json','quality_sequence.json',
                     'failure_diagnostic_sequence.json'):
            shutil.copyfile(source/name,target/name)
    for step in steps:
        if step in commands:
            command=[v.replace(old,str(target)) for v in commands[step]]
        elif step=='analyze':
            command=['taskset','-c','0-3,6-31',sys.executable,'-m',
                     'tools.worldstream_m123.wrcps_semantic_quality','--root',str(target),'--stage','analyze']
        else:
            command=['taskset','-c','0-3,6-31',sys.executable,'-m',
                     'tools.worldstream_m123.wrcps_semantic_experiment','--root',str(target),'--stage',step]
        print(shlex.join(command),flush=True)
        if not args.execute:continue
        from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
        from tools.worldstream_m2.global_residual_benchmark import execute,wait_gpu
        with gpu_job_lock():
            wait_gpu()
            execute(target,step,command[4:])
        if step=='qualify':
            passed=json.loads((target/'stable_core_analysis.json').read_text())['stage_one_pass']
            if not passed:raise RuntimeError('New qualification failed; preserve evidence and stop')
    if args.execute:
        summary=json.loads((target/'zero_quality_summary.json').read_text())
        print('QUALITY_GATE',summary['stage_two_pass'],'No ExactKV/performance campaign scheduled.',flush=True)


if __name__=='__main__':main()
