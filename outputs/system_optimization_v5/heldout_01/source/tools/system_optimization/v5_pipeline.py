"""Finish frozen v5 GPU/CPU work in order, with observe-only GPU waiting."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .audit import write
from .v5 import ROOT


def idle_gpu():
    while True:
        proc=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                            capture_output=True,text=True)
        try:ready=proc.returncode==0 and int(proc.stdout.splitlines()[0])<500
        except (ValueError,IndexError):ready=False
        if ready:return
        print('Waiting for idle GPU; no processes stopped',proc.stdout.strip(),flush=True)
        time.sleep(20)


def run(module,log):
    print('START',module,flush=True)
    with (ROOT/'logs'/log).open('x') as handle:
        subprocess.run([sys.executable,'-u','-m',module],stdout=handle,stderr=subprocess.STDOUT,check=True)
    print('DONE',module,flush=True)


def main():
    if set(os.sched_getaffinity(0)) & {4,5}:raise RuntimeError('Forbidden CPUs')
    while True:
        state=json.loads((ROOT/'gpu_status.json').read_text())['status']
        if state=='failed':raise RuntimeError('Compact experiment failed; preserve and resolve before continuing')
        if state=='completed':break
        time.sleep(20)
    idle_gpu()
    run('tools.system_optimization.v5_system','system_gpu.log')
    # The completed model process releases memory before remaining GPU tests.
    idle_gpu()
    with (ROOT/'logs/regression_gpu.log').open('x') as handle:
        subprocess.run([sys.executable,'-m','pytest','-q','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3'],
            stdout=handle,stderr=subprocess.STDOUT,check=True)
    run('tools.system_optimization.v5_facts','facts.log')
    run('tools.system_optimization.v5_report','report.log')


if __name__=='__main__':
    write(ROOT/'pipeline_status.json',dict(status='running'))
    try:main()
    except BaseException as error:
        write(ROOT/'pipeline_status.json',dict(status='failed',error=repr(error)));raise
    else:write(ROOT/'pipeline_status.json',dict(status='completed'))
