"""Queue the requested CUDA regression after complete-model GPU ownership ends."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from tools.worldstream_m2.source_state_refinement_cuda_batch import wait_idle
from tools.worldstream_m1.io_utils import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/m3_state_guided_verified_decode')
    parser.add_argument('--finalize',action='store_true');args=parser.parse_args();root=Path(args.output)
    dest=root/'regression_preflight';dest.mkdir(exist_ok=False)
    deadline=time.monotonic()+4*3600
    while not (root/'validation_all/status.json').exists():
        log=(root/'validation.log').read_text() if (root/'validation.log').exists() else ''
        if 'Traceback (most recent call last)' in log:
            raise RuntimeError('Complete-model validation failed; see validation.log')
        if time.monotonic()>deadline:raise TimeoutError('Complete-model validation has not finished')
        time.sleep(15)
    validation=json.loads((root/'validation_all/status.json').read_text())
    if validation.get('completed') is not True:raise RuntimeError('Incomplete validation')
    wait_idle(dest,None)
    command=[sys.executable,'-m','pytest','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','-q',
             '--junitxml='+str(root/'regression_cuda.xml')]
    with (root/'regression.log').open('x') as log:
        status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    write_json(dest/'status.json',dict(completed=True,returncode=status,command=command))
    if status:raise SystemExit(status)
    if args.finalize:
        subprocess.run([sys.executable,'-m','tools.worldstream_m3.sgvd.finalize','--output',str(root)],check=True)


if __name__=='__main__':main()
