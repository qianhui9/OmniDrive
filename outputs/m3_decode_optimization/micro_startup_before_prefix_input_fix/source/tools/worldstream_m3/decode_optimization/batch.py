"""Serialize follow-up workloads after the full service; never compete for GPU."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('outputs/m3_decode_optimization')
PY='/home/qh/miniconda3/envs/OmniDrive/bin/python'
CPUS='0-3,6-31'


def main():
    while not (ROOT/'service/status.json').exists():
        if 'Traceback (most recent call last)' in (ROOT/'service_final.log').read_text(errors='replace'):
            raise RuntimeError('Service failed; inspect retained log')
        time.sleep(5)
    for name,args in [
        ('micro',['-m','tools.worldstream_m3.decode_optimization.micro']),
        ('prefix',['-m','tools.worldstream_m3.decode_optimization.prefix']),
        ('regression',['-m','pytest','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','-q',
                       '--junitxml='+str(ROOT/'regression_cuda.xml')])]:
        with (ROOT/(name+'_final.log')).open('w') as f:
            result=subprocess.run(['taskset','-c',CPUS,PY,*args],stdout=f,stderr=subprocess.STDOUT)
        (ROOT/(name+'_exit.json')).write_text(json.dumps(dict(exit_code=result.returncode)))
        if result.returncode:raise RuntimeError(name+' failed; raw log retained')
    print('All sequential CUDA workloads completed',flush=True)


if __name__=='__main__':main()
