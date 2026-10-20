"""Serialize follow-up workloads after the full service; never compete for GPU."""
import json
import os
from pathlib import Path
import subprocess
import time
import hashlib

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
        if name=='regression':
            from tools.worldstream_m2.source_state_refinement_cuda_batch import wait_idle
            resource=ROOT/'regression_resources';resource.mkdir(exist_ok=True)
            wait_idle(resource,None)
            test_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for group in ('m1','m2','m3') for p in Path('tests/worldstream_'+group).rglob('*.py')}
            (ROOT/'regression_sources.json').write_text(json.dumps(test_sources,indent=2))
        with (ROOT/(name+'_final.log')).open('w') as f:
            result=subprocess.run(['taskset','-c',CPUS,PY,*args],stdout=f,stderr=subprocess.STDOUT)
        (ROOT/(name+'_exit.json')).write_text(json.dumps(dict(exit_code=result.returncode)))
        if name=='regression':
            (ROOT/'regression_source_integrity.json').write_text(json.dumps(dict(unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in test_sources.items()))))
        if result.returncode:raise RuntimeError(name+' failed; raw log retained')
    print('All sequential CUDA workloads completed',flush=True)


if __name__=='__main__':main()
