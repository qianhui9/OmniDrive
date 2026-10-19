"""Wait for timed GPU work, verify final-source regression, then archive."""
import json
import os
import subprocess
import time
from .state_aware_feasibility import ROOT
from .source_state_refinement_cuda_batch import wait_idle,CPUS,PYTHON
from tools.worldstream_m1.io_utils import write_json


def main():
    root=ROOT/'gpu_validation'
    while True:
        state=json.loads((root/'status.json').read_text())
        if state['status']=='completed':break
        if state['status']=='failed':raise RuntimeError('Primary GPU batch failed; inspect preserved logs')
        time.sleep(30)
    gpu=wait_idle(root)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index']),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MPLCONFIGDIR='/tmp/m2_state_aware_mpl')
    status=dict(status='running',steps=[]);write_json(root/'final_source_status.json',status)
    try:
        for name,args in [('final_regression',['-m','pytest','tests/worldstream_m1','tests/worldstream_m2',
                'tests/worldstream_m3','-q','-o','junit_family=legacy','--junitxml='+str(root/'regression_final_cuda.xml')]),
                ('finalize',['-m','tools.worldstream_m2.state_aware_finalize'])]:
            command=['taskset','-c',CPUS,PYTHON,*args];print('Starting',name,flush=True)
            with (root/(name+'.log')).open('w') as log:
                run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            status['steps'].append(dict(name=name,command=command,exit_code=run.returncode))
            write_json(root/'final_source_status.json',status)
            if run.returncode:raise RuntimeError(name+' failed; inspect log')
        status['status']='completed'
    except Exception as exc:
        status.update(status='failed',error=repr(exc));raise
    finally:write_json(root/'final_source_status.json',status)


if __name__=='__main__':main()
