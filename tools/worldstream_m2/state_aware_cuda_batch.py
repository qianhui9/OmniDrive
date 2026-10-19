"""Fresh CUDA regression and full VLM comparisons, with no GPU preemption."""
from datetime import datetime,timezone
import json
import os
import subprocess
from .source_state_refinement_cuda_batch import wait_idle,CPUS,PYTHON
from .state_aware_feasibility import ROOT
from tools.worldstream_m1.io_utils import write_json


def main():
    root=ROOT/'gpu_validation';root.mkdir(exist_ok=False)
    state=dict(status='preflight',started=datetime.now(timezone.utc).isoformat(),steps=[],
        python=PYTHON,affinity=sorted(os.sched_getaffinity(0)))
    write_json(root/'status.json',state)
    try:
        gpu=wait_idle(root);state['gpu']=gpu
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index']),OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/m2_state_aware_mpl')
        for name,args in [('regression',['-m','pytest','tests/worldstream_m1','tests/worldstream_m2',
                'tests/worldstream_m3','-q','-o','junit_family=legacy','--junitxml='+str(root/'regression_cuda.xml')]),
            ('integration',['-u','-m','tools.worldstream_m2.state_aware_integration']),
            ('execution_profile',['-u','-m','tools.worldstream_m2.state_aware_execution_profile'])]:
            wait_idle(root,gpu['index'])
            row=dict(name=name,command=['taskset','-c',CPUS,PYTHON,*args],started=datetime.now(timezone.utc).isoformat())
            state.update(status='running');state['steps'].append(row);write_json(root/'status.json',state)
            print('Starting',name,flush=True)
            with (root/(name+'.log')).open('w') as log:
                run=subprocess.run(row['command'],env=env,stdout=log,stderr=subprocess.STDOUT)
            row.update(exit_code=run.returncode,finished=datetime.now(timezone.utc).isoformat());write_json(root/'status.json',state)
            if run.returncode:raise RuntimeError(name+' failed; see archived log')
            print('Finished',name,flush=True)
        state['status']='completed'
    except Exception as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:
        state['finished']=datetime.now(timezone.utc).isoformat();write_json(root/'status.json',state)


if __name__=='__main__':main()
