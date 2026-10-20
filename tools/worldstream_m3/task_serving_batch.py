"""Serial authorized GPU work; never terminates an occupied GPU process."""
import json
import os
from pathlib import Path
import subprocess
from tools.worldstream_m2.source_state_refinement_cuda_batch import wait_idle,CPUS,PYTHON

ROOT=Path(os.environ.get('OMNIDRIVE_TASK_SERVING_OUTPUT','outputs/m2_freeze_m3_task_serving'))

def main():
    steps=[('regression',['-m','pytest','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','-q','-o','junit_family=legacy','--junitxml='+str(ROOT/'regression_cuda.xml')]),
        ('service',['-u','-m','tools.worldstream_m3.task_serving_profile','--phase','service','--scenes','2','--frames','4','--questions','3','--warmup','0','--repeats','1']),
        ('m2_oracle',['-u','-m','tools.worldstream_m3.m2_freeze_oracle'])]
    status=dict(status='running',steps=[])
    for name,args in steps:
        gpu=wait_idle(ROOT);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index']),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/m2m3_mpl')
        cmd=['taskset','-c',CPUS,PYTHON,*args];print('START',name,flush=True)
        row=dict(name=name,command=cmd);status['steps'].append(row)
        (ROOT/'batch_status.json').write_text(json.dumps(status,indent=2))
        with (ROOT/(name+'.log')).open('w') as log:row['exit_code']=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if row['exit_code']:
            status['status']='failed';(ROOT/'batch_status.json').write_text(json.dumps(status,indent=2));raise RuntimeError(name+' failed; raw log retained')
        print('DONE',name,flush=True)
    status['status']='completed';(ROOT/'batch_status.json').write_text(json.dumps(status,indent=2))

if __name__=='__main__':main()
