"""Fresh final-source validation after the serial exploratory batch completes."""
import json
import os
from pathlib import Path
import subprocess
import time
from tools.worldstream_m2.source_state_refinement_cuda_batch import wait_idle,CPUS,PYTHON
ROOT=Path(os.environ.get('OMNIDRIVE_TASK_SERVING_OUTPUT','outputs/m2_freeze_m3_task_serving'))

def main():
    while True:
        status=json.loads((ROOT/'batch_status.json').read_text())
        if status['status']=='completed':break
        if status['status']=='failed':raise RuntimeError('Earlier batch failed; do not compete with it')
        time.sleep(30)
    steps=[('final_regression',['-m','pytest','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','-q','-o','junit_family=legacy','--junitxml='+str(ROOT/'final_regression_cuda.xml')]),
        ('final_source_validation',['-u','-m','tools.worldstream_m3.task_serving_profile','--phase','final_source_validation','--scenes','1','--frames','1','--questions','3'])]
    state=dict(status='running',steps=[])
    for name,args in steps:
        gpu=wait_idle(ROOT);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index']),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/m2m3_mpl')
        command=['taskset','-c',CPUS,PYTHON,*args];row=dict(name=name,command=command);state['steps'].append(row)
        (ROOT/'final_validation_status.json').write_text(json.dumps(state,indent=2))
        with (ROOT/(name+'.log')).open('w') as log:row['exit_code']=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if row['exit_code']:state['status']='failed';break
    else:state['status']='completed'
    (ROOT/'final_validation_status.json').write_text(json.dumps(state,indent=2))

if __name__=='__main__':main()
