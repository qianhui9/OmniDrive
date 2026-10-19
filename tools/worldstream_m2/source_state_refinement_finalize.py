"""Wait for the primary job and an idle GPU, then verify final source revisions."""
import json
import os
from pathlib import Path
import subprocess
import time
from .state_interface_cuda_batch import wait_idle,write,PYTHON,CPUS


def main():
    root=Path('outputs/module1_source_state_refinement/gpu_validation_v2')
    state=dict(status='waiting_for_primary',steps=[])
    write(root/'final_source_status.json',state)
    while True:
        primary=json.loads((root/'status.json').read_text())
        if primary['status']=='completed':break
        if primary['status'] in ('failed','blocked'):raise RuntimeError('Primary GPU job did not complete')
        print('Waiting for primary full-task comparison before final-source checks',flush=True)
        time.sleep(30)
    # Final scope metadata uses exact historical descriptor crops. Rerun the
    # full M1 population before GPU timings to keep both measurements isolated.
    cpu_command=['taskset','-c',CPUS,PYTHON,'-u','-m','tools.worldstream_m1.source_state_refinement_replay',
        '--scenes','8','--frames','12','--output','outputs/module1_source_state_refinement/replay_final']
    cpu_env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/state_mpl')
    with (root/'replay_final.log').open('w') as handle:
        completed=subprocess.run(cpu_command,env=cpu_env,stdout=handle,stderr=subprocess.STDOUT)
    if completed.returncode:raise RuntimeError('Final CPU scope replay failed')
    selected=wait_idle(root)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(selected['index']),OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/state_mpl',OPENCV_FOR_THREADS_NUM='1')
    commands=[('regression_current',['-m','pytest','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3',
        '-q','-o','junit_family=legacy','--junitxml='+str(root/'regression_current_cuda.xml')]),
        ('current_full_task',['-u','-m','tools.worldstream_m2.source_state_refinement_current_recheck'])]
    state['status']='running'
    try:
        for name,args in commands:
            wait_idle(root,selected['index'])
            command=['taskset','-c',CPUS,PYTHON,*args]
            row=dict(name=name,command=command);state['steps'].append(row);write(root/'final_source_status.json',state)
            with (root/(name+'.log')).open('w') as handle:
                result=subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
            row['exit_code']=result.returncode
            if result.returncode:raise RuntimeError(name+' failed')
        for folder in ('current_source_recheck','current_repeat_recheck'):
            if json.loads((root/folder/'status.json').read_text()).get('quality_pass') is not True:
                raise RuntimeError(folder+' failed quality gate')
        state['status']='completed'
    except Exception as exc:
        state.update(status='failed',error=repr(exc));write(root/'final_source_status.json',state);raise
    write(root/'final_source_status.json',state)


if __name__=='__main__':main()
