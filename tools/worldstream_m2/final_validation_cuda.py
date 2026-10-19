"""Fresh host CUDA regression and real OmniDrive tasks; wait for other jobs."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .source_state_refinement_cuda_batch import wait_idle,PYTHON,CPUS,write

ROOT=Path('outputs/module1_final_validation_refinement')


def integration():
    from . import final_validation_integration as runner
    model,dataset=runner.load();runner.load=lambda:(model,dataset)
    for case,scenes in (('natural',2),('repeat',1)):
        sys.argv=['final-validation','--case',case,'--scenes',str(scenes),'--frames','8',
            '--warmup','1','--repeats','1','--replay-protocol',str(ROOT/'replay/protocol.json'),
            '--output',str(ROOT/'gpu_validation'/case)]
        if case=='repeat':sys.argv+=['--lease-experiment','--selective-probe']
        runner.main()


def main():
    root=ROOT/'gpu_validation';root.mkdir(exist_ok=False)
    state=dict(status='waiting_CPU_replay',steps=[],cpu_affinity=sorted(os.sched_getaffinity(0)))
    write(root/'status.json',state)
    while not (ROOT/'replay/completion.json').exists():
        print('Waiting for final paired CPU replay',flush=True);time.sleep(30)
    try:
        gpu=wait_idle(root);state['gpu']=gpu
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index']),MPLCONFIGDIR='/tmp/final_mpl',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OPENCV_FOR_THREADS_NUM='1')
        witness=subprocess.run(['taskset','-c',CPUS,PYTHON,'-c',
            'import torch,json; torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32=False; '
            'x=torch.randn(16,16); error=float(((x.cuda()@x.cuda()).cpu()-x@x).abs().max()); '
            'assert error<1e-4; print(json.dumps(dict(device=torch.cuda.get_device_name(0),max_abs=error)))'],
            env=env,capture_output=True,text=True)
        (root/'kernel_witness.txt').write_text(witness.stdout+witness.stderr)
        if witness.returncode:raise RuntimeError('CUDA witness failed')
        state['status']='running'
        steps=[('regression',['-m','pytest','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','-q',
            '-o','junit_family=legacy','--junitxml='+str(root/'regression_cuda.xml')]),
            ('complete_model',['-u','-m','tools.worldstream_m2.final_validation_cuda','--integration'])]
        for name,args in steps:
            wait_idle(root,gpu['index'])
            command=['taskset','-c',CPUS,PYTHON,*args]
            row=dict(name=name,command=command,start=datetime.now(timezone.utc).isoformat());state['steps'].append(row)
            write(root/'status.json',state)
            with (root/(name+'.log')).open('w') as handle:
                result=subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
            row['exit_code']=result.returncode;write(root/'status.json',state)
            if result.returncode:raise RuntimeError(name+' failed')
        for case in ('natural','repeat'):
            if not json.loads((root/case/'status.json').read_text()).get('quality_pass'):raise RuntimeError(case+' quality failed')
        state['status']='completed'
    except Exception as exc:
        state.update(status='failed',error=repr(exc));write(root/'status.json',state);raise
    write(root/'status.json',state)


if __name__=='__main__':
    integration() if '--integration' in sys.argv else main()
