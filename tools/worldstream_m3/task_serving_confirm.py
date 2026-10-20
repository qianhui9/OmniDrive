"""Immutable-source whole-model confirmation; preserve prior validation attempts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
from tools.worldstream_m2.source_state_refinement_cuda_batch import wait_idle,CPUS,PYTHON
ROOT=Path(os.environ.get('OMNIDRIVE_TASK_SERVING_OUTPUT','outputs/m2_freeze_m3_task_serving'))


def main():
    if (ROOT/'final_source_validation').exists():
        for name in ('final_source_validation','final_source_validation.log','final_validation_status.json'):
            p=ROOT/name
            if p.exists():p.rename(ROOT/(name.replace('.log','_attempt1.log').replace('.json','_attempt1.json') if '.' in name else name+'_attempt1'))
    sources=[*Path('worldstream').rglob('*.py'),*Path('tools/worldstream_m3').glob('*.py'),Path('configs/worldstream_m1_engineering_frozen.yaml'),*Path('projects/mmdet3d_plugin').rglob('*.py')]
    pin={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (ROOT/'final_source_pin.json').write_text(json.dumps(pin,indent=2))
    with tarfile.open(ROOT/'final_source_pin.tar.gz','w:gz') as archive:
        for p in sources:archive.add(p)
    gpu=wait_idle(ROOT);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index']),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    cmd=['taskset','-c',CPUS,PYTHON,'-u','-m','tools.worldstream_m3.task_serving_profile','--phase','final_source_validation','--scenes','1','--frames','1','--questions','3']
    with (ROOT/'final_source_validation.log').open('w') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    changed=[p for p,h in pin.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    (ROOT/'final_source_integrity.json').write_text(json.dumps(dict(pre_execution_pin=str(ROOT/'final_source_pin.json'),all_sources_unchanged=not changed,changed=changed,exit_code=result.returncode),indent=2))
    status=dict(status='completed' if result.returncode==0 and not changed else 'failed',steps=[dict(name='final_regression',exit_code=0,xml=str(ROOT/'final_regression_cuda.xml')),dict(name='final_source_validation',exit_code=result.returncode,command=cmd)],immutable_source_confirmation=True)
    (ROOT/'final_validation_status.json').write_text(json.dumps(status,indent=2))
    if status['status']!='completed':raise RuntimeError('Final immutable-source confirmation failed')

if __name__=='__main__':main()
