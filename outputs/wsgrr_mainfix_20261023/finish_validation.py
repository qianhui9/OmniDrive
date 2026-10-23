"""Finish the authorized sole-main validation serially, without sharing the GPU."""
import json
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
PYTHON='/home/qh/miniconda3/envs/OmniDrive/bin/python'
PREFIX=['taskset','-c','0-3,6-31']

def read(path):
    return json.loads(path.read_text()) if path.exists() else {}
def status(phase,**fields):
    value=dict(phase=phase,**fields)
    (ROOT/'finish_status.json').write_text(json.dumps(value,indent=2))
    print(value,flush=True)

status('WAITING_FULL_OFFICIAL_EVALUATION')
while True:
    worker=read(ROOT/'main_full/status.json')
    official=read(ROOT/'main_full/official_eval/status.json')
    if worker.get('status')=='FAILED' or official.get('status')=='FAILED':
        status('FAILED_PREREQUISITE',worker=worker,official=official);raise SystemExit(1)
    if worker.get('status')=='COMPLETED' and official.get('status')=='COMPLETED':break
    time.sleep(50)
assert read(ROOT/'main_full/official_eval/summary.json')['frames']==6019

# ABBA order balances the two current timing repetitions. Native quality is
# reused from the hash-checked full evaluation, not recomputed on full-val.
for name,condition,repeat in [('Native_r0','B0',0),('main_r0','B2',0),('main_r1','B2',1),('Native_r1','B0',1)]:
    while True:
        active=subprocess.check_output(PREFIX+['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
        if not active:break
        status('WAITING_GPU',next_run=name,active=active);time.sleep(50)
    status('RUNNING_REQUEST_TIMING',run=name)
    command=PREFIX+[PYTHON,'-m','tools.worldstream_m123.benchmark.worker','--output',str(ROOT/name),
        '--run-root',str(ROOT),'--condition',condition,'--workload','language',
        '--checkpoint','ckpts/iter_10548.pth','--limit','40','--warmup','1',
        '--questions-per-frame','1','--seed','0','--repeat',str(repeat),'--measurement-set','main']
    if condition=='B2':command+=['--offline-wsgrr','D','--window-budget','3']
    with (ROOT/(name+'.log')).open('w') as handle:
        code=subprocess.call(command,cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
    measured=read(ROOT/name/'metrics.json')
    if code or measured.get('frames')!=40 or measured.get('questions')!=40:
        status('FAILED_REQUEST_TIMING',run=name,returncode=code);raise SystemExit(1)
status('SUMMARIZING')
subprocess.run(PREFIX+[PYTHON,str(ROOT/'summarize.py')],cwd=REPO,check=True)
status('COMPLETED',full_validation_frames=6019,request_repeats=2,frames_per_repeat=40)
