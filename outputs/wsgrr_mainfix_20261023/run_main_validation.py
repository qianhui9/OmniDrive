"""Run the one completed final full-val evaluation after final continuous checks."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
PREFIX=['taskset','-c','0-3,6-31']
PYTHON='/home/qh/miniconda3/envs/OmniDrive/bin/python'
def state(phase,**fields):
    v=dict(phase=phase,**fields)
    (ROOT/'run_main_status.json').write_text(json.dumps(v,indent=2));print(v,flush=True)

try:
    state('WAITING_FINAL_CONTINUOUS_VALIDATION')
    while not (ROOT/'visual40_main/results.json').exists():time.sleep(50)
    visual=json.loads((ROOT/'visual40_main/results.json').read_text())
    assert visual['frames']==40
    assert all(a['cache_transition_audit']['checked'] and
        all(a['cache_transition_audit']['checks'].values()) for a in visual['natural'])
    manifest=json.loads((ROOT/'final_source_manifest.json').read_text())
    assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in manifest['files'].items())
    while subprocess.check_output(PREFIX+['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip():
        state('WAITING_GPU');time.sleep(50)
    state('RUNNING_FINAL_FULL_INFERENCE',frames=6019)
    command=PREFIX+[PYTHON,'-m','tools.worldstream_m123.benchmark.worker',
        '--output',str(ROOT/'main_full'),'--run-root',str(ROOT),
        '--condition','B2','--workload','language','--checkpoint','ckpts/iter_10548.pth',
        '--limit','0','--warmup','1','--questions-per-frame','1','--seed','0','--repeat','0',
        '--measurement-set','main','--offline-wsgrr','D','--window-budget','3','--perception-only']
    with (ROOT/'main_full.log').open('x') as handle:
        subprocess.run(command,cwd=REPO,stdout=handle,stderr=subprocess.STDOUT,check=True)
    state('WAITING_OFFICIAL_AND_REQUEST_REPEATS')
    with (ROOT/'finish_validation_final.log').open('x') as handle:
        subprocess.run(PREFIX+[PYTHON,str(ROOT/'finish_validation.py')],cwd=REPO,
            stdout=handle,stderr=subprocess.STDOUT,check=True)
    state('COMPLETED')
except BaseException as exc:
    state('FAILED',error=repr(exc));raise
