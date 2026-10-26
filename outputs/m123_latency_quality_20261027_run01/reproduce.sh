#!/usr/bin/env bash
set -euo pipefail
repo=/data/qh/phdCode/work4/OmniDrive
cd "$repo"
root=${1:?Specify a fresh output directory}
reference=outputs/m123_latency_quality_20261027_run01
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
py=/home/qh/miniconda3/envs/OmniDrive/bin/python
# All subprocesses inherit affinity; this entry may also be launched with taskset.
exec taskset -c 0-3,6-31 "$py" -u - "$root" "$reference" <<'PY'
import json,sys,shutil,hashlib
from pathlib import Path
from tools.worldstream_m2.global_residual_benchmark import execute,wait_gpu
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
from tools.worldstream_m123.latency_quality import calibrate,verify
root,reference=map(Path,sys.argv[1:]);root.mkdir(parents=True,exist_ok=False)
for name in ('development_sequence.json','validation_sequence.json','profile_sequence.json','quality_budget.json','experiment_plan.json','environment.json','before_config.py','protocol_history.json','frozen_m3_policy.json'):
    shutil.copy2(reference/name,root/name)
for c in ('m1','m2','m3'):
    shutil.copytree(reference/('before_'+c),root/('before_'+c),ignore=shutil.ignore_patterns('__pycache__'))
if hashlib.sha256((root/'frozen_m3_policy.json').read_bytes()).digest()!=hashlib.sha256(Path('outputs/final_route/m3_reuse_policy.json').read_bytes()).digest():
    raise ValueError('M3 policy differs from the frozen experiment')
# The source snapshot is the original before-code baseline, not the new checkout.
with gpu_job_lock():
    execute(root,'historical_reassessment',['-m','tools.worldstream_m2.global_residual_analysis','--root',str(root/'historical_quality_reassessment'),'--stage','reassess','--evidence-root','outputs/m2_global_residual_20261026_run03','--quality-budget','0.10'])
    for before in (True,False):
        name='before_profile' if before else 'after_profile';wait_gpu()
        args=['-m','tools.worldstream_m123.benchmark.frozen_worker','--snapshot-root',str(root)] if before else ['-m','tools.worldstream_m123.benchmark.worker']
        args += ['--output',str(root/name),'--run-root',str(root),'--condition','B3','--workload','language','--limit','4','--warmup','1','--seed','0','--repeat','0','--measurement-set','efficiency','--sequence-file',str(root/'profile_sequence.json'),'--m2-global-anchor-interval','4','--m2-global-drift-threshold','0.9120883047580719','--runtime-profile','--profile-frame','1','--capture-m3-inputs','--m3-strategy','wsdr_kv','--trace-m3-decisions']
        if not before:args+=['--capture-m1-controls']
        execute(root,name,args)
    execute(root,'m1_control',['-m','tools.worldstream_m123.latency_controls','--root',str(root),'--stage','m1'])
# Controls lock themselves; release the parent lock before launching one.
execute(root,'decode_control_run',['-m','tools.worldstream_m123.latency_controls','--root',str(root),'--stage','decode'])
execute(root,'m3_diagnostics',['-m','tools.worldstream_m123.latency_diagnostics','--root',str(root)])
with gpu_job_lock():
    calibrate(root)
    verify(root)
execute(root,'final_report',['-m','tools.worldstream_m123.latency_report','--root',str(root)])
from tools.worldstream_m123.artifact_index import build_index
build_index(root)
print('Offline evidence complete; this reproduction did not modify formal defaults.')
PY
