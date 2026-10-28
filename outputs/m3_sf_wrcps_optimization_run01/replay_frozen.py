"""Replay frozen SF development data, without selecting new logs or thresholds.

Default is a command preview. Final acceptance/performance never auto-admit.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',required=True)
p.add_argument('--execute',action='store_true')
p.add_argument('--through',choices=['qualification','adaptation','core-quality'],default='core-quality')
a=p.parse_args();source=Path(__file__).resolve().parent;dest=Path(a.root)
if {4,5}&set(os.sched_getaffinity(0)):p.error('Use documented taskset affinity')
if dest.resolve()==source:p.error('Never overwrite original evidence')
stages=[('tools.worldstream_m123.sf_wrcps_experiment','capture'),
        ('tools.worldstream_m123.sf_wrcps_experiment','qualify')]
if a.through!='qualification':stages.append(('tools.worldstream_m123.sf_wrcps_experiment','adaptation'))
if a.through=='core-quality':stages.append(('tools.worldstream_m123.sf_wrcps_diagnostics','core-quality'))
commands=[['taskset','-c','0-3,6-31',sys.executable,'-u','-m',module,'--root',str(dest),'--stage',stage]
          for module,stage in stages]
print(json.dumps(dict(frozen_source=str(source),output=str(dest),commands=commands,
    acceptance='Separate manual critical review and all quality gates required; this replay does not open final acceptance logs.'),indent=2),flush=True)
if a.execute:
    if dest.exists():p.error('Replay output must be a new directory')
    dest.mkdir(parents=True)
    for name in ['experiment_plan.json','historical_data_audit.json','semantic_type_audit.json',
                 'qualification_train_sequence.json','qualification_val_sequence.json','qualification_sequence.json',
                 'target_sequence.json','acceptance_reserved_sequence.json','adaptation_sequence.json']:
        shutil.copyfile(source/name,dest/name)
    for command in commands:
        subprocess.run(command,check=True)
