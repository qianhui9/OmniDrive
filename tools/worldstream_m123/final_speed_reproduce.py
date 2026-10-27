"""Repeat the fixed final-speed protocol in a fresh output directory."""
import argparse
from pathlib import Path
import shutil


def reproduce(root,evidence):
    root.mkdir(parents=True,exist_ok=False)
    for component in ('m1','m2','m3'):
        shutil.copytree(evidence/('before_'+component),root/('before_'+component),ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('before_config.py','environment.json','frozen_m3_policy.json','risk0.json','rms.json',
                 'quality_budget.json','equivalence_sequence.json','sequence.json','experiment_plan.json'):
        shutil.copy2(evidence/name,root/name)
    from tools.worldstream_m123.final_speed_experiment import run
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    with gpu_job_lock():run(root,'all')
    from tools.worldstream_m123.artifact_index import build_index
    build_index(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--evidence-root',default='outputs/m123_final_speed_validation_20261027_run01');a=p.parse_args()
    reproduce(Path(a.root),Path(a.evidence_root))
