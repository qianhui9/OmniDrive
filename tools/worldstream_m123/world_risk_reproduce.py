"""Reproduce the frozen world-risk protocol into a fresh, never reused root."""
import argparse
from pathlib import Path
import shutil
from tools.worldstream_m2.global_residual_benchmark import execute


def reproduce(root,evidence):
    root.mkdir(parents=True,exist_ok=False)
    for name in ('before_m1','before_m2','before_m3'):
        shutil.copytree(evidence/name,root/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('before_config.py','environment.json','frozen_m3_policy.json','experiment_plan.json',
                 'quality_budget.json','development_sequence.json','validation_sequence.json','profile_sequence.json'):
        shutil.copy2(evidence/name,root/name)
    from tools.worldstream_m123.world_risk_experiment import service,develop,verify,controls
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    with gpu_job_lock():
        service(root,'after_profile',root/'profile_sequence.json',formal=True,
            extra=('--runtime-profile','--profile-frame','1','--capture-m1-controls','--capture-m3-inputs'))
        service(root,'teacher',root/'development_sequence.json',formal=True,preload=True,extra=('--shadow-global-labels',))
        develop(root)
        if not (root/'selection.json').exists():return
        verify(root);controls(root)
    execute(root,'risk_microbenchmark',['-m','tools.worldstream_m123.risk_microbench','--root',str(root)])
    execute(root,'residual_diagnostics',['-m','tools.worldstream_m123.world_risk_diagnostics','--root',str(root)])
    execute(root,'report_generation',['-m','tools.worldstream_m123.world_risk_report','--root',str(root)])
    from tools.worldstream_m123.artifact_index import build_index
    build_index(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--evidence-root',default='outputs/m123_world_risk_optimization_20261027_run01');a=p.parse_args()
    reproduce(Path(a.root),Path(a.evidence_root))
