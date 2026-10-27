"""Reproduce the frozen recovery protocol in a fresh directory."""
import argparse
from pathlib import Path
import shutil
from tools.worldstream_m2.global_residual_benchmark import execute,wait_gpu
from tools.worldstream_m123.m3_recovery_experiment import run
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock


def reproduce(root,evidence):
    root.mkdir(parents=True,exist_ok=False)
    for component in ('m1','m2','m3'):
        shutil.copytree(evidence/('before_'+component),root/('before_'+component),ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('before_config.py','frozen_m3_policy.json','historical_sequence24.json','single_sequence48.json',
            'multi_sequence24.json','experiment_plan.json','baseline_evidence.json','historical_source.diff','semantic_failure_review.json','qualified_m3_policy.json'):
        shutil.copy2(evidence/name,root/name)
    with gpu_job_lock():
        for name,sequence,count in [('historical_current','historical_sequence24.json',3),('single_before','single_sequence48.json',1)]:
            wait_gpu()
            execute(root,name,['-m','tools.worldstream_m123.benchmark.frozen_worker','--snapshot-root',str(root),
                '--output',str(root/name),'--run-root',str(root),'--condition','B3','--workload','language',
                '--limit',str(24 if count==3 else 48),'--warmup','1','--seed','0','--questions-per-frame',str(count),
                '--sequence-file',str(root/sequence),'--m3-strategy','wsdr_kv','--m3-policy',str(root/'frozen_m3_policy.json'),
                '--trace-m3-decisions','--capture-m3-inputs'])
        run(root,'all')
        qualification=root/'qualification';qualification.mkdir(exist_ok=False)
        for component in ('m1','m2','m3'):
            shutil.copytree(evidence/'qualification'/('before_'+component),qualification/('before_'+component),ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('multi_sequence24.json','frozen_m3_policy.json','experiment_plan.json'):
            shutil.copy2(evidence/'qualification'/name,qualification/name)
        from tools.worldstream_m123.m3_recovery_qualification import run as qualify
        qualify(qualification,'all')
        wait_gpu();execute(root,'tests_gpu_qualified',['-m','pytest','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','tests/worldstream_m123','-q'])
        from tools.worldstream_m123.m3_recovery_final_report import report as final_report
        final_report(root)
    from tools.worldstream_m123.artifact_index import build_index
    build_index(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--evidence-root',default='outputs/m3_reuse_recovery_20261027_run01');a=p.parse_args()
    reproduce(Path(a.root),Path(a.evidence_root))
