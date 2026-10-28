"""SF layout quality only after qualified target-domain semantic support."""
import argparse
import json
import os
from pathlib import Path
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
from tools.worldstream_m2.global_residual_benchmark import wait_gpu,execute,evaluate
from tools.worldstream_m123.wrcps_semantic_quality import analyze_zero_quality


def quality(root):
    if not json.loads((root/'stable_core_analysis.json').read_text())['stage_one_pass']:
        raise ValueError('Independent core integrity gate failed')
    if not (root/'adapter_training/adapter.pt').exists():raise ValueError('Trained SF adapter required')
    review_path=root/'core_valid_quality/critical_review.json'
    if not review_path.exists() or not json.loads(review_path.read_text()).get('development_core_quality_pass'):
        raise ValueError('Core-valid development representation and critical semantic review must pass first')
    review=json.loads(review_path.read_text())
    from tools.worldstream_m123.benchmark.common import sha256
    if review.get('adapter_sha256')!=sha256(root/'adapter_training/adapter.pt'):
        raise ValueError('Development quality review belongs to a different adapter')
    seq=root/'target_sequence.json';count=len(json.loads(seq.read_text())['tokens'])
    for name,condition in [('quality_native','B0'),('quality_zero','B3')]:
        if (root/name).exists():raise FileExistsError('Preserve measured quality output')
        args=['-m','tools.worldstream_m123.benchmark.worker','--output',str(root/name),'--run-root',str(root),
            '--condition',condition,'--workload','language','--limit',str(count),'--warmup','1','--repeat','0',
            '--sequence-file',str(seq),'--questions-per-frame','3','--measurement-set','efficiency']
        if condition=='B3':args+=['--m3-strategy','exact_prefix','--m3-zero-reuse','--sf-wrcps',
            '--wrcps-adapter',str(root/'adapter_training/adapter.pt'),'--semantic-qualification',str(root/'semantic_qualification.json')]
        with gpu_job_lock():wait_gpu();execute(root,name,args)
    for name in ('quality_native','quality_zero'):
        with gpu_job_lock():wait_gpu();evaluate(root,name)
    analyze_zero_quality(root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args()
    if set(os.sched_getaffinity(0))&{4,5}:p.error('Use taskset -c 0-3,6-31')
    quality(Path(a.root))


if __name__=='__main__':main()
