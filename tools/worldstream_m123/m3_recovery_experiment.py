"""Frozen natural single/multi-question workloads and actual equal-token controls."""
import argparse
import hashlib
from pathlib import Path
from tools.worldstream_m2.global_residual_benchmark import read,rows,execute,wait_gpu,evaluate
from tools.worldstream_m123.benchmark.common import write


def service(root,workload,kind,repeat=0,fixed=False):
    mode='fixed' if fixed else 'free';name=f'{workload}_{mode}_r{repeat}_{kind}'
    target=root/name;sequence=root/(workload+'_sequence'+('24' if workload=='multi' else '48')+'.json')
    if target.exists():
        if read(target/'status.json')['status']!='COMPLETED':raise ValueError('Preserve partial run; use a fresh name: '+name)
        print('PRESERVE',name,flush=True);return
    wait_gpu()
    args=['-m','tools.worldstream_m123.benchmark.worker','--output',str(target),'--run-root',str(root),
        '--condition','B0' if kind=='native' else 'B3','--workload','language',
        '--sequence-file',str(sequence),'--limit',str(24 if workload=='multi' else 48),
        '--warmup','1','--seed','0','--repeat',str(repeat),'--questions-per-frame',str(3 if workload=='multi' else 1),
        '--measurement-set','efficiency']
    if kind!='native':args+=['--m3-strategy','wsdr_kv','--m3-policy',str(root/'frozen_m3_policy.json')]
    if kind=='zero':args+=['--m3-zero-reuse']
    if fixed:args+=['--fixed-token-reference',str(root/(workload+'_fixed_reference.json'))]
    elif kind=='zero':args+=['--capture-m3-inputs']
    execute(root,name,args)


def reference(root,workload):
    path=root/f'{workload}_free_r0_native/per_question_metrics.jsonl';frames={}
    for row in rows(path):
        ids=row['token_ids'][0]
        if ids[0]!=1 or len(ids)-1!=row['generated_tokens']:raise ValueError('Unexpected native BOS')
        frames.setdefault(row['frame_id'],[]).append(dict(question=row['question'],tokens=ids[1:]))
    write(root/(workload+'_fixed_reference.json'),dict(source='Native free generation on same frozen natural workload',
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_file=str(path),frames=frames))


def run(root,stage):
    plan=read(root/'experiment_plan.json')
    for name,digest in plan['manifest_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen input changed '+name)
    # Each worker explicitly loads this SHA-frozen benchmark policy. The
    # current deployment may later be conservatively restricted; historical
    # reproduction must not require restoring an unsafe deployment file.
    if stage in ('all','tests'):
        wait_gpu();execute(root,'tests_gpu_qualified' if (root/'tests_gpu_delivery.log').exists() else 'tests_gpu_delivery' if (root/'tests_gpu_final.log').exists() else 'tests_gpu_final',['-m','pytest','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','tests/worldstream_m123','-q'])
    if stage in ('all','free'):
        for workload in ('multi','single'):
            for kind in ('native','zero','reuse'):service(root,workload,kind)
            reference(root,workload)
    if stage in ('all','fixed'):
        for workload in ('multi','single'):
            for repeat,order in enumerate(plan['performance']['orders']):
                for kind in order:service(root,workload,kind,repeat,True)
    if stage in ('all','control'):
        wait_gpu();execute(root,'same_input_control',['-m','tools.worldstream_m123.m3_recovery_control','--root',str(root)])
    if stage in ('all','quality'):
        for workload in ('multi','single'):
            for kind in ('native','reuse'):
                name=f'{workload}_free_r0_{kind}'
                if not (root/(name+'_official/summary.json')).exists():evaluate(root,name)
    if stage in ('all','analyze'):
        execute(root,'analysis',['-m','tools.worldstream_m123.m3_recovery_analysis','--root',str(root)])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['all','tests','free','fixed','control','quality','analyze'],default='all');a=p.parse_args()
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    with gpu_job_lock():run(Path(a.root),a.stage)
