"""Frozen Native/RMS/Risk0 free and equal-token full-request comparisons."""
import argparse
import hashlib
from pathlib import Path
from tools.worldstream_m2.global_residual_benchmark import read,rows,wait_gpu,execute,evaluate
from tools.worldstream_m2.global_residual_analysis import write


def validate(root):
    plan=read(root/'experiment_plan.json')
    for name,key in [('sequence.json','sequence_sha256'),('risk0.json','risk0_sha256'),('rms.json','rms_sha256')]:
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=plan[key]:raise ValueError('Frozen input changed: '+name)
    manifest=read(root/'sequence.json')
    if set(manifest['scene_splits'])&set(plan['excluded_scenes']):raise ValueError('Old scene leakage')
    if len(manifest['tokens'])!=plan['frames'] or set(manifest['scene_splits'])!=set(plan['scenes']):raise ValueError('Unexpected sample scope')
    if read(root/'risk0.json')!=read(Path('outputs/m123_world_risk_optimization_20261027_run01/risk_0.json')):raise ValueError('Risk0 was retuned')
    if (root/'frozen_m3_policy.json').read_bytes()!=Path('outputs/final_route/m3_reuse_policy.json').read_bytes():raise ValueError('M3 policy changed')
    return plan


def service(root,name,kind,*,sequence=None,repeat=0,fixed=False,extra=()):
    sequence=sequence or root/'sequence.json';target=root/name
    if target.exists():
        if not (target/'status.json').exists() or read(target/'status.json')['status']!='COMPLETED':
            raise ValueError('Incomplete existing run; preserve it and use a fresh attempt name: '+name)
        invocation=read(target/'invocation.json')
        if invocation['sequence_file']!=str(sequence) or bool(invocation.get('fixed_token_reference'))!=fixed:raise ValueError('Resume boundary mismatch')
        print('PRESERVE COMPLETED',name,flush=True);return
    wait_gpu()
    args=['-m','tools.worldstream_m123.benchmark.worker','--output',str(target),'--run-root',str(root),
        '--condition','B0' if kind=='native' else 'B3','--workload','language','--limit',str(len(read(sequence)['tokens'])),
        '--warmup','1','--seed','0','--repeat',str(repeat),'--measurement-set','efficiency','--sequence-file',str(sequence)]
    if kind!='native':args+=['--m2-risk-config',str(root/(kind+'.json'))]
    if fixed:args+=['--fixed-token-reference',str(root/'fixed_reference.json')]
    execute(root,name,args+list(extra))


def reference(root):
    records=rows(root/'free_r0_native/per_question_metrics.jsonl');frames={}
    for r in records:
        ids=r['token_ids'][0]
        if ids[0]!=1 or len(ids)-1!=r['generated_tokens']:raise ValueError('Unexpected native BOS convention')
        frames.setdefault(r['frame_id'],[]).append(dict(question=r['question'],tokens=ids[1:]))
    value=dict(source='Native normal free generation, round0, same checkpoint and new scene sequence',
        source_file='free_r0_native/per_question_metrics.jsonl',
        source_sha256=hashlib.sha256((root/'free_r0_native/per_question_metrics.jsonl').read_bytes()).hexdigest(),frames=frames)
    path=root/'fixed_reference.json'
    if path.exists() and read(path)!=value:raise ValueError('Fixed reference changed')
    write(path,value)


def run(root,stage):
    plan=validate(root)
    for mode in (('free','fixed') if stage=='all' else (stage,)):
        if mode=='fixed':reference(root)
        for repeat,order in enumerate(plan['timing']['orders']):
            for kind in order:
                name=f'{mode}_r{repeat}_{kind}'
                service(root,name,kind,repeat=repeat,fixed=mode=='fixed')
                if mode=='free' and repeat==0:
                    if not (root/(name+'_official/summary.json')).exists():evaluate(root,name)
            if mode=='free' and repeat==0:reference(root)
    if stage=='all':
        # These diagnostic forwards never contribute headline timings.
        service(root,'equivalence_development','risk0',sequence=root/'equivalence_sequence.json',
            extra=('--risk-equivalence-reference',str(root/'before_m2')))
        service(root,'residual_labels_new','risk0',extra=('--shadow-global-labels',))
        wait_gpu();execute(root,'tests_gpu_final',['-m','pytest','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','tests/worldstream_m123','-q'])
        execute(root,'final_analysis',['-m','tools.worldstream_m123.final_speed_analysis','--root',str(root)])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--stage',choices=['all','free','fixed'],default='all');a=p.parse_args()
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    with gpu_job_lock():run(Path(a.root),a.stage)
