"""Reproducible offline calibration and paired service runs of one M2 algorithm."""
import argparse
import hashlib
from .quality_budget import quality_budgets, DEFAULT_QUALITY_BUDGET
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def execute(root, name, arguments):
    command=['taskset','-c','0-3,6-31',sys.executable,'-u',*arguments]
    journal=root/'commands.json';records=read(journal) if journal.exists() else []
    record=dict(name=name,command=command,start=datetime.now(timezone.utc).isoformat(),log=str(root/(name+'.log')))
    records.append(record);journal.write_text(json.dumps(records,indent=2))
    print('START',name,flush=True)
    with Path(record['log']).open('w') as handle:
        result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode,end=datetime.now(timezone.utc).isoformat())
    journal.write_text(json.dumps(records,indent=2))
    if result.returncode:raise RuntimeError('Experiment failed: '+record['log'])
    print('DONE',name,flush=True)


def wait_gpu():
    while True:
        status=subprocess.check_output(['taskset','-c','0-3,6-31','nvidia-smi',
            '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if int(status.strip().splitlines()[0])<500:break
        print('WAIT GPU occupied; leave other processes untouched',flush=True)
        time.sleep(30)


def service(root, name, sequence, *, interval=None, threshold=None, repeat=0):
    if (root/name).exists():raise ValueError('Measured output already exists; use a new experiment root')
    wait_gpu()
    count=len(read(sequence)['tokens'])
    args=['-m','tools.worldstream_m123.benchmark.worker','--output',str(root/name),'--run-root',str(root),
        '--condition','B3' if interval is not None else 'B0','--workload','language',
        '--limit',str(count),'--warmup','1','--seed','0','--repeat',str(repeat),
        '--measurement-set','efficiency','--sequence-file',str(sequence)]
    if interval is not None:args+=['--m2-global-anchor-interval',str(interval),'--m2-global-drift-threshold',str(threshold)]
    execute(root,name,args)


def evaluate(root,name):
    if (root/(name+'_official')).exists():raise ValueError('Official output already exists; preserve the previous evaluation')
    execute(root,name+'_official',['-m','tools.worldstream_m2.task_quality',
        '--run',str(root/name),'--output',str(root/(name+'_official'))])


def run(args):
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    if set(os.sched_getaffinity(0))&{4,5}:raise ValueError('Use the documented taskset affinity')
    budgets=quality_budgets(args.quality_budget,args.vlm_quality_budget)
    budget_path=root/'quality_budget.json'
    if budget_path.exists() and read(budget_path)['quality_budgets']!=budgets:
        raise ValueError('An experiment root cannot change its frozen quality budgets')
    budget_path.write_text(json.dumps(dict(quality_budgets=budgets,units='absolute native-minus-candidate score loss'),indent=2))
    if args.stage=='calibration':
        names=['cal_native']+['cal_age'+str(age) for age in (2,3,4)]
        if any((root/(name+suffix)).exists() for name in names for suffix in ('','_official')):
            raise ValueError('Calibration output already exists; use a new experiment root')
        prior=Path('outputs/m2_unified_wsgrr_20261026_run01')
        sequence=read(prior/'sequence40.json')
        sequence['questions']={r['frame_id']:[r['question']] for r in rows(prior/'formal40/per_question_metrics.jsonl')}
        sequence['split']='development_only'
        path=root/'development_sequence40.json';path.write_text(json.dumps(sequence,indent=2))
        drift_path=Path(args.drift_source) if args.drift_source else root/'drift_calibration40/frames.jsonl'
        drift=rows(drift_path)
        if not drift or any(r['frame_id'] not in set(sequence['tokens']) for r in drift):
            raise ValueError('Drift calibration must contain development frames only')
        values=sorted(r['global_state']['global_feature_drift'] for r in drift
            if r['strategy']=='AR2' and r['global_state']['global_anchor_age']==1)
        import numpy as np
        if not 0 <= args.drift_quantile <= 1:raise ValueError('Invalid drift quantile')
        threshold=float(np.quantile(values,args.drift_quantile))
        (root/'calibration_parameters.json').write_text(json.dumps(dict(threshold=threshold,
            definition='Development-only quantile of age-1 maximum-camera relative H2 RMS',
            drift_quantile=args.drift_quantile,drift_source=str(drift_path),
            drift_source_sha256=hashlib.sha256(drift_path.read_bytes()).hexdigest(),
            age1_drifts=values,candidates=[2,3,4],independent_validation_used=False),indent=2))
        service(root,'cal_native',path);evaluate(root,'cal_native')
        for age in (2,3,4):
            service(root,'cal_age'+str(age),path,interval=age,threshold=threshold)
            evaluate(root,'cal_age'+str(age))
    elif args.stage=='service':
        selection=read(root/'selection.json');age=selection['interval'];threshold=selection['threshold']
        for repeat in range(3):
            for formal in ((False,True) if repeat%2==0 else (True,False)):
                service(root,'r'+str(repeat)+('_formal' if formal else '_native'),root/'development_sequence40.json',
                    interval=age if formal else None,threshold=threshold,repeat=repeat)
        evaluate(root,'r0_formal');evaluate(root,'r0_native')
    elif args.stage=='validation':
        selection=read(root/'selection.json')
        path=root/'validation_sequence24.json'
        if args.validation_sequence:
            original=read(args.validation_sequence)
            if path.exists() and read(path)!=original:raise ValueError('Frozen validation manifest differs')
            path.write_text(json.dumps(original,indent=2))
        if not path.exists():raise ValueError('Declare an independent validation sequence before execution')
        original=read(path)
        plan_path=root/'experiment_plan.json'
        if plan_path.exists():
            expected=read(plan_path)['validation']['manifest_sha256']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('Predeclared heldout manifest was modified')
        development_scenes=set(read(root/'development_sequence40.json').get('scene_splits',{}))
        if not development_scenes:
            development_scenes={r['scene_id'] for r in rows(root/'drift_calibration40/frames.jsonl')}
        if development_scenes & set(original['scene_splits']):raise ValueError('Validation scene leakage')
        if set(original.get('excluded_scenes',[])) & set(original['scene_splits']):
            raise ValueError('Validation contains an explicitly excluded previous scene')
        service(root,'validation_native',path);evaluate(root,'validation_native')
        service(root,'validation_formal',path,interval=selection['interval'],threshold=selection['threshold'])
        evaluate(root,'validation_formal')
    elif args.stage=='tests':
        wait_gpu()
        execute(root,'tests_gpu_final',['-m','pytest','tests/worldstream_m1','tests/worldstream_m2',
            'tests/worldstream_m3','tests/worldstream_m123','-q'])
    elif args.stage=='verify':
        # Selection is already frozen on development data. Validate once, then
        # retain that same policy for every timing round even if quality fails.
        validation_args=argparse.Namespace(**vars(args));validation_args.stage='validation'
        run(validation_args)
        selection=read(root/'selection.json')
        test_args=argparse.Namespace(**vars(args));test_args.stage='tests'
        run(test_args)
        wait_gpu()
        execute(root,'paired40',['-m','tools.worldstream_m2.window_validate','--wsgrr','--main-only',
            '--formal','--snapshot-boundary','--frames','40','--repeats','3','--warmup','1',
            '--anchor-intervals',str(selection['interval']),'--global-drift-threshold',str(selection['threshold']),
            '--reference-runtime',str(root/'baseline_m2'),'--reference-state-manager',str(root/'baseline_state_manager.py'),
            '--followup-profile','--m1-predictions',
            'outputs/wsgrr_mainfix_20261023/references/wsgrrr_20261022_followup/Native_full/official_eval/detection/pts_bbox/results_nusc.json',
            '--output',str(root/'paired40')])
        service_args=argparse.Namespace(**vars(args));service_args.stage='service'
        run(service_args)
        execute(root,'final_analysis',['-m','tools.worldstream_m2.global_residual_analysis','--root',str(root),'--stage','final',
            '--quality-budget',str(args.quality_budget),*(['--vlm-quality-budget',str(args.vlm_quality_budget)] if args.vlm_quality_budget is not None else [])])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['calibration','service','validation','verify','tests'],required=True)
    p.add_argument('--quality-budget',type=float,default=DEFAULT_QUALITY_BUDGET,help='Maximum absolute loss per task metric; offline evaluation only (default: 0.05)')
    p.add_argument('--vlm-quality-budget',type=float,default=None,help='Legacy reproduction override only')
    p.add_argument('--drift-quantile',type=float,default=.5,help='Development-only H2 drift quantile (default: 0.5)')
    p.add_argument('--drift-source',help='Existing development drift JSONL (never heldout data)')
    p.add_argument('--validation-sequence',help='Predeclared heldout manifest, frozen before validation')
    run(p.parse_args())
