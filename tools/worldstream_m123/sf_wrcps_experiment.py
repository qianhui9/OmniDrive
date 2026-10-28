"""Bounded SF-WRCPS data collection; freezes splits before seeing outputs."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import pickle
import shutil
import numpy as np
from tools.worldstream_m123.benchmark.common import write,sha256
from tools.worldstream_m123.benchmark.sequence import validate_sequence
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
from tools.worldstream_m2.global_residual_benchmark import execute,wait_gpu

PRIOR=Path('outputs/m3_semantic_core_wrcps_run01')


def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def prepare(root):
    from mmcv import Config
    if (root/'experiment_plan.json').exists():raise FileExistsError('Preserve frozen plan')
    root.mkdir(parents=True,exist_ok=True)
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    table=json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())
    logs={s['token']:s['log_token'] for s in table}
    previous=[json.loads((PRIOR/n).read_text()) for n in ('experiment_plan.json','adaptation_sequence.json')]
    used=set(v for m in previous for v in m['scene_logs'].values())
    from tools.worldstream_m123.benchmark.independent_validation import historical_audit
    infos={s:pickle.load(open(cfg.data.train.ann_file if s=='train' else cfg.data.test.ann_file,'rb'))['infos'] for s in ('train','val')}
    audit=historical_audit('outputs',root,infos['train']+infos['val'])
    write(root/'historical_data_audit.json',audit)
    old_language_logs={logs[s] for s in audit['excluded_scenes'] if s in logs}
    def select(split,count,exclude):
        groups=defaultdict(list)
        for r in infos[split]:groups[r['scene_token']].append(r)
        result=[];taken=set()
        for scene,ss in sorted(groups.items()):
            log=logs[scene]
            if log in exclude or log in taken:continue
            segment=sorted(ss,key=lambda r:r['timestamp'])[:8]
            if len(segment)!=8:continue
            if any(not Path(f'data/nuscenes/conv/{split}/{r["token"]}.json').exists() for r in segment):continue
            pairs=[json.loads(Path(f'data/nuscenes/conv/{split}/{r["token"]}.json').read_text()) for r in segment]
            if any(len(p)<3 for p in pairs):continue
            validate_sequence(segment,scene)
            if not all(a.get('next')==b['token'] and b.get('prev')==a['token'] for a,b in zip(segment,segment[1:])):continue
            result.append((scene,segment,pairs));taken.add(log)
            if len(result)==count:break
        if len(result)!=count:raise ValueError(f'Insufficient {split} logs: {len(result)}/{count}')
        return result
    # VAL has only 18 log units. Old six VAL diagnostic logs are excluded.
    # Remaining logs may have older non-SF M3 exposure; disclose, do not call
    # this a globally pristine acceptance set. Only development sees outputs
    # before the policy is frozen; final six logs are held out this round.
    target=select('val',12,used)
    train=select('train',18,used|old_language_logs|{logs[s] for s,_,_ in target})
    manifests={n:dict(tokens=[],questions={},scene_splits={},scene_logs={}) for n in ('qualification_train','qualification_val','target','adaptation')}
    roles={};datasets={}
    for j,(scene,segment,pairs) in enumerate(train+target):
        if j<8:name='qualification_train';role='calibration' if j<4 else 'qualification_validation';data='train'
        elif j<18:name='adaptation';role='train' if j<14 else 'validation';data='train'
        else:
            q=j-18;name='qualification_val' if q<6 else 'target';data='val'
            role='calibration' if q<3 else 'qualification_validation' if q<6 else 'independent_acceptance'
        m=manifests[name];m['scene_splits'][scene]=role;m['scene_logs'][scene]=logs[scene]
        roles[scene]=role;datasets[scene]=data
        for r,p in zip(segment,pairs):
            m['tokens'].append(r['token']);m['questions'][r['token']]=[x['question'] for x in p[:3]]
    for name,m in manifests.items():write(root/(name+'_sequence.json'),m)
    qual=dict(tokens=[],questions={},scene_splits={},scene_logs={})
    for name in ('qualification_train','qualification_val'):
        for key in ('tokens',):qual[key]+=manifests[name][key]
        for key in ('questions','scene_splits','scene_logs'):qual[key].update(manifests[name][key])
    write(root/'qualification_sequence.json',qual)
    write(root/'acceptance_reserved_sequence.json',manifests['target'])
    write(root/'adaptation_sequence.json',manifests['adaptation'])
    write(root/'experiment_plan.json',dict(base_commit='0932a377b199e6249b87e15881c6e3c960f3cf3b',seed=20261028,
        scene_splits=roles,scene_logs={s:logs[s] for s in roles},scene_datasets=datasets,
        selector=dict(core_size=1,missing_observation_grace_frames=0),
        calibration=dict(trials=3,max_observed_wrong_edges=0,max_unknown_fraction=.05,min_scored_edges=30,min_scenes=3),
        fixed_car_rule_source=str(PRIOR/'semantic_qualification.json'),fixed_car_rule_sha256=sha256(PRIOR/'semantic_qualification.json'),
        compression=dict(primary_K=64,optional_K=32,rank=128,optional_trigger='Confirmed spatial regression on development core-valid requests only; no acceptance tuning'),
        quality_budgets=dict(mAP=.1,NDS=.1,DET_l=.1,Token_F1=.1),
        historical_exposure=dict(qualification_train_new_logs=True,old_EQ_logs_excluded=True,
            target_logs_with_prior_language_exposure=sorted({logs[s] for s in manifests['target']['scene_splits']}&old_language_logs),
            limitation='VAL acceptance log units held out this round, but older M3 work saw most VAL logs; limited log-heldout evidence, not globally unseen data'),
        sequence_sha256=sha256(root/'qualification_sequence.json')))
    write(root/'semantic_type_audit.json',dict(object='Current detector class, current observation, accepted association residual/margin; eligible for independent object partial-oracle evaluation',
        map='M1 adapter publishes lane_control_points without class scores, GT topology ID, or actual lane semantic class; original VLM map queries are latent tokens. Not qualified for stable road topology/traffic rules.',
        region='No current independently evaluated identity/class producer; no semantic admission',
        M2='Current refresh/change diagnostics are invalidation covariates, not identity or map correctness certificates',
        physical_certification=False))


def capture(root):
    with gpu_job_lock():
        for split in ('train','val'):
            name='qualification_'+split;seq=root/(name+'_sequence.json')
            if (root/(name+'_capture')).exists():raise FileExistsError('Preserve capture')
            wait_gpu();execute(root,name+'_capture',['-m','tools.worldstream_m123.benchmark.worker',
                '--output',str(root/(name+'_capture')),'--run-root',str(root),'--condition','B3','--workload','language',
                '--limit',str(len(json.loads(seq.read_text())['tokens'])),'--warmup','1','--dataset-split',split,
                '--sequence-file',str(seq),'--questions-per-frame','1','--m3-zero-reuse','--m3-strategy','exact_prefix',
                '--capture-semantic-candidates'])


def qualify(root):
    from tools.worldstream_m123.wrcps_semantic_qualification import attach_labels,subset,summarize,acceptable,calibration_anchor_hits,qualify as publish
    if (root/'qualification_capture').exists():raise FileExistsError('No retuning/overwriting qualification')
    dest=root/'qualification_capture';dest.mkdir()
    for filename in ('semantic_candidates.jsonl','semantic_labels_offline.jsonl'):
        with (dest/filename).open('w') as f:
            for split in ('train','val'):
                run=root/('qualification_'+split+'_capture')
                if json.loads((run/'status.json').read_text())['status']!='COMPLETED':raise ValueError('Incomplete capture')
                f.write((run/filename).read_text())
    write(dest/'status.json',dict(status='COMPLETED',scope='Ordered concatenation of separate actual train/val captures'))
    plan=json.loads((root/'experiment_plan.json').read_text());frames=rows(dest/'semantic_candidates.jsonl')
    labels={r['frame_id']:r for r in rows(dest/'semantic_labels_offline.jsonl')}
    split=plan['scene_splits'];items=attach_labels(frames,labels,split)
    cal=[r for r in items if r['split']=='calibration'];calframes=[f for f in frames if split[f['context']['scene_id']]=='calibration']
    limits=plan['calibration'];selected={};trials=[]
    old=json.loads(Path(plan['fixed_car_rule_source']).read_text())
    if sha256(plan['fixed_car_rule_source'])!=plan['fixed_car_rule_sha256']:raise ValueError('Old fixed car rule changed')
    for cls in sorted({r['record']['semantic_class'] for r in cal}):
        sample=[r for r in cal if r['record']['semantic_class']==cls];feasible=[]
        if cls=='car':
            rules=[{k:old['rules']['car'][k] for k in ('min_score','max_residual_m','min_observations','trial')}]
        else:
            scores=[r['record']['evidence']['score'] for r in sample];res=[r['record']['evidence']['residual_m'] for r in sample]
            rules=[dict(min_score=float(np.quantile(scores,sq)),max_residual_m=float(np.quantile(res,rq)),min_observations=obs,trial=t) for t,(sq,rq,obs) in enumerate(((.5,.75,2),(.75,.5,3),(.9,.25,3)))]
        for rule in rules:
            rule=dict(rule,admitted=True);metrics=summarize(subset(sample,rule));hits=calibration_anchor_hits(calframes,cls,rule)
            trials.append(dict(semantic_class=cls,rule=rule,metrics=metrics,calibration_anchor_hits=hits,fixed_old_rule=cls=='car'))
            if acceptable(metrics,limits):feasible.append((hits,metrics['candidates'],rule,metrics))
        if feasible:
            _,_,rule,metrics=max(feasible,key=lambda v:(v[0],v[1],-v[2]['trial']));selected[cls]=dict(rule,calibration=metrics)
    write(root/'calibration_frozen_rules.json',dict(rules=selected,trials=trials,validation_consulted=False,car_rule_unchanged=True))
    publish(root)


def target_capture(root):
    if not json.loads((root/'stable_core_analysis.json').read_text())['stage_one_pass']:
        raise ValueError('No independently usable semantic core; stop training/performance admission')
    seq=root/'target_sequence.json'
    with gpu_job_lock():
        wait_gpu();execute(root,'target_native_capture',['-m','tools.worldstream_m123.benchmark.worker',
            '--output',str(root/'target_native_capture'),'--run-root',str(root),'--condition','B3','--workload','language',
            '--limit','48','--warmup','1','--sequence-file',str(seq),'--questions-per-frame','3',
            '--m3-zero-reuse','--m3-strategy','exact_prefix','--capture-semantic-candidates',
            '--capture-m3-inputs','--capture-stable-world','--semantic-qualification',str(root/'semantic_qualification.json')])


def adaptation(root):
    from tools.worldstream_m123.wrcps_semantic_experiment import training_manifest
    if not json.loads((root/'stable_core_analysis.json').read_text())['stage_one_pass']:
        raise ValueError('Independent core integrity gate failed; training is forbidden')
    timeline=rows(root/'stable_anchor_timeline.jsonl');plan=json.loads((root/'experiment_plan.json').read_text())
    target_hits=sum(r['anchor_hit'] for r in timeline if r['split']=='qualification_validation'
        and plan['scene_datasets'][r['scene_id']]=='val')
    if not target_hits:raise ValueError('No correct independent VAL-domain core hit: training admission denied')
    seq=root/'adaptation_sequence.json';count=len(json.loads(seq.read_text())['tokens'])
    with gpu_job_lock():
        wait_gpu();execute(root,'adaptation_capture',['-m','tools.worldstream_m123.benchmark.worker',
            '--output',str(root/'adaptation_capture'),'--run-root',str(root),'--condition','B3','--workload','language',
            '--limit',str(count),'--warmup','1','--dataset-split','train','--sequence-file',str(seq),'--questions-per-frame','3',
            '--m3-zero-reuse','--m3-strategy','exact_prefix','--capture-m3-inputs','--capture-stable-world',
            '--semantic-qualification',str(root/'semantic_qualification.json')])
    training_manifest(root)
    with gpu_job_lock():
        wait_gpu();execute(root,'adapter_training',['-m','tools.worldstream_m123.wrcps_train','--manifest',str(root/'training_manifest.json'),
            '--output',str(root/'adapter_training'),'--stable-tokens','64','--rank','128','--epochs','3','--seed','20261028','--spatial'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['prepare','capture','qualify','target','adaptation'],required=True);args=p.parse_args()
    if set(os.sched_getaffinity(0))&{4,5}:p.error('Use taskset -c 0-3,6-31')
    {'prepare':prepare,'capture':capture,'qualify':qualify,'target':target_capture,'adaptation':adaptation}[args.stage](Path(args.root))


if __name__=='__main__':main()
