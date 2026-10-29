"""WRSAR v4: finite DEV calibration and source-frozen real M123 experiments."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout,redirect_stderr
import json
from pathlib import Path
import pickle
from .benchmark.common import write,sha256

ROOT=Path('outputs/m3_target_complete_bypass_run01')
PRIOR=Path('outputs/m3_state_calibrated_bypass_run01')
CONFIGS=[dict(target_completeness_enabled=False,min_confidence=.45,max_binding_risk=.75),
    dict(target_completeness_enabled=True,min_confidence=.45,max_binding_risk=.75,
        position_min_confidence=.55,max_target_risk=.50,center_nearest_min_cameras=2),
    dict(target_completeness_enabled=True,min_confidence=.45,max_binding_risk=.75,
        position_min_confidence=.65,max_target_risk=.35,center_nearest_min_cameras=2),
    dict(target_completeness_enabled=True,min_confidence=.45,max_binding_risk=.75,
        position_min_confidence=.45,max_target_risk=.65,center_nearest_min_cameras=1)]


def source_questions(token,split='train'):
    questions=[]
    for folder in ('conv','vqa'):
        path=Path('data/nuscenes')/folder/split/(token+'.json')
        if path.exists():questions.extend(p['question'] for p in json.loads(path.read_text()))
    return questions


def prepare(root):
    from tools.worldstream_m1.dataset import resolve_image
    assert (root/'nearest_target_failure_audit.json').exists(),'P0 must finish first'
    assert not (root/'experiment_plan.json').exists(),'Never replace a frozen split'
    prior=json.loads((PRIOR/'experiment_plan.json').read_text())
    scenes={x['token']:x for x in json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())}
    all_logs={s['log_token'] for s in scenes.values()};excluded=set();plans={}
    info_path='data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl'
    infos=pickle.load(open(info_path,'rb'))['infos']
    frame_scene={x['token']:x['scene_token'] for x in infos}
    def walk(x):
        if isinstance(x,dict):
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
        elif isinstance(x,str):
            if x in scenes:excluded.add(scenes[x]['log_token'])
            if x in all_logs:excluded.add(x)
            if x in frame_scene:excluded.add(scenes[frame_scene[x]]['log_token'])
    for p in sorted(Path('outputs').glob('m3*/experiment_plan.json')):
        walk(json.loads(p.read_text()));plans[str(p)]=sha256(p)
    grouped=defaultdict(list)
    for x in infos:grouped[x['scene_token']].append(x)
    chosen=[];logs=set()
    for sid in sorted(grouped):
        log=scenes[sid]['log_token']
        if log in excluded or log in logs:continue
        frames=sorted(grouped[sid],key=lambda f:f['timestamp'])[:8]
        if len(frames)!=8 or any(len(source_questions(f['token']))<3 for f in frames):continue
        if not all(resolve_image(c['data_path'],'data/nuscenes').exists() for f in frames for c in f['cams'].values()):continue
        chosen.append(dict(scene_id=sid,log_id=log,tokens=[f['token'] for f in frames]));logs.add(log)
        if len(chosen)==6:break
    assert len(chosen)==6,'Need two new DEV and four new TEST logs; never reuse old TEST logs'
    plan=dict(prior,version='wrsar4',base_commit='b89f6aeb3be4dfde9edf1aabe2c6fed8aa3899ad',
        development_scenes=chosen[:2],test_scenes=chosen[2:],diagnostic_scenes=[],rounds=1,
        dataset_infos=info_path,qa_split='train',experiment_seed=20261029,
        configs=CONFIGS,prior_excluded_log_ids=sorted(excluded),prior_plan_hashes=plans,
        historical_validation_partition_exhausted=True,
        base_model_training_exposure='These are new experimental DEV/TEST logs from the physical training partition; base-model exposure is possible. No claim of official held-out model generalization.',
        natural_questions={t:source_questions(t)[:3] for s in chosen for t in s['tokens']},
        workloads=['natural','fixed','addressable','mixed'],question_counts=[1,3],
        selection='First six sorted image/QA-complete unused train logs:2 DEV,4 TEST,first8 contiguous frames. Exclude historical plan scene/log/frame metadata before looking at detections/GT/answers.',
        mixed_protocol='Preset1/3 introduced structured slot, cycling by frame_index%3; two original natural questions. Not a natural population distribution.',
        prior_diagnostics_only=True)
    write(root/'experiment_plan.json',plan)
    write(root/'protocol_freeze.json',dict(plan_sha256=sha256(root/'experiment_plan.json'),
        configs=CONFIGS,quality_budgets=[.01,.02,.03],default_budget=.02,
        research_admission=dict(average_quality='Known paired loss and scene-CI upper <=budget; conservative all-request upper <=budget',
            severe='New severe rate and scene-CI upper <=0.02 of scored requests; no post-TEST change',
            max_new_severe_rate=.02,minimum_scored_fast_unique_frames=3,
            performance='Positive B-to-C speed in every TEST log and upper scene delta CI<0',
            independent_contribution='Same evidence C-basic/C-target quality or severe-risk improvement, independently measured E2E tradeoff',
            deployment_safety='NOT_QUALIFIED; research eligibility never authorizes safety-critical deployment'),
        sampling_unchanged=True,temporal_risk_enabled=False,no_test_tuning=True,
        no_unconditional_position_ban=True,competition_score_floor=.15))
    print('PREPARED',json.dumps(dict(development=chosen[:2],test=chosen[2:],excluded_logs=len(excluded))),flush=True)


def stage_jobs(root,stage,plan):
    if stage=='shadow':return [('addressable',3,0,'B','shadow',None)]
    if stage=='pareto':
        jobs=[]
        for si,s in enumerate(plan['development_scenes']):
            for repeat in range(2):
                order=[('B',None)]+[(f'C{i}',root/f'candidate_{i}.json') for i in range(4)]
                offset=(si+repeat*2)%len(order);order=order[offset:]+order[:offset]
                jobs.extend(('addressable',3,repeat,c,'off' if c=='B' else 'online',p,s['scene_id']) for c,p in order)
        return jobs
    if stage=='ablation':
        selected=json.loads((root/'selected_policy.json').read_text())
        other='C-basic' if selected['target_completeness_enabled'] else 'C-target'
        policy=root/('basic_ablation_policy.json' if other=='C-basic' else 'target_ablation_policy.json')
        return [(w,n,0,other,'online',policy,s['scene_id']) for s in plan['test_scenes']
            for w,n in [('addressable',1),('addressable',3),('mixed',3)]]
    jobs=[]
    for wi,w in enumerate(plan['workloads']):
        for n in ((3,) if w=='mixed' else (1,3)):
            for si,s in enumerate(plan['test_scenes']):
                order=['A','B','C'];offset=(si+wi+(n==3))%3;order=order[offset:]+order[:offset]
                jobs.extend((w,n,0,c,'online' if c=='C' else 'off',root/'selected_policy.json' if c=='C' else None,s['scene_id']) for c in order)
    return jobs


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--stage',required=True,choices=['prepare','shadow','calibrate','pareto','freeze','run','ablation','report']);a=p.parse_args()
    if a.stage=='prepare':prepare(a.root)
    elif a.stage in ('calibrate','freeze','report'):
        from . import target_complete_report as report
        getattr(report,a.stage)(a.root)
    else:
        from .selective_answer_reuse import run
        with (a.root/f'{a.stage}_execution.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):run(a.root,a.stage)
