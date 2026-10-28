"""Real six-camera EQ-SSC qualification, with scene AND log separation."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import pickle
import subprocess
from tools.worldstream_m123.benchmark.common import sha256,write
from tools.worldstream_m123.benchmark.independent_validation import historical_audit
from tools.worldstream_m123.benchmark.sequence import validate_sequence
from tools.worldstream_m123.crossframe_e2e import native_pairs
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
from tools.worldstream_m2.global_residual_benchmark import wait_gpu,execute


def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def prepare(root):
    from mmcv import Config
    root.mkdir(parents=True,exist_ok=True)
    if (root/'experiment_plan.json').exists():raise FileExistsError('Preserve frozen experiment plan')
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    infos=pickle.load(open(cfg.data.test.ann_file,'rb'))['infos']
    train_infos=pickle.load(open(cfg.data.train.ann_file,'rb'))['infos']
    scenes=json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())
    logs={s['token']:s['log_token'] for s in scenes}
    if (root/'historical_data_audit.json').exists():
        (root/'historical_data_audit.json').rename(root/'initial_val_only_historical_audit.json')
    audit=historical_audit('outputs',root,infos+train_infos);write(root/'historical_data_audit.json',audit)
    excluded_logs={logs[s] for s in audit['excluded_scenes'] if s in logs}
    def pairs(token,split):
        path=Path('data/nuscenes/conv')/split/(token+'.json')
        return json.loads(path.read_text()) if path.exists() else []
    def choose(values,split,count,excluded):
        grouped=defaultdict(list)
        for info in values:
            if info['scene_token'] not in audit['excluded_scenes'] and logs[info['scene_token']] not in excluded:
                grouped[info['scene_token']].append(info)
        chosen=[];used=set()
        for scene,ss in sorted(grouped.items()):
            segment=sorted(ss,key=lambda x:x['timestamp'])[:8]
            if len(segment)!=8 or logs[scene] in used or any(not pairs(i['token'],split) for i in segment):continue
            validate_sequence(segment,scene)
            if not all(a.get('next')==b['token'] and b.get('prev')==a['token'] for a,b in zip(segment,segment[1:])):continue
            chosen.append((scene,segment));used.add(logs[scene])
            if len(chosen)==count:break
        if len(chosen)<count:raise ValueError('Insufficient distinct logs for '+split+': '+str(len(chosen)))
        return chosen
    # Historical M3 work exposed 17/18 validation logs. Reusing those logs for
    # semantic fitting would invalidate log-level independence. Fit/validate on
    # twelve separate official TRAIN logs; reserve six VAL logs for later task
    # acceptance, disjoint from this qualification and future adapter training.
    acceptance=choose(infos,'val',6,set())
    acceptance_logs={logs[s] for s,_ in acceptance}
    development=choose(train_infos,'train',12,excluded_logs|acceptance_logs)
    manifest=dict(tokens=[],questions={},scene_splits={},scene_logs={})
    reserve=dict(tokens=[],questions={},scene_splits={},scene_logs={})
    for j,(scene,segment) in enumerate(development+acceptance):
        split='calibration' if j<6 else 'qualification_validation' if j<12 else 'independent_acceptance'
        target=manifest if j<12 else reserve
        target['scene_splits'][scene]=split;target['scene_logs'][scene]=logs[scene]
        for info in segment:
            target['tokens'].append(info['token']);target['questions'][info['token']]=[pairs(info['token'],'train' if j<12 else 'val')[0]['question']]
    write(root/'qualification_sequence.json',manifest);write(root/'acceptance_reserved_sequence.json',reserve)
    write(root/'experiment_plan.json',dict(initial_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scene_splits=dict(manifest['scene_splits'],**reserve['scene_splits']),scene_logs=dict(manifest['scene_logs'],**reserve['scene_logs']),
        qualification_frames=96,reserved_acceptance_frames=48,frames_per_scene=8,
        selector=dict(core_size=1,missing_observation_grace_frames=0),
        calibration=dict(trials=3,max_observed_wrong_edges=0,max_unknown_fraction=.05,min_scored_edges=30,min_scenes=3,
            rules='score quantiles .50/.75/.90; residual quantiles .75/.50/.25; consecutive counts 2/3/3',
            qualification_scope='empirical identity and observed class continuity; no physical identity or motion claim',
            uncertainty='scene-cluster descriptive intervals; unobserved population risk is not guaranteed'),
        quality_budgets=dict(mAP=.1,NDS=.1,DET_l=.1,Token_F1=.1),seed=20261028,
        sequence_sha256=sha256(root/'qualification_sequence.json'),acceptance_sha256=sha256(root/'acceptance_reserved_sequence.json')))
    print('Prepared 12 train-log qualification scenes and 6 separate reserved validation-log scenes',flush=True)


def capture(root):
    plan=json.loads((root/'experiment_plan.json').read_text())
    if sha256(root/'qualification_sequence.json')!=plan['sequence_sha256']:raise ValueError('Frozen sequence changed')
    if (root/'qualification_capture').exists():raise FileExistsError('Preserve previous capture')
    with gpu_job_lock():
        wait_gpu();execute(root,'qualification_capture',['-m','tools.worldstream_m123.benchmark.worker',
            '--output',str(root/'qualification_capture'),'--run-root',str(root),
            '--condition','B3','--workload','language','--limit','96','--warmup','1',
            '--sequence-file',str(root/'qualification_sequence.json'),'--questions-per-frame','1',
            '--capture-semantic-candidates','--dataset-split','train','--m3-zero-reuse','--m3-strategy','exact_prefix'])


def extend_validation(root):
    """One bounded, metadata-selected sample extension; frozen rule unchanged."""
    import shutil
    from mmcv import Config
    if (root/'validation_extension_sequence.json').exists():raise FileExistsError('Only one extension is permitted')
    original=json.loads((root/'semantic_qualification.json').read_text())
    if any(r.get('admitted') for r in original['rules'].values()):raise ValueError('Do not enlarge validation after a qualified result')
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    infos=pickle.load(open(cfg.data.train.ann_file,'rb'))['infos']
    scenes=json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text());logs={s['token']:s['log_token'] for s in scenes}
    plan=json.loads((root/'experiment_plan.json').read_text());forbidden=set(plan['scene_logs'].values())
    grouped=defaultdict(list)
    for info in infos:
        if logs[info['scene_token']] not in forbidden:grouped[info['scene_token']].append(info)
    seq=dict(tokens=[],questions={},scene_splits={},scene_logs={});used=set()
    for scene,ss in sorted(grouped.items()):
        segment=sorted(ss,key=lambda i:i['timestamp'])[:8]
        paths=[Path('data/nuscenes/conv/train')/(i['token']+'.json') for i in segment]
        if len(segment)!=8 or logs[scene] in used or any(not p.exists() or not json.loads(p.read_text()) for p in paths):continue
        validate_sequence(segment,scene)
        if not all(a.get('next')==b['token'] and b.get('prev')==a['token'] for a,b in zip(segment,segment[1:])):continue
        seq['scene_splits'][scene]='qualification_validation';seq['scene_logs'][scene]=logs[scene];used.add(logs[scene])
        for i,p in zip(segment,paths):seq['tokens'].append(i['token']);seq['questions'][i['token']]=[json.loads(p.read_text())[0]['question']]
        if len(used)==12:break
    if len(used)!=12:raise ValueError('Insufficient additional independent logs')
    preserved=root/'initial_qualification';preserved.mkdir(exist_ok=True)
    for name in ('semantic_qualification.json','stable_core_analysis.json','stable_anchor_timeline.jsonl','calibration_frozen_rules.json'):
        if (preserved/name).exists() and sha256(preserved/name)!=sha256(root/name):
            raise ValueError('Initial qualification archive differs from original')
        shutil.copy2(root/name,preserved/name)
    # Retain the initial decision, but require a new one-shot publication after
    # the extension; calibration_frozen_rules.json is never rewritten.
    (root/'semantic_qualification.json').rename(preserved/'initial_policy_original.json')
    write(root/'validation_extension_sequence.json',seq)
    write(root/'qualification_extensions.json',dict(sequence='validation_extension_sequence.json',
        capture='validation_extension_capture',frames=96,new_log_units=12,
        sequence_sha256=sha256(root/'validation_extension_sequence.json'),
        frozen_calibration_sha256=sha256(root/'calibration_frozen_rules.json'),
        reason='Fixed car rule scored only 5 heldout edges in one scene; increase coverage without threshold retuning',
        selection='next metadata-sorted 12 unused official training logs, no detector outputs or GT used in selection'))
    plan['scene_splits'].update(seq['scene_splits']);plan['scene_logs'].update(seq['scene_logs'])
    plan['validation_extension']='qualification_extensions.json';write(root/'experiment_plan.json',plan)
    with gpu_job_lock():
        wait_gpu();execute(root,'validation_extension_capture',['-m','tools.worldstream_m123.benchmark.worker',
            '--output',str(root/'validation_extension_capture'),'--run-root',str(root),'--condition','B3',
            '--workload','language','--limit','96','--warmup','1','--dataset-split','train',
            '--sequence-file',str(root/'validation_extension_sequence.json'),'--questions-per-frame','1',
            '--m3-zero-reuse','--m3-strategy','exact_prefix','--capture-semantic-candidates'])


def prepare_adapter_data(root):
    if not json.loads((root/'stable_core_analysis.json').read_text())['stage_one_pass']:
        raise ValueError('Independent Stable Core quality/lifetime gate did not pass')
    from mmcv import Config
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    infos=pickle.load(open(cfg.data.train.ann_file,'rb'))['infos']
    scene_table=json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())
    logs={s['token']:s['log_token'] for s in scene_table}
    plan=json.loads((root/'experiment_plan.json').read_text())
    forbidden=set(plan['scene_logs'].values())
    grouped=defaultdict(list)
    for info in infos:
        if logs[info['scene_token']] not in forbidden:grouped[info['scene_token']].append(info)
    selected=[];used=set()
    for scene,ss in sorted(grouped.items()):
        segment=sorted(ss,key=lambda i:i['timestamp'])[:8]
        if len(segment)<8 or logs[scene] in used:continue
        paths=[Path('data/nuscenes/conv/train')/(i['token']+'.json') for i in segment]
        if any(not p.exists() or len(json.loads(p.read_text()))<3 for p in paths):continue
        validate_sequence(segment,scene)
        if not all(a.get('next')==b['token'] and b.get('prev')==a['token'] for a,b in zip(segment,segment[1:])):continue
        selected.append((scene,segment,paths));used.add(logs[scene])
        if len(selected)==8:break
    if len(selected)!=8:raise ValueError('Insufficient separate train/validation logs for adapter')
    manifest=dict(tokens=[],questions={},scene_splits={},scene_logs={})
    for j,(scene,segment,paths) in enumerate(selected):
        manifest['scene_splits'][scene]='train' if j<4 else 'validation';manifest['scene_logs'][scene]=logs[scene]
        for i,p in zip(segment,paths):
            manifest['tokens'].append(i['token']);manifest['questions'][i['token']]=[p['question'] for p in json.loads(p.read_text())[:3]]
    if (root/'adaptation_sequence.json').exists():raise FileExistsError('Preserve adapter selection')
    write(root/'adaptation_sequence.json',manifest)


def adapter_capture(root):
    if not json.loads((root/'stable_core_analysis.json').read_text())['stage_one_pass']:raise ValueError('Stable core gate failed')
    if (root/'adaptation_capture').exists():raise FileExistsError('Preserve previous capture')
    with gpu_job_lock():
        wait_gpu();execute(root,'adaptation_capture',['-m','tools.worldstream_m123.benchmark.worker',
            '--output',str(root/'adaptation_capture'),'--run-root',str(root),'--condition','B3',
            '--workload','language','--limit','64','--warmup','1','--dataset-split','train',
            '--sequence-file',str(root/'adaptation_sequence.json'),'--questions-per-frame','3',
            '--m3-zero-reuse','--m3-strategy','exact_prefix','--capture-m3-inputs','--capture-stable-world',
            '--semantic-qualification',str(root/'semantic_qualification.json')])


def training_manifest(root):
    from tools.worldstream_m123.latency_controls import load_capture
    run=root/'adaptation_capture'
    if json.loads((run/'status.json').read_text())['status']!='COMPLETED':raise ValueError('Capture incomplete')
    seq=json.loads((root/'adaptation_sequence.json').read_text())
    acceptance=json.loads((root/'acceptance_reserved_sequence.json').read_text())
    samples=[];skipped=[];base_hash=None
    from dataclasses import asdict
    for token in seq['tokens']:
        capture=run/'m3_inputs'/(token+'.pt');value=load_capture(capture)
        snapshot=value['stable_snapshot']
        if not snapshot.facts:skipped.append(token)
        source=Path('data/nuscenes/conv/train')/(token+'.json');pairs=json.loads(source.read_text())
        training_pairs=[]
        for qi,question in enumerate(value['questions']):
            pair=next(p for p in pairs if p['question']==question)
            training_pairs.append(dict(question_index=qi,question=question,answer=pair['answer']))
        samples.append(dict(scene_id=snapshot.scene_id,frame_id=token,capture=str(capture),capture_sha256=sha256(capture),
            stable_snapshot=asdict(snapshot),question_source=str(source),question_source_sha256=sha256(source),
            pairs=training_pairs))
        base_hash=value['context']['weights_sha256']
    manifest=dict(scene_splits=dict(seq['scene_splits'],**acceptance['scene_splits']),scene_logs=dict(seq['scene_logs'],**acceptance['scene_logs']),
        nuscenes_scene_table='data/nuscenes/v1.0-trainval/scene.json',semantic_qualification=str(root/'semantic_qualification.json'),
        base_weights_sha256=base_hash,samples=samples,real_cold_start_frames_without_facts=skipped,
        note='Real empty-core frames are trained for cold starts; they never count as stable anchor hits')
    write(root/'training_manifest.json',manifest)


def train_adapter(root):
    if not json.loads((root/'stable_core_analysis.json').read_text())['stage_one_pass']:raise ValueError('Stable core gate failed')
    with gpu_job_lock():
        wait_gpu();execute(root,'adapter_training',['-m','tools.worldstream_m123.wrcps_train',
            '--manifest',str(root/'training_manifest.json'),'--output',str(root/'adapter_training'),
            '--stable-tokens','128','--rank','128','--epochs','3','--seed','20261028'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['prepare','capture','qualify','extend-validation','prepare-adapter','capture-adapter','training-manifest','train'],required=True);a=p.parse_args()
    if set(os.sched_getaffinity(0)) & {4,5}:p.error('Use taskset -c 0-3,6-31')
    root=Path(a.root)
    if a.stage=='prepare':prepare(root)
    elif a.stage=='capture':capture(root)
    elif a.stage=='extend-validation':extend_validation(root)
    elif a.stage=='qualify':
        from tools.worldstream_m123.wrcps_semantic_qualification import qualify
        qualify(root)
    elif a.stage=='prepare-adapter':prepare_adapter_data(root)
    elif a.stage=='capture-adapter':adapter_capture(root)
    elif a.stage=='training-manifest':training_manifest(root)
    elif a.stage=='train':train_adapter(root)


if __name__=='__main__':main()
