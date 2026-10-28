"""Bounded TRAIN-only SF-WRCPS event audit, without LLM decoding or new rules."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
from time import perf_counter

from tools.worldstream_m123.benchmark.common import write,sha256

PRIOR=Path('outputs/m3_sf_wrcps_optimization_run01')


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


def prepare(root):
    from mmcv import Config
    from tools.worldstream_m123.benchmark.sequence import validate_sequence
    root.mkdir(parents=True,exist_ok=False)
    policy=json.loads((PRIOR/'semantic_qualification.json').read_text())
    adaptation=json.loads((PRIOR/'adaptation_sequence.json').read_text())
    target=json.loads((PRIOR/'target_sequence.json').read_text())
    scene_table=json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())
    logs={s['token']:s['log_token'] for s in scene_table}
    forbidden=set(policy['calibration_log_ids']+policy['validation_log_ids'])
    forbidden.update(adaptation['scene_logs'][s] for s,v in adaptation['scene_splits'].items() if v=='validation')
    forbidden.update(target['scene_logs'].values())
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    infos=pickle.load(open(cfg.data.train.ann_file,'rb'))['infos'];groups=defaultdict(list)
    for r in infos:groups[r['scene_token']].append(r)
    # Current qualification/validation/acceptance are protected. Historical
    # TRAIN exposure is disclosed, not mechanically excluded from training.
    exposure=defaultdict(list);positives=Counter()
    for old in ('m3_semantic_core_wrcps_run01','m3_sf_wrcps_optimization_run01'):
        p=Path('outputs')/old
        for name in ('qualification_sequence.json','adaptation_sequence.json'):
            f=p/name
            if f.exists():
                for s in json.loads(f.read_text())['scene_splits']:exposure[s].append(str(f))
        f=p/'stable_anchor_timeline.jsonl'
        if f.exists():
            for r in rows(f):
                if r['anchor_hit']:positives[r['scene_id']]+=1
        f=p/'adaptation_capture/stable_world_snapshots.jsonl'
        if f.exists():
            for r in rows(f):
                if r['facts']:positives[r['scene_id']]+=1
    pool=[];eligible={}
    for scene,ss in sorted(groups.items()):
        ss=sorted(ss,key=lambda r:r['timestamp']);segment=ss[:32]
        reason=None
        if len(segment)<32:reason='less_than_32_frames'
        elif any(not a.get('next')==b['token'] or not b.get('prev')==a['token'] for a,b in zip(segment,segment[1:])):reason='not_continuous'
        elif any(len(r['cams'])!=6 or any(not Path(c['data_path']).exists() for c in r['cams'].values()) for r in segment):reason='camera_data_missing'
        elif any(not (Path('data/nuscenes/conv/train')/(r['token']+'.json')).exists() for r in segment):reason='language_source_missing'
        else:
            pairs=[json.loads((Path('data/nuscenes/conv/train')/(r['token']+'.json')).read_text()) for r in segment]
            if any(len(p)<3 or any(not q.get('question') or not q.get('answer') for q in p[:3]) for p in pairs):reason='less_than_three_original_QA'
        heldout=logs[scene] in forbidden
        pool.append(dict(scene_id=scene,log_id=logs[scene],frames_available=len(ss),first_32_data_failure=reason,
            training_eligible=not heldout and reason is None,protected_log=heldout,
            known_recent_exposure=exposure[scene],historical_positive_count=positives[scene],
            exposure_scope='Recent semantic/SF manifests; older general exposure may exist and is not called unseen'))
        if reason is None:eligible[scene]=segment
    original=[s for s,v in adaptation['scene_splits'].items() if v=='train']
    validation=[s for s,v in adaptation['scene_splits'].items() if v=='validation']
    chosen=list(original);used={logs[s] for s in chosen}
    ranked=sorted((r for r in pool if r['training_eligible'] and r['scene_id'] not in chosen),
        key=lambda r:(-r['historical_positive_count'],r['scene_id']))
    for r in ranked:
        if r['log_id'] in used:continue
        chosen.append(r['scene_id']);used.add(r['log_id'])
        if len(chosen)==14:break
    if len(chosen)!=14 or any(s not in eligible for s in chosen+validation):raise ValueError('Bounded 32-frame data unavailable; inspect pool, do not replace from heldout outputs')
    sequence=dict(tokens=[],scene_splits={},scene_logs={},questions={})
    for s in chosen+validation:
        role='train' if s in chosen else 'validation';sequence['scene_splits'][s]=role;sequence['scene_logs'][s]=logs[s]
        validate_sequence(eligible[s],s)
        for r in eligible[s]:
            t=r['token'];sequence['tokens'].append(t)
            sequence['questions'][t]=[p['question'] for p in json.loads((Path('data/nuscenes/conv/train')/(t+'.json')).read_text())[:3]]
    write(root/'official_train_pool_audit.json',dict(frames=len(infos),scenes=len(groups),logs=len({logs[s] for s in groups}),
        protected_logs=sorted(forbidden),eligible_scenes=sum(r['training_eligible'] for r in pool),scenes_detail=pool))
    write(root/'audit_sequence.json',sequence)
    write(root/'experiment_plan.json',dict(base_commit='ec4c667736a72afb2cafb1635540d9db613f333c',seed=20261028,
        qualification_path=str(PRIOR/'semantic_qualification.json'),qualification_sha256=sha256(PRIOR/'semantic_qualification.json'),
        original_train_scenes=original,expanded_train_scenes=chosen[6:],validation_scenes=validation,
        training_logs=sorted(used),protected_logs=sorted(forbidden),windows=[8,16,32],
        bounded_audit=dict(train_logs=14,validation_logs=4,frames=576,selection='Old six train logs plus eight distinct legal TRAIN logs ranked by existing positive evidence; all first32 frames'),
        enrichment=dict(rule='Keep complete causal event windows, no duplicate frames; retain no-core boundaries and original first three QA',
            min_training_positive_logs=3,min_training_core_sequences=6,min_training_valid_frames=24,min_training_extensions=12,
            min_validation_positive_logs=1,min_validation_extensions=3),
        micro=dict(repeats=9,warmup=2,seed=20261028,primary_K=64,controls=['full','text','text_plus64'],
            note='Controlled exact-prefix upper bound, not trained SF or service quality; same executor'),
        training=dict(K=64,rank=128,epochs=3,conditional=True),
        quality_budgets=dict(mAP=.1,NDS=.1,DET_l=.1,Token_F1=.1),
        audit_sequence_sha256=sha256(root/'audit_sequence.json'),acceptance_sequence_path=str(PRIOR/'target_sequence.json'),
        acceptance_sequence_sha256=sha256(PRIOR/'target_sequence.json')))


def audit(root):
    import random
    import hashlib
    import torch
    import projects.mmdet3d_plugin
    from mmcv.parallel import collate,scatter
    from mmdet3d.datasets import build_dataset
    from tools.worldstream_m123.benchmark.worker import effective_config,load_model,clear_questions,select_named_questions,reset
    from tools.worldstream_m123.benchmark.observe import Observer,m1_record
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    from worldstream.m3.semantic_core import SemanticQualifier,committed_semantic_candidates
    from worldstream.m3.stable_world import freeze_stable_world,StableAnchorManager
    from tools.worldstream_m1.dataset import load_oracle
    from tools.worldstream_m1.identity_metrics import strict_labels
    plan=json.loads((root/'experiment_plan.json').read_text());seq=json.loads((root/'audit_sequence.json').read_text())
    assert sha256(root/'audit_sequence.json')==plan['audit_sequence_sha256']
    assert sha256(plan['qualification_path'])==plan['qualification_sha256']
    dest=root/'state_audit';dest.mkdir(exist_ok=False);(dest/'m3_inputs').mkdir()
    write(dest/'status.json',dict(status='RUNNING',scope='Offline M1/M2 causal audit with zero language generation; not service efficiency'))
    with gpu_job_lock():
        wait_gpu();torch.manual_seed(plan['seed'])
        cfg=effective_config('B3','language',dest,'native_full')
        # Only tool-side data/output overrides. Keep the native perception,
        # M1 association and M2 execution; zero prompts avoids LLM decoding.
        cfg.data.test.ann_file=cfg.data.train.ann_file
        for node in cfg.data.test.pipeline:
            if node['type']=='LoadAnnoatationVQATest':
                node['base_conv_path']='./data/nuscenes/conv/train/'
                node['base_vqa_path']='./data/nuscenes/vqa/train/'
        write(dest/'effective_config.json',dict(config=cfg._cfg_dict.to_dict(),
            scope='Offline TRAIN state/capture audit, M3 disabled and zero prompts; never a service latency claim'))
        dataset=build_dataset(cfg.data.test);lookup={r['token']:r for r in dataset.data_infos}
        selected=[lookup[t] for t in seq['tokens']];dataset.data_infos=selected
        model=load_model(cfg,'ckpts/iter_10548.pth',dest).eval()
        qualifier=SemanticQualifier(plan['qualification_path']);observer=Observer(model)
        annotations=load_oracle(selected,'data/nuscenes');previous_labels={};manager=StableAnchorManager()
        buffers={};handles=[]
        for name,head in [('object',model.pts_bbox_head),('map',model.map_head)]:
            def collect(module,args,output,_name=name):buffers[_name]=output[1].detach()
            handles.append(head.register_forward_hook(collect))
        previous_scene=None;birth={};last_ids=set()
        with (dest/'frames.jsonl').open('w') as stream:
            for ordinal,info in enumerate(selected):
                seed=int(hashlib.sha256(info['token'].encode()).hexdigest()[:8],16)%(2**31)
                random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                observer.clear();buffers.clear()
                if info['scene_token']!=previous_scene:reset(model);last_ids=set()
                data=scatter(collate([dataset[ordinal]],samples_per_gpu=1),[0])[0]
                # Retain original question encodings before clearing only this
                # audit invocation. No shortened language service is measured.
                select_named_questions(data,seq['questions'][info['token']])
                original_ids=data['input_ids'][0][0]
                ids=[t.detach().cpu() for t in original_ids[:3]]
                clear_questions(data)
                tick=perf_counter()
                with torch.no_grad():model(return_loss=False,rescale=True,**data)
                torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
                if len(observer.commits)!=1 or observer.native_generations:raise ValueError('Audit must commit M1 once and decode zero LLM requests')
                frame,anchors,dense=m1_record(model,observer.commits[-1]);m=model.m1_manager
                context=dict(scene_id=frame.scene_id,frame_id=frame.frame_id,timestamp=frame.timestamp,
                    state_epoch=m.store.generation,state_version=m.store.commit_count,state_execution_signature=m.store.execution_signature)
                candidates=committed_semantic_candidates(m,context);selection=qualifier.select(candidates)
                snapshot=freeze_stable_world(m,context,qualifier);anchor,hit=manager.update(snapshot,context)
                labels=strict_labels(anchors,annotations.get(frame.frame_id,[]),gate=2.)
                current={aid:labels.get(i) for aid,i in zip(dense['commit']['committed_anchor_ids'],dense['commit']['selected_prediction_indices'])}
                if not hit:birth={f[0]:current.get(f[0]) for f in anchor.facts}
                checks=[dict(anchor_id=f[0],status='unknown' if birth.get(f[0]) is None or current.get(f[0]) is None
                    else 'correct' if birth[f[0]]==current[f[0]] else 'wrong') for f in anchor.facts]
                # Independent per-filter observations, not just first rejection.
                detail=[]
                for record in candidates['records']:
                    a=m.store.anchors[record['anchor_id']];history=a.association_history;last=history[-1] if history else {}
                    detail.append(dict(anchor_id=a.anchor_id,semantic_class=a.semantic_class,score=a.source_score,
                        current_observation_support=len(a.observation_support),source_is_previous_frame=bool(last.get('source_is_previous_frame')),
                        residual_m=last.get('association_residual_m'),eligible=record['eligible_observation']))
                current_ids={r['anchor_id'] for r in candidates['records']}
                row=dict(ordinal=ordinal,scene_id=frame.scene_id,frame_id=frame.frame_id,log_id=seq['scene_logs'][frame.scene_id],
                    split=seq['scene_splits'][frame.scene_id],timestamp=frame.timestamp,candidates=candidates,selection=selection,
                    anchor=asdict(anchor),anchor_hit=hit,core_fact_checks_offline=checks,filter_details=detail,
                    previous_current_ids_lost=sorted(last_ids-current_ids) if previous_scene==frame.scene_id else [],
                    new_current_ids=sorted(current_ids-last_ids) if previous_scene==frame.scene_id else [],
                    snapshot=asdict(snapshot),m1_commit=dense['commit'],m1_precheck_diagnostics=dense['precheck']['diagnostics'],
                    state_audit_model_wall_ms=wall,GT_online=False,LLM_decode_requests=0)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                carriers=torch.cat([buffers['object'],buffers['map']],1).cpu()
                if carriers.shape!=(1,513,4096) or len(ids)!=3:raise ValueError('Native carrier/question capture contract failed')
                capture=dict(carriers=carriers,ids=ids,questions=seq['questions'][frame.frame_id],stable_snapshot=snapshot,
                    stable_context=context,context=dict(context,weights_sha256='6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d',
                        model_version='6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d',camera_ids=list(info['cams'])),
                    generation=dict(do_sample=True,temperature=.1,top_p=.75,num_beams=1,max_new_tokens=320,use_cache=True),
                    rng=[(torch.get_rng_state(),torch.cuda.get_rng_state()) for _ in range(3)],
                    source_scope='Real causal M1/M2 capture, zero LLM decode; not service timing')
                torch.save(capture,dest/'m3_inputs'/(frame.frame_id+'.pt'))
                last_ids=current_ids;previous_scene=frame.scene_id
                if (ordinal+1)%8==0:
                    print('STATE_AUDIT',ordinal+1,'/',len(selected),flush=True)
                    write(dest/'progress.json',dict(completed_frames=ordinal+1,target_frames=len(selected)))
        for h in handles:h.remove()
        write(dest/'status.json',dict(status='COMPLETED',frames=len(selected),LLM_decode_requests=0,
            qualification_sha256=qualifier.sha256,scope='Native perception/M1/M2 audit; not formal VLM service'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['prepare','audit'],required=True);a=p.parse_args()
    if {4,5}&set(os.sched_getaffinity(0)):p.error('Use taskset -c 0-3,6-31')
    (prepare if a.stage=='prepare' else audit)(Path(a.root))


if __name__=='__main__':main()
