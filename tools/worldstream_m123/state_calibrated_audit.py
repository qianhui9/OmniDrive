"""Offline evidence audits; never imported by online WRSAR decisions."""
import argparse
from collections import Counter, defaultdict
import json
import math
import subprocess
from pathlib import Path
from .benchmark.common import write,sha256
from .selective_answer_reuse_report import lines,label,grade
from .selective_answer_reuse_audit import match_annotation

ROOT=Path('outputs/m3_state_calibrated_bypass_run01')
PRIOR=Path('outputs/m3_temporal_selective_bypass_run01')


def prior_veto_audit(root):
    root.mkdir(parents=True,exist_ok=True)
    rows=lines(PRIOR/'run/requests.jsonl');ab=lines(PRIOR/'ablation/requests.jsonl')
    objects=json.loads((PRIOR/'current_frame_facts.json').read_text())['frames']
    cs={(r['scene_id'],r['frame_id'],r['round']):r for r in rows
        if r['condition']=='C' and r['workload']=='addressable' and r['question_count']==3}
    previous={r['frame_id']:r for r in rows if r['condition']=='C' and r['workload']=='addressable'
              and r['question_count']==3 and r['round']==0}
    grouped={};records=[]
    for r in ab:
        if r['question_count']!=3:continue
        t=cs[(r['scene_id'],r['frame_id'],r['round'])]
        for qi,(cur,temp) in enumerate(zip(r['requests'],t['requests'])):
            if not cur['llm_skipped'] or temp['llm_skipped']:continue
            f=cur['fast_answer']['facts'][0]
            # Reset-generated IDs differ across jobs; selected target is bound
            # by current class/geometry/source, not arbitrary store generation.
            key=(r['scene_id'],r['frame_id'],cur['question'],f['semantic_class'],tuple(f['center_ego']))
            records.append(dict(scene_id=r['scene_id'],frame_id=r['frame_id'],round=r['round'],question_index=qi))
            if key in grouped:
                grouped[key]['rounds'].append(r['round']);continue
            q=cur['question'];current_answer=r['answers'][qi]['A'][0];temporal_answer=t['answers'][qi]['A'][0]
            cg=grade(q,current_answer,r['frame_id'],objects[r['frame_id']]);tg=grade(q,temporal_answer,r['frame_id'],objects[r['frame_id']])
            category=('unnecessary_veto' if cg['known'] and cg['score']==1 else
                      'beneficial_veto' if cg['known'] and cg['score']==0 and tg['known'] and tg['score']==1 else 'unknown_veto')
            history=[]
            for a in temp['fast_answer']['target_binding']['temporal_competitors']:
                prior=previous.get(a['source_frame']);anchor=next((x for x in prior['committed_world']['anchors']
                    if x['anchor_id']==a['anchor_id']),None) if prior else None
                old_gt=match_annotation(anchor['center_ego'],objects[a['source_frame']]) if anchor else None
                now=next((g for g in objects[r['frame_id']] if old_gt and g['instance_id']==old_gt['instance_id']),None)
                history.append(dict(**a,previous_annotation=old_gt,current_same_instance=now,
                    current_physical_presence=bool(now) if old_gt else None,
                    current_visible=now['visible'] if now else None,
                    current_direct_ROI=bool(now and now['center_ego'][0]>0 and abs(now['center_ego'][1])<=4.
                        and math.hypot(*now['center_ego'][:2])<=50.) if now else None,
                    ego_transform_source=dict(previous_pose=prior['committed_world']['ego_to_world'] if prior else None,
                        current_pose=t['committed_world']['ego_to_world'],dt_s=t['timestamp']-a['source_timestamp'])))
            from worldstream.m3.task_conditioner import question_facts
            from worldstream.m3.answer_validity import fact_region
            from worldstream.m3.world_delta import VEHICLES
            eligible=sorted((g for g in objects[r['frame_id']] if g['visible'] and g['semantic_class'] in VEHICLES
                and fact_region(g['center_ego'],question_facts(q))),key=lambda g:math.hypot(*g['center_ego'][:2]))
            grouped[key]=dict(scene_id=r['scene_id'],frame_id=r['frame_id'],question=q,question_index=qi,
                selected_target=f,rounds=[r['round']],current_only_answer=current_answer,temporal_answer=temporal_answer,
                current_quality=cg,temporal_quality=tg,current_visible_ROI_annotations=eligible,history=history,
                veto_class=category,reason=temp['fast_answer']['reason'],
                added_llm_forward_calls=temp['llm_forward_calls']-cur['llm_forward_calls'],
                added_request_wall_ms=temp['end_to_end_ms']-cur['end_to_end_ms'],
                human_review='PENDING_NOT_PERFORMED')
    cases=list(grouped.values());counts=dict(Counter(c['veto_class'] for c in cases))
    names=['implementation_summary.md','result_answers.md','temporal_state_contribution.json','temporal_competitor_analysis.json',
        'target_binding_analysis.json','old_binding_regression.json','task_quality.json','severe_error_audit.json',
        'quality_latency_pareto.json','runtime_cost_report.md','go_no_go.json']
    write(root/'prior_evidence_read.json',{n:dict(sha256=sha256(PRIOR/n),bytes=len((PRIOR/n).read_bytes())) for n in names})
    write(root/'temporal_veto_quality_audit.json',dict(status='OFFLINE_PRIOR_TEST_DIAGNOSTIC',raw_three_Q_records=len(records),
        independent_cases=len(cases),independent_scenes=len({c['scene_id'] for c in cases}),counts=counts,cases=cases,
        unchanged_label_protocol='Current visible sample annotations, exact question ROI, >=2m nearest-class ambiguity retained; no UNKNOWN as beneficial',
        decision='No proven beneficial veto: retain Current-Only and evaluate temporal optionality from DEV only' if not counts.get('beneficial_veto') else 'Measured benefit warrants bounded DEV calibration',
        human_review_performed=False))
    (root/'temporal_veto_case_review.md').write_text('# Prior temporal-veto evidence\n\n'+
        f'{len(records)} three-question records reduce to {len(cases)} independent Scene/Frame/Question/Selected-Target cases on {len({c["scene_id"] for c in cases})} scene. Counts: {counts}.\n\n'+
        'All original nearest-class ambiguity and visibility conditions remain. Current QA mentions multiple vehicles but does not resolve the nearest target. Exact coordinate nearest is not promoted to a reliable visual semantic label when multiple different classes lie within the frozen2m ambiguity margin. No human review has been performed. UNKNOWN is neither beneficial nor unnecessary. No new Temporal Risk formula is justified by this audit alone.\n\n'+
        '\n\n'.join(f"- {c['frame_id']}: {c['current_only_answer']} -> {c['temporal_answer']}; {c['veto_class']}; {c['current_quality']['reference']}; {c['added_llm_forward_calls']} additional LLM forwards." for c in cases)+'\n')
    render_cases(root,cases,objects,previous)
    print(json.dumps(dict(raw_records=len(records),independent_cases=len(cases),counts=counts)),flush=True)


def render_cases(root,cases,objects,previous):
    """Deterministic review preparation from actual images and pose projections."""
    import os,pickle
    os.environ.setdefault('MPLCONFIGDIR','/tmp/omnidrive-wrsar-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from pyquaternion import Quaternion
    from tools.worldstream_m1.dataset import resolve_image
    infos={i['token']:i for i in pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']}
    def pose(r,t):
        p=np.eye(4);p[:3,:3]=Quaternion(r).rotation_matrix;p[:3,3]=t;return p
    dest=root/'review_images';dest.mkdir(exist_ok=True)
    for case in cases:
        token=case['frame_id'];info=infos[token];world=pose(info['ego2global_rotation'],info['ego2global_translation'])
        fig,axes=plt.subplots(2,3,figsize=(16,6.8))
        for ax,(cid,cam) in zip(axes.flat,info['cams'].items()):
            ax.imshow(Image.open(resolve_image(cam['data_path'],'data/nuscenes')))
            w2c=np.linalg.inv(pose(cam['ego2global_rotation'],cam['ego2global_translation'])@
                             pose(cam['sensor2ego_rotation'],cam['sensor2ego_translation']))
            def point(center):
                c=w2c@world@np.r_[center,1.];p=np.asarray(cam['cam_intrinsic'])@c[:3]
                return p[:2]/p[2] if c[2]>0 else None
            for g in objects[token]:
                if g['semantic_class'] not in ('car','truck','construction_vehicle'):continue
                p=point(g['center_ego'])
                if p is None or not (0<=p[0]<1600 and 0<=p[1]<900):continue
                color='lime' if g['visible'] else 'orange'
                ax.scatter(*p,s=15,c=color);ax.text(*p,g['semantic_class'],color=color,fontsize=6)
            for a in previous[token]['committed_world']['anchors']:
                if not a['observed'] or a['anchor_type']!='object':continue
                p=point(a['center_ego'])
                if p is not None and 0<=p[0]<1600 and 0<=p[1]<900:ax.scatter(*p,s=13,c='cyan',marker='x')
            ax.set_title(cid);ax.axis('off')
        fig.suptitle(token+' | GT centers: green=visible, orange=uncertain; M1=cyan x | REVIEW PENDING')
        fig.tight_layout();fig.savefig(dest/(token+'.png'),dpi=130);plt.close(fig)


def first_frame_audit(root):
    """CPU reconstruction from actual recorded raw detector scalars, not GT inputs."""
    import numpy as np
    import torch
    import projects.mmdet3d_plugin
    from mmcv import Config
    from mmcv.parallel import collate,scatter
    from mmdet3d.datasets import build_dataset
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from worldstream.m1 import WorldStateManager
    from worldstream.m1.omnidrive_adapter import frame_from_model,outputs_from_model
    from worldstream.m1.geometry import project_anchor
    from worldstream.m3.world_state import freeze_committed_state
    from worldstream.m3.answer_validity import propose_fact_answer
    from worldstream.m3.answer_memory import AnswerMemory
    from .benchmark.worker import effective_config
    config_out=root/'cpu_reconstruction_config';config_out.mkdir(exist_ok=True)
    cfg=effective_config('B3','language',config_out,'native_full')
    dataset=build_dataset(cfg.data.test);index={x['token']:i for i,x in enumerate(dataset.data_infos)}
    captures=json.loads((root/'detection/requests.json').read_text())['rows'];results=[]
    manager=WorldStateManager(cfg.model.m1_config)
    for row in captures:
        data=scatter(collate([dataset[index[row['frame_id']]]],samples_per_gpu=1),[-1])[0]
        meta=data['img_metas'][0]
        plain={k:(v[0] if k in ('img','input_ids') else v[0][0].unsqueeze(0)) for k,v in data.items() if k!='img_metas'}
        plain['img']=plain['img'].half()
        frame=frame_from_model(meta,plain,manager.config)
        raw=row['raw_queries'];raw_centers=np.asarray(raw['boxes'])[:,:3]
        boxes=[];scores=[];labels=[]
        for c in row['decoded_candidates']:
            j=int(np.linalg.norm(raw_centers-np.asarray(c['center_ego']),axis=1).argmin())
            b=list(raw['boxes'][j]);b[2]-=b[5]/2
            boxes.append(b);scores.append(c['score']);labels.append(c['class_id'])
        result=[dict(boxes_3d=LiDARInstance3DBoxes(torch.tensor(boxes),box_dim=len(boxes[0])),
                     scores_3d=torch.tensor(scores),labels_3d=torch.tensor(labels))]
        manager.reset();pre=manager.precheck(frame)
        outputs=outputs_from_model(frame,result,[],{},manager.config)
        manager.commit(frame,outputs,pre)
        old=freeze_committed_state(manager,frame);new=freeze_committed_state(manager,frame,current_detection_support=True)
        ctx=dict(scene_id=frame.scene_id,frame_id=frame.frame_id,timestamp=frame.timestamp,state_epoch=new.epoch,
            state_execution_signature=new.execution_signature,current_state_version=new.version,camera_ids=list(frame.cameras),frame_index=0)
        policy=json.loads((PRIOR/'selected_policy.json').read_text())
        old_p=propose_fact_answer('What type of vehicle is directly ahead?',old,ctx,AnswerMemory(),('diagnostic',),policy,None,True,False)
        new_p=propose_fact_answer('What type of vehicle is directly ahead?',new,ctx,AnswerMemory(),('diagnostic',),policy,None,True,False)
        projections=[]
        for anchor in manager.store.anchors.values():
            center=np.linalg.inv(frame.ego_to_world)@np.r_[anchor.geometry_world.center,1.]
            if anchor.semantic_class!='car' or not (0<center[0]<15):continue
            for cid,cam in frame.cameras.items():
                p=project_anchor(anchor,cam,manager.config)
                projections.append(dict(anchor_id=anchor.anchor_id,center_ego=center[:3].tolist(),camera_id=cid,
                    timestamp=cam.timestamp,frame_timestamp=frame.timestamp,timing_reliable=cam.timing_reliable,
                    valid=p.valid,coverage=p.coverage,reason=p.reason,
                    descriptor_exists=manager.validator.descriptor(cam.image,p.region) is not None if p.valid else False))
        results.append(dict(frame_id=frame.frame_id,scene_id=frame.scene_id,projections=projections,
            old_proposal=old_p,new_proposal=new_p,old_observed_objects=sum(a.observed for a in old.anchors),
            new_observed_objects=sum(a.observed for a in new.anchors),
            actual_online_network='Previous captured detector scalar outputs; CPU replay, no extra network forward'))
    write(root/'first_frame_detection_chain.json',dict(results=results,scope='Diagnostic actual-output replay; not final online E2E or generalization evidence'))
    (root/'first_frame_detection_audit.md').write_text('# First-frame Car/Truck chain\n\n'+
        'The actual current head detects Car at(8.507,-3.859,0.942)m with score0.874642. It survives decoder,0.3 M1 score filter, semantic mapping, Ego/world transformation and commit. M1 stores it as a current actual_perception anchor, but no reusable photometric observation exists. The old M3 snapshot requires that observation to call the current detection observed and discards it from answering. Both GTy=-3.939 and predictedy=-3.859 are inside the unchanged4m direct ROI. This is a snapshot evidence mismatch, not a detector miss or DQTrack identity deletion.\n\n'+
        'The minimal opt-in M3 fix accepts current committed detector evidence with calibrated current camera-center projection, actual exposure-time pose and current source receipt. No historical prediction becomes a positive fact, and no M1/M2 reuse certificate is created or loosened. All detector/score/ROI/model thresholds remain unchanged. Exact per-camera whole-box coverage and descriptor results, plus replay decisions on four actual captures, are in first_frame_detection_chain.json. CPU replay is not counted as a new GPU model call or as proof on new TEST.\n')
    print(json.dumps([dict(frame_id=r['frame_id'],old=r['old_proposal']['answer_text'],new=r['new_proposal']['answer_text'],
        old_observed=r['old_observed_objects'],new_observed=r['new_observed_objects']) for r in results]),flush=True)


def verify(root,checkpoint=False,manifest=False):
    if not checkpoint and not manifest:
        assert (root/'artifact_manifest.json').exists(),'Finalize the artifact manifest first'
        assert (root/'checkpoint_verification.json').exists(),'Rehash the checkpoint first'
        previous=json.loads((root/'integrity_audit.json').read_text())
        assert previous['audit_tool_sha256']==sha256(Path(__file__))
    freeze=json.loads((root/'POLICY_FREEZE.json').read_text())
    assert sha256(root/'selected_policy.json')==freeze['selected_policy_sha256']
    assert all(sha256(p)==h for p,h in freeze['sources'].items())
    for p,h in freeze['sources'].items():assert sha256(root/'frozen_sources'/p)==h
    plan=json.loads((root/'experiment_plan.json').read_text());protocol=json.loads((root/'protocol_freeze.json').read_text())
    assert sha256(root/'experiment_plan.json')==protocol['plan_sha256']
    logs={s['log_id'] for s in plan['test_scenes']}
    assert len(logs)==3 and not logs&set(plan['prior_excluded_log_ids'])
    assert not logs&{s['log_id'] for s in plan['development_scenes']}
    stages={};skips=Counter();expected=json.loads(Path('outputs/m3_world_guided_decode_run01/native_decode_profile.json').read_text())['model_sha256']
    for stage in ('detection','shadow','pareto','run','ablation'):
        out=root/stage;status=json.loads((out/'run_status.json').read_text());assert status['status']=='COMPLETED'
        sources=json.loads((out/'execution_sources.json').read_text())['sources']
        for p,h in sources.items():
            p=Path(p);rel=p.relative_to(Path.cwd()) if p.is_absolute() else p
            assert sha256(out/'executed_sources'/rel)==h
            if stage!='detection':assert sha256(p)==h
        if stage=='detection':
            rs=json.loads((out/'requests.json').read_text())['rows'];assert all(r['execution_completed'] for r in rs)
        else:
            rs=lines(out/'requests.jsonl');assert status['checkpoint_sha256']==expected
            for row in rs:
                assert row['execution_completed'] and row['m1_commit_completed'] and row['m2_completed']
                assert row['observed_llm_forward_calls']==sum(q['llm_forward_calls'] for q in row['requests'])
                for q in row['requests']:
                    if q['llm_skipped']:
                        skips[stage]+=1
                        assert q['llm_forward_calls']==0 and q['layout_preparation_ms']==0.
                        assert not q['prefill_completed'] and not q['decode_completed']
                        assert q['fast_answer']['supported'] and q['fast_answer']['admitted']
                        for f in q['fast_answer']['facts']:
                            assert f['source_frame']==row['frame_id'] and f['source_timestamp']==row['timestamp']
                            assert f['evidence_origin']=='actual_perception'
                    elif row['condition']!='A':
                        assert q['executor']=='PrefillExecutor' and q['prefill_completed'] and q['decode_completed']
                        assert q['question_rng']['enabled']
                if stage=='shadow':assert not any(q['llm_skipped'] for q in row['requests'])
        assert len(rs)==status['full_model_requests']
        assert len(rs)==dict(detection=4,shadow=18,pareto=180,run=1008,ablation=96)[stage]
        if stage!='detection':
            from .state_calibrated_bypass import stage_jobs
            scenes=plan['development_scenes'] if stage in ('shadow','pareto') else plan['test_scenes']
            expected_rows={(w,n,rep,c,s['scene_id'],t) for w,n,rep,c,_,_ in stage_jobs(root,stage,plan)
                for s in scenes for t in s['tokens']}
            actual_rows={(r['workload'],r['question_count'],r['round'],r['condition'],r['scene_id'],r['frame_id']) for r in rs}
            assert len(actual_rows)==len(rs) and actual_rows==expected_rows
            assert all(len(r['camera_ids'])==6 and len(r['requests'])==r['question_count'] for r in rs)
        stages[stage]=dict(full_M123_calls=len(rs),executed_sources_verified=len(sources))
    checks=json.loads((root/'current_vs_temporal_ablation.json').read_text());assert checks['current_evidence_equal']
    result=json.loads((root/'validation_gpu/summary.json').read_text())
    assert result['exit_code']==0 and result['CUDA_available'] and not {4,5}&set(result['cpu_affinity'])
    cpu=json.loads((root/'validation_cpu/summary.json').read_text())
    assert cpu['exit_code']==0 and not {4,5}&set(cpu['cpu_affinity'])
    cp=dict(expected_sha256=expected,actual_hash_verified=False)
    if checkpoint:
        h=sha256('ckpts/iter_10548.pth');assert h==expected
        cp.update(actual_sha256=h,actual_hash_verified=True);write(root/'checkpoint_verification.json',cp)
    elif (root/'checkpoint_verification.json').exists():
        cp=json.loads((root/'checkpoint_verification.json').read_text());assert cp['actual_hash_verified'] and cp['actual_sha256']==expected
    audit=dict(status='PASS',stages=stages,full_M123_calls=sum(s['full_M123_calls'] for s in stages.values()),
        actual_skips=dict(skips),formal_skips=skips['run'],regression_tests=result['counts'],checkpoint=cp,
        frozen_policy_quality_and_sources=True,new_disjoint_logs=True,same_current_detections_ablation=True,
        actual_forward_hooks_verified=True,current_perception_receipts_verified=True,deployment_enabled=False,
        audit_tool_sha256=sha256(Path(__file__)))
    if manifest:
        write(root/'integrity_audit.json',audit);files={}
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.name=='artifact_manifest.json' or any('q_r' in x or x.endswith('_answers') for x in p.parts):continue
            assert p.suffix not in ('.pt','.pth','.pkl','.npy','.npz'),'Duplicated tensor: '+str(p)
            files[str(p)]=dict(bytes=p.stat().st_size,sha256=sha256(p))
        write(root/'artifact_manifest.json',dict(status='VERIFIED_DELIVERY',files=files,total_bytes=sum(v['bytes'] for v in files.values())))
    if (root/'artifact_manifest.json').exists():
        m=json.loads((root/'artifact_manifest.json').read_text())
        assert all(sha256(p)==v['sha256'] and Path(p).stat().st_size==v['bytes'] for p,v in m['files'].items())
        audit['manifest_files_verified']=len(m['files'])
    print(json.dumps(audit),flush=True)


def support_contribution(root):
    """Counterfactual removes only the newly exposed current evidence, offline."""
    from dataclasses import replace
    from .state_calibrated_report import world
    from .selective_answer_reuse_report import load_labels
    from worldstream.m3.answer_memory import AnswerMemory
    from worldstream.m3.answer_validity import propose_fact_answer
    rows=[r for r in lines(root/'run/requests.jsonl') if r['condition']=='C' and r['workload']=='addressable' and r['question_count']==3 and r['round']==0]
    objects=json.loads((root/'current_frame_facts.json').read_text())['frames'];policy=json.loads((root/'selected_policy.json').read_text());comparisons=[];previous=None
    for row in rows:
        current=world(row['committed_world'])
        if previous is not None and previous.scene_id!=current.scene_id:previous=None
        legacy=replace(current,anchors=tuple(replace(a,observed=False,cameras=()) if 'center_projection' in a.observation_kind else a for a in current.anchors))
        ctx=dict(scene_id=current.scene_id,frame_id=current.frame_id,timestamp=current.timestamp,state_epoch=current.epoch,
            state_execution_signature=current.execution_signature,current_state_version=current.version,camera_ids=row['camera_ids'],frame_index=row['frame_index'])
        for qi,request in enumerate(row['requests']):
            p=request.get('fast_answer') or {}
            if not p.get('facts'):continue
            old=propose_fact_answer(request['question'],legacy,ctx,AnswerMemory(),('offline',),policy,previous,True,policy['temporal_risk_enabled'])
            comparisons.append(dict(frame_id=row['frame_id'],scene_id=row['scene_id'],question=request['question'],
                actual_new_proposal=p,legacy_snapshot_counterfactual=old,
                new_quality=grade(request['question'],p['answer_text'],row['frame_id'],objects[row['frame_id']]),
                old_quality=grade(request['question'],old['answer_text'],row['frame_id'],objects[row['frame_id']]) if old['supported'] else None,
                actual_llm_skipped=request['llm_skipped']))
        previous=current
    write(root/'observation_support_contribution.json',dict(status='OFFLINE_SAME_ACTUAL_DETECTIONS_COUNTERFACTUAL',comparisons=comparisons,
        scope='No GPU/E2E claim from counterfactual. Remove only current-center-projection witnesses; unknown stays unknown. Actual online skips are independently measured in run/requests.jsonl.'))
    print('OBSERVATION_SUPPORT_COUNTERFACTUAL_SAVED',len(comparisons),flush=True)


def summarize(root):
    """Supplementary tables/plots from frozen raw results, no policy changes."""
    from .selective_answer_reuse_report import mean
    from .temporal_selective_report import performance
    from worldstream.m3.task_conditioner import question_facts
    rows=lines(root/'run/requests.jsonl');events=[e for e in lines(root/'quality_requests.jsonl') if e['stage']=='run']
    objects=json.loads((root/'current_frame_facts.json').read_text())['frames']
    groups=defaultdict(list)
    for e in events:groups[(e['workload'],e['question_count'],e['condition'],e['scene_id'])].append(e)
    tables=[]
    for (w,n,c,scene),es in sorted(groups.items()):
        known=[e for e in es if e['quality']['known']]
        tables.append(dict(workload=w,question_count=n,condition=c,scene_id=scene,requests=len(es),scored=len(known),
            UNKNOWN=len(es)-len(known),score=mean([e['quality']['score'] for e in known]),
            severe=sum(bool(e['quality']['severe_error']) for e in known),actual_skips=sum(e['llm_skipped'] for e in es),
            independent_frame_questions=len({(e['frame_id'],e['question']) for e in known})))
    write(root/'quality_by_scene.json',dict(groups=tables,
        CI_estimand='Scene bootstrap uses equal-weight scene mean paired losses; overall task scores are weighted by known requests. Repeated rounds are not independent scenes.'))
    nearest=[];sources=defaultdict(list)
    for e in events:
        if not e['llm_skipped']:continue
        f=e['proposal']['facts'][0];sources[f['observation_kind']].append(e)
        if question_facts(e['question']).selector!='nearest':continue
        selected=match_annotation(f['center_ego'],objects[e['frame_id']]);reference=e['quality']['reference']
        known=bool(reference.get('known') and selected and reference.get('selected_annotation'))
        nearest.append(dict(frame_id=e['frame_id'],scene_id=e['scene_id'],round=e['round'],question_count=e['question_count'],
            question=e['question'],known=known,selected_match=selected,reference=reference,
            correct=selected['annotation_id']==reference['selected_annotation'] if known else None,
            fact_score=e['quality']['score'],observation_kind=f['observation_kind']))
    write(root/'nearest_target_audit.json',dict(events=nearest,scored=sum(e['known'] for e in nearest),
        correct=sum(e['correct'] is True for e in nearest),UNKNOWN=sum(not e['known'] for e in nearest),
        scope='Actual fast nearest choices only; class-blind unique current visible GT match<2m and second-match gap>0.5m. Task-fact correctness and target-identity correctness are separate. Unmatched is UNKNOWN.'))
    errors=json.loads((root/'target_binding_error_analysis.json').read_text())
    raw={(r['workload'],r['question_count'],r['round'],r['frame_id']):r for r in rows if r['condition']=='C'}
    causes=[]
    for error in errors['errors']:
        cause='normal_target_generation';details={}
        if error['llm_skipped']:
            r=raw[(error['workload'],error['question_count'],error['round'],error['frame_id'])]
            proposal=r['requests'][error['request_index']]['fast_answer'];fact=proposal['facts'][0]
            selected=match_annotation(fact['center_ego'],objects[error['frame_id']])
            reference_id=error['quality']['reference'].get('selected_annotation')
            current_matches=[match_annotation(a['center_ego'],objects[error['frame_id']])
                for a in r['committed_world']['anchors'] if a['anchor_type']=='object' and a['observed']]
            reference_observed=bool(reference_id and any(a and a['annotation_id']==reference_id for a in current_matches))
            if selected is None:cause='unmatched_current_detection_identity_UNKNOWN'
            elif reference_id and selected['annotation_id']!=reference_id:
                cause='nearest_binding_or_candidate_exclusion' if reference_observed else 'reference_not_in_current_supported_candidates'
            elif selected['semantic_class']!=fact['semantic_class']:cause='current_detector_semantic_class_mismatch'
            else:cause='current_fact_or_answer_interpretation_error'
            details=dict(selected_gt_match=selected,reference_annotation_id=reference_id,
                reference_in_current_observed_candidates=reference_observed,proposal=proposal)
        causes.append(dict(**error,offline_cause=cause,cause_evidence=details))
    errors.update(offline_causes=causes,cause_counts=dict(Counter(e['offline_cause'] for e in causes)),
        cause_scope='Offline geometry matching only, no GT online. Missing unique identity matches remain UNKNOWN; absence of a supported reference does not prove a raw-network miss.')
    write(root/'target_binding_error_analysis.json',errors)
    write(root/'fast_quality_by_observation_source.json',dict(groups=[dict(observation_kind=k,requests=len(v),
        scored=sum(e['quality']['known'] for e in v),UNKNOWN=sum(not e['quality']['known'] for e in v),
        score=mean([e['quality']['score'] for e in v if e['quality']['known']]),
        errors=sum(e['quality']['score']<1 for e in v if e['quality']['known']),
        unique_scored_frame_questions=len({(e['frame_id'],e['question']) for e in v if e['quality']['known']}),
        independent_scenes=len({e['scene_id'] for e in v})) for k,v in sources.items()]))
    write(root/'steady_state_performance.json',performance([r for r in rows if r['frame_index']>0],[('A','B'),('B','C'),('A','C')]))
    breakdown=json.loads((root/'abc_comparison.json').read_text())['module_breakdown'];deltas=[]
    for w,n in sorted({(m['workload'],m['question_count']) for m in breakdown}):
        b=next(m for m in breakdown if (m['workload'],m['question_count'],m['condition'])==(w,n,'B'))
        c=next(m for m in breakdown if (m['workload'],m['question_count'],m['condition'])==(w,n,'C'))
        keys=['e2e_wall_ms','m1_wall_ms','m2_wall_ms','heads_wall_ms','snapshot_wall_ms','m3_wall_ms',
            'prefill_cuda_ms_per_question','decode_cuda_ms_per_question','rng_management_wall_ms',
            'answer_memory_selection_wall_ms','answer_memory_management_wall_ms']
        deltas.append(dict(workload=w,question_count=n,C_minus_B={k:c[k]-b[k] for k in keys},
            C_miss_decision_wall_ms=c['miss_decision_wall_ms'],C_parsing_wall_ms=c['question_fact_extraction_wall_ms_per_question']))
    write(root/'unsupported_overhead_attribution.json',dict(deltas=deltas,
        caution='Module spans overlap and are not summed to reconstruct E2E. Observational differences do not identify all CPU/GPU/order causes. End-to-end scene CIs and steady/cold comparisons accompany these means.'))
    paired_misses=[]
    baseline={(r['workload'],r['question_count'],r['round'],r['frame_id']):r for r in rows if r['condition']=='B'}
    for r in rows:
        if r['condition']!='C':continue
        b=baseline[(r['workload'],r['question_count'],r['round'],r['frame_id'])]
        for qi,(br,cr) in enumerate(zip(b['requests'],r['requests'])):
            if cr['llm_skipped']:continue
            paired_misses.append(dict(workload=r['workload'],question_count=r['question_count'],round=r['round'],
                scene_id=r['scene_id'],frame_id=r['frame_id'],request_index=qi,
                same_seed=br['question_rng']['seed']==cr['question_rng']['seed'],same_tokens=br['tokens']==cr['tokens'],
                same_executor=br['executor']==cr['executor']=='PrefillExecutor',
                same_draft_candidates=br.get('draft_candidates')==cr.get('draft_candidates')))
    write(root/'miss_pairing_diagnostic.json',dict(rows=paired_misses,
        all_same_seed=all(r['same_seed'] for r in paired_misses),all_same_executor=all(r['same_executor'] for r in paired_misses),
        same_tokens=sum(r['same_tokens'] for r in paired_misses),miss_requests=len(paired_misses),
        caveat='Question RNG starts agree; earlier skips may change AnswerMemory and draft rejection schedules. Token identity is a diagnostic, not the optimization objective. No causal replay was added.'))
    validation=json.loads((root/'validation_gpu/summary.json').read_text());counts=json.loads((root/'run_counts.json').read_text())
    write(root/'validation_summary.json',dict(regression=validation,full_model_experiments=counts,
        distinction='Regression process full_M123_GPU_experiments0 does not replace the actual experiment counts.'))
    (root/'validation_summary.md').write_text(f"# Validation\n\n{validation['counts']} on RTX3090; CPU results in validation_cpu/summary.json. Actual full M123 calls {counts['full_M123_calls']}; formal requests {counts['formal_question_requests']}; formal skips {counts['actual_formal_skips']}. No additional RNG causal replay was required in v3.\n")
    enrich_answers(root)
    interpret_results(root)
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/omnidrive-wrsar-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points=json.loads((root/'quality_latency_pareto.json').read_text())['points'];chosen=json.loads((root/'selected_policy.json').read_text())['online_dev_metrics']['index']
    fig,ax=plt.subplots(figsize=(7,4))
    for p in points:
        select=p['index']==chosen
        ax.scatter(p['mean_e2e_ms'],-100*p['loss'],s=100 if select else 50,marker='*' if select else '^' if p['temporal_risk_enabled'] else 'o')
        ax.annotate(f"{'T' if p['temporal_risk_enabled'] else 'Current'}: {p['min_confidence']}/{p['max_binding_risk']}",
            (p['mean_e2e_ms'],-100*p['loss']),xytext=(5,-16 if select else 8),textcoords='offset points',fontsize=8)
    ax.margins(.3,.2);ax.set_xlabel('DEV real full M123 E2E (ms/frame)');ax.set_ylabel('Scored task-fact improvement vs B (pp)')
    ax.set_title('Four DEV candidates; one frozen TEST policy');ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(root/'quality_latency_pareto.png',dpi=180);fig.savefig(root/'quality_latency_pareto.pdf');plt.close(fig)
    print('SUPPLEMENTARY_TABLES_SAVED',flush=True)


def enrich_answers(root):
    """Expand presentation from frozen scores; never alter a label or policy."""
    perf=json.loads((root/'abc_comparison.json').read_text())
    ab=json.loads((root/'current_vs_temporal_ablation.json').read_text())
    severe=json.loads((root/'severe_error_audit.json').read_text())
    support=json.loads((root/'observation_support_contribution.json').read_text())['comparisons']
    known=[x for x in support if x['old_quality'] and x['old_quality']['known'] and x['new_quality']['known']]
    changed=[x for x in known if x['old_quality']['score']!=x['new_quality']['score']]
    improved=sum(x['new_quality']['score']>x['old_quality']['score'] for x in changed)
    worsened=len(changed)-improved
    m=next(x for x in perf['module_breakdown'] if (x['workload'],x['question_count'],x['condition'])==('addressable',3,'C'))
    extras={
        7:f'同实际检测的离线旧/新快照对照：{len(known)}个双方可评分候选中，{improved}个由错变对、{worsened}个由对变错；仅移除新暴露的中心投影证据，不代表另一次在线基线或E2E实验。详见observation_support_contribution.json。',
        10:'实际Current→Temporal：'+ '; '.join(f"{x['question_count']}Q E2E比值{x['e2e_speedup']:.6f}×、时延差CI{x['delta_scene_CI95_ms']}ms" for x in ab['performance']['summaries'])+'。比值小于1表示时序较慢；消融与主实验在不同时间阶段，三场景区间仅支持有限判断。',
        13:f"Addressable3Q C：快照{m['snapshot_wall_ms']:.4f}ms/帧；解析{m['question_fact_extraction_wall_ms_per_question']:.6f}、当前绑定{m['current_binding_wall_ms_per_question']:.6f}、时序{m['temporal_risk_wall_ms_per_question']:.6f}、RNG管理{m['rng_management_wall_ms']:.4f}ms/问。绑定/时序包含在有效性时间内，不相加重算。",
        15:'本次点估计'+('达到或超过' if next(x for x in perf['summaries'] if (x['workload'],x['question_count'],x['comparison'])==('addressable',3,'B->C'))['e2e_speedup']>=1.314639 else '低于')+'上一轮1.314639×；数据流不同，不能据此给出同样本提升因果结论。',
        16:'B→C各负载：'+ '; '.join(f"{x['workload']}{x['question_count']}Q {x['e2e_speedup']:.6f}×，差CI{x['delta_scene_CI95_ms']}ms" for x in perf['summaries'] if x['comparison']=='B->C')+'。完整E2E包含同步和RNG成本；观测模块差异不能单独证明GPU顺序或解析的因果作用。',
        18:f"新增严重输出按路径{severe['paths']}，去重Frame/Question为{severe['unique_new_severe']}；重复Round和不同负载不是独立场景。所有严重输出和对应基线保留在severe_error_audit.json。",
        19:''}
    policies={b:json.loads((root/f'policy_{b}pp.json').read_text()) for b in (1,2,3)}
    extras[19]='1%、2%、3%预算的DEV最优配置：'+ '; '.join(
        f"{b}%: {'Temporal' if p['temporal_risk_enabled'] else 'Current-Only'}，confidence={p['min_confidence']}，risk={p['max_binding_risk']}，expected_normal_m3={p['expected_normal_m3_ms']:.6f}ms，min_net_gain={p['min_net_gain_ms']}ms"
        for b,p in policies.items())+ '。这些是DEV工作点，TEST未重新选择。'
    path=root/'result_answers.md';parts=path.read_text().split('\n\n')
    for i,part in enumerate(parts):
        for number,extra in extras.items():
            if part.startswith(str(number)+'. '):parts[i]=part.split(' 补充实测：')[0]+' 补充实测：'+extra
    path.write_text('\n\n'.join(parts))
    write(root/'snapshot_support_summary.json',dict(paired_known_candidates=len(known),wrong_to_right=improved,
        right_to_wrong=worsened,UNKNOWN_or_missing_legacy=len(support)-len(known),
        scope='Offline candidate comparison from same actual detection scalars; not an online speed or independent baseline.'))


def interpret_results(root):
    """Interpret retained measurements without changing any acceptance rule."""
    def read(name):return json.loads((root/name).read_text())
    decision=read('go_no_go.json');severe=read('severe_error_audit.json')
    source=read('fast_quality_by_observation_source.json')['groups']
    quality=read('task_quality.json');ab=read('current_vs_temporal_ablation.json')
    perf=read('abc_comparison.json');support=read('snapshot_support_summary.json')
    causes=read('target_binding_error_analysis.json')['offline_causes']
    def group(cond,work='addressable',count=3):
        return next(g for g in quality['groups'] if (g['workload'],g['question_count'],g['condition'],g['execution'])==(work,count,cond,'all'))
    b=group('B');c=group('C');mixed=group('C','mixed')
    def speed(work):return next(s for s in perf['summaries'] if (s['workload'],s['question_count'],s['comparison'])==(work,3,'B->C'))
    unique={}
    for e in severe['new_severe_vs_B']:unique.setdefault((e['frame_id'],e['question']),e)
    detail=[]
    for (frame,q),e in unique.items():
        cause=next(x for x in causes if x['frame_id']==frame and x['question']==q and x['new_severe_error'])
        evidence=cause['cause_evidence'];proposal=evidence.get('proposal',{});facts=proposal.get('facts',[])
        detail.append(f"- Frame `{frame}`, {q}: C={e['answer']!r}; B={e['baseline_answer']!r}; reference={e['quality']['reference']!r}; path={e['path']}. Current facts={facts!r}; offline cause={cause['offline_cause']}; reference in current supported candidates={evidence.get('reference_in_current_observed_candidates')}. An UNKNOWN identity match does not erase a KNOWN opposite-direction fact error.")
        temporal=[r for r in lines(root/'ablation/requests.jsonl') if r['frame_id']==frame and r['question_count']==e['question_count'] and r['round']==e['round']]
        for row in temporal:
            detail.append(f"  Temporal companion: {row['answers'][e['request_index']]!r}; actual skip={row['requests'][e['request_index']]['llm_skipped']}.")
    unique_veto={k:len({(v['scene_id'],v['frame_id'],v['question']) for v in ab['vetoes'] if v['veto_class']==k})
        for k in ('beneficial_veto','unnecessary_veto','unknown_veto')}
    ac=next(s for s in ab['performance']['summaries'] if s['question_count']==3)
    policy=read('selected_policy.json')
    parts=[
        f"Final verdict **{decision['status']}**. The acceptance rules were frozen before TEST, including no new severe errors. There are {len(severe['new_severe_vs_B'])} newly severe outputs, {len(unique)} unique frame/question cases; correlated rounds are not independent cases.",
        f"Addressable3Q B→C {speed('addressable')['e2e_speedup']:.6f}x. Known required-fact accuracy B={b['score']:.6%}, C={c['score']:.6%}, coverage={c['scored']}/{c['requests']}, UNKNOWN={c['UNKNOWN']}. Conservative all-request B-minus-C loss bounds={decision['paired_quality']['conservative_all_request_loss_bounds']}. Severe totals B={b['severe']}, C={c['severe']}; reductions in other cases do not cancel newly severe outputs.\n\n"+'\n'.join(detail),
        'Fast evidence sources (formal events; repetitions/workloads are correlated):\n\n'+'\n'.join(f"- {g['observation_kind']}: events={g['requests']}, scored={g['scored']}, UNKNOWN={g['UNKNOWN']}, score={g['score']}, unique scored frame/questions={g['unique_scored_frame_questions']}, scenes={g['independent_scenes']}" for g in source),
        f"The offline same-detection old/new snapshot comparison has {support['paired_known_candidates']} jointly scored candidates, {support['wrong_to_right']} wrong-to-right and {support['right_to_wrong']} right-to-wrong changes; UNKNOWN/missing legacy={support['UNKNOWN_or_missing_legacy']}. This does not prove an independent online E2E or net generalization improvement. Resolving the old Car diagnostic is insufficient. TEST outcomes never change the policy.",
        f"Temporal3Q veto events={ab['veto_counts']}; unique frame/question cases={unique_veto}. Temporal independent scored quality gain={ab['temporal_independent_quality_gain']}. Current→Temporal3Q speed ratio={ac['e2e_speedup']:.6f}x; delta sceneCI={ac['delta_scene_CI95_ms']}ms. Separate timing phases and three logs limit interpretation.",
        f"Natural3Q actual skips={sum(g['requests'] for g in quality['groups'] if (g['workload'],g['condition'],g['question_count'],g['execution'])==('natural','C',3,'fast_hit'))}; no established natural-QA speed benefit. Preset mixed3Q B→C={speed('mixed')['e2e_speedup']:.6f}x, skips={next(g['requests'] for g in quality['groups'] if (g['workload'],g['condition'],g['question_count'],g['execution'])==('mixed','C',3,'fast_hit'))}/{mixed['requests']}; the introduced1/3 structured slots are not a natural-distribution estimate.",
        f"DEV selects temporal={policy['temporal_risk_enabled']}, confidence={policy['min_confidence']}, risk={policy['max_binding_risk']}. This is the best observed of four predeclared DEV candidates, not a TEST-qualified deployment policy or a global optimum. Default research switches stay off. TEST has {decision['independent_test_scenes']} independent logs/{decision['independent_test_frames']} frames; rounds are correlated. UNKNOWNs do not certify complete driving, visual appearance, absence or motion quality. Implementation and experiments are complete; research acceptance is a separate outcome."]
    extra='\n\n## Frozen acceptance outcome and limits\n\n'+'\n\n'.join(parts)+'\n'
    (root/'final_research_assessment.md').write_text('# Final research assessment'+extra)
    path=root/'implementation_summary.md'
    path.write_text(path.read_text().split('\n\n## Frozen acceptance outcome')[0]+extra)
    answer=root/'result_answers.md';marker='\n\n详见final_research_assessment.md：'
    answer.write_text(answer.read_text().split(marker)[0].rstrip()+marker+
        f"新增严重输出{len(severe['new_severe_vs_B'])}条，去重{len(unique)}个Frame/Question；最终{decision['status']}。总质量或严重数的改善不能抵消新增关键错误；新暴露证据的独立净收益仍不足，未做TEST调参。\n")


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--prior-veto',action='store_true');p.add_argument('--first-frame',action='store_true')
    p.add_argument('--support-contribution',action='store_true');p.add_argument('--verify',action='store_true')
    p.add_argument('--summarize',action='store_true')
    p.add_argument('--checkpoint',action='store_true');p.add_argument('--manifest',action='store_true');a=p.parse_args()
    if a.prior_veto:prior_veto_audit(a.root)
    elif a.first_frame:first_frame_audit(a.root)
    elif a.support_contribution:support_contribution(a.root)
    elif a.summarize:summarize(a.root)
    else:verify(a.root,a.checkpoint,a.manifest)
