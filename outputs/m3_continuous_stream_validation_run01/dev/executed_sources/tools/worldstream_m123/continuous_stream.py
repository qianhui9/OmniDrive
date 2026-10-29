"""Real continuous-stream benchmark; GT is confined to offline report modules."""
import argparse
from collections import defaultdict,Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import random
import re
import subprocess
import time
import traceback

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/omnidrive-continuous-mpl')

from .benchmark.common import write,sha256,diagnostic_json
from .selective_answer_reuse import QUESTIONS,append,encode_questions,validate_native_prompt_context

ROOT=Path('outputs/m3_continuous_stream_validation_run01')
SCOPE='WITHIN_LOG_EXTENDED_EVALUATION'


def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def source_hashes():
    paths=[p for folder in ('worldstream','projects/mmdet3d_plugin','projects/configs/OmniDrive') for p in Path(folder).rglob('*.py')]
    paths += list(Path('tools/worldstream_m123').rglob('*.py'))
    # Final presentation assembly cannot affect inference or DEV grading. Its
    # source is attested separately when assembling the final report.
    return {str(p):sha256(p) for p in sorted(set(paths)) if 'frozen' not in p.parts
        and p.name!='continuous_stream_analysis.py'}


def question_schedule(workload,frequency,fi,natural,fixed):
    if frequency in ('every4','every2'):
        period=int(frequency[-1])
        if fi%period:return []
        count=1;ordinal=fi//period
    else:count=3 if frequency=='three' else 1;ordinal=fi
    if workload=='addressable':return list(QUESTIONS) if count==3 else [QUESTIONS[ordinal%3]]
    if workload=='mixed':return [QUESTIONS[fi%3]]+list(natural[:2])
    return list((fixed if workload=='fixed' else natural)[:count])


def inventory(root,infos,scenes):
    valid={i['token'] for i in infos};used=defaultdict(lambda:defaultdict(list));sources=[]
    actual_names={'requests.jsonl','per_frame_metrics.jsonl','e2e_trials.jsonl'}
    evidence_names={'per_question_metrics.jsonl','request_records.jsonl','service_evidence.jsonl'}
    for p in sorted(Path('outputs').rglob('*.jsonl')):
        if root in p.parents or not any(s.startswith(('m3_','m123_')) for s in p.parts):continue
        if p.name not in actual_names|evidence_names:continue
        hit=0
        for no,row in enumerate(lines(p),1):
            token=row.get('frame_id');trace=row.get('module_execution') or {}
            if token not in valid:continue
            measured=(row.get('execution_completed') is True and row.get('e2e_wall_ms',0)>0) or (
                trace.get('completed') is True and row.get('model_ms',0)>0)
            if measured:
                stages=row.get('stages') or {};requests=row.get('requests') or (row.get('m3') or {}).get('requests',[])
                language=bool(requests or any(k in stages for k in ('native_generation','prefill_executor_generate','prefill')))
                used[token]['actual_visual_GPU'].append(dict(path=str(p),line=no))
                if language:used[token]['completed_M123_language'].append(dict(path=str(p),line=no))
                hit+=1
            elif p.name in evidence_names:
                used[token]['language_execution_evidence'].append(dict(path=str(p),line=no));hit+=1
            if measured or p.name in evidence_names:
                lower=str(p).lower()
                if any(x in lower for x in ('shadow','pareto','calibrat','development','adaptation','training')):
                    used[token]['method_development'].append(str(p))
                if any(x in lower for x in ('/run/','test','validation','qualification')):
                    used[token]['quality_evaluation'].append(str(p))
                if any(x in lower for x in ('diagnostic','error','regression')):used[token]['error_diagnostic'].append(str(p))
        if hit:sources.append(dict(path=str(p),sha256=sha256(p),matching_rows=hit))
    # Plans are tracked separately and NEVER promoted to executed GPU frames.
    pattern=re.compile(r'(?<![a-z0-9])[0-9a-f]{32}(?![a-z0-9])')
    for p in sorted(Path('outputs').rglob('*.json')):
        if root in p.parents or not any(s.startswith(('m3_','m123_')) for s in p.parts):continue
        if not ('plan' in p.name.lower() or 'sequence' in p.name.lower() or 'manifest' in p.name.lower()):continue
        tokens=set(pattern.findall(p.read_text(errors='replace')))&valid
        if tokens:sources.append(dict(path=str(p),sha256=sha256(p),planned_frames=len(tokens)))
        for t in tokens:used[t]['planned'].append(str(p))
    records=[]
    for i in infos:
        sid=i['scene_token'];s=scenes[sid];u=used[i['token']]
        records.append(dict(scene_id=sid,log_id=s['log_token'],frame_id=i['token'],timestamp=i['timestamp']/1e6,
            original_partition='official_val',base_model_training='not_in_configured_nuscenes_train; exact_pretraining_exposure_unknown',
            historical_WRSAR_log=True,completed_M123_language=bool(u.get('completed_M123_language')),
            actual_visual_GPU=bool(u.get('actual_visual_GPU')),language_execution_evidence=bool(u.get('language_execution_evidence')),
            method_development=bool(u.get('method_development')),quality_evaluation=bool(u.get('quality_evaluation')),
            severe_error_diagnostic=bool(u.get('error_diagnostic')),planned=bool(u.get('planned')),
            evidence={k:sorted(set(v)) if v and isinstance(v[0],str) else v for k,v in u.items()}))
    write(root/'historical_execution_inventory.json',dict(scope='Historical M3/M123 frame receipts, before this run',
        val_frames=len(infos),completed_M123_language_frames=sum(r['completed_M123_language'] for r in records),
        never_completed_M123_frames=sum(not r['completed_M123_language'] for r in records),
        never_observed_visual_or_language_frames=sum(not(r['actual_visual_GPU'] or r['language_execution_evidence']) for r in records),
        planned_only_frames=sum(r['planned'] and not(r['actual_visual_GPU'] or r['language_execution_evidence']) for r in records),
        classification='Completed outer model/trace receipts establish full visual execution; per-question-only evidence stays separate and is conservatively excluded. Plans do not establish execution. Development/evaluation roles inferred from retained stage paths, with exact references.',
        sources=sources,frames=records))
    return {r['frame_id']:r for r in records}


def prepare(root):
    from tools.worldstream_m1.io_utils import iter_json_array
    from tools.worldstream_m1.dataset import resolve_image
    from .benchmark.sequence import validate_sequence
    from .final_fact_bypass import historical_manifest
    root.mkdir(exist_ok=True,parents=True)
    assert not (root/'extended_frame_selection.json').exists(),'Do not replace frozen selection'
    infos=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']
    scenes={s['token']:s for s in iter_json_array('data/nuscenes/v1.0-trainval/scene.json')}
    write(root/'historical_manifest.json',historical_manifest(root))
    usage=inventory(root,infos,scenes);grouped=defaultdict(list)
    for i in infos:grouped[i['scene_token']].append(i)
    candidates=[];rejections=[]
    for sid,fs in sorted(grouped.items()):
        fs.sort(key=lambda i:i['timestamp']);segments=[];current=[]
        for f in fs:
            u=usage[f['token']]
            paths=[resolve_image(c['data_path'],'data/nuscenes') for c in f['cams'].values()]
            qa=Path('data/nuscenes/conv/val')/(f['token']+'.json')
            legal=not(u['actual_visual_GPU'] or u['language_execution_evidence']) and len(paths)==6 and all(p.is_file() for p in paths)
            legal=legal and qa.is_file() and len(json.loads(qa.read_text()))>=3
            if legal and current and (current[-1].get('next')!=f['token'] or f.get('prev')!=current[-1]['token']):
                segments.append(current);current=[]
            if legal:current.append(f)
            elif current:segments.append(current);current=[]
        if current:segments.append(current)
        for seg in segments:
            if len(seg)<16:continue
            try:v=validate_sequence(seg,sid)
            except ValueError as e:rejections.append(dict(scene_id=sid,reason=str(e)));continue
            candidates.append(dict(scene_id=sid,log_id=scenes[sid]['log_token'],tokens=[f['token'] for f in seg],
                available_length=len(seg),continuity=v))
    chosen=[];logs=set()
    for c in candidates:
        if c['log_id'] in logs:continue
        n=16 if len(chosen)<2 else min(24,c['available_length'])
        chosen.append(dict(c,tokens=c['tokens'][:n],selected_length=n,role='extended_DEV' if len(chosen)<2 else 'extended_evaluation'))
        logs.add(c['log_id'])
        if len(chosen)==5:break
    assert len(chosen)==5,'Need five usable distinct-log segments; report missing resources rather than fabricate streams'
    tokens={t for c in chosen for t in c['tokens']}
    write(root/'extended_frame_selection.json',dict(scope=SCOPE,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        rule='First sorted scene/first chronological eligible segment on five distinct logs, first2 DEV16 and next3 evaluation up to24 frames. Exclude every actual visual GPU or per-question language receipt; require real six cameras, linked poses and >=3 original Conv QA. No detector output, GT correctness or answer filtering.',
        candidate_segments=len(candidates),candidate_unique_frames=sum(len(c['tokens']) for c in candidates),
        candidates=candidates,selected=chosen,selected_frames=[dict(usage[i['token']],extended_evaluation=True,
            camera_paths={k:str(resolve_image(v['data_path'],'data/nuscenes')) for k,v in i['cams'].items()},
            ego_translation=i['ego2global_translation']) for i in infos if i['token'] in tokens],rejections=rejections,
        independent_test=False,scene_log_isolation_from_history=False,new_frame_isolation=True))
    write(root/'protocol_freeze.json',dict(scope=SCOPE,selection_sha256=sha256(root/'extended_frame_selection.json'),
        seed=20261030,cold_frame=0,steady_start_index=4,default_budget=.02,budgets=[.01,.02,.03],
        request_frequencies=['every4','every2','one','three'],workloads=['addressable','natural','fixed','mixed'],
        DEV='B shadow then CPU restricted gate fitting from DEV only; compare online policies only if admission changes; compare fallback if any Position hit',
        selection_criterion='Minimum measured DEV E2E among policies with paired known answer loss<=budget; report severe and UNKNOWN independently; no strict 2% certification',
        restricted_fit='Just above maximum incorrect admitted Position support on extended DEV; if no known Position errors, retain the prior .45 floor and mark indistinguishable',
        evaluation_jobs='A/B/C on 4 addressable frequencies; A/B/C natural1Q/fixed1Q/mixed3Q; one meaningful Position companion on addressable3Q',
        same_visual_stream=True,no_idle_visual_skip=True,no_sampling_changes=True,no_GT_online=True,strict_generalization_GO=False))
    (root/'continuous_qa_protocol.md').write_text('''# Continuous QA protocol (frozen before GPU)

WITHIN_LOG_EXTENDED_EVALUATION. Official val, historically used logs, previously unexecuted frames. Selection is metadata-only. Five distinct logs: two retrospective DEV16-frame streams and three evaluation streams up to24 frames. Generic question text and logs are historically exposed; this is not independent TEST.

Category: What type of vehicle is directly ahead? Positive Existence: Is there a car in front? Position: Where is the nearest pedestrian relative to the vehicle? Ego x forward, y left, meters; maximum planar distance50m; directly |y|<=4m and x>0. Nearest means nearest legal visible current GT instance, never nearest M1 detection. Longitudinal/lateral answer sectors use a +/-1m center band; .5m boundary ambiguity remains UNKNOWN. Current nuScenes annotations, camera exposure poses/center projection, visibility>=3 and nonzero lidar+radar support define a visibility proxy, not human image adjudication. A closer camera-projected but visibility-uncertain candidate prevents nearest certification. GT pairs within2m remain identity-ambiguous; common answer labels do not imply unique identity. No annotation witness never establishes absence. Object matching remains class-independent one-to-one, distance<2m and .5m ambiguity margin; duplicates cannot share a correctness receipt. Object/target/answer labels remain separate. All six UNKNOWN reasons are retained.

Every4/every2 asks one rotating Category/Existence/Position question at fi%period==0, using that frame's inputs/GT. One asks one rotating question every frame. Three asks all3 every frame. Mixed3 inserts one rotating structured question plus the current frame's first2 original natural questions: preset1/3, not a natural population distribution. Natural1 uses the current frame's first original Conv QA; fixed1 uses the scene's first question on current images, scored against current labels/QA only. No new templates. No-QA frames still run six-camera vision, M1 precheck/commit, M2 and heads; M3 lifecycle runs with zero prompts and zero forwards. Empty-prompt permission is explicit benchmark opt-in, default contract unchanged.

Each condition begins with an empty sequence state, then processes every selected frame in order. Cold=start frame0; transition1-3; steady=fi>=4 of the same unreset stream. No warmup frame removal/reset to manufacture steady state. A uses native visual backbone and native generation while retaining measured M1 bookkeeping; B/C use the same deployed incremental M2 and Target-Aware K8, C adds current-only bypass. Thus A-to-C includes M2+M3 effects; B-to-C isolates the language-policy change within the stateful system. Question-level RNG is enabled for all conditions; original do_sample=True, temperature=.1, top_p=.75, max_new_tokens320 remain. Real outer synchronized E2E includes answer export, excludes equally applied input preparation and post-run audit serialization. P99 is descriptive with its sample count. Scene/log-cluster paired CIs do not prove rare-error safety.
''')
    write(root/'prepare_sources.json',source_hashes())
    print(json.dumps(dict(status='PREPARED',candidates=len(candidates),segments=[(s['role'],s['scene_id'],len(s['tokens'])) for s in chosen])),flush=True)


def verify_sources(root,filename):
    expected=json.loads((root/filename).read_text());changed=[p for p,h in expected.items() if sha256(p)!=h]
    assert not changed,changed


def run(root,stage):
    import torch
    import projects.mmdet3d_plugin
    from mmcv.parallel import collate,scatter,DataContainer
    from mmdet3d.datasets import build_dataset
    from .benchmark.worker import effective_config,load_model,reset
    from .benchmark.observe import Observer
    from .benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    from worldstream.m3 import M3Runtime
    from worldstream.m3.visual_state import base_model
    selection=json.loads((root/'extended_frame_selection.json').read_text());protocol=json.loads((root/'protocol_freeze.json').read_text())
    assert sha256(root/'extended_frame_selection.json')==protocol['selection_sha256']
    scenes=[s for s in selection['selected'] if s['role']==('extended_DEV' if stage=='dev' else 'extended_evaluation')]
    out=root/stage;out.mkdir(exist_ok=False)
    hashes=source_hashes();write(out/'execution_sources.json',hashes)
    with gpu_job_lock():
        wait_gpu();assert torch.cuda.is_available() and not {4,5}&os.sched_getaffinity(0)
        torch.manual_seed(20261030);x=torch.arange(256.,device='cuda').reshape(16,16);witness=float((x@x.T).sum().item())
        write(out/'kernel_witness.json',dict(gpu=torch.cuda.get_device_name(0),seed=20261030,result=witness,cpu_affinity=sorted(os.sched_getaffinity(0))))
        cfg=effective_config('B3','language',out,'native_full')
        model=load_model(cfg,'ckpts/iter_10548.pth',out).eval().requires_grad_(False)
        cp=json.loads(Path('outputs/m3_final_fact_bypass_run01/checkpoint_verification.json').read_text())['actual_sha256']
        actual=sha256('ckpts/iter_10548.pth');assert cp==actual
        model.worldstream_audit_identity=dict(checkpoint_sha256=actual)
        dataset=build_dataset(cfg.data.test);original={i['token']:i for i in dataset.data_infos}
        dataset.data_infos=[original[t] for s in scenes for t in s['tokens']];index={f['token']:i for i,f in enumerate(dataset.data_infos)}
        validate_native_prompt_context(model,dataset,index,scenes,out)
        observer=Observer(model);calls=[]
        hook=base_model(model.lm_head).register_forward_pre_hook(lambda *a:calls.append(model.m3_runtime.last_diagnostics.get('active_request_index') if model.m3_runtime else None))
        if stage=='dev':
            from .continuous_stream_report import calibrate_dev,freeze_dev
            shadow=dict(main_path='current_only',position_mode='basic',min_confidence=.45,max_binding_risk=.75,
                expected_normal_m3_ms=1000.,policy_id='DEV_shadow',status='FROZEN_RETROSPECTIVE_POLICY',
                evaluation_scope=SCOPE,automatic_deployment=False,calibration_sha256=sha256(root/'protocol_freeze.json'))
            write(root/'shadow_policy.json',shadow)
            jobs=[dict(workload='addressable',frequency='three',condition='B-shadow',mode='shadow',policy=root/'shadow_policy.json',scene=s) for s in scenes]
        else:
            verify_sources(root,'policy_sources.json')
            policy=root/'selected_retrospective_policy.json';freeze=json.loads((root/'POLICY_FREEZE.json').read_text())
            assert sha256(policy)==freeze['selected_policy_sha256']
            jobs=[]
            workloads=[('addressable',f) for f in ('every4','every2','one','three')]+[('natural','one'),('fixed','one'),('mixed','three')]
            for wi,(workload,frequency) in enumerate(workloads):
                for si,s in enumerate(scenes):
                    order=['A','B','C'];offset=(wi+si)%3;order=order[offset:]+order[:offset]
                    jobs += [dict(workload=workload,frequency=frequency,condition=c,mode='online' if c=='C' else 'off',policy=policy if c=='C' else None,scene=s) for c in order]
            companion=freeze.get('companion_policy')
            if companion:jobs += [dict(workload='addressable',frequency='three',condition='C-companion',mode='online',policy=Path(companion),scene=s) for s in scenes]
        completed=0;jobindex=0
        with torch.no_grad():
            while jobindex<len(jobs):
                job=jobs[jobindex];condition=job['condition'];scene=job['scene'];workload=job['workload'];frequency=job['frequency'];policy=job['policy']
                model.m3_runtime=None if condition=='A' else M3Runtime(dict(reuse_strategy='exact_prefix',decode_strategy='world_guided',
                    target_aware_exact_only=True,experimental_speculative_sampling=True,selective_answer_mode=job['mode'],
                    selective_answer_policy=str(policy) if policy else '',question_rng_enabled=True,target_binding_enabled=True,
                    temporal_risk_enabled=False,target_completeness_enabled=False,current_detection_support=True))
                model.worldstream_question_rng_enabled=True;model.worldstream_require_m123=condition!='A';model.worldstream_allow_idle_frames=True
                model.worldstream_profile=True;model.worldstream_capture_m3_inputs=False
                # A is the native visual+language reference. B/C share the same
                # deployed M2; retain M1 in A for honest state-maintenance cost.
                if not hasattr(model,'_continuous_m2'):model._continuous_m2=model.m2_runtime
                model.m2_runtime=None if condition=='A' else model._continuous_m2
                dest=out/f'{jobindex:03d}_{workload}_{frequency}_{condition}';dest.mkdir();model.save_path=str(dest)+'/'
                reset(model);fixed=None
                for fi,token in enumerate(scene['tokens']):
                    torch.manual_seed(20261030+fi);random.seed(20261030+fi)
                    model.worldstream_request_rng_context=dict(repeat=0,experiment_seed=20261030)
                    prep=time.perf_counter();data=scatter(collate([dataset[index[token]]],samples_per_gpu=1),[0])[0]
                    meta=data['img_metas'][0][0];natural=list(meta['vlm_labels'].data)[1:4]
                    if fixed is None:fixed=list(natural)
                    questions=question_schedule(workload,frequency,fi,natural,fixed)
                    ids=encode_questions(model,questions,original[token]['location']) if questions else []
                    data['input_ids'][0][0]=ids;meta['vlm_labels']=DataContainer(questions,stack=False)
                    input_ms=(time.perf_counter()-prep)*1000
                    observer.clear();calls.clear();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                    tick=time.perf_counter();outputs=model(return_loss=False,rescale=True,**data);torch.cuda.synchronize();wall=(time.perf_counter()-tick)*1000
                    stages=observer.collect();answers=outputs[0]['text_out'];diag=model.m3_diagnostics or {};requests=[]
                    if condition=='A':
                        assert len(observer.native_generations)==len(questions)
                        for q,g in zip(questions,observer.native_generations):
                            phases=g['phase_events'];pre=[e for e in phases if e[0]=='prefill'];dec=[e for e in phases if e[0]=='decode']
                            requests.append(dict(question=q,tokens=g['output'][0,1:].cpu().tolist(),llm_forward_calls=len(pre)+len(dec),
                                llm_skipped=False,answer_source='current_target_model_generation',fast_answer=None,
                                prefill_cuda_ms=sum(a.elapsed_time(b) for _,a,b,_ in pre),decode_cuda_ms=sum(a.elapsed_time(b) for _,a,b,_ in dec),
                                prefill_wall_ms=sum(e[3] for e in pre),decode_wall_ms=sum(e[3] for e in dec),end_to_end_ms=g['host_wall_ms']))
                    else:
                        assert diag['execution_completed'] and len(diag['requests'])==len(questions)
                        for qi,(q,d) in enumerate(zip(questions,diag['requests'])):
                            assert calls.count(qi)==d['llm_forward_calls']
                            if d['llm_skipped']:assert calls.count(qi)==0 and d['layout_preparation_ms']==0. and not d['prefill_completed'] and not d['decode_completed']
                            requests.append(dict(d,question=q,tokens=d['output_token_ids'],prefill_cuda_ms=d['prefill_including_cache_gpu_ms'],decode_cuda_ms=d['decode_forward_cuda_ms'],
                                prefill_wall_ms=d.get('prefill_including_cache_cpu_enqueue_ms',0.),decode_wall_ms=d['decode_loop_gpu_span_ms']))
                    trace=model.worldstream_execution;snapshot=getattr(model,'m3_committed_state',None)
                    assert trace['completed'] and trace['m1']['commit_completed']
                    if condition!='A':assert trace['m2']['execution_completed']
                    if not questions:assert len(calls)==0 and not requests and not answers
                    row=dict(stage=stage,scope=SCOPE,job=jobindex,condition=condition,workload=workload,frequency=frequency,
                        scene_id=scene['scene_id'],log_id=scene['log_id'],frame_id=token,frame_index=fi,timestamp=original[token]['timestamp']/1e6,
                        phase='cold' if fi==0 else 'steady' if fi>=4 else 'transition',questions=questions,question_count=len(questions),requests=requests,answers=answers,
                        e2e_wall_ms=wall,input_preparation_ms=input_ms,m3_wall_ms=diag.get('runtime_host_elapsed_ms',sum(g['host_wall_ms'] for g in observer.native_generations)),
                        m1_wall_ms=sum(stages.get(k,{}).get('host_wall_ms',0.) for k in ('m1_precheck','m1_commit')),
                        m1_including_adapters_ms=model.m1_manager.last_commit_diagnostics.get('m1_total_including_adapters_ms'),
                        m2_wall_ms=stages.get('m2_total' if condition!='A' else 'eva_native',{}).get('host_wall_ms'),
                        heads_wall_ms=sum(stages.get(k,{}).get('host_wall_ms',0.) for k in ('detection','map')),
                        snapshot_wall_ms=getattr(model,'m3_committed_snapshot_wall_ms',0.),answer_export_ms=model.language_export_wall_ms,
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(),committed_world=asdict(snapshot) if snapshot else None,
                        camera_ids=list(meta['cam_infos']),observed_llm_forward_calls=len(calls),execution_completed=True,
                        m1_commit_completed=True,m2_completed=condition!='A',m2=diagnostic_json(model.m2_diagnostics),
                        m1_commit_count=model.m1_manager.store.commit_count,stages=stages,policy_file=str(policy) if policy else None)
                    append(out/'requests.jsonl',row);completed+=1
                    if fi%4==0:print('PROGRESS',stage,jobindex+1,len(jobs),condition,workload,frequency,fi+1,len(scene['tokens']),flush=True)
                jobindex+=1
                if stage=='dev' and jobindex==len(scenes):
                    candidates=calibrate_dev(root)
                    for ci,p in enumerate(candidates):
                        for si,s in enumerate(scenes):jobs.append(dict(workload='addressable',frequency='three',condition=p.stem,mode='online',policy=p,scene=s))
                    write(out/'planned_jobs.json',[{k:str(v) if isinstance(v,Path) else v for k,v in j.items()} for j in jobs])
        hook.remove();assert all(sha256(p)==h for p,h in hashes.items()),'Source changed during GPU stage'
        write(out/'run_status.json',dict(status='COMPLETED',gpu=torch.cuda.get_device_name(0),full_M123_frame_calls=completed,
            question_requests=sum(len(r['requests']) for r in lines(out/'requests.jsonl')),source_hashes=hashes,checkpoint_sha256=actual))
        if stage=='dev':freeze_dev(root)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','dev','evaluation','report','verify']);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    if {4,5}&os.sched_getaffinity(0):p.error('Use taskset -c 0-3,6-31')
    if a.stage=='prepare':prepare(a.root)
    elif a.stage in ('dev','evaluation'):
        with (a.root/(a.stage+'_execution.log')).open('x') as log:
            from contextlib import redirect_stdout,redirect_stderr
            with redirect_stdout(log),redirect_stderr(log):
                try:run(a.root,a.stage)
                except BaseException:
                    traceback.print_exc();write(a.root/(a.stage+'_failure.json'),dict(error=traceback.format_exc()));raise
    else:
        from .continuous_stream_report import report,verify
        (report if a.stage=='report' else verify)(a.root)


if __name__=='__main__':main()
