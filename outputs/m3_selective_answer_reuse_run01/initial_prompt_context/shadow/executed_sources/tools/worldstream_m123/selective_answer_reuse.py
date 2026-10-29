"""Bounded WRSAR full-model GPU worker; labels live in the offline report module."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import time

from .benchmark.common import write,sha256

ROOT=Path('outputs/m3_selective_answer_reuse_run01')
QUESTIONS=[
    'What type of vehicle is directly ahead?',
    'Is there a car in front?',
    'Where is the nearest pedestrian relative to the vehicle?',
]
DIAGNOSTIC_QUESTIONS=['Are there pedestrians on the left?',
                      'Is the nearest car moving or stationary?']
CONFIDENCE_CANDIDATES=(.45,.65,.85)


def append(path,value):
    with Path(path).open('a') as handle:handle.write(json.dumps(value,allow_nan=False)+'\n')


def prepare(root):
    from .world_guided_decode_v2 import prepare as select_scenes
    root.mkdir(parents=True,exist_ok=False)
    dev=json.loads(Path('outputs/m3_world_guided_decode_run02/experiment_plan.json').read_text())
    exclusions=[]
    for p in [Path('outputs/m3_world_guided_decode_run03/experiment_plan.json'),
              Path('outputs/m3_world_guided_decode_run03/initial_scope_mismatch/experiment_plan.json')]:
        if p.exists():exclusions.extend(s['scene_id'] for s in json.loads(p.read_text())['scenes'])
    select_scenes(root/'heldout_selection',extra_excluded_scenes=[s['scene_id'] for s in dev['scenes']]+exclusions)
    test=json.loads((root/'heldout_selection/experiment_plan.json').read_text())
    # Fixed before any model outcome: 3 contiguous frames per independent test
    # log, all 4 retained DEV frames. No object/GT/candidate filtering.
    test_scenes=[dict(s,tokens=s['tokens'][:3]) for s in test['scenes']]
    assert not ({s['log_id'] for s in dev['scenes']}&{s['log_id'] for s in test_scenes})
    write(root/'experiment_plan.json',dict(version='wrsar1',base_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),development_scenes=dev['scenes'],test_scenes=test_scenes,
        natural_questions=test['questions'],addressable_questions=QUESTIONS,
        diagnostic_questions=DIAGNOSTIC_QUESTIONS,confidence_candidates=CONFIDENCE_CANDIDATES,
        quality_budgets=[.01,.02,.03],default_budget=.02,rounds=3,
        workloads=['natural','fixed','addressable'],question_counts=[1,3],
        generation=test['generation'],independent_scene_and_log_splits=True,
        B='Existing target-aware K8 exact-history candidate selector, no M1 fast answer',
        C='Identical B generator on misses; explicit current-committed-M1 selective fast answers',
        selection='Existing sorted image-complete scene selector, no answer/GT or fast-hit filtering',
        quality='Offline current-frame visible positive annotation facts; absence UNKNOWN; exact current QA for original workloads.'))
    (root/'m1_addressable_qa_protocol.md').write_text('''# M1-Addressable Continuous QA protocol

Three fixed questions per real frame: vehicle category directly ahead; positive car existence in front; nearest pedestrian position. Single-question trials use the first question. A separate DEV shadow diagnostic checks pedestrian-left existence and reliable simple vehicle motion. These questions form an independent subset and never replace original natural QA. Fixed QA repeats each scene's first actual natural questions on different current images.

DEV reuses the three v2 development logs (four contiguous frames each). Final TEST uses three new scene/log streams, three contiguous frames each, selected before inference by sorted scene ID and image/QA availability only. Neither GT answers nor M1 detections affect scene/question selection. A/B/C use the same six camera inputs, preprocessing, model/checkpoint and generation parameters (max_new_tokens320). Three rotating A/B/C orders give paired scene timings; duplicate rounds are not independent label samples.

Supported facts: actual current detector class/ego geometry, positive observation and reliable velocity when present. Nearest targets are selected before confidence thresholding; class-ambiguous nearest ties abstain. Front/rear use ego x, left/right use ego y, 50m ROI (near questions 30m), directly |y|<=4m. Position sectors have a 1m center band; boundary cases abstain. No same-class identity assertion is made. Global safety, counts, plural category inventories, signal colors, appearance and road roles require normal generation. Existing Map head outputs geometric polylines, not signal colors or complete road semantics.

Offline labels distinguish physical annotations, visible/sensor-supported projections, question ROI and actual M1 observations. Positive current annotation witnesses use actual camera exposure poses. No witness cannot prove visual absence. Occlusion, uncertain nearest targets and unsupported motion labels are UNKNOWN. Required category/existence/direction facts must be present in the output; missing required facts are errors when the current reference is known. Color or verbosity is not the metric. No human image review is claimed.

Shadow still runs the full target-aware LLM and saves proposals BEFORE generation. Shadow timing is not skip acceleration. Three a-priori detector confidence floors (.45/.65/.85) are compared on DEV paired task-fact loss relative to B, for absolute budgets .01/.02/.03. Detector confidence and old validity rule scores are NOT error probabilities. Empirical bins include counts/unknowns and uncertainty. At least three independently framed scored correct proposals are required before the research online gate; all missing labels remain visible in quality coverage.

Policy is frozen before TEST. Research online outputs distinguish full_answer_reuse (closed grammar, all assertions currently supported, original generation age retained) from fact_grounded_response (current facts; no new target-validation witness). Default deployment remains Native. Detailed temporal entry/loss/identity/ego changes are audited after inference rather than used to cherry-pick streams.
''')


def source_files():
    return [Path(__file__)]+[
        Path('worldstream/m3')/n for n in ('runtime.py','config.py','answer_validity.py','answer_memory.py','task_conditioner.py','world_state.py','prefill_executor.py')]


def run(root,stage):
    import torch
    import projects.mmdet3d_plugin  # noqa: F401
    from mmcv.parallel import collate,scatter,DataContainer
    from mmdet3d.datasets import build_dataset
    from .benchmark.worker import effective_config,load_model,reset
    from .benchmark.observe import Observer
    from .benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    from worldstream.m3 import M3Runtime
    plan=json.loads((root/'experiment_plan.json').read_text())
    out=root/stage;out.mkdir(exist_ok=False)
    hashes={str(p):sha256(p) for p in source_files()}
    write(out/'execution_sources.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=hashes))
    with gpu_job_lock():
        wait_gpu();assert not {4,5}&os.sched_getaffinity(0)
        cfg=effective_config('B3','language',out,'native_full')
        model=load_model(cfg,'ckpts/iter_10548.pth',out).eval().requires_grad_(False)
        # Same original checkpoint identity as v3; no new checkpoint writes.
        checkpoint_sha=json.loads(Path('outputs/m3_world_guided_decode_run01/native_decode_profile.json').read_text())['model_sha256']
        model.worldstream_audit_identity=dict(checkpoint_sha256=checkpoint_sha)
        dataset=build_dataset(cfg.data.test)
        original={i['token']:i for i in dataset.data_infos}
        scenes=plan['development_scenes'] if stage=='shadow' else plan['test_scenes']
        dataset.data_infos=[original[t] for s in scenes for t in s['tokens']]
        index={f['token']:i for i,f in enumerate(dataset.data_infos)}
        observer=Observer(model)
        calls=[]
        from worldstream.m3.visual_state import base_model
        hook=base_model(model.lm_head).register_forward_pre_hook(lambda *a:calls.append(
            model.m3_runtime.last_diagnostics.get('active_request_index') if model.m3_runtime else None))
        jobs=[]
        if stage=='shadow':
            jobs=[('addressable',3,0,'B','shadow',None),('diagnostic',2,0,'B','shadow',None)]
        else:
            if not (root/'selected_policy.json').exists():raise RuntimeError('Shadow/calibration gate not passed')
            for workload in plan['workloads']:
                for count in plan['question_counts']:
                    for repeat in range(plan['rounds']):
                        order=['A','B','C'];order=order[repeat:]+order[:repeat]
                        jobs.extend((workload,count,repeat,c,'online' if c=='C' else 'off',root/'selected_policy.json' if c=='C' else None) for c in order)
        with torch.no_grad():
            for workload,count,repeat,condition,mode,policy in jobs:
                model.m3_runtime=None if condition=='A' else M3Runtime(dict(reuse_strategy='exact_prefix',
                    decode_strategy='world_guided',target_aware_exact_only=True,
                    experimental_speculative_sampling=True,selective_answer_mode=mode,
                    selective_answer_policy=str(policy) if policy else ''))
                model.worldstream_profile=True
                model.worldstream_capture_m3_inputs=False
                dest=out/f'{workload}_{count}q_r{repeat}_{condition}';dest.mkdir()
                model.save_path=str(dest)+'/'
                for scene in scenes:
                    reset(model);fixed_questions=None
                    for fi,token in enumerate(scene['tokens']):
                        torch.manual_seed(20261029+repeat*1000+index[token]);random.seed(20261029+index[token])
                        data=scatter(collate([dataset[index[token]]],samples_per_gpu=1),[0])[0]
                        meta=data['img_metas'][0][0]
                        assert len(meta['cam_infos'])==6
                        original_questions=list(meta['vlm_labels'].data)[1:4]
                        if fixed_questions is None:fixed_questions=list(original_questions)
                        if workload in ('addressable','diagnostic'):
                            questions=(QUESTIONS if workload=='addressable' else DIAGNOSTIC_QUESTIONS)[:count]
                            questions=list(questions)
                            ids=encode_questions(model,questions,original[token]['location'])
                        elif workload=='fixed':
                            questions=fixed_questions[:count];ids=encode_questions(model,questions,original[token]['location'])
                            if fi==0:assert all(torch.equal(a,b) for a,b in zip(ids,data['input_ids'][0][0][1:1+count]))
                        else:
                            questions=original_questions[:count];ids=data['input_ids'][0][0][1:1+count]
                        data['input_ids'][0][0]=ids;meta['vlm_labels']=DataContainer(questions,stack=False)
                        observer.clear();calls.clear();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                        tick=time.perf_counter();outputs=model(return_loss=False,rescale=True,**data)
                        torch.cuda.synchronize();wall=(time.perf_counter()-tick)*1000
                        stages=observer.collect();answers=outputs[0]['text_out'];diag=model.m3_diagnostics or {}
                        requests=[]
                        if condition=='A':
                            assert len(observer.native_generations)==count
                            for q,g in zip(questions,observer.native_generations):
                                phases=g['phase_events'];pre=[e for e in phases if e[0]=='prefill'];dec=[e for e in phases if e[0]=='decode']
                                requests.append(dict(question=q,tokens=g['output'][0,1:].cpu().tolist(),
                                    llm_forward_calls=len(pre)+len(dec),llm_skipped=False,answer_source='current_target_model_generation',
                                    prefill_cuda_ms=sum(a.elapsed_time(b) for _,a,b,_ in pre),decode_cuda_ms=sum(a.elapsed_time(b) for _,a,b,_ in dec),
                                    prefill_wall_ms=sum(e[3] for e in pre),decode_wall_ms=pre[-1][2].elapsed_time(g['end']),
                                    end_to_end_ms=g['host_wall_ms'],fast_answer=None,draft_status='none'))
                        else:
                            assert diag['execution_completed'] and len(diag['requests'])==count
                            for qi,(q,d) in enumerate(zip(questions,diag['requests'])):
                                assert calls.count(qi)==d['llm_forward_calls'],(qi,calls,d)
                                if d['llm_skipped']:assert calls.count(qi)==0 and d['layout_preparation_ms']==0.
                                requests.append(dict(d,question=q,tokens=d['output_token_ids'],
                                    prefill_cuda_ms=d['prefill_including_cache_gpu_ms'],decode_cuda_ms=d['decode_forward_cuda_ms'],
                                    prefill_wall_ms=d.get('prefill_including_cache_cpu_enqueue_ms',0.),decode_wall_ms=d['decode_loop_gpu_span_ms']))
                        snapshot=getattr(model,'m3_committed_state',None)
                        row=dict(stage=stage,condition=condition,workload=workload,question_count=count,round=repeat,
                            scene_id=scene['scene_id'],log_id=scene['log_id'],frame_id=token,frame_index=fi,
                            timestamp=float(original[token]['timestamp'])/1e6,questions=questions,requests=requests,answers=answers,
                            e2e_wall_ms=wall,m3_wall_ms=diag.get('runtime_host_elapsed_ms',sum(g['host_wall_ms'] for g in observer.native_generations)),
                            m1_wall_ms=sum(stages.get(k,{}).get('host_wall_ms',0.) for k in ('m1_precheck','m1_commit')),
                            m1_including_adapters_ms=model.m1_manager.last_commit_diagnostics.get('m1_total_including_adapters_ms'),
                            m2_wall_ms=stages.get('m2_total',{}).get('host_wall_ms'),
                            heads_wall_ms=sum(stages.get(k,{}).get('host_wall_ms',0.) for k in ('detection','map')),
                            snapshot_wall_ms=getattr(model,'m3_committed_snapshot_wall_ms',0.),
                            peak_gpu_bytes=torch.cuda.max_memory_allocated(),committed_world=asdict(snapshot) if snapshot else None,
                            camera_ids=list(meta['cam_infos']),observed_llm_forward_calls=len(calls),
                            m1_commit_completed=model.worldstream_execution['m1']['commit_completed'],
                            m2_completed=model.m2_diagnostics['execution_completed'],execution_completed=model.worldstream_execution['completed'])
                        assert row['execution_completed'] and row['m1_commit_completed'] and row['m2_completed']
                        append(out/'requests.jsonl',row)
                    print('WRSAR',stage,workload,count,repeat,condition,scene['scene_id'],flush=True)
        hook.remove()
        assert hashes=={str(p):sha256(p) for p in source_files()},'Measured source changed during run'
        write(out/'run_status.json',dict(status='COMPLETED',gpu=torch.cuda.get_device_name(0),checkpoint_sha256=checkpoint_sha,
            full_model_requests=sum(1 for _ in (out/'requests.jsonl').open()),weights_changed=False))


def encode_questions(model,questions,location):
    """Exactly the deployed Llava conversation prompt, not bare question IDs."""
    from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import preprocess,DEFAULT_IMAGE_TOKEN
    sources=[[dict(**{'from':'human'},value=DEFAULT_IMAGE_TOKEN+'\n'+f'You are driving in {location}. '+q),
              dict(**{'from':'gpt'},value='')] for q in questions]
    ids=preprocess(sources,model.tokenizer,True,False)['input_ids']
    device=next(model.lm_head.parameters()).device
    return [v.to(device) for v in ids]


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=['prepare','shadow','calibrate','run','report'],required=True)
    a=parser.parse_args()
    if a.stage=='prepare':prepare(a.root)
    elif a.stage in ('shadow','run'):run(a.root,a.stage)
    else:
        from .selective_answer_reuse_report import calibrate,report
        (calibrate if a.stage=='calibrate' else report)(a.root)
