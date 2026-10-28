"""WSG-SD v2: original data inputs, causal own-history A/B/C, full model E2E."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
from functools import wraps
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from tools.worldstream_m123.benchmark.common import write,sha256
from worldstream.m3.task_conditioner import question_dependencies

ROOT=Path('outputs/m3_world_guided_decode_run02')
PRIOR=Path('outputs/m3_world_guided_decode_run01')
QUALIFICATION=Path('outputs/m3_sf_wrcps_optimization_run01/semantic_qualification.json')


def source_questions(token):
    """Read question strings only; answers never enter experiment scheduling."""
    questions=[]
    for folder in ('conv/val','vqa/val'):
        path=Path('data/nuscenes')/folder/(token+'.json')
        if path.exists():questions.extend(p['question'] for p in json.loads(path.read_text()))
    return questions


def prepare(root):
    import pickle
    from mmcv import Config
    from tools.worldstream_m1.dataset import resolve_image
    root.mkdir(exist_ok=False,parents=True)
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    with open(cfg.data.test.ann_file,'rb') as handle:infos=pickle.load(handle)['infos']
    old=json.loads((PRIOR/'sequence.json').read_text())
    qualification=json.loads(QUALIFICATION.read_text())
    excluded=set(old['scene_splits'])|set(qualification['calibration_scene_ids'])|set(qualification['validation_scene_ids'])
    scene_table={s['token']:s for s in json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())}
    excluded_logs={scene_table[s]['log_token'] for s in excluded}
    grouped=defaultdict(list)
    for info in infos:grouped[info['scene_token']].append(info)
    chosen=[];logs=set();manifest=[]
    for scene in sorted(grouped):
        log=scene_table[scene]['log_token']
        if scene in excluded or log in excluded_logs or log in logs:continue
        seq=sorted(grouped[scene],key=lambda f:f['timestamp'])[:4]
        if len(seq)!=4 or any(len(source_questions(f['token']))<3 for f in seq):continue
        if not all(resolve_image(c['data_path'],'data/nuscenes').exists() for f in seq for c in f['cams'].values()):continue
        chosen.extend(seq);logs.add(log)
        manifest.append(dict(scene_id=scene,log_id=log,tokens=[f['token'] for f in seq]))
        if len(manifest)==3:break
    if len(manifest)!=3:raise RuntimeError('Need three image-complete disjoint scene/log streams')
    questions={f['token']:source_questions(f['token'])[:3] for f in chosen}
    write(root/'experiment_plan.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source='Original validation conversation/VQA question order, first three real questions; loader-generated overview excluded from primary workload.',
        scenes=manifest,questions=questions,independent_of_prior_scenes_and_logs=True,
        qualification_path=str(QUALIFICATION),qualification_sha256=sha256(QUALIFICATION),
        frames=12,rounds=3,workloads=['natural','fixed'],questions_per_frame=[1,3],conditions=['A','B','C'],
        generation=dict(do_sample=True,temperature=.1,top_p=.75,num_beams=1,max_new_tokens=320,use_cache=True),
        candidate_length=8,quality_budget_absolute=.02,
        quality='Offline only after inference: original QA task facts; independent nuScenes annotation audit. No native-answer-as-truth and no text-similarity accuracy.',
        selection='First three sorted VAL scenes with complete images/three QA, excluding prior scene/logs and M1 qualification calibration/validation scene/logs. First four contiguous frames. No output/outcome selection.',
        task_counts=dict(Counter(question_dependencies(q).intent for qs in questions.values() for q in qs))))
    print('PREPARED',manifest,flush=True)


def append(path,row):
    with path.open('a') as handle:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def runtime_control(root,model):
    """Same-carrier OLD/new complete runtime wall, original failure included."""
    from worldstream.m3 import M3Runtime,CarrierLayout
    dest=root/'runtime_control';dest.mkdir()
    text=subprocess.check_output(['git','show','4f3fe1d:worldstream/m3/runtime.py'],text=True)
    source=dest/'baseline_runtime.py';source.write_text(text)
    spec=importlib.util.spec_from_file_location('worldstream.m3._measured_v1_runtime',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old_profile=json.loads((PRIOR/'native_decode_profile.json').read_text())
    frames=old_profile['frames'][:2]
    rows=[]
    def timed(fn):
        torch.cuda.synchronize();tick=time.perf_counter();result=fn();torch.cuda.synchronize()
        return result,(time.perf_counter()-tick)*1000
    for repeat in range(3):
        for name,cls in ([('old',module.M3Runtime),('new',M3Runtime)] if repeat%2==0 else [('new',M3Runtime),('old',module.M3Runtime)]):
            runtime=cls(dict(reuse_strategy='exact_prefix',decode_strategy='history_draft'))
            for i,f in enumerate(frames):
                cap=torch.load(PRIOR/'native_profile'/(f['frame_id']+'.pt'),weights_only=False)
                source_cap=torch.load(PRIOR/'native_profile'/(frames[0]['frame_id']+'.pt'),weights_only=False)
                ctx=dict(scene_id=f['scene_id'],frame_id=f['frame_id'],timestamp=cap['timestamp'],
                    state_epoch=0,state_version=i+1,camera_ids=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'],
                    model_version=old_profile['model_sha256'],profile=True)
                current_images=cap['carriers'].cuda();current_ids=[v.cuda() for v in source_cap['ids']]
                (answers,diag),wall=timed(lambda:runtime.generate(model.lm_head,current_ids,current_images,
                    context=ctx,questions=source_cap['questions'],tokenizer=model.tokenizer,layout=CarrierLayout(256,1,256),
                    **dict(cap['generation'],do_sample=False)))
                rows.append(dict(implementation=name,round=repeat,frame_id=f['frame_id'],source_frame=i==0,
                    m3_wall_ms=wall,diagnostics=diag,tokens=[a[0,1:].cpu().tolist() for a in answers]))
            runtime.reset()
    write(root/'runtime_before_after.json',dict(status='COMPLETED',rows=rows,baseline_source_sha256=sha256(source),
        scope='Three alternating rounds over same two old real carriers, three original loader questions including known long failure. Full runtime includes assembly, metadata, retrieval and cache management; excludes visual backbone/heads.'))


@torch.no_grad()
def run(root):
    import projects.mmdet3d_plugin  # noqa
    from mmcv.parallel import collate,scatter,DataContainer
    from mmdet3d.datasets import build_dataset
    from tools.worldstream_m123.benchmark.worker import effective_config,load_model,reset
    from tools.worldstream_m123.benchmark.observe import Observer
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    from worldstream.m3 import M3Runtime
    plan=json.loads((root/'experiment_plan.json').read_text())
    dest=root/'e2e';dest.mkdir(exist_ok=False)
    sources=[Path(__file__),Path('worldstream/m3/runtime.py'),Path('worldstream/m3/answer_memory.py'),
        Path('worldstream/m3/world_state.py'),Path('worldstream/m3/world_delta.py'),Path('worldstream/m3/task_conditioner.py'),
        Path('worldstream/m3/config.py'),Path('worldstream/m3/prefill_executor.py'),Path('projects/mmdet3d_plugin/models/detectors/petr3d.py')]
    source_hashes={str(p):sha256(p) for p in sources}
    write(root/'execution_sources.json',dict(stage='run',code_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=source_hashes))
    with gpu_job_lock():
        wait_gpu()
        assert not {4,5}&os.sched_getaffinity(0)
        cfg=effective_config('B3','language',dest,'native_full')
        model=load_model(cfg,'ckpts/iter_10548.pth',dest).eval().requires_grad_(False)
        checkpoint_sha=json.loads((PRIOR/'native_decode_profile.json').read_text())['model_sha256']
        model.worldstream_audit_identity=dict(checkpoint_sha256=checkpoint_sha)
        dataset=build_dataset(cfg.data.test)
        by_token={f['token']:f for f in dataset.data_infos}
        tokens=[t for s in plan['scenes'] for t in s['tokens']]
        dataset.data_infos=[by_token[t] for t in tokens]
        index={f['token']:i for i,f in enumerate(dataset.data_infos)}
        # Freeze loader inputs, not outcomes. We explicitly discard its generated
        # overview. Preparation/collation is outside MODEL request timing equally.
        runtime_control(root,model)
        print('P0_RUNTIME_CONTROL_DONE',flush=True)
        observer=Observer(model)
        development_coverage(root,model,dataset,observer,by_token)
        captures=root/'captures';captures.mkdir()
        rows=[]
        for count in (1,3):
            for workload in ('natural','fixed'):
                for repeat in range(3):
                    order=['A','B','C'];order=order[repeat:]+order[:repeat]
                    for condition in order:
                        model.m3_runtime=(None if condition=='A' else M3Runtime(dict(reuse_strategy='exact_prefix',
                            decode_strategy='history_draft' if condition=='B' else 'world_guided',
                            experimental_speculative_sampling=True,
                            semantic_qualification=str(QUALIFICATION) if condition=='C' else '')))
                        model.worldstream_capture_m3_inputs=False
                        model.worldstream_profile=True
                        out=dest/(str(count)+'q_'+workload+'_r'+str(repeat)+'_'+condition)
                        out.mkdir();model.save_path=str(out)+'/'
                        for scene in plan['scenes']:
                            reset(model);fixed_ids=fixed_questions=None
                            for frame_i,token in enumerate(scene['tokens']):
                                torch.manual_seed(20261028+repeat*1000+index[token]);random.seed(20261028+index[token])
                                data=scatter(collate([dataset[index[token]]],samples_per_gpu=1),[0])[0]
                                meta=data['img_metas'][0][0]
                                labels=list(meta['vlm_labels'].data)
                                assert labels[1:4]==plan['questions'][token], 'Actual original QA order changed'
                                ids=data['input_ids'][0][0][1:1+count];questions=labels[1:1+count]
                                if fixed_ids is None:
                                    fixed_ids=[v.clone() for v in ids];fixed_questions=list(questions)
                                if workload=='fixed':ids=fixed_ids;questions=fixed_questions
                                data['input_ids'][0][0]=ids
                                meta['vlm_labels']=DataContainer(questions,stack=False)
                                # Capture ids separately: native forward converts
                                # temporal containers in place, as documented in v1.
                                saved_ids=[v.cpu() for v in ids]
                                observer.clear();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                                tick=time.perf_counter()
                                outputs=model(return_loss=False,rescale=True,**data)
                                torch.cuda.synchronize();wall=(time.perf_counter()-tick)*1000
                                stages=observer.collect()
                                text=outputs[0]['text_out']
                                diag=model.m3_diagnostics or {}
                                req=[]
                                if condition=='A':
                                    assert len(observer.native_generations)==count
                                    for q,g in zip(questions,observer.native_generations):
                                        pre=[e for e in g['phase_events'] if e[0]=='prefill'];dec=[e for e in g['phase_events'] if e[0]=='decode']
                                        req.append(dict(question=q,tokens=g['output'][0,1:].cpu().tolist(),
                                            prefill_cuda_ms=sum(a.elapsed_time(b) for _,a,b,_ in pre),
                                            decode_cuda_ms=sum(a.elapsed_time(b) for _,a,b,_ in dec),
                                            decode_wall_ms=pre[-1][2].elapsed_time(g['end']),
                                            prefill_host_ms=sum(e[3] for e in pre),decode_host_ms=sum(e[3] for e in dec),
                                            target_forward_calls=1+len(dec),draft_status='none'))
                                    m3_wall=sum(g['host_wall_ms'] for g in observer.native_generations)
                                else:
                                    assert diag['execution_completed'] and len(diag['requests'])==count
                                    for q,a,d in zip(questions,text,diag['requests']):
                                        # Actual tokens are owned by the runtime's
                                        # answer memory, already transferred for its
                                        # required causal history. No GT is involved.
                                        req.append(dict(d,question=q,tokens=d['output_token_ids'],
                                            prefill_cuda_ms=d['prefill_including_cache_gpu_ms'],
                                            decode_cuda_ms=d['decode_forward_cuda_ms'],decode_wall_ms=d['decode_loop_gpu_span_ms'],
                                            prefill_host_ms=d['prefill_including_cache_cpu_enqueue_ms'],
                                            decode_host_ms=d['decode_forward_host_enqueue_ms'],target_forward_calls=1+d['decode_forward_calls']))
                                    m3_wall=diag['runtime_host_elapsed_ms']
                                snapshot=getattr(model,'m3_committed_state',None)
                                row=dict(condition=condition,round=repeat,workload=workload,question_count=count,
                                    scene_id=scene['scene_id'],log_id=scene['log_id'],frame_id=token,frame_index=frame_i,
                                    cold_scene_frame=frame_i==0,e2e_wall_ms=wall,m3_wall_ms=m3_wall,
                                    m1_wall_ms=sum(stages.get(k,{}).get('host_wall_ms',0) for k in ('m1_precheck','m1_commit')),
                                    m1_including_adapters_ms=model.m1_manager.last_commit_diagnostics.get('m1_total_including_adapters_ms'),
                                    m2_wall_ms=stages.get('m2_total',{}).get('host_wall_ms'),
                                    heads_wall_ms=sum(stages.get(k,{}).get('host_wall_ms',0) for k in ('detection','map')),
                                    snapshot_wall_ms=getattr(model,'m3_committed_snapshot_wall_ms',0.),
                                    peak_gpu_bytes=torch.cuda.max_memory_allocated(),requests=req,answers=text,
                                    committed_world=asdict(snapshot) if snapshot else None,
                                    execution_completed=model.worldstream_execution['completed'])
                                assert row['execution_completed'] and model.worldstream_execution['m1']['commit_completed']
                                append(root/'e2e_trials.jsonl',row);rows.append(row)
                                if repeat==0 and workload=='natural' and count==3:
                                    if condition=='A':
                                        torch.save(dict(ids=saved_ids,questions=questions,carriers=observer.native_generations[0]['audit_carriers'].cpu()),captures/(token+'.pt'))
                                    if condition=='C':write(captures/(token+'_world.json'),row['committed_world'])
                            print('E2E',count,workload,repeat,condition,scene['scene_id'],'rows',len(rows),flush=True)
        # Greedy probes on original QA using fresh current carriers and the same
        # committed M1 snapshots from actual C, no repeated backbone benchmark.
        greedy_probe(root,model,plan)
        write(root/'run_status.json',dict(status='COMPLETED',e2e_rows=len(rows),gpu=torch.cuda.get_device_name(0),
            generation=plan['generation'],checkpoint_sha256=checkpoint_sha))
        if source_hashes!={str(p):sha256(p) for p in sources}:raise RuntimeError('Executed source changed during measured run')


def greedy_probe(root,model,plan):
    from worldstream.m3 import M3Runtime,CarrierLayout
    from worldstream.m3.world_state import CommittedWorldState,CommittedAnchor
    rows=[]
    # Whole four-frame streams, three original questions: causal retrieval can
    # see only this probe's earlier generated answers and committed observations.
    for condition in ('A','B','C'):
        runtime=None if condition=='A' else M3Runtime(dict(reuse_strategy='exact_prefix',
            decode_strategy='history_draft' if condition=='B' else 'world_guided'))
        for scene in plan['scenes']:
            if runtime:runtime.reset()
            for token in scene['tokens']:
                cap=torch.load(root/'captures'/(token+'.pt'),weights_only=False)
                raw=json.loads((root/'captures'/(token+'_world.json')).read_text())
                world=CommittedWorldState(**dict(raw,anchors=tuple(CommittedAnchor(**dict(a,
                    center_ego=tuple(a['center_ego']),velocity_ego=tuple(a['velocity_ego']),cameras=tuple(a['cameras']))) for a in raw['anchors']),ego_to_world=tuple(raw['ego_to_world'])))
                ctx=dict(scene_id=world.scene_id,frame_id=world.frame_id,timestamp=world.timestamp,state_epoch=world.epoch,
                    current_state_version=world.version,state_execution_signature=world.execution_signature,
                    camera_ids=list(dataset_cameras()),model_version='iter_10548_frozen',profile=True)
                gen=dict(plan['generation'],do_sample=False)
                images=cap['carriers'].cuda();ids=[v.cuda() for v in cap['ids']]
                if runtime is None:
                    answers=[model.lm_head.generate(inputs=v[None] if v.ndim==1 else v,images=images,**gen) for v in ids]
                    diag={}
                else:
                    answers,diag=runtime.generate(model.lm_head,ids,images,context=ctx,questions=cap['questions'],
                        tokenizer=model.tokenizer,layout=CarrierLayout(256,1,256),committed_world_state=world if condition=='C' else None,**gen)
                rows.append(dict(condition=condition,scene_id=world.scene_id,frame_id=token,questions=cap['questions'],
                    tokens=[a[0,1:].cpu().tolist() for a in answers],
                    answers=[model.tokenizer.decode(a[0,1:],skip_special_tokens=True) for a in answers],diagnostics=diag))
    write(root/'greedy_quality_probe.json',dict(status='COMPLETED',rows=rows))


def development_coverage(root,model,dataset,observer,by_token):
    """Same 48 original v1 QA inputs: own online history, real current commits."""
    from mmcv.parallel import collate,scatter,DataContainer
    from tools.worldstream_m123.benchmark.worker import reset
    from worldstream.m3 import M3Runtime
    prior=json.loads((PRIOR/'sequence.json').read_text())
    manifest=json.loads((PRIOR/'question_manifest.json').read_text())['frames']
    original_infos=dataset.data_infos
    dataset.data_infos=[by_token[t] for t in prior['tokens']]
    model.m3_runtime=M3Runtime(dict(reuse_strategy='exact_prefix',decode_strategy='world_guided',
        experimental_speculative_sampling=True,semantic_qualification=str(QUALIFICATION)))
    out=root/'development_coverage';out.mkdir();model.save_path=str(out)+'/'
    model.worldstream_profile=True
    previous=None
    for i,info in enumerate(dataset.data_infos):
        if previous!=info['scene_token']:reset(model)
        torch.manual_seed(20261028+i);random.seed(20261028+i)
        data=scatter(collate([dataset[i]],samples_per_gpu=1),[0])[0]
        frozen=manifest[info['token']]
        data['input_ids'][0][0]=[torch.tensor(v,device='cuda') for v in frozen['ids'][1:3]]
        data['img_metas'][0][0]['vlm_labels']=DataContainer(frozen['questions'][1:3],stack=False)
        observer.clear();torch.cuda.synchronize();tick=time.perf_counter()
        outputs=model(return_loss=False,rescale=True,**data)
        torch.cuda.synchronize()
        append(root/'development_coverage.jsonl',dict(frame_id=info['token'],scene_id=info['scene_token'],
            e2e_wall_ms=(time.perf_counter()-tick)*1000,answers=outputs[0]['text_out'],
            diagnostics=model.m3_diagnostics,committed_world=asdict(model.m3_committed_state)))
        previous=info['scene_token']
    dataset.data_infos=original_infos
    print('DEVELOPMENT_48_QA_DONE',flush=True)


def dataset_cameras():
    return ('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=['prepare','run'],required=True)
    args=parser.parse_args();{'prepare':prepare,'run':run}[args.stage](args.root)
