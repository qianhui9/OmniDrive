"""WRSAR v3 bounded experiments over the existing full-M123 execution path."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
from . import selective_answer_reuse as worker
from .benchmark.common import write,sha256

ROOT=Path('outputs/m3_state_calibrated_bypass_run01')
PRIOR=Path('outputs/m3_temporal_selective_bypass_run01')
CONFIGS=[dict(min_confidence=.45,max_binding_risk=.35,temporal_risk_enabled=False),
         dict(min_confidence=.45,max_binding_risk=.75,temporal_risk_enabled=False),
         dict(min_confidence=.65,max_binding_risk=.75,temporal_risk_enabled=False),
         dict(min_confidence=.45,max_binding_risk=.75,temporal_risk_enabled=True)]


def prepare(root):
    import pickle
    from collections import defaultdict
    from .world_guided_decode_v2 import source_questions
    from tools.worldstream_m1.dataset import resolve_image
    root.mkdir(parents=True,exist_ok=True);assert not (root/'experiment_plan.json').exists()
    previous=json.loads((PRIOR/'experiment_plan.json').read_text())
    excluded=set(previous['prior_excluded_log_ids'])|{s['log_id'] for s in previous['test_scenes']}
    tables={s['token']:s for s in json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())}
    grouped=defaultdict(list)
    for f in pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']:grouped[f['scene_token']].append(f)
    def eligible(sid,n):
        fs=sorted(grouped[sid],key=lambda f:f['timestamp'])[:n]
        if len(fs)!=n or any(len(source_questions(f['token']))<3 for f in fs):return None
        if not all(resolve_image(c['data_path'],'data/nuscenes').exists() for f in fs for c in f['cams'].values()):return None
        return dict(scene_id=sid,log_id=tables[sid]['log_token'],tokens=[f['token'] for f in fs])
    scenes=[];logs=set()
    for sid in sorted(grouped):
        log=tables[sid]['log_token']
        if log in excluded or log in logs:continue
        s=eligible(sid,8)
        if s:scenes.append(s);logs.add(log)
        if len(scenes)==3:break
    assert len(scenes)==3,'Need new disjoint logs; no old TEST substitution'
    dev=[eligible(s['scene_id'],6) for s in previous['development_scenes']];assert all(dev)
    plan=dict(previous,version='wrsar3',development_scenes=dev,test_scenes=scenes,rounds=2,
        natural_questions={t:source_questions(t)[:3] for s in scenes for t in s['tokens']},
        workloads=['natural','fixed','addressable','mixed'],question_counts=[1,3],configs=CONFIGS,
        prior_excluded_log_ids=sorted(excluded),experiment_seed=20261029,
        selection='First three sorted new image/QA-complete disjoint logs,8 first contiguous frames; DEV6 frames per prior development log. No object/GT/hit/outcome selection.',
        mixed_protocol='Only3Q: one declared structured question, cycling category/existence/position by frame_index%3, plus first two original natural questions. Preset1/3 structured-question slots, not a natural QA population claim.',
        B='Existing exact K8 Target-Aware and unchanged request RNG',
        C='DEV-selected current-only or existing temporal-risk research option; current-camera detector evidence separate from photometric reuse support')
    write(root/'experiment_plan.json',plan)
    write(root/'protocol_freeze.json',dict(plan_sha256=sha256(root/'experiment_plan.json'),configs=CONFIGS,
        quality_budgets=[.01,.02,.03],default_budget=.02,
        temporal_rule='Do not choose temporal without measured DEV improvement in quality/E2E; never treat UNKNOWN vetoes beneficial',
        no_test_threshold_or_seed_tuning=True,current_positive_only=True,production_defaults_unchanged=True))
    print('PREPARED',json.dumps(scenes),flush=True)


def stage_jobs(root,stage,plan):
    if stage=='shadow':return [('addressable',3,0,'B','shadow',None)]
    if stage=='pareto':
        jobs=[]
        for repeat in range(2):
            order=[('B',None)]+[(f'C{i}',root/f'candidate_{i}.json') for i in range(len(CONFIGS))]
            if repeat:order.reverse()
            jobs.extend(('addressable',3,repeat,c,'off' if c=='B' else 'online',p) for c,p in order)
        return jobs
    selected=json.loads((root/'selected_policy.json').read_text())
    if stage=='ablation':
        other='C-current' if selected['temporal_risk_enabled'] else 'C-temporal'
        return [('addressable',n,r,other,'online',root/'selected_policy.json') for n in (1,3) for r in range(plan['rounds'])]
    jobs=[]
    for wi,w in enumerate(plan['workloads']):
        for n in ((3,) if w=='mixed' else (1,3)):
            for r in range(plan['rounds']):
                order=['A','B','C'];offset=(wi+(n==3)+r)%3;order=order[offset:]+order[:offset]
                jobs.extend((w,n,r,c,'online' if c=='C' else 'off',root/'selected_policy.json' if c=='C' else None) for c in order)
    return jobs


def detection_capture(root):
    import os
    import numpy as np
    import torch
    import projects.mmdet3d_plugin
    from mmcv.parallel import collate,scatter,DataContainer
    from mmdet3d.datasets import build_dataset
    from .benchmark.worker import effective_config,load_model,reset
    from .benchmark.observe import Observer
    from .benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    from worldstream.m3 import M3Runtime
    from worldstream.m1.omnidrive_adapter import CLASS_NAMES
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    plan=json.loads((PRIOR/'experiment_plan.json').read_text())
    old=json.loads(Path('outputs/m3_selective_answer_reuse_run01/experiment_plan.json').read_text())
    bad=next(s for s in old['test_scenes'] if '79070dca796f4ffb868d84f1b067f9c2' in s['tokens'])
    scenes=[dict(bad,tokens=bad['tokens'][:1]),dict(plan['development_scenes'][0],tokens=plan['development_scenes'][0]['tokens'][:3])]
    out=root/'detection';out.mkdir(exist_ok=False)
    sources=worker.source_files()+[Path(__file__),Path('worldstream/m1/omnidrive_adapter.py'),Path('worldstream/m1/state_manager.py'),
        Path('projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py')]
    for p in sources:
        rel=p.relative_to(Path.cwd()) if p.is_absolute() else p
        dst=out/'executed_sources'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    write(out/'execution_sources.json',dict(sources={str(p):sha256(p) for p in sources}))
    with gpu_job_lock():
        wait_gpu();assert not {4,5}&os.sched_getaffinity(0)
        cfg=effective_config('B3','language',out,'native_full')
        model=load_model(cfg,'ckpts/iter_10548.pth',out).eval().requires_grad_(False)
        model.m3_runtime=M3Runtime(dict(reuse_strategy='exact_prefix',decode_strategy='world_guided',
            target_aware_exact_only=True,experimental_speculative_sampling=True,question_rng_enabled=True))
        model.worldstream_require_m123=True;model.worldstream_profile=True
        observer=Observer(model);observer.capture_m1_controls=True
        dataset=build_dataset(cfg.data.test);original={i['token']:i for i in dataset.data_infos}
        dataset.data_infos=[original[t] for s in scenes for t in s['tokens']]
        index={f['token']:i for i,f in enumerate(dataset.data_infos)}
        raw={};decoded=[]
        def hook(module,args,result):
            outs=result[0];scores=outs['all_cls_scores'][-1,0].sigmoid().detach()
            boxes=denormalize_bbox(outs['all_bbox_preds'][-1,0].detach(),module.bbox_coder.pc_range)
            raw.update(class_scores=scores.cpu().float().tolist(),boxes=boxes.cpu().float().tolist(),
                       top_classes=scores.argmax(-1).cpu().tolist())
        handle=model.pts_bbox_head.register_forward_hook(hook)
        original_decode=model.pts_bbox_head.get_bboxes
        def decode(*args,**kwargs):
            result=original_decode(*args,**kwargs);decoded.clear()
            for boxes,scores,labels in result:
                decoded.extend(dict(center_ego=c,score=s,semantic_class=CLASS_NAMES[l],class_id=l)
                    for c,s,l in zip(boxes.gravity_center.cpu().float().tolist(),scores.cpu().tolist(),labels.cpu().tolist()))
            return result
        model.pts_bbox_head.get_bboxes=decode
        rows=[]
        with torch.no_grad():
            for scene in scenes:
                reset(model)
                for fi,t in enumerate(scene['tokens']):
                    torch.manual_seed(20261029+index[t])
                    data=scatter(collate([dataset[index[t]]],samples_per_gpu=1),[0])[0]
                    meta=data['img_metas'][0][0];q=worker.QUESTIONS[:1]
                    data['input_ids'][0][0]=worker.encode_questions(model,q,original[t]['location'])
                    meta['vlm_labels']=DataContainer(q,stack=False)
                    model.save_path=str(out/(t+'_answers'))+'/'
                    model.worldstream_request_rng_context=dict(experiment_seed=20261029,repeat=0)
                    observer.clear();result=model(return_loss=False,rescale=True,**data);torch.cuda.synchronize()
                    frame=observer.m1_control['frame'];outputs=observer.m1_control['outputs']
                    inverse=np.linalg.inv(frame.ego_to_world)
                    anchors=[dict(semantic_class=a.semantic_class,score=a.source_score,
                        center_ego=(inverse[:3,:3]@a.geometry_world.center+inverse[:3,3]).tolist())
                        for a in outputs.anchors if a.anchor_type=='object']
                    rows.append(dict(frame_id=t,scene_id=scene['scene_id'],frame_index=fi,raw_queries=dict(raw),
                        decoded_candidates=list(decoded),adapter_anchors=anchors,
                        committed_world=asdict(model.m3_committed_state),answers=result[0]['text_out'],
                        m1_min_detection_score=model.m1_manager.config.min_detection_score,
                        decoder=dict(max_num=model.pts_bbox_head.bbox_coder.max_num,score_threshold=model.pts_bbox_head.bbox_coder.score_threshold,
                            post_center_range=model.pts_bbox_head.bbox_coder.post_center_range.cpu().tolist()),
                        execution_completed=model.worldstream_execution['completed']))
                    print('DETECTION_AUDIT',t,len(decoded),len(anchors),flush=True)
        handle.remove()
        write(out/'requests.json',dict(scope='Four actual full M123 diagnostic calls; raw scalar candidates retained, no repeated Tensor/model data. Capture overhead is outside formal latency qualification.',rows=rows))
        write(out/'run_status.json',dict(status='COMPLETED',full_model_requests=len(rows),gpu=torch.cuda.get_device_name(0)))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--stage',choices=['detection','prepare','shadow','calibrate','pareto','freeze','run','ablation','report'],required=True);a=p.parse_args()
    if a.stage=='prepare':prepare(a.root)
    elif a.stage in ('calibrate','freeze','report'):
        from .state_calibrated_report import calibrate,freeze,report
        dict(calibrate=calibrate,freeze=freeze,report=report)[a.stage](a.root)
    else:
        with (a.root/f'{a.stage}_execution.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
            if a.stage=='detection':detection_capture(a.root)
            else:worker.run(a.root,a.stage)


if __name__=='__main__':main()
