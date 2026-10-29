"""Capture actual detector stages for prior v3 failures, never online GT."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from .benchmark.common import write,sha256

ROOT=Path('outputs/m3_target_complete_bypass_run01')
PRIOR=Path('outputs/m3_state_calibrated_bypass_run01')


def capture(root):
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
    from .selective_answer_reuse import source_files,QUESTIONS,encode_questions
    from worldstream.m3 import M3Runtime
    from worldstream.m1.omnidrive_adapter import CLASS_NAMES
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    plan=json.loads((PRIOR/'experiment_plan.json').read_text())
    old=json.loads(Path('outputs/m3_selective_answer_reuse_run01/experiment_plan.json').read_text())
    car=next(s for s in old['test_scenes'] if '79070dca796f4ffb868d84f1b067f9c2' in s['tokens'])
    scenes=plan['test_scenes']+[dict(car,tokens=car['tokens'][:1])]
    out=root/'diagnostic';out.mkdir(exist_ok=False)
    sources=list(dict.fromkeys(source_files()+[Path(__file__),Path('worldstream/m1/omnidrive_adapter.py'),
        Path('worldstream/m1/state_manager.py'),Path('projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py')]))
    hashes={str(p):sha256(p) for p in sources}
    for p in sources:
        rel=p.relative_to(Path.cwd()) if p.is_absolute() else p
        dst=out/'executed_sources'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    write(out/'execution_sources.json',dict(sources=hashes))
    with gpu_job_lock():
        wait_gpu();assert not {4,5}&os.sched_getaffinity(0)
        cfg=effective_config('B3','language',out,'native_full')
        model=load_model(cfg,'ckpts/iter_10548.pth',out).eval().requires_grad_(False)
        model.m3_runtime=M3Runtime(dict(reuse_strategy='exact_prefix',decode_strategy='world_guided',
            target_aware_exact_only=True,experimental_speculative_sampling=True,question_rng_enabled=True,
            current_detection_support=True,target_binding_enabled=True))
        model.worldstream_require_m123=True;model.worldstream_profile=True
        observer=Observer(model);observer.capture_m1_controls=True
        dataset=build_dataset(cfg.data.test);original={i['token']:i for i in dataset.data_infos}
        dataset.data_infos=[original[t] for s in scenes for t in s['tokens']]
        index={f['token']:i for i,f in enumerate(dataset.data_infos)}
        raw={};decoded=[]
        def hook(module,args,result):
            outs=result[0];scores=outs['all_cls_scores'][-1,0].sigmoid().detach()
            boxes=denormalize_bbox(outs['all_bbox_preds'][-1,0].detach(),module.bbox_coder.pc_range)
            raw.clear();raw.update(class_scores=scores.cpu().float().tolist(),boxes=boxes.cpu().float().tolist())
        handle=model.pts_bbox_head.register_forward_hook(hook)
        decode_original=model.pts_bbox_head.get_bboxes
        def decode(*args,**kwargs):
            result=decode_original(*args,**kwargs);decoded.clear()
            for boxes,scores,labels in result:
                decoded.extend(dict(center_ego=c,score=s,semantic_class=CLASS_NAMES[l],class_id=l)
                    for c,s,l in zip(boxes.gravity_center.cpu().float().tolist(),scores.cpu().tolist(),labels.cpu().tolist()))
            return result
        model.pts_bbox_head.get_bboxes=decode;rows=[]
        with torch.no_grad():
            for scene in scenes:
                reset(model)
                for fi,t in enumerate(scene['tokens']):
                    torch.manual_seed(20261029+index[t])
                    data=scatter(collate([dataset[index[t]]],samples_per_gpu=1),[0])[0]
                    meta=data['img_metas'][0][0];questions=list(QUESTIONS[:3])
                    data['input_ids'][0][0]=encode_questions(model,questions,original[t]['location'])
                    meta['vlm_labels']=DataContainer(questions,stack=False)
                    model.save_path=str(out/(t+'_answers'))+'/'
                    model.worldstream_request_rng_context=dict(experiment_seed=20261029,repeat=0)
                    observer.clear();result=model(return_loss=False,rescale=True,**data);torch.cuda.synchronize()
                    frame=observer.m1_control['frame'];outputs=observer.m1_control['outputs']
                    inverse=np.linalg.inv(frame.ego_to_world)
                    anchors=[dict(semantic_class=a.semantic_class,score=a.source_score,
                        center_ego=(inverse[:3,:3]@a.geometry_world.center+inverse[:3,3]).tolist())
                        for a in outputs.anchors if a.anchor_type=='object']
                    cameras={cid:dict(intrinsic=c.intrinsic.tolist(),world_to_camera=c.world_to_camera.tolist(),
                        image_shape=list(c.image.shape),timestamp=c.timestamp,timing_reliable=c.timing_reliable)
                        for cid,c in frame.cameras.items()}
                    row=dict(frame_id=t,scene_id=scene['scene_id'],frame_index=fi,timestamp=frame.timestamp,
                        raw_queries=dict(raw),decoded_candidates=list(decoded),adapter_anchors=anchors,
                        committed_world=asdict(model.m3_committed_state),cameras=cameras,
                        m1_min_detection_score=model.m1_manager.config.min_detection_score,
                        decoder=dict(max_num=model.pts_bbox_head.bbox_coder.max_num,
                            score_threshold=model.pts_bbox_head.bbox_coder.score_threshold,
                            post_center_range=model.pts_bbox_head.bbox_coder.post_center_range.cpu().tolist()),
                        answers=result[0]['text_out'],execution_completed=model.worldstream_execution['completed'])
                    rows.append(row)
                    with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print('DIAGNOSTIC',len(rows),t,flush=True)
        handle.remove();assert all(sha256(p)==h for p,h in hashes.items())
        write(out/'run_status.json',dict(status='COMPLETED',full_model_requests=len(rows),
            checkpoint_sha256=sha256('ckpts/iter_10548.pth'),gpu=torch.cuda.get_device_name(0),
            scope='Prior TEST diagnostics only;25 full M123 calls, scalar stages, no inference tuning on new TEST'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    with (a.root/'diagnostic_execution.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):capture(a.root)
