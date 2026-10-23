"""Real GT evaluation using unchanged nuScenes/OpenLane evaluator functions.

Subset SDK metrics are prominently scoped and are never full-val scores.
"""
import argparse
import copy
import json
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
from .common import write
from tools.worldstream_m123.benchmark.metrics import finite_metrics


def prepare_detection_evaluation(nusc,cfg,prediction_file,tokens,full_tokens):
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.common.loaders import load_gt,load_prediction,add_center_dist,filter_eval_boxes
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.common.data_classes import EvalBoxes
    prediction,meta=load_prediction(str(prediction_file),cfg.max_boxes_per_sample,DetectionBox,verbose=False)
    raw_prediction_indices={id(box):i for token in prediction.sample_tokens for i,box in enumerate(prediction[token])}
    gt=load_gt(nusc,'val',DetectionBox,verbose=False)
    if set(gt.sample_tokens)!=set(full_tokens):raise ValueError('Official full validation GT does not match configured dataset')
    if set(prediction.sample_tokens)!=set(tokens):raise ValueError('Prediction/declared subset token mismatch')
    selected=EvalBoxes()
    for token in tokens:selected.add_boxes(token,gt[token])
    # Same official filtering and DetectionEval.evaluate implementation as the
    # constructor. Only the explicitly declared evaluation sample set is sliced.
    evaluation=DetectionEval.__new__(DetectionEval)
    evaluation.nusc=nusc;evaluation.cfg=cfg;evaluation.verbose=False
    evaluation._raw_prediction_indices=raw_prediction_indices
    evaluation.pred_boxes=filter_eval_boxes(nusc,add_center_dist(nusc,prediction),cfg.class_range,verbose=False)
    evaluation.gt_boxes=filter_eval_boxes(nusc,add_center_dist(nusc,selected),cfg.class_range,verbose=False)
    evaluation.sample_tokens=evaluation.gt_boxes.sample_tokens
    return evaluation


def detection_metrics(nusc,cfg,prediction_file,tokens,full_tokens):
    evaluation=prepare_detection_evaluation(nusc,cfg,prediction_file,tokens,full_tokens)
    metrics,curves=evaluation.evaluate()
    return metrics.serialize(),curves.serialize(),evaluation


def map_metrics(gt,infos,outputs):
    from .map_eval import evaluate_map,has_geometry_cache
    keys=[tuple(info['lane_info']) for info in infos]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate map frame identity')
    missing=[k for k in keys if k not in gt]
    if missing:raise ValueError('Missing actual map GT: '+str(missing[:5]))
    predictions=dict(method='OmniDrive',authors=[],**{'e-mail':'','institution / company':'','country / region':'China'},results={})
    for key,result in zip(keys,outputs):
        lines,scores=result['lane_results'][0]
        if np.asarray(scores).shape[-1:] not in ((1,),):raise ValueError('Only actual single-class lane head is supported')
        count=len(lines)
        predictions['results'][key]=dict(predictions=dict(
            lane_centerline=[dict(id=i,points=np.asarray(line,dtype=np.float32),confidence=float(np.asarray(score).reshape(-1)[0])) for i,(line,score) in enumerate(zip(lines,scores))],
            traffic_element=[],topology_lclc=np.zeros((count,count)),topology_lcte=np.zeros((count,0))))
    # No scores or distances are filtered/relaxed here. Exact existing evaluator.
    workers=8 if len(keys)>100 else (-1 if has_geometry_cache() else 0)
    return evaluate_map({key:copy.deepcopy(gt[key]) for key in keys},predictions,workers=workers)


def evaluate_run(run_dir,output_dir):
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from nuscenes import NuScenes
    import projects.mmdet3d_plugin
    run_dir,output_dir=Path(run_dir),Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    write(output_dir/'status.json',dict(status='RUNNING',prediction_run=str(run_dir.resolve())))
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py');cfg.data.test.test_mode=True
    cfg.data.test.pipeline=[]
    dataset=build_dataset(cfg.data.test)
    all_infos=sorted(dataset.data_infos,key=lambda i:(i['scene_token'],i['timestamp']))
    by_token={i['token']:i for i in all_infos}
    tokens=json.loads((run_dir/'sequence.json').read_text())['tokens']
    if len(set(tokens))!=len(tokens) or not set(tokens)<=set(by_token):raise ValueError('Invalid sample identity')
    infos=[by_token[token] for token in tokens]
    with (run_dir/'predictions.pkl').open('rb') as handle:outputs=pickle.load(handle)
    if len(outputs)!=len(tokens):raise ValueError('Incomplete actual prediction coverage')
    dataset.data_infos=infos
    paths,tmp=dataset.format_results(outputs,jsonfile_prefix=str(output_dir/'detection'))
    nusc=NuScenes(version='v1.0-trainval',dataroot='data/nuscenes',verbose=False)
    metrics,curves,evaluation=detection_metrics(nusc,dataset.eval_detection_configs,paths['pts_bbox'],tokens,by_token)
    def population(boxes):
        counts=Counter(box.detection_name for token in tokens for box in boxes[token])
        return dict(total=sum(counts.values()),by_class=dict(sorted(counts.items())))
    write(output_dir/'detection_populations.json',dict(
        ground_truth=population(evaluation.gt_boxes),prediction=population(evaluation.pred_boxes),
        scope='After unchanged official class-range/point-count/bike-rack filtering; no diagnostic confidence cutoff'))
    clean,undefined=finite_metrics(metrics);write(output_dir/'detection_metrics.json',dict(metrics=clean,undefined=undefined))
    curve_values,curve_undefined=finite_metrics(curves)
    write(output_dir/'detection_curves.json',dict(curves=curve_values,undefined=curve_undefined))
    from .official_detection_audit import audit
    audit(evaluation,output_dir/'detection_matches_2m')
    with open(dataset.lane_anno_file,'rb') as handle:gt=pickle.load(handle)
    map_value,map_undefined=finite_metrics(map_metrics(gt,infos,outputs))
    write(output_dir/'map_metrics.json',dict(metrics=map_value,undefined=map_undefined,
        applicability=dict(DET_l='actual lane-centerline head',DET_t='no traffic-element prediction head',
            TOP_ll='no topology head; zero adjacency placeholder, not learned topology',TOP_lt='no traffic-element/topology prediction head')))
    scenes={}
    # Report GT-based per-scene Detection and Map using the same functions.
    # Detection slices the already officially filtered boxes, preserving criteria.
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.evaluate import DetectionEval
    for scene in sorted({i['scene_token'] for i in infos}):
        indices=[i for i,info in enumerate(infos) if info['scene_token']==scene]
        ev=DetectionEval.__new__(DetectionEval);ev.cfg=evaluation.cfg;ev.nusc=nusc;ev.verbose=False
        ev.gt_boxes=EvalBoxes();ev.pred_boxes=EvalBoxes()
        for index in indices:
            token=tokens[index];ev.gt_boxes.add_boxes(token,evaluation.gt_boxes[token]);ev.pred_boxes.add_boxes(token,evaluation.pred_boxes[token])
        ev.sample_tokens=ev.gt_boxes.sample_tokens
        scene_det,_=ev.evaluate()
        scenes[scene]=dict(frames=len(indices),detection=scene_det.serialize(),
            map=map_metrics(gt,[infos[i] for i in indices],[outputs[i] for i in indices]))
    scenes,scene_undefined=finite_metrics(scenes);write(output_dir/'scenes.json',dict(scenes=scenes,undefined=scene_undefined))
    full=set(tokens)==set(by_token)
    result=dict(status='COMPLETED',frames=len(tokens),scenes=len(scenes),full_validation=full,
        scope='FULL configured validation set' if full else 'OFFICIAL EVALUATOR ON DECLARED SUBSET, NOT full-val mAP/NDS/Map Score',
        detection=clean,map=map_value,task_quality_certified=False,production_approvals_added=0,
        detector_criteria=dataset.eval_detection_configs.serialize(),sample_tokens=tokens,
        prediction_run=str(run_dir.resolve()))
    write(output_dir/'summary.json',result)
    write(output_dir/'status.json',dict(status='COMPLETED',frames=len(tokens),full_validation=full))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    already_exists=Path(args.output).exists()
    try:r=evaluate_run(args.run,args.output)
    except Exception:
        import traceback
        if not already_exists and Path(args.output).exists():
            (Path(args.output)/'failure.log').write_text(traceback.format_exc())
            write(Path(args.output)/'status.json',dict(status='FAILED',error=traceback.format_exc()))
        raise
    print('COMPLETED',r['frames'],r['scope'],'mAP',r['detection']['mean_ap'],'NDS',r['detection']['nd_score'],'Map',r['map'])
