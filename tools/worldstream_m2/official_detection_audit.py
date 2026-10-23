"""Read-only attribution of the actual SDK score-ordered 2 m matching pass.

No matching algorithm is reimplemented: callbacks record the unchanged distance
function's candidate visits, and a return hook verifies the SDK's TP decisions.
These uninterpolated counts are not AP or NDS.
"""
import json
from pathlib import Path
import sys
import numpy as np
from nuscenes.eval.detection.algo import accumulate
from .common import write


def sdk_matches(gt,pred,class_name,dist_fcn,threshold):
    visits={};captured={}
    def distance(a,b):
        value=dist_fcn(a,b)
        previous=visits.get(id(b))
        if value<(previous[1] if previous is not None else np.inf):visits[id(b)]=(a,float(value))
        return value
    previous=sys.getprofile()
    def returned(frame,event,arg):
        if frame.f_code is accumulate.__code__ and event=='return':
            captured.update({key:frame.f_locals[key] for key in ('npos','taken','tp','sortind','pred_boxes_list') if key in frame.f_locals})
        if previous is not None:previous(frame,event,arg)
    try:
        sys.setprofile(returned)
        data=accumulate(gt,pred,class_name,distance,threshold,verbose=False)
    finally:sys.setprofile(previous)
    if captured.get('npos',0)==0:
        return dict(assessed=False,matches=[],unmatched_predictions=[],taken=set(),metric_data=data)
    order=[captured['pred_boxes_list'][i] for i in captured['sortind']]
    # SDK converts TP to cumulative counts only after at least one match.
    values=np.asarray(captured['tp']);flags=np.diff(np.r_[0,values]).astype(bool) if captured['taken'] else values.astype(bool)
    if len(flags)!=len(order):raise ValueError('Official TP audit coverage differs')
    matches=[];unmatched=[]
    for box,matched in zip(order,flags):
        nearest=visits.get(id(box))
        if bool(nearest and nearest[1]<threshold)!=bool(matched):raise ValueError('Audit disagrees with actual SDK TP decision')
        if matched:matches.append((box,nearest[0],nearest[1]))
        else:unmatched.append(box)
    if len(matches)!=len(captured['taken']):raise ValueError('Official one-to-one GT count differs')
    return dict(assessed=True,matches=matches,unmatched_predictions=unmatched,taken=captured['taken'],metric_data=data)


def audit(evaluation,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    gt,pred=evaluation.gt_boxes,evaluation.pred_boxes
    raw=getattr(evaluation,'_raw_prediction_indices',{})
    position={id(box):(token,i) for token in gt.sample_tokens for i,box in enumerate(gt[token])}
    prediction_position={id(box):(token,i) for token in pred.sample_tokens for i,box in enumerate(pred[token])}
    summary={};pred_status={token:[] for token in pred.sample_tokens};missed=[]
    from tools.worldstream_m123.benchmark.metrics import finite_metrics
    def emit(handle,value):
        clean,_=finite_metrics(value);handle.write(json.dumps(clean,allow_nan=False)+'\n')
    with (directory/'matched_objects.jsonl').open('w') as matched_file:
        for label in evaluation.cfg.class_names:
            result=sdk_matches(gt,pred,label,evaluation.cfg.dist_fcn_callable,evaluation.cfg.dist_th_tp)
            positives=[b for b in gt.all if b.detection_name==label]
            predictions=[b for b in pred.all if b.detection_name==label]
            if not result['assessed']:
                summary[label]=dict(gt=0,predictions=len(predictions),tp=None,fp=None,fn=None,recall=None,precision=None,
                    reason='SDK returns no_predictions for absent GT class; predictions are not scored here')
                for box in predictions:pred_status[box.sample_token].append(dict(filtered_index=prediction_position[id(box)][1],raw_index=raw.get(id(box)),class_name=label,matched=None))
                continue
            for box,target,distance in result['matches']:
                token,index=prediction_position[id(box)];_,gt_index=position[id(target)]
                pred_status[token].append(dict(filtered_index=index,raw_index=raw.get(id(box)),class_name=label,score=box.detection_score,matched=True,gt_filtered_index=gt_index,xy_distance_m=distance))
                emit(matched_file,dict(frame_id=token,prediction_raw_index=raw.get(id(box)),prediction_filtered_index=index,
                    gt_filtered_index=gt_index,xy_distance_m=distance,prediction=box.serialize(),ground_truth=target.serialize()))
            for box in result['unmatched_predictions']:
                token,index=prediction_position[id(box)]
                pred_status[token].append(dict(filtered_index=index,raw_index=raw.get(id(box)),class_name=label,score=box.detection_score,matched=False))
            unmatched_gt=[b for b in positives if position[id(b)] not in result['taken']]
            missed.extend(dict(frame_id=b.sample_token,gt_filtered_index=position[id(b)][1],ground_truth=b.serialize()) for b in unmatched_gt)
            tp,fp,fn=len(result['matches']),len(result['unmatched_predictions']),len(unmatched_gt)
            summary[label]=dict(gt=len(positives),predictions=len(predictions),tp=tp,fp=fp,fn=fn,
                recall=tp/(tp+fn) if tp+fn else None,precision=tp/(tp+fp) if tp+fp else None)
    with (directory/'unmatched_ground_truth.jsonl').open('w') as handle:
        for row in missed:emit(handle,row)
    with (directory/'prediction_status.jsonl').open('w') as handle:
        for token,values in pred_status.items():emit(handle,dict(frame_id=token,predictions=sorted(values,key=lambda x:x['filtered_index'])))
    totals={key:sum(value[key] for value in summary.values() if value.get(key) is not None) for key in ('gt','predictions','tp','fp','fn')}
    totals['recall']=totals['tp']/totals['gt'] if totals['gt'] else None
    totals['precision_scored_classes']=totals['tp']/(totals['tp']+totals['fp']) if totals['tp']+totals['fp'] else None
    write(directory/'summary.json',dict(threshold_m=evaluation.cfg.dist_th_tp,classes=summary,totals=totals,
        scope='Actual unchanged SDK score-ordered XY matching, all retained scores, strict distance < threshold; raw uninterpolated counts, NOT mAP/NDS or prediction agreement. Indices refer to formatted prediction order and filtered official GT list.'))
    return totals


if __name__=='__main__':
    import argparse
    from nuscenes import NuScenes
    from nuscenes.eval.detection.data_classes import DetectionConfig
    from .task_quality import prepare_detection_evaluation
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-dir',required=True)
    parser.add_argument('--full-sequence',required=True)
    args=parser.parse_args();directory=Path(args.official_dir)
    summary=json.loads((directory/'summary.json').read_text())
    tokens=json.loads((directory.parent/'sequence.json').read_text())['tokens']
    full_tokens=json.loads(Path(args.full_sequence).read_text())['tokens']
    cfg=DetectionConfig.deserialize(summary['detector_criteria'])
    nusc=NuScenes(version='v1.0-trainval',dataroot='data/nuscenes',verbose=False)
    evaluation=prepare_detection_evaluation(nusc,cfg,directory/'detection/pts_bbox/results_nusc.json',tokens,full_tokens)
    print(audit(evaluation,directory/'detection_matches_2m'))
