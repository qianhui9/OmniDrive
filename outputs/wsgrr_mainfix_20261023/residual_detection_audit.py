"""Inspect paired official GT changes; no inference or evaluation thresholds change."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.worldstream_m1.io_utils import read_prediction_subset

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'references/wsgrrr_20261022_followup/Native_full/official_eval'
NEW=ROOT/'main_full/official_eval'
CLASSES=('car','bicycle','pedestrian','motorcycle')
def rows(path):
    with path.open() as f:
        for line in f:yield json.loads(line)
def key(r):return r['frame_id'],r['gt_filtered_index']
native={key(r):r for r in rows(OLD/'detection_matches_2m/matched_objects.jsonl') if r['ground_truth']['detection_name'] in CLASSES}
current={key(r):r for r in rows(NEW/'detection_matches_2m/matched_objects.jsonl') if r['ground_truth']['detection_name'] in CLASSES}
unmatched={key(r):r for r in rows(NEW/'detection_matches_2m/unmatched_ground_truth.jsonl') if r['ground_truth']['detection_name'] in CLASSES}
lost=sorted(set(native)-set(current));restored=sorted(set(current)-set(native))
for k in lost:assert native[k]['ground_truth']==unmatched[k]['ground_truth']
selected={}
counts={}
for c in CLASSES:
    a=[k for k in lost if native[k]['ground_truth']['detection_name']==c]
    b=[k for k in restored if current[k]['ground_truth']['detection_name']==c]
    counts[c]=dict(native_TP_to_current_FN=len(a),native_FN_to_current_TP=len(b))
    if a:selected[c]=min(a,key=lambda k:(-native[k]['prediction']['detection_score'],k))
tokens={k[0] for k in selected.values()}
predictions=read_prediction_subset(NEW/'detection/pts_bbox/results_nusc.json',tokens)
statuses={r['frame_id']:r for r in rows(NEW/'detection_matches_2m/prediction_status.jsonl') if r['frame_id'] in tokens}
runtime={r['frame_id']:r for r in rows(ROOT/'main_full/per_frame_metrics.jsonl') if r['frame_id'] in tokens}
cases={}
for c,k in selected.items():
    n=native[k];gt=n['ground_truth'];target=np.asarray(gt['translation'])
    pool=[s for s in statuses[k[0]]['predictions'] if s['class_name']==c]
    nearest=min(pool,key=lambda s:np.linalg.norm(np.asarray(predictions[k[0]][s['raw_index']]['translation'])[:2]-target[:2])) if pool else None
    cand=predictions[k[0]][nearest['raw_index']] if nearest else None
    distance=float(np.linalg.norm(np.asarray(cand['translation'])[:2]-target[:2])) if cand else None
    d=runtime[k[0]]['m2'];stages=[s for s in d['stages'] if 'total_windows' in s]
    all_fresh=all(s['executed_windows']==s['total_windows'] for s in stages)
    reason=('no_retained_candidate_class' if not cand else 'nearest_candidate_outside_unchanged_2m_gate' if distance>=2.
        else 'nearby_candidate_assigned_to_other_GT' if nearest['matched'] and nearest.get('gt_filtered_index')!=k[1]
        else 'official_matching_requires_further_review')
    cases[c]=dict(frame_id=k[0],gt_filtered_index=k[1],ground_truth=gt,native_matched_prediction=n,
        candidate_nearest_same_class=cand,candidate_nearest_official_status=nearest,
        candidate_nearest_GT_xy_distance_m=distance,
        prediction_center_change_m=float(np.linalg.norm(np.asarray(cand['translation'])-n['prediction']['translation'])) if cand else None,
        observed_failure=reason,
        all_current_cached_prefix_windows_fresh=all_fresh,
        cache_context=d['cache_context'],full_update_reason=d['full_update_reason'],
        prefix_stages=stages,region_refresh=d['region_refresh'],
        attribution_limit=('Current frame reuses no cached prefix outputs; current-region omission alone cannot explain this residual. Temporal downstream head history and matching competition remain possible.' if all_fresh else
            'Partial prefix reuse exists. Window metadata alone does not prove this object ROI was reused or causally produced the error; no new counterfactual run was made.'))
result=dict(scope='Paired unchanged official 2m GT audit. Select highest Native-confidence TP->candidate-FN case per requested class; retain all GT changes separately. No candidate threshold, inference or production policy changed.',counts=counts,cases=cases)
(ROOT/'residual_detection_cases.json').write_text(json.dumps(result,indent=2,allow_nan=False))
(ROOT/'residual_detection_gt_changes.json').write_text(json.dumps(dict(
    lost=[dict(frame_id=k[0],gt_filtered_index=k[1],native=native[k]) for k in lost],
    restored=[dict(frame_id=k[0],gt_filtered_index=k[1],current=current[k]) for k in restored]),indent=2,allow_nan=False))
print('Residual cases', {c:dict(frame_id=v['frame_id'],reason=v['observed_failure'],all_fresh=v['all_current_cached_prefix_windows_fresh'],distance=v['candidate_nearest_GT_xy_distance_m']) for c,v in cases.items()})
