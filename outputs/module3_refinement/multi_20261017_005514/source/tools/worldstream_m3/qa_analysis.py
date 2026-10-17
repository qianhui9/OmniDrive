"""Offline QA and native planning metrics from actual saved GPU answers.

No generator, carrier selector or online runtime imports this analysis. Ground
truth is matched by sample token, and missing/unalignable targets remain missing.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import re
import numpy as np
from .experiment import write
from .llm_probe import quality
from worldstream.m3.task_conditioner import task_type
from .offline_reference import resolve_reference


def planning_quality(answer, info):
    if 'gt_planning' not in info or 'gt_planning_mask' not in info:
        return dict(status='missing_ground_truth')
    gt=np.asarray(info['gt_planning'])[0,:,:2]
    mask=np.asarray(info['gt_planning_mask'])[0].all(axis=-1).astype(bool)
    pairs=re.findall(r'\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)',answer)
    result=dict(ground_truth=gt.tolist(),valid_mask=mask.tolist(),predicted_pair_count=len(pairs))
    if len(pairs)!=len(gt):
        return dict(result,status='malformed_or_wrong_trajectory_length',ade_m=None,fde_m=None)
    xy=np.asarray(pairs,float)
    distance=np.linalg.norm(xy-gt,axis=-1)
    return dict(result,status='parsed',trajectory=xy.tolist(),
        ade_m=float(distance[mask].mean()) if mask.any() else None,
        fde_m=float(distance[-1]) if mask[-1] else None)


def analyze(run, annotations):
    run=Path(run)
    rows=[json.loads(s) for s in (run/'frames.jsonl').read_text().splitlines()]
    with Path(annotations).open('rb') as handle:payload=pickle.load(handle)
    targets={r['token']:r for r in payload['infos']}
    results=[]
    recording_normalizations=0
    for row in rows:
        if len(row['generated_ids'])!=len(row['questions']):
            from .refinement_report import normalize_recordings
            recording_normalizations+=normalize_recordings(row)
        for qi,question in enumerate(row['questions']):
            answer=' '.join(str(x) for x in row['answers'][qi]['A'])
            reference,reference_source=resolve_reference(row['token'],question,row['ground_truth'])
            record=dict(mode=row['mode'],repeat=row['repeat'],warmup=row['warmup'],
                sample_token=row['token'],question=question,task=task_type(question),answer=answer,
                token_ids=row['generated_ids'][qi],
                ground_truth=reference,
                ground_truth_source=reference_source,
                qa_diagnostics=quality(answer,reference) if reference is not None else None)
            if 'planning trajectory' in question.lower():
                record['planning']=planning_quality(answer,targets.get(row['token'],{}))
            results.append(record)
    grouped=defaultdict(list)
    for record in results:grouped[(record['mode'],record['task'],record['warmup'])].append(record)
    summaries=[]
    for (mode,task,warmup),values in grouped.items():
        annotated=[r['qa_diagnostics'] for r in values if r['qa_diagnostics'] is not None]
        numeric=[r for r in annotated if r['numeric_mae'] is not None]
        plans=[r['planning'] for r in values if 'planning' in r]
        ade=[r['ade_m'] for r in plans if r.get('ade_m') is not None]
        fde=[r['fde_m'] for r in plans if r.get('fde_m') is not None]
        summaries.append(dict(mode=mode,task=task,warmup=warmup,questions=len(values),
            annotated_questions=len(annotated),
            normalized_answer_exact=sum(r['exact'] for r in annotated),
            mean_gt_token_f1=float(np.mean([r['token_f1'] for r in annotated])) if annotated else None,
            numeric_aligned_samples=len(numeric),
            numeric_count_mismatch=sum(not r['numeric_count_match'] for r in annotated),
            numeric_list_mae=float(np.mean([r['numeric_mae'] for r in numeric])) if numeric else None,
            planning_samples=len(plans),planning_parsed=sum(r['status']=='parsed' for r in plans),
            planning_ade_m=float(np.mean(ade)) if ade else None,planning_fde_m=float(np.mean(fde)) if fde else None))
    baselines={(r['repeat'],r['sample_token'],r['question']):r for r in results if r['mode']=='baseline'}
    paired=defaultdict(list)
    for r in results:
        if r['mode']=='baseline':continue
        reference=baselines.get((r['repeat'],r['sample_token'],r['question']))
        if reference is not None:paired[(r['mode'],r['task'],r['warmup'])].append((reference,r))
    paired_summaries=[]
    for (mode,task,warmup),values in paired.items():
        gt=[(a['qa_diagnostics'],b['qa_diagnostics']) for a,b in values if a['qa_diagnostics'] is not None and b['qa_diagnostics'] is not None]
        plans=[(a['planning'],b['planning']) for a,b in values if a.get('planning',{}).get('ade_m') is not None and b.get('planning',{}).get('ade_m') is not None]
        paired_summaries.append(dict(mode=mode,task=task,warmup=warmup,questions=len(values),
            token_ids_equal=sum(a['token_ids']==b['token_ids'] for a,b in values),annotated_pairs=len(gt),
            baseline_gt_token_f1=float(np.mean([a['token_f1'] for a,b in gt])) if gt else None,
            method_gt_token_f1=float(np.mean([b['token_f1'] for a,b in gt])) if gt else None,
            planning_ade_pairs=len(plans),
            baseline_planning_ade_m=float(np.mean([a['ade_m'] for a,b in plans])) if plans else None,
            method_planning_ade_m=float(np.mean([b['ade_m'] for a,b in plans])) if plans else None,
            planning_max_abs_ade_delta_m=max((abs(a['ade_m']-b['ade_m']) for a,b in plans),default=None)))
    return dict(source=str(run),source_sha256=hashlib.sha256((run/'frames.jsonl').read_bytes()).hexdigest(),
        duplicate_fallback_recordings_normalized=recording_normalizations,
        annotation_source=str(annotations),rows=results,summary=summaries,paired_summary=paired_summaries,
        comparison_rule='Use paired_summary for method comparisons. Raw summaries have unequal sample counts in incomplete runs.',
        quality_limits='Token F1 and numeric-list alignment are descriptive diagnostics, not semantic QA accuracy or safety certification. Planning uses actual native ground truth/mask by sample token.',
        performance_validity='This analysis does not promote an interrupted/OOM experiment into a valid final latency benchmark.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--annotations',default='data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    args=p.parse_args();result=analyze(args.run,args.annotations)
    write(Path(args.run)/'qa_ground_truth_analysis.json',result)


if __name__=='__main__':main()
