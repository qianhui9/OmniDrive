"""Natural Native/ZeroKV quality gate before any WRCPS ExactKV campaign."""
import argparse
import json
import os
from pathlib import Path
import re
import numpy as np
from tools.worldstream_m123.benchmark.common import write,sha256
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
from tools.worldstream_m2.global_residual_benchmark import execute,wait_gpu,evaluate
from tools.worldstream_m123.crossframe_e2e import native_pairs
from tools.worldstream_m123.crossframe_report import traffic_claims
from tools.worldstream_m123.benchmark.quality_calibration import paired_quality


def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def critical_motion_claims(text,question=''):
    """Offline review aids, never a substitute for natural scene evaluation."""
    claims=set()
    for sentence in re.split(r'[.!?;]',text.lower()):
        if not re.search(r'\b(pedestrians?|people|person|persons)\b',sentence+' '+question.lower()):continue
        if re.search(r'\b(crossing|cross(?:es)? the (?:road|street)|walking across)\b',sentence):
            negative=bool(re.search(r'\b(no pedestrians? (?:are )?crossing|not (?:currently )?crossing|none (?:are )?crossing)\b',sentence))
            claims.add(('pedestrian_crossing',not negative))
        if re.search(r'\b(waiting|standing|stationary)\b',sentence):claims.add(('pedestrian_waiting',True))
    return claims


def service(root,name,kind,sequence,*,repeat=0,fixed_reference=None,policy=None,teacher_diagnostic=False):
    if (root/name).exists():raise FileExistsError('Preserve previous service run')
    adapter=root/'adapter_training/adapter.pt'
    if kind!='native' and not adapter.exists():raise ValueError('Trained adapter required')
    count=len(json.loads(Path(sequence).read_text())['tokens'])
    args=['-m','tools.worldstream_m123.benchmark.worker','--output',str(root/name),'--run-root',str(root),
        '--condition','B0' if kind=='native' else 'B3','--workload','language','--limit',str(count),
        '--warmup','1','--repeat',str(repeat),'--sequence-file',str(sequence),'--questions-per-frame','3',
        '--measurement-set','efficiency']
    if kind!='native':
        args+=['--m3-strategy','exact_prefix','--wrcps-adapter',str(adapter),
            '--semantic-qualification',str(root/'semantic_qualification.json')]
        if policy:args+=['--m3-policy',str(policy)]
    if kind=='zero':args+=['--m3-zero-reuse']
    if kind=='exact' and not policy:raise ValueError('ExactKV needs newly calibrated policy')
    if fixed_reference:args+=['--fixed-token-reference',str(fixed_reference)]
    if teacher_diagnostic:args+=['--capture-m3-inputs','--wrcps-teacher-diagnostic']
    wait_gpu();execute(root,name,args)


def run_zero_quality(root):
    if not json.loads((root/'stable_core_analysis.json').read_text())['stage_one_pass']:raise ValueError('Core gate failed')
    if json.loads((root/'adapter_training/training_result.json').read_text())['trained_steps']<1:raise ValueError('Untrained adapter')
    sequence=json.loads((root/'acceptance_reserved_sequence.json').read_text())
    sequence['questions']={token:[p['question'] for p in native_pairs(token)[:3]] for token in sequence['tokens']}
    if any(len(q)!=3 for q in sequence['questions'].values()):raise ValueError('Need three original labelled questions')
    write(root/'quality_sequence.json',sequence)
    with gpu_job_lock():
        for kind in ('native','zero'):service(root,'quality_'+kind,kind,root/'quality_sequence.json')
    for kind in ('native','zero'):evaluate(root,'quality_'+kind)
    analyze_zero_quality(root)


def analyze_zero_quality(root):
    native=rows(root/'quality_native/per_question_metrics.jsonl');zero=rows(root/'quality_zero/per_question_metrics.jsonl')
    if len(native)!=len(zero):raise ValueError('Unpaired quality requests')
    paired=[]
    for a,b in zip(native,zero):
        if (a['frame_id'],a['question'],a['reference'])!=(b['frame_id'],b['question'],b['reference']):raise ValueError('Different quality inputs')
        if b['actual_reuse_length']!=0:raise ValueError('ZeroKV actually reused history')
        quality=paired_quality(b['answer'],a['answer'],a['reference'])
        baseline=critical_motion_claims(a['answer'],a['question'])
        candidate=critical_motion_claims(b['answer'],b['question'])
        gold=critical_motion_claims(a['reference'],a['question'])
        new=candidate-baseline
        contradicted=[c for c in new if (c[0],not c[1]) in gold]
        review=[c for c in new if c not in gold]
        traffic=traffic_claims(b['answer']);old_traffic=traffic_claims(a['answer']);gold_traffic=traffic_claims(a['reference'])
        changed_traffic={k:v for k,v in traffic.items() if v!=old_traffic.get(k) and v!=gold_traffic.get(k)}
        paired.append(dict(scene_id=a['scene_id'],frame_id=a['frame_id'],question=a['question'],reference=a['reference'],
            native_answer=a['answer'],zero_answer=b['answer'],**quality,
            new_pedestrian_claims_needing_review=review,contradicted_pedestrian_claims=contradicted,
            new_traffic_claims_needing_review=changed_traffic))
    write(root/'zero_quality_pairs.json',paired)
    scores={}
    for kind,rr in [('native',native),('zero',zero)]:
        official=json.loads((root/('quality_'+kind+'_official/summary.json')).read_text())
        scores[kind]=dict(mAP=official['detection']['mean_ap'],NDS=official['detection']['nd_score'],
            DET_l=official['map']['OpenLane-V2 Score']['DET_l'],Token_F1=float(np.mean([r['quality']['token_f1'] for r in rr])))
    losses={k:scores['native'][k]-scores['zero'][k] for k in scores['native']}
    critical=sum(r['critical_regressions'] for r in paired)
    contradicted=sum(bool(r['contradicted_pedestrian_claims']) for r in paired)
    unresolved=sum(bool(r['new_pedestrian_claims_needing_review'] or r['new_traffic_claims_needing_review']) for r in paired)
    passed=all(v<=.1 for v in losses.values()) and critical==0 and contradicted==0 and unresolved==0
    write(root/'zero_quality_summary.json',dict(stage_two_pass=passed,scores=scores,absolute_losses=losses,
        budget=.1,requests=len(paired),scenes=len({r['scene_id'] for r in paired}),
        critical_proxy_regressions=critical,contradicted_pedestrian_requests=contradicted,
        unresolved_critical_review_requests=unresolved,adapter_sha256=sha256(root/'adapter_training/adapter.pt'),
        semantic_qualification_sha256=sha256(root/'semantic_qualification.json'),
        scope='Official evaluator on declared six-scene subset; not full-validation metrics',
        limitation='Parsers are offline review aids. Unresolved new high-impact claims cannot grant quality acceptance.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['zero','analyze','diagnostic'],required=True);a=p.parse_args()
    if set(os.sched_getaffinity(0)) & {4,5}:p.error('Use taskset -c 0-3,6-31')
    root=Path(a.root)
    if a.stage=='diagnostic':
        with gpu_job_lock():service(root,'failure_diagnostic','zero',root/'failure_diagnostic_sequence.json',teacher_diagnostic=True)
    else:(run_zero_quality if a.stage=='zero' else analyze_zero_quality)(root)


if __name__=='__main__':main()
