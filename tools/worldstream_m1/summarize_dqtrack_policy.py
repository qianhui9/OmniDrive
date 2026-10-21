"""Summarize frozen development runs without selecting/tuning an online policy."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from .dqtrack_policy_eval import percentiles,rates


def aggregate_strata(rows):
    totals={}
    for row in rows:
        for key,values in row['strata'].items():
            total=totals.setdefault(key,{})
            for k,v in values.items():
                if isinstance(v,int):total[k]=total.get(k,0)+v
    return {k:rates(v) for k,v in totals.items()}


def bootstrap_ratio(rows,numerator,denominator):
    rng=np.random.RandomState(20261021);values=[]
    for _ in range(2000):
        sampled=[rows[i]['metrics'] for i in rng.randint(0,len(rows),len(rows))]
        denom=sum(row[denominator] for row in sampled)
        if denom:values.append(sum(row[numerator] for row in sampled)/denom)
    return np.percentile(values,[2.5,97.5]).tolist() if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--paired-directory',default='paired_final')
    p.add_argument('--summary-output',default='analysis_summary.json')
    a=p.parse_args()
    root=Path(a.root);paired=root/a.paired_directory;summary=json.loads((paired/'summary.json').read_text())
    report=dict(methods={},failure_cases=[],policy_rejections={},timings={},
        deployment='RETAIN_CONSERVATIVE_DEFAULT_OFF; no qualified calibration artifact generated',
        interval_scope='descriptive scene bootstrap; four exposed scenes, not a risk bound')
    for method in ('geometry','shadow','conservative','research_tiebreak'):
        rows=[json.loads(p.read_text()) for p in sorted(paired.glob('*/'+method+'.json'))]
        traces=[t for row in rows for t in row['traces']]
        reject=collections.Counter();reasons=collections.Counter()
        for t in traces:
            for edge in t['edges']:
                if edge['status']=='wrong':report['failure_cases'].append(dict(method=method,frame_id=t['frame_id'],scene_id=t['scene_id'],**edge))
            for d in (t['dqtrack'] or {}).get('policy',{}).get('rejected',[]):reject.update(d['reasons'])
            for d in (t['dqtrack'] or {}).get('unaligned_records',[]):reasons[d['reason']]+=1
        report['methods'][method]=dict(metrics=summary['methods'][method]['metrics'],
            effective_recall_scene_interval=bootstrap_ratio(rows,'correct','eligible'),
            wrong_rate_scene_interval=bootstrap_ratio(rows,'wrong','scored'),
            strata=aggregate_strata(rows),unaligned_reasons=dict(reasons),
            m1=percentiles([t['m1_ms'] for t in traces]),
            bridge_alignment=percentiles([t['dqtrack']['elapsed_ms'] for t in traces if t['dqtrack']]),
            certified_observations=sum(len(t['certified']) for t in traces))
        report['policy_rejections'][method]=dict(reject)
    tracker=json.loads((root/'dqtrack/run_report.json').read_text())
    scenes=tracker['scenes'];ts=[t for s in scenes for t in s['frames']]
    report['timings']['dqtrack_forward']=percentiles([t['forward_ms'] for t in ts])
    report['timings']['dqtrack_forward_and_export']=percentiles([t['elapsed_ms'] for t in ts])
    report['timings']['dqtrack_extra_gpu_compute_total_ms']=sum(t['forward_ms'] for t in ts)
    report['timings']['dqtrack_peak_gpu_allocated_bytes']=max(s['peak_gpu_allocated_bytes'] for s in scenes)
    observations=json.loads((root/'omnidrive/model_observation_provenance.json').read_text())
    report['timings']['omnidrive_detection']=percentiles(observations['perception_wall_ms'])
    report['timings']['omnidrive_detection_peak_gpu_allocated_bytes']=max(t['allocated_bytes'] for t in observations['gpu_peak_memory'])
    vlm=root/'vlm_smoke'
    if (vlm/'native/results.json').exists():
        native=json.loads((vlm/'native/results.json').read_text())
        calls=json.loads((vlm/'language_timings.json').read_text())['calls']
        report['timings']['native_full_model']={}
        for mode in ('baseline','observe_only'):
            generation=[c['wall_ms'] for c in calls if '/'+mode+'/' in c['language_directory']]
            report['timings']['native_full_model'][mode]=dict(
                model_call=percentiles([f['end_to_end_wall_ms'] for f in native[mode]['frame_measurements']]),
                actual_generate=percentiles(generation),peak_allocated_bytes=native[mode]['peak_allocated_bytes'])
        report['timings']['native_scope']='two frames per mode, full OmniDrive+M1+Planning; external DQTrack and batch IO excluded'
    report['timings']['joint_dqtrack_vlm_serving_wall_ms']=None
    report['timings']['joint_scope']='separate-process staged experiments; no jointly scheduled full pipeline measured'
    official=json.loads((root/'official_subset/SUBSET_DIAGNOSTIC.json').read_text())
    report['tracking_subset']=dict(scope=official['scope'],scenes=official['scenes'],frames=official['frames'],
        metrics={k:official['metrics'].get(k) for k in ('amota','amotp','recall','ids','frag','mota','tp','fp','fn')})
    with (root/a.summary_output).open('x') as f:json.dump(report,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
