"""Summarize recorded executions only; missing runs remain explicitly pending."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'references/wsgrrr_20261022_followup'

def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def dist(values):
    a=np.asarray(list(values),dtype=float)
    return dict(n=len(a),mean=float(a.mean()),p50=float(np.median(a)),p95=float(np.percentile(a,95)),
                minimum=float(a.min()),maximum=float(a.max())) if len(a) else None
def delta(a,b):return dict(absolute=b-a,relative_percent=(b/a-1)*100 if a else None)

def visual(path):
    result=read(path/'results.json');data=[r for r in result['natural'] if r['continuous'] and r['strategy']=='D3']
    stages=[s for r in data for s in r['stages'] if 'total_windows' in s]
    total=sum(s['total_windows'] for s in stages);updated=sum(s['executed_windows'] for s in stages)
    candidate=[v for r in data for v in r['incremental']['wall_samples_ms']]
    native=[v for r in data for v in r['native']['wall_samples_ms']]
    paired=[(r['native']['wall_ms']-r['incremental']['wall_ms'])/r['native']['wall_ms']*100 for r in data]
    value=dict(frames=result['frames'],continuous_frames=len(data),samples_per_frame=data[0]['incremental']['repeats'],
        scope=result['timing_scope'],updated_windows=updated,total_windows=total,update_rate=updated/total,reuse_rate=1-updated/total,
        visual_wall_ms=dict(native=dist(native),wsgrr=dist(candidate)),
        paired_frame_speedup_percent=dist(paired),
        feature_mean_abs=dist(r['difference']['mean_abs'] for r in data),
        feature_max_abs=dist(r['difference']['max_abs'] for r in data),
        mandatory_windows=dist(r['region_refresh']['mandatory_windows'] for r in data),
        ordinary_windows=dist(r['region_refresh']['ordinary_windows'] for r in data),
        host_phase_ms={key:dist(r[key] for r in data if key!='M1_precheck_ms' or
            (r['ordinal']!=1 and not r.get('M1_precheck_profiled')))
            for key in ('M1_precheck_ms','M1_state_read_ms','region_scoring_host_ms','window_and_suffix_host_ms','cache_publication_host_ms')},
        host_timing_caveat='Asynchronous boundaries overlap pending GPU work. Cache publication host time includes finite-check synchronization.',
        recognized_mandatory_misses=sum(len(set(r['region_refresh']['mandatory_window_ids'])-
            set(r['stages'][0]['executed_window_ids'])) for r in data),
        max_age_before=max(max(r['region_refresh'].get('ages',[0])) for r in data),
        ordinary_budget=3)
    if 'cache_transition_audit' in data[0]:
        audits=[r['cache_transition_audit'] for r in data]
        value['cache_audit']=dict(checked_frames=sum(a['checked'] for a in audits),
            checks=sum(len(a.get('checks',{})) for a in audits),all_passed=all(all(a.get('checks',{}).values()) for a in audits))
        value['max_age_after']=max(max(a) for r in data for a in r['window_ages_after'].values())
        value['uncertainty_required_windows']=dist(len(r['region_refresh']['uncertainty_required_window_ids'])
            for r in data if 'uncertainty_required_window_ids' in r['region_refresh'])
    return value

report=dict(native_reuse=read(ROOT/'native_reuse_check.json'),ordinary_budget=3,
    policy='Sole current WSGRR; no new ablation, budget scan, approval or precision change.',
    visual_before=visual(OLD/'visual40_final'))
if (ROOT/'visual40_main/results.json').exists():report['visual_after']=visual(ROOT/'visual40_main')
else:report['visual_after']='PENDING'

native=read(OLD/'Native_full/official_eval/summary.json')
before=read(OLD/'D3_full/official_eval/summary.json')
def scores(summary):
    d=summary['detection'];m=summary['map']
    return dict(mAP=d['mean_ap'],NDS=d['nd_score'],mATE=d['tp_errors']['trans_err'],
        mASE=d['tp_errors']['scale_err'],mAOE=d['tp_errors']['orient_err'],
        mAVE=d['tp_errors']['vel_err'],mAAE=d['tp_errors']['attr_err'],
        DET_l=m['OpenLane-V2 Score']['DET_l'],lane_Fscore=m['F-Score for 3D Lane']['score'])
official=dict(native=scores(native),before=scores(before),frames=6019,scenes=150,
    map_scope='Actual lane-centerline head; no learned traffic-element or topology heads.',
    native_reference=str(OLD/'Native_full/official_eval/summary.json'))
p=ROOT/'main_full/official_eval/summary.json'
if p.exists():
    after=read(p);assert after['full_validation'] and after['frames']==6019
    assert after['sample_tokens']==native['sample_tokens']
    official.update(after=scores(after),after_vs_native={k:delta(v,scores(after)[k]) for k,v in scores(native).items()},
        after_vs_before={k:delta(v,scores(after)[k]) for k,v in scores(before).items()},
        per_class={c:dict(native=v,before=before['detection']['mean_dist_aps'][c],
            after=after['detection']['mean_dist_aps'][c],
            after_vs_native=delta(v,after['detection']['mean_dist_aps'][c]))
            for c,v in native['detection']['mean_dist_aps'].items()})
    ns=read(OLD/'Native_full/official_eval/scenes.json')['scenes']
    bs=read(OLD/'D3_full/official_eval/scenes.json')['scenes']
    cs=read(ROOT/'main_full/official_eval/scenes.json')['scenes']
    scene_rows=[dict(scene_id=s,frames=cs[s]['frames'],
        native_mAP=ns[s]['detection']['mean_ap'],before_mAP=bs[s]['detection']['mean_ap'],
        after_mAP=cs[s]['detection']['mean_ap'],
        after_vs_native_mAP=cs[s]['detection']['mean_ap']-ns[s]['detection']['mean_ap'],
        after_vs_before_mAP=cs[s]['detection']['mean_ap']-bs[s]['detection']['mean_ap']) for s in sorted(cs)]
    (ROOT/'paired_scenes.json').write_text(json.dumps(scene_rows,indent=2))
    official['per_scene_mAP_change']=dist(s['after_vs_native_mAP'] for s in scene_rows)
else:official['after']='PENDING_COMPLETE_OFFICIAL_EVALUATOR'
report['official']=official

p=ROOT/'main_full/metrics.json'
if p.exists():
    frames=rows(ROOT/'main_full/per_frame_metrics.jsonl')
    assert len(frames)==6019
    stages=[s for r in frames for s in r['m2']['stages'] if 'total_windows' in s]
    total=sum(s['total_windows'] for s in stages);updated=sum(s['executed_windows'] for s in stages)
    reasons={};ordinary_violations=misses=source_time_violations=0
    for r in frames:
        d=r['m2'];f=d['region_refresh'];stage=d['stages'][0]
        reason=str(d['full_update_reason']);reasons[reason]=reasons.get(reason,0)+1
        misses+=len(set(f['mandatory_window_ids'])-set(stage['executed_window_ids']))
        cameras=d['cache_context']['camera_ids'];per_camera=stage['total_windows']//len(cameras)
        ordinary_violations+=sum(sum(j//per_camera==c for j in f['ordinary_window_ids'])>3 for c in range(len(cameras)))
        for entries in d['output_window_provenance'].values():
            source_time_violations+=sum(e['computation_timestamp']>r['timestamp'] for e in entries)
    report['full_execution']=dict(frames=len(frames),scenes=len({r['scene_id'] for r in frames}),
        total_windows=total,updated_windows=updated,update_rate=updated/total,reuse_rate=1-updated/total,
        mandatory_windows=sum(r['m2']['region_refresh']['mandatory_windows'] for r in frames),
        ordinary_windows=sum(r['m2']['region_refresh']['ordinary_windows'] for r in frames),
        recognized_mandatory_misses=misses,per_camera_ordinary_budget_violations=ordinary_violations,
        future_source_timestamps=source_time_violations,
        cache_publication_errors=sum(r['m2'].get('cache_publication_error') is not None for r in frames),
        full_refresh_reasons=reasons,
        request_ms=dist(r['request_ms'] for r in frames),model_ms=dist(r['model_ms'] for r in frames),
        M2_wall_ms=dist(r['m2']['M2_wall_ms'] for r in frames),
        scope='All 6019 perception-only requests, including scene cold starts; descriptive timings, paired current complete-request repeats are separate.',
        event_caveat='Recall of recognized mandatory constraints only; not real-world change recall.')
    assert misses==ordinary_violations==source_time_violations==0

timings={}
for name in ('Native_r0','Native_r1','main_r0','main_r1'):
    p=ROOT/name/'per_frame_metrics.jsonl'
    if (ROOT/name/'metrics.json').exists():
        r=rows(p);timings[name]=dict(frames=len(r),request_ms=dist(v['request_ms'] for v in r),
            quality=read(ROOT/name/'metrics.json').get('quality'),
            scope=r[0]['timing_scope'])
report['complete_request_repeats']=timings or 'PENDING'
if len(timings)==4:
    cohorts={}
    for label,names in [('Native',['Native_r0','Native_r1']),('WSGRR',['main_r0','main_r1'])]:
        measured=[r for name in names for r in rows(ROOT/name/'per_frame_metrics.jsonl')]
        cohorts[label]=dict(request_ms=dist(r['request_ms'] for r in measured),
            model_ms=dist(r['model_ms'] for r in measured),unique_frames=len({r['frame_id'] for r in measured}),
            observations=len(measured),repeats=2)
    report['complete_request_pooled']=cohorts
    report['complete_request_p50_change']=delta(cohorts['Native']['request_ms']['p50'],cohorts['WSGRR']['request_ms']['p50'])
(ROOT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print('Recorded summary:',ROOT/'summary.json')
