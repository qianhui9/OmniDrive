"""Rebuild direct WSGRR serving evidence from existing measured artifacts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from tools.worldstream_m123.benchmark.metrics import distribution
from tools.worldstream_m123.benchmark.quality_calibration import paired_quality
from .window_validate import compare_task_runs
from .common import write


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def window_stats(frames):
    stages=[s for f in frames for s in f['stages'] if 'total_windows' in s]
    total=sum(s['total_windows'] for s in stages)
    reused=sum(s['reused_windows'] for s in stages)
    reuse_frames=sum(f['actual_skipped_windows']>0 for f in frames)
    return dict(frames=len(frames),total_windows=total,executed_windows=total-reused,
        reused_windows=reused,reuse_rate=reused/total,reuse_frames=reuse_frames,
        reuse_frame_fraction=reuse_frames/len(frames),
        per_block={str(i):dict(total_windows=sum(s['total_windows'] for s in stages if s['block']==i),
            reused_windows=sum(s['reused_windows'] for s in stages if s['block']==i),
            reuse_rate=sum(s['reused_windows'] for s in stages if s['block']==i)/sum(s['total_windows'] for s in stages if s['block']==i))
            for i in sorted({s['block'] for s in stages})},
        cache_publications=sum(f['cache_saved'] for f in frames),cache_versions=[f['cache_version'] for f in frames],
        full_refresh_reasons=dict(Counter(f.get('full_recompute_reason') or 'partial' for f in frames)))


def run(root,historical):
    root=Path(root)
    visual=read(root/'visual40/results.json')['natural']
    old=read(Path(historical)/'results.json')['natural']
    if len(visual)!=len(old):raise ValueError('Incomplete historical pairing')
    pair=[]
    for a,b in zip(visual,old):
        def ids(r,key):return [(s['block'],s[key]) for s in r['stages'] if 'total_windows' in s]
        pair.append(dict(frame_id=a['frame_id'],same_frame=a['frame_id']==b['frame_id'],
            same_plan=ids(a,'planned_window_ids')==ids(b,'planned_window_ids'),
            same_execution=ids(a,'executed_window_ids')==ids(b,'executed_window_ids'),
            same_cache_version=a['cache_version']==b['cache_version'],
            same_reuse=a['actual_skipped_windows']==b['actual_skipped_windows']))
    write(root/'d3_pairing.json',dict(rows=pair,all_equal=all(all(v for k,v in r.items() if k!='frame_id') for r in pair)))
    native=rows(root/'native40/per_frame_metrics.jsonl');formal=rows(root/'formal40/per_frame_metrics.jsonl')
    nq=rows(root/'native40/per_question_metrics.jsonl');fq=rows(root/'formal40/per_question_metrics.jsonl')
    if [f['frame_id'] for f in native]!=[f['frame_id'] for f in formal]:raise ValueError('Service frame identity mismatch')
    if [(q['frame_id'],q['question']) for q in nq]!=[(q['frame_id'],q['question']) for q in fq]:raise ValueError('Question pairing mismatch')
    quality=[dict(frame_id=b['frame_id'],question=b['question'],
        **paired_quality(b['answer'],a['answer'],b['reference'])) for a,b in zip(nq,fq)]
    write(root/'paired_vlm_quality.json',dict(rows=quality,scope='40 labelled questions in one development scene; lexical/critical-label proxies, not semantic certification'))
    task=compare_task_runs(root/'native40',root/'formal40',root/'task_differences.json')
    performance={}
    for name,ff,qq,label in [('native',native,nq,'eva_native'),('formal',formal,fq,'m2_total')]:
        performance[name]=dict(request_ms=distribution(f['request_ms'] for f in ff),
            model_ms=distribution(f['model_ms'] for f in ff),
            visual_cuda_span_ms=distribution(f['stages'][label]['gpu_span_ms'] for f in ff),
            visual_host_submission_ms=distribution(f['stages'][label]['host_wall_ms'] for f in ff),
            prefill_ms=distribution(f['stages']['prefill']['gpu_span_ms'] for f in ff),
            decode_ms=distribution(f['stages'].get('decode',{}).get('gpu_span_ms',0.) for f in ff),
            phase_timing_scope='Sum of Observer CUDA forward spans per frame; one question/frame; decode excludes sampling gaps',
            generated_tokens=sum(q['generated_tokens'] for q in qq),
            peak_gpu_allocated_bytes=max(f['peak_gpu_allocated_bytes'] for f in ff),
            peak_gpu_reserved_bytes=max(f['peak_gpu_reserved_bytes'] for f in ff))
    performance['visual_cuda_speedup']=performance['native']['visual_cuda_span_ms']['mean']/performance['formal']['visual_cuda_span_ms']['mean']
    performance['e2e_mean_speedup']=performance['native']['request_ms']['mean']/performance['formal']['request_ms']['mean']
    replay_timing={}
    for key in ('wall_ms','gpu_span_ms'):
        replay_timing[key]=dict(native=distribution(r['native'][key] for r in visual),
            formal=distribution(r['incremental'][key] for r in visual))
        replay_timing[key]['mean_speedup']=replay_timing[key]['native']['mean']/replay_timing[key]['formal']['mean']
    deviations=[]
    for a,b in zip(visual,formal):
        d=b['m2'];ra=a['region_refresh'];rb=d['region_refresh']
        if a['actual_skipped_windows']!=d['actual_skipped_windows']:
            deviations.append(dict(frame_id=b['frame_id'],replay_reused=a['actual_skipped_windows'],live_reused=d['actual_skipped_windows'],
                replay_mandatory=ra['mandatory_windows'],live_mandatory=rb['mandatory_windows'],
                replay_regions=a['m1_regions'],live_regions=rb.get('region_state_count'),
                replay_event_classes=dict(Counter(e.get('class_id') for e in ra.get('event_regions',[]))),
                live_event_classes=dict(Counter(e.get('class_id') for e in rb.get('event_regions',[]))),
                replay_full_reason=a['full_update_reason'],live_full_reason=d['full_update_reason']))
    write(root/'replay_vs_live_m1.json',dict(rows=deviations,scope='Historical Detection replay M1 versus live Detection/Map commit M1; not identical state inputs'))
    official={}
    for name in ('native','formal'):
        path=root/(name+'40_official_subset/summary.json')
        if path.exists():
            value=read(path);official[name]=dict(detection_mAP=value['detection']['mean_ap'],detection_NDS=value['detection']['nd_score'],map=value['map'],scope=value['scope'])
    result=dict(visual_replay_all=window_stats(visual),visual_replay_continuous=window_stats(visual[1:]),
        historical_pairing_all_equal=all(all(v for k,v in r.items() if k!='frame_id') for r in pair),
        full_service_all=window_stats([f['m2'] for f in formal]),
        full_service_continuous=window_stats([f['m2'] for f in formal[1:]]),
        service_performance=performance,replay_visual_performance=replay_timing,
        vlm_quality=dict(questions=len(quality),native_gt_token_f1=float(np.mean([q['baseline_f1'] for q in quality])),
            formal_gt_token_f1=float(np.mean([q['f1'] for q in quality])),
            f1_loss=float(np.mean([q['f1_loss'] for q in quality])),identical_answers=sum(q['answer_equal'] for q in quality),
            critical_proxy_regression_questions=sum(q['critical_regressions']>0 for q in quality),
            critical_proxy_regression_checks=sum(q['critical_regressions'] for q in quality)),
        official_subset=official,
        detection_mean_matched_center_distance_m=task['detection_mean_matched_center_distance_m'],
        map_mean_matched_point_distance_m=task['map_mean_matched_point_distance_m'],
        m3_reused_tokens=sum(q.get('reuse_diagnostics',{}).get('reused_tokens',0) for q in fq),
        m1_integrity=read(root/'formal40/m1_scene_integrity.json'),
        m1_full_refresh_reasons=dict(Counter(reason for f in formal for reason in (f.get('m1_precheck') or {}).get('full_refresh_reasons',[]))),
        runtime_breakdown_ms={key:distribution(f['m2'][key] for f in formal) for key in
            ('M2_validation_ms','M2_region_scoring_ms','M2_execution_ms','M2_cache_publication_ms','M2_total_ms')},
        m1_cpu_ms={key:distribution(f['m1_cpu_timings_ms'].get(key,0.) for f in formal) for key in
            ('precheck_total',)},
        m1_total_including_adapters_ms=distribution(f['m1_commit_timings_ms'].get('m1_total_including_adapters_ms',0.) for f in formal),
        m3_full_prefill_reasons=dict(Counter((q.get('reuse_diagnostics') or {}).get('full_prefill_reason','unknown') for q in fq)),
        replay_feature_difference=dict(mean_abs=float(np.mean([f['difference']['mean_abs'] for f in visual])),
            max_abs=max(f['difference']['max_abs'] for f in visual)),
        replay_vs_live_changed_frames=len(deviations),
        limitations=['Single development scene, one sampled question per frame, one end-to-end timing pass per condition',
            'CUDA event spans include host submission gaps; not active kernel time',
            'Historical replay states differ from live Detection/Map states',
            'M3 policy unchanged; prior independent quality on full M2 does not certify new visual inputs'])
    write(root/'comparison_summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--historical-d3',required=True)
    args=parser.parse_args();result=run(args.root,args.historical_d3)
    print(json.dumps({k:v for k,v in result.items() if k not in ('m1_integrity',)},ensure_ascii=False,indent=2))
