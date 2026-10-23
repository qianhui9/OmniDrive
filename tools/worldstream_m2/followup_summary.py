"""Aggregate measured followup artifacts; missing measurements stay explicit."""
import argparse
from collections import Counter, defaultdict
import json
import re
from pathlib import Path
import numpy as np
from .common import write
from tools.worldstream_m123.benchmark.metrics import distribution, quality_summary


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def delta(a,b):
    if a is None or b is None:return dict(native=a,candidate=b,absolute=None,relative_percent=None)
    return dict(native=a,candidate=b,absolute=b-a,relative_percent=(b/a-1)*100 if a else None)


def quality_delta(a,b):
    result={key:delta(a['detection'][key],b['detection'][key]) for key in ('mean_ap','nd_score')}
    result['tp_errors']={key:delta(value,b['detection']['tp_errors'][key]) for key,value in a['detection']['tp_errors'].items()}
    result['class_AP']={key:delta(value,b['detection']['mean_dist_aps'][key]) for key,value in a['detection']['mean_dist_aps'].items()}
    result['map']={group:{key:delta(value,b['map'][group][key]) for key,value in values.items()} for group,values in a['map'].items()}
    return result


def numeric_diagnostic(answer,reference):
    """Additional text-number diagnostic; original benchmark metric is retained."""
    if not isinstance(reference,str):return None
    pattern=r'(?<!\w)[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    a,b=([float(x) for x in re.findall(pattern,s)] for s in (answer,reference))
    if not np.isfinite(a+b).all():return dict(valid=False,reason='nonfinite_or_overflow_numeric_token')
    same=len(a)==len(b)
    return dict(valid=True,answer_numbers=a,reference_numbers=b,numeric_applicable=bool(b),
        numeric_count_match=same if b else None,numeric_mae=float(np.abs(np.asarray(a)-b).mean()) if b and same else None)


def visual_pairs(root):
    values={};frame_sets={}
    paths=dict(before=root.parent/'wsgrrr_20261022/continuous40',after=root/'visual40_final',baseline_final=root/'baseline_final40_control')
    control_alias=read(root/'baseline_control_alias.json') or {}
    if control_alias.get('baseline_final40_control'):
        paths['baseline_final']=root/control_alias['baseline_final40_control']
    for name,directory in paths.items():
        if not (directory/'results.json').exists():continue
        frames=[r for r in rows(directory/'frames.jsonl') if r.get('strategy',r['case'].split('_')[0])=='D3' and r['ordinal']>0]
        frame_sets[name]={r['frame_id']:r for r in frames}
        native=[r['native']['wall_ms'] for r in frames];candidate=[r['incremental']['wall_ms'] for r in frames]
        values[name]=dict(frames=len(frames),repeats=frames[0]['incremental']['repeats'],source=str(directory),
            native_frame_medians_ms=distribution(native),candidate_frame_medians_ms=distribution(candidate),
            paired_frame_delta_ms=distribution(b-a for a,b in zip(native,candidate)),
            mean_latency_change_percent=(np.mean(candidate)/np.mean(native)-1)*100,
            candidate_all_repeat_samples_ms=distribution(t for r in frames for t in r['incremental']['wall_samples_ms']),
            native_all_repeat_samples_ms=distribution(t for r in frames for t in r['native']['wall_samples_ms']),
            update_ratio=sum(sum(s['executed_windows'] for s in r['stages'] if 'total_windows'in s) for r in frames)/sum(sum(s['total_windows'] for s in r['stages'] if 'total_windows'in s) for r in frames),
            host_ranges={key:distribution(r[key] for r in frames) for key in ['M1_precheck_ms','M1_state_read_ms','region_scoring_host_ms','window_and_suffix_host_ms','cache_publication_host_ms']})
    write(root/'visual_before_after.json',values)
    if 'baseline_final' in frame_sets:
        baseline,current=frame_sets['baseline_final'],frame_sets['after']
        if set(baseline)!=set(current):raise ValueError('Baseline/current visual frame coverage differs')
        comparisons=[]
        for token,a in baseline.items():
            b=current[token];x=a['stages'][0]['executed_window_ids'];y=b['stages'][0]['executed_window_ids']
            comparisons.append(dict(frame_id=token,ordinal=a['ordinal'],baseline_windows=x,current_windows=y,
                exact_selected_ids_equal=x==y,baseline_events=len(a['region_refresh'].get('event_regions',[])),current_events=len(b['region_refresh'].get('event_regions',[]))))
        manifests=[read(paths[name]/'results.json') for name in ['baseline_final','after']]
        write(root/'baseline_current_selection_comparison.json',dict(frames=len(comparisons),
            same_selected_window_ids=sum(r['exact_selected_ids_equal'] for r in comparisons),
            same_input_tokens=manifests[0]['sample_tokens']==manifests[1]['sample_tokens'],
            same_replay_sha256=manifests[0]['m1_prediction_sha256']==manifests[1]['m1_prediction_sha256'],rows=comparisons,
            scope='Observed selection counts/IDs, never silently force equality or assume identical work from equal ordinary budgets'))


def request_control(root):
    result=dict(status='WAITING',conditions={},scope='Exact Git-baseline vs final code, same complete request scope and two repeated 40-frame runs. Native paired within each code phase; generation lengths remain explicit.')
    frame_sets={};question_sets={}
    for label in ('BaselineNative','BaselineD3','Native','D3'):
        fs=[];qs=[]
        for repeat in range(2):
            folder=root/f'{label}_r{repeat}'
            if (read(folder/'status.json') or {}).get('status')!='COMPLETED':break
            fs.extend(rows(folder/'per_frame_metrics.jsonl'));qs.extend(rows(folder/'per_question_metrics.jsonl'))
        if len(fs)!=80:continue
        fields={key:distribution(r[key] for r in fs) for key in ('request_ms','model_ms')}
        for key in ('M1_M2_state_read_ms','M2_wall_ms'):
            fields[key]=distribution(r['m2'][key] for r in fs if r['m2'] and key in r['m2'])
        result['conditions'][label]=dict(timing=fields,quality=quality_summary(qs),
            generated_tokens=distribution(q['generated_tokens'] for q in qs),
            official_subset=read(root/f'{label}_r0'/'subset_official/summary.json'))
        frame_sets[label]={(r['repeat'],r['frame_id']):r for r in fs}
        question_sets[label]={(r['repeat'],r['frame_id'],r['question_index']):r for r in qs}
    if len(frame_sets)==4:
        result['status']='COMPLETED';result['pairs']={}
        for before,after in [('BaselineNative','Native'),('BaselineD3','D3')]:
            a,b=frame_sets[before],frame_sets[after]
            if set(a)!=set(b):raise ValueError('Before/after complete-request input coverage differs')
            qa,qb=question_sets[before],question_sets[after]
            if set(qa)!=set(qb) or any(qa[k]['question']!=qb[k]['question'] for k in qa):raise ValueError('Before/after questions differ')
            result['pairs'][after]=dict(frames=len(a),same_seed_frames=sum(a[k]['seed']==b[k]['seed'] for k in a),
                same_perception_digests=sum(a[k]['output_digest']==b[k]['output_digest'] for k in a),
                same_answers=sum(qa[k]['answer']==qb[k]['answer'] for k in qa),
                paired_request_delta_ms=distribution(b[k]['request_ms']-a[k]['request_ms'] for k in a))
    write(root/'controlled_request_before_after.json',result)
    return result


def summarize(root):
    root=Path(root);result=dict(scope='Actual recorded runs only; no quality or speed certification',runs={})
    visual_pairs(root)
    result['controlled_request_before_after']=request_control(root)
    aliases=read(root/'run_aliases.json') or {}
    schedules={}
    for name in ('Native','D3','NoM1Region','NoM1RegionMatched'):
        frames=[];questions=[];repeats={};schedule={}
        for repeat in range(2):
            canonical=f'{name}_r{repeat}'
            folder=root/aliases.get(canonical,canonical)
            # The failed first matched attempt is retained and explicitly aliased.
            if name=='NoM1RegionMatched' and repeat==0 and not (folder/'metrics.json').exists():
                folder=root/'NoM1RegionMatched_r0_retry01'
            status=read(folder/'status.json')
            if not status or status['status']!='COMPLETED':continue
            fs=rows(folder/'per_frame_metrics.jsonl');qs=rows(folder/'per_question_metrics.jsonl')
            frames.extend(fs);questions.extend(qs)
            repeats[str(repeat)]=dict(directory=str(folder),request_ms=distribution(r['request_ms'] for r in fs),quality=quality_summary(qs))
            for row in fs:schedule[repeat,row['frame_id']]=row
        if not frames:continue
        groups=defaultdict(list)
        for q in questions:groups[q['task_type']].append(q)
        strata={
            'spatial_relation':('left','right','where','relative','located','position'),
            'object_state':('current activity','moving','parked','stopped','state of'),
            'critical_driving':('should','would happen','driving decisions','driving behavior','challenges','safe','cautious')}
        strata_results={}
        for key,terms in strata.items():
            selected=[q for q in questions if any(term in q['question'].lower() for term in terms)]
            values=quality_summary(selected)
            values['reference_exact_mismatch_rate']=1-values['answer_exact_match'] if selected else None
            strata_results[key]=values
        timing_keys=('preprocessing_ms','transfer_and_collate_ms','model_ms','output_transfer_ms','request_ms','gpu_stream_span_ms')
        summary=dict(frames=len(frames),unique_frames=len({r['frame_id'] for r in frames}),repeats=repeats,
            timing={key:distribution(r[key] for r in frames) for key in timing_keys},quality=quality_summary(questions),
            generated_tokens=distribution(q['generated_tokens'] for q in questions if q.get('generated_tokens') is not None),
            input_tokens=distribution(q['input_tokens'] for q in questions if q.get('input_tokens') is not None),
            question_categories={key:quality_summary(qs) for key,qs in groups.items()},
            question_text_strata=strata_results,strata_keywords=strata,
            strata_note='Exploratory overlapping text tags, not official benchmark categories. Semantic correctness is not inferred from lexical mismatch.',
            category_note='Question-text task_type tags; 1-ExactMatch is lexical reference mismatch, NOT adjudicated semantic error')
        number_rows=[dict(frame_id=q['frame_id'],question=q['question'],legacy=q['quality'],
            additional=numeric_diagnostic(q['answer'],q['reference'])) for q in questions if q['repeat']==0]
        numeric=[q['additional'] for q in number_rows if q['additional'] and q['additional']['valid'] and q['additional']['numeric_applicable']]
        errors=[q['numeric_mae'] for q in numeric if q['numeric_mae'] is not None]
        summary['additional_numeric_diagnostic']=dict(unique_input_question_pairs=len(number_rows),numeric_questions=len(numeric),
            numeric_count_matches=sum(q['numeric_count_match'] for q in numeric),numeric_mae=float(np.mean(errors)) if errors else None,
            numeric_mae_denominator=len(errors),rows=number_rows,
            scope='New text parser supports leading decimals and scientific notation; legacy unchanged. Equal number counts do NOT prove entity alignment or numeric correctness. MAE remains undefined when counts differ.')
        stages=defaultdict(list);host_stages=defaultdict(list)
        for row in frames:
            for key,value in row['stages'].items():
                stages[key].append(value['gpu_span_ms']);host_stages[key].append(value['host_wall_ms'])
        summary['stage_cuda_event_spans_ms']={k:distribution(v) for k,v in stages.items()}
        summary['stage_host_wall_ms']={k:distribution(v) for k,v in host_stages.items()}
        if name!='Native':
            counts=Counter();errors=[];forced=[];per_frame=[];m2_times=defaultdict(list);m1_times=defaultdict(list)
            for row in frames:
                diag=row['m2'];prefix=[s for s in diag['stages'] if 'total_windows' in s]
                refresh=diag.get('region_refresh',{});executed=set(prefix[0]['executed_window_ids'])
                must=set(refresh.get('mandatory_window_ids',[]));event=set(j for e in refresh.get('event_regions',[]) for j in e['windows'])
                counts.update(total=sum(s['total_windows'] for s in prefix),executed=sum(s['executed_windows'] for s in prefix),
                    required_unique=len(must),required_recalled=len(must&executed),event_unique=len(event),event_recalled=len(event&executed),
                    ordinary_unique=refresh.get('ordinary_windows',0),pointwise_tokens=sum(s.get('pointwise_tokens',0) for s in prefix),
                    qkv_tokens=sum(s.get('qkv_tokens',0) for s in prefix),event_rows=len(refresh.get('event_regions',[])))
                forced.append(len(must))
                per_frame.append(dict(repeat=row['repeat'],frame_id=row['frame_id'],executed=len(executed),total=prefix[0]['total_windows'],
                    forced=len(must),ordinary=refresh.get('ordinary_windows',0),event_unique=len(event),event_recalled=len(event&executed),
                    update_ratio=len(executed)/prefix[0]['total_windows'],reuse_ratio=1-len(executed)/prefix[0]['total_windows']))
                errors.extend(row.get('m2_reuse_checks',[]))
                for key,value in diag.items():
                    if key.endswith('_ms'):m2_times[key].append(value)
                for key,value in row.get('m1_precheck',{}).get('timings_ms',{}).items():m1_times[key].append(value)
            summary.update(window_counts=dict(counts),window_update_ratio=counts['executed']/counts['total'],
                reuse_ratio=1-counts['executed']/counts['total'],required_recall=counts['required_recalled']/counts['required_unique'] if counts['required_unique'] else None,
                event_coverage=counts['event_recalled']/counts['event_unique'] if counts['event_unique'] else None,
                forced_windows_per_frame=distribution(forced),per_frame_windows=per_frame,
                m2_host_ranges_ms={k:distribution(v) for k,v in m2_times.items()},m1_precheck_cpu_components_ms={k:distribution(v) for k,v in m1_times.items()},
                feature_errors={key:distribution(e[key] for e in errors) for key in ('mean_abs','max_abs','p95_abs')},
                feature_error_scope='Only actual-reuse frames; full-refresh frames are exact and excluded from this conditional distribution',
                event_caveat='Coverage of identified mandatory/event windows only, never true-world change recall; temporal frames/repeats are not independent samples')
        schedules[name]=schedule;result['runs'][name]=summary
    if 'D3' in schedules:
        comparisons={}
        for name in ('NoM1Region','NoM1RegionMatched'):
            if name not in schedules:continue
            count=hit=same=0;per_frame=[]
            for key,d3 in schedules['D3'].items():
                candidate=schedules[name].get(key)
                if candidate is None:continue
                required=set(j for e in d3['m2']['region_refresh'].get('event_regions',[]) for j in e['windows'])
                target=d3['m2']['stages'][0];other=candidate['m2']['stages'][0]
                chosen=set(other['executed_window_ids']);count+=len(required);hit+=len(required&chosen)
                cams=len(d3['m2']['cache_context']['camera_ids']);per_cam=target['total_windows']//cams
                target_counts=[sum(j//per_cam==i for j in target['executed_window_ids']) for i in range(cams)]
                actual_counts=[sum(j//per_cam==i for j in chosen) for i in range(cams)]
                same+=target_counts==actual_counts
                per_frame.append(dict(repeat=key[0],frame_id=key[1],d3_per_camera=target_counts,candidate_per_camera=actual_counts,
                    reference_events=len(required),recalled=len(required&chosen)))
            comparisons[name]=dict(reference_d3_event_windows=count,reference_event_recalled=hit,reference_coverage=hit/count if count else None,
                same_per_camera_window_count_frames=same,paired_frames=len(per_frame),per_frame=per_frame,
                caveat='D3 trajectory event footprints are a common audit reference, not independently detected ground-truth events')
        result['window_budget_comparisons']=comparisons
    feature_cohorts={name:{key:row['m2_reuse_checks'][0] for key,row in schedule.items() if row.get('m2_reuse_checks')}
                     for name,schedule in schedules.items() if name!='Native'}
    if len(feature_cohorts)==3:
        common=set.intersection(*(set(values) for values in feature_cohorts.values()))
        result['shared_reuse_feature_cohort']=dict(paired_frames_with_repeats=len(common),unique_frames=len({key[1] for key in common}),
            groups={name:{metric:distribution(values[key][metric] for key in sorted(common)) for metric in ('mean_abs','max_abs','p95_abs')}
                    for name,values in feature_cohorts.items()},
            scope='Intersection of actual measured reuse checks in all three candidates, same frames/repeats. Full-refresh frames without separate feature checks are not filled with invented measured zeros.')
    result['official']={}
    for scope,folders in [('subset40',dict(Native='Native_r0',D3='D3_r0',NoM1Region='NoM1Region_r0',NoM1RegionMatched='NoM1RegionMatched_r0_retry01')),
                          ('full6019',dict(Native='Native_full',D3='D3_full',NoM1Region='NoM1Region_full'))]:
        measured={}
        for name,folder in folders.items():
            p=root/folder/('subset_official' if scope=='subset40' else 'official_eval')
            value=read(p/'summary.json')
            measured[name]=value if value else dict(status='NOT_COMPLETED',detail=read(p/'status.json'),inference_progress=read(root/folder/'progress.json'))
        result['official'][scope]=measured
        if measured['Native'].get('status')=='COMPLETED':
            result['official'][scope+'_deltas']={name:quality_delta(measured['Native'],v) for name,v in measured.items() if name!='Native' and v.get('status')=='COMPLETED'}
            scene_deltas={}
            base_dir=root/folders['Native']/('subset_official' if scope=='subset40' else 'official_eval')
            base_scenes=(read(base_dir/'scenes.json') or {}).get('scenes',{})
            for name,folder in folders.items():
                if name=='Native':continue
                candidate_dir=root/folder/('subset_official' if scope=='subset40' else 'official_eval')
                candidate_scenes=(read(candidate_dir/'scenes.json') or {}).get('scenes',{})
                scene_deltas[name]={}
                for scene,a in base_scenes.items():
                    b=candidate_scenes.get(scene)
                    if b is None:continue
                    if a['frames']!=b['frames']:raise ValueError('Paired scene coverage differs')
                    scene_deltas[name][scene]=dict(frames=a['frames'],**quality_delta(a,b))
            write(root/(scope+'_paired_scenes.json'),scene_deltas)
    result['prediction_diagnostics']={}
    result['official_detection_operating_points']={}
    result['official_map_operating_points_subset40']={}
    for name in ('Native_r0','D3_r0','NoM1Region_r0','NoM1RegionMatched_r0_retry01'):
        value=read(root/name/'map_gt_audit40/summary.json')
        if value is not None:result['official_map_operating_points_subset40'][name]=value
    result['official_map_operating_points_full6019']={}
    for name in ('Native_full','D3_full','NoM1Region_full'):
        value=read(root/name/'map_gt_audit_full/summary.json')
        if value is not None:result['official_map_operating_points_full6019'][name]=value
    for name in ('Native_r0','D3_r0','NoM1Region_r0','NoM1RegionMatched_r0_retry01','Native_full','D3_full','NoM1Region_full'):
        directory=root/name/('official_eval' if name.endswith('_full') else 'subset_official')/'detection_matches_2m'
        value=read(directory/'summary.json')
        if value is not None:result['official_detection_operating_points'][name]=value
    for name in ('D3_r0','NoM1Region_r0','NoM1RegionMatched_r0_retry01'):
        directory=root/(name+'_diagnostic')
        if (root/(name+'_diagnostic_v2')/'summary.json').exists():directory=root/(name+'_diagnostic_v2')
        maps=read(directory/'maps.json');detections=read(directory/'detections.json')
        if maps is None or detections is None:continue
        result['prediction_diagnostics'][name]=dict(
            legacy_map_m=distribution(row['legacy_cost_m'] for frame in maps for row in frame['legacy']),
            symmetric_gated_map_m=distribution(row['symmetric_m'] for frame in maps for row in frame['symmetric_gated_matches']),
            map_native_count=sum(len(f['native']) for f in maps),map_candidate_count=sum(len(f['candidate']) for f in maps),
            map_native_unmatched=sum(len(f['unmatched_native']) for f in maps),map_candidate_unmatched=sum(len(f['unmatched_candidate']) for f in maps),
            detection_center_m=distribution(row['distance_m'] for frame in detections for row in frame['legacy']),
            detection_gated_matches=sum(f['gated_match_count'] for f in detections),
            detection_native_unmatched=sum(len(f['unmatched_native_indices']) for f in detections),
            detection_candidate_unmatched=sum(len(f['unmatched_candidate_indices']) for f in detections),
            scope='Prediction-to-prediction diagnostics at unchanged score>=0.3 cutoff, never GT accuracy')
    result['full_execution']={}
    for name in ('Native_full','D3_full','NoM1Region_full'):
        folder=root/name
        if (read(folder/'status.json') or {}).get('status')!='COMPLETED':continue
        counts=Counter();modes=Counter();reasons=Counter();scenes=defaultdict(Counter)
        requests=[];models=[];times=defaultdict(list)
        with (folder/'per_frame_metrics.jsonl').open() as handle:
            for line in handle:
                row=json.loads(line);counts['frames']+=1;requests.append(row['request_ms']);models.append(row['model_ms'])
                diag=row['m2'] or {};modes[diag.get('mode','native')]+=1
                reasons[str(diag.get('full_update_reason'))]+=1
                counts['cache_publication_errors']+=diag.get('cache_publication_error') is not None
                for stage in diag.get('stages',[]):
                    if 'total_windows' not in stage:continue
                    total,executed=stage['total_windows'],stage['executed_windows']
                    counts.update(total_windows=total,executed_windows=executed,pointwise_tokens=stage.get('pointwise_tokens',0),qkv_tokens=stage.get('qkv_tokens',0))
                    scenes[row['scene_id']].update(total_windows=total,executed_windows=executed)
                refresh=diag.get('region_refresh',{})
                counts.update(mandatory_unique=refresh.get('mandatory_windows',0),ordinary_unique=refresh.get('ordinary_windows',0),
                    identified_event_missed_windows=refresh.get('event_missed_windows',0))
                for key,value in row.get('m1_cpu_timings_ms',{}).items():times['M1/'+key].append(value)
                for key,value in diag.items():
                    if key.endswith('_ms'):times['M2/'+key].append(value)
        result['full_execution'][name]=dict(counts=dict(counts),modes=dict(modes),full_update_reasons=dict(reasons),
            request_ms=distribution(requests),model_ms=distribution(models),
            host_timing_ranges_ms={k:distribution(v) for k,v in times.items()},
            update_ratio=counts['executed_windows']/counts['total_windows'] if counts['total_windows'] else 1.,
            per_scene_windows={k:dict(v) for k,v in scenes.items()},
            scope='Complete perception-only inference; may overlap prior group CPU evaluation. Latencies are archived, not used to claim a cross-group speedup. Identified event recall is not real-world change recall.')
    write(root/'measured_summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    result=summarize(parser.parse_args().root)
    for name,value in result['runs'].items():
        print(name,'request p50/p95',value['timing']['request_ms']['p50'],value['timing']['request_ms']['p95'],'reuse',value.get('reuse_ratio'),'F1',value['quality']['token_f1'])
