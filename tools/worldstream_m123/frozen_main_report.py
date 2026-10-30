"""Receipt-only reporting for the frozen A/C benchmark; never selects parameters."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import pickle
import re
import time
import traceback

import numpy as np

from .benchmark.common import write, sha256
from .frozen_main_benchmark import ROOT, AUTHORITY, WORKLOADS, read, content_hash, validate_frozen
from .continuous_stream_report import grade
from .fact_bypass_identity import question_target, match_instances, request_identity
from .final_quality_analysis import raw_anchors, current_anchors, eligible, detection_frame, aggregate_detection, association
from worldstream.m3.task_conditioner import question_facts


def rows(path):
    path=Path(path)
    opener=path.open if path.exists() else lambda mode:gzip.open(str(path)+'.gz',mode)
    archived_handle=None;archive_name=None;archive_index=-1
    with opener('rt') as handle:
        try:
            for line in handle:
                if not line.strip():continue
                item=json.loads(line)
                if 'receipt_archive' not in item:
                    yield item;continue
                name=path.parent/item['receipt_archive']
                if name!=archive_name:
                    if archived_handle:archived_handle.close()
                    archived_handle=gzip.open(name,'rb');archive_name=name;archive_index=-1
                assert item['archive_row']>archive_index,'Archive index must remain chronological'
                while archive_index<item['archive_row']:
                    raw=archived_handle.readline();archive_index+=1
                import hashlib
                assert hashlib.sha256(raw).hexdigest()==item['record_sha256']
                record=json.loads(raw);assert record['frame_id']==item['frame_id']
                yield record
        finally:
            if archived_handle:archived_handle.close()


def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def distribution(values):
    values=[v for v in values if v is not None]
    if not values:return dict(n=0,mean=None,p50=None,p95=None,p99=None)
    assert np.isfinite(values).all()
    return dict(n=len(values),mean=float(np.mean(values)),**{
        'p'+str(q):float(np.percentile(values,q)) for q in (50,95,99)})


def ratio(a,b):
    return a/b if a is not None and b else None


def cluster_bootstrap(pairs, repeats=2000):
    """Paired resampling of entire Log totals, never individual frames."""
    logs=defaultdict(lambda:np.zeros(3))
    for p in pairs:
        logs[p['log_id']]+=np.array([p['A_ms'],p['C_ms'],1.])
    if len(logs)<2:return dict(logs=len(logs),CI95=None,limitation='Fewer than two Log clusters')
    values=np.stack(list(logs.values()));rng=np.random.default_rng(20261030)
    totals=values[rng.integers(0,len(values),size=(repeats,len(values)))].sum(1)
    speeds=totals[:,0]/totals[:,1]
    return dict(logs=len(logs),repeats=repeats,unit='Log',
        speedup_CI95=np.quantile(speeds,[.025,.975]).tolist(),
        latency_reduction_CI95=np.quantile(1-1/speeds,[.025,.975]).tolist(),
        mean_latency_difference_ms_CI95=np.quantile((totals[:,0]-totals[:,1])/totals[:,2],[.025,.975]).tolist(),
        limitation='Sampling variability over exposed validation logs, not independent TEST or causal isolation of any module. One timing repeat per arm.')


def quality_summary(events):
    known=[e for e in events if e['quality']['known']]
    return dict(requests=len(events),scored=len(known),UNKNOWN=len(events)-len(known),
        UNKNOWN_fraction=ratio(len(events)-len(known),len(events)),
        scored_answer_accuracy=mean([e['quality']['score'] for e in known]),
        fully_correct_answers=sum(e['quality']['score']==1 for e in known),
        answer_scoring_coverage=ratio(len(known),len(events)),
        returned_answer_coverage=ratio(sum(bool(e['answer'].strip()) for e in events),len(events)),
        FAST_answers=sum(e['llm_skipped'] for e in events),FAST_coverage=ratio(sum(e['llm_skipped'] for e in events),len(events)),
        absolute_severe=sum(bool(e['quality']['severe_error']) for e in known),
        unknown_reasons=dict(Counter(e['quality'].get('reason') or (e['quality'].get('reference') or {}).get('reason') or 'unavailable_GT_or_semantics'
            for e in events if not e['quality']['known'])))


def quality_bootstrap(pairs):
    groups=defaultdict(lambda:np.zeros(4))
    for p in pairs:groups[p['log_id']]+=np.array([p['A'],p['C'],p['new_severe'],1.])
    if len(groups)<2:return dict(logs=len(groups),CI95=None)
    values=np.stack(list(groups.values()));rng=np.random.default_rng(20261030)
    sums=values[rng.integers(0,len(values),size=(2000,len(values)))].sum(1)
    return dict(logs=len(groups),unit='Log',repeats=2000,
        loss_A_minus_C_CI95=np.quantile((sums[:,0]-sums[:,1])/sums[:,3],[.025,.975]).tolist(),
        new_severe_rate_CI95=np.quantile(sums[:,2]/sums[:,3],[.025,.975]).tolist(),
        scope='Known paired labels only; UNKNOWN not imputed, zero-observed severe CI is not a zero-risk guarantee.')


def performance(part):
    ms=[r['e2e_ms'] for r in part];total=sum(ms)
    return dict(frames=len(part),logs=len({r['log_id'] for r in part}),
        E2E_ms=distribution(ms),model_E2E_ms=distribution([r['model_ms'] for r in part]),
        input_preparation_ms=distribution([r['input_ms'] for r in part]),
        QA_frame_E2E_ms=distribution([r['e2e_ms'] for r in part if r['questions']]),
        idle_frame_E2E_ms=distribution([r['e2e_ms'] for r in part if not r['questions']]),
        FPS=ratio(1000*len(part),total),QPS=ratio(1000*sum(r['questions'] for r in part),total),
        peak_GPU_bytes=max((r['peak_gpu_bytes'] for r in part),default=0),
        M1_ms=distribution([r['M1_ms'] for r in part]),visual_CUDA_ms=distribution([r['visual_CUDA_ms'] for r in part]),
        M3_ms=distribution([r['M3_ms'] for r in part]),
        prefill_CUDA_ms_per_frame=distribution([r['prefill_CUDA_ms'] for r in part]),
        decode_CUDA_ms_per_frame=distribution([r['decode_CUDA_ms'] for r in part]))


def pair_summary(part, events):
    a=performance([r for r in part if r['arm']=='A']);c=performance([r for r in part if r['arm']=='C'])
    speed=ratio(a['E2E_ms']['mean'],c['E2E_ms']['mean'])
    aa=[e for e in events if e['arm']=='A'];cc=[e for e in events if e['arm']=='C']
    key=lambda e:(e['workload_id'],e['frame_id'],e['request_index'])
    bases={key(e):e for e in aa};known=[]
    for e in cc:
        b=bases.get(key(e))
        if b and e['quality']['known'] and b['quality']['known']:
            assert e['question']==b['question']
            known.append(dict(A=b['quality']['score'],C=e['quality']['score'],
                new_severe=bool(e['quality']['severe_error']) and not b['quality']['severe_error'],
                C_severe=bool(e['quality']['severe_error']),log_id=e['log_id'],
                frame_id=e['frame_id'],request_index=e['request_index']))
    frame_key=lambda r:(r['workload_id'],r['frame_id'])
    native={frame_key(r):r for r in part if r['arm']=='A'}
    pairs=[dict(log_id=r['log_id'],A_ms=native[frame_key(r)]['e2e_ms'],C_ms=r['e2e_ms'])
        for r in part if r['arm']=='C' and frame_key(r) in native]
    target_coverage={}
    for arm in ('A','C'):
        es=[e for e in events if e['arm']==arm and e['target']['known']]
        fs=[r for r in part if r['arm']==arm]
        gt=sum(r['raw_visible_GT']['GT'] for r in fs);tp=sum(r['raw_visible_GT']['TP'] for r in fs)
        target_coverage[arm]=dict(known_targets=len(es),
            raw_target_recall=mean([e['raw_target_recalled'] for e in es]),
            eligible_target_recall=mean([e['eligible_target_recalled'] for e in es]) if arm=='C' else None,
            raw_visible_object_GT=gt,raw_visible_object_TP=tp,raw_visible_object_recall=ratio(tp,gt))
    return dict(A=a,C=c,speedup=speed,latency_reduction=1-1/speed if speed else None,
        bootstrap=cluster_bootstrap(pairs),A_quality=quality_summary(aa),C_quality=quality_summary(cc),
        target_coverage=target_coverage,
        quality_pairs=dict(scored=len(known),all_requests=len(cc),UNKNOWN=len(cc)-len(known),
            A_accuracy=mean([r['A'] for r in known]),C_accuracy=mean([r['C'] for r in known]),
            loss_A_minus_C=mean([r['A']-r['C'] for r in known]),
            new_severe=sum(r['new_severe'] for r in known),C_absolute_severe=sum(r['C_severe'] for r in known),
            log_bootstrap=quality_bootstrap(known),
            quality_scope='Known-question fact accuracy; natural unsupported questions may have fractional fact recall; never general free-form accuracy certification'))


def append(handle,value):
    handle.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def lane_evaluate(root,destination,lane_rows):
    """Actual OpenLane evaluator on this run's matched S1 subset only."""
    from contextlib import redirect_stdout,redirect_stderr
    from .benchmark.metrics import finite_metrics
    from tools.worldstream_m2.map_eval import evaluate_map
    info=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']
    tokens={r['frame_id'] for r in lane_rows};by={i['token']:i for i in info if i['token'] in tokens}
    path=Path('data/nuscenes/data_dict_subset_B_val.pkl')
    result=dict(status='NOT_CERTIFIED',official_full_dataset_certification=False,
        scope='Current-run S1 paired predictions, one complete scene per sampled Log; not full val.',
        supported_metrics=['OpenLane-V2 DET_l','3D Lane F-Score'],
        unscored=['traffic elements','topology','composite score'],
        reason='Empty traffic/zero topology placeholders cannot support those quality claims.',
        GT_source=str(path),GT_sha256=sha256(path),evaluator='tools.worldstream_m2.map_eval.evaluate_map',
        evaluator_sha256=sha256('tools/worldstream_m2/map_eval.py'),groups={})
    full=pickle.load(path.open('rb'));missing=[t for t in tokens if by[t]['lane_info'] not in full]
    if missing:
        result['missing_GT_frames']=missing;write(destination/'map_quality_metrics.json',result);return result
    subset={by[t]['lane_info']:full[by[t]['lane_info']] for t in sorted(tokens)}
    result.update(status='SUBSET_EVALUATED',GT_frames=len(subset),
        GT_lane_observations=sum(len(v['annotation']['lane_centerline']) for v in subset.values()))
    for arm in ('A','C'):
        part=[r for r in lane_rows if r['arm']==arm]
        predictions=dict(method='OmniDrive',authors=[],**{'e-mail':'','institution / company':'','country / region':'China'},results={})
        for r in part:
            lane=r['lane_predictions'];n=len(lane['scores'])
            predictions['results'][by[r['frame_id']]['lane_info']]=dict(predictions=dict(
                lane_centerline=[dict(id=i,points=np.asarray(points,dtype=np.float32),confidence=float(np.asarray(score).item()))
                    for i,(points,score) in enumerate(zip(lane['points'],lane['scores']))],
                traffic_element=[],topology_lclc=np.zeros((n,n)),topology_lcte=np.zeros((n,0))))
        try:
            with (destination/f'map_{arm}_evaluation.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
                value=evaluate_map(subset,predictions,workers=4)
            metrics,nonfinite=finite_metrics(value)
            result['groups'][arm]=dict(metrics=metrics,nonfinite_metrics_unavailable=nonfinite,
                predictions=len(part),extra_filtering=False)
        except Exception:
            result['status']='PARTIAL_OR_UNAVAILABLE';result['groups'][arm]=dict(status='NOT_CERTIFIED',error=traceback.format_exc())
        write(destination/'map_quality_metrics.json',result)
    return result


def report(root, pilot=False):
    protocol=validate_frozen(root)
    source=root/'pilot/paired_gpu_receipts.jsonl' if pilot else root/'paired_gpu_receipts.jsonl'
    status=read(root/('pilot' if pilot else 'main')/'run_status.json')
    assert status['status']=='COMPLETED'
    destination=root/'pilot_analysis' if pilot else root
    destination.mkdir(exist_ok=True)
    objects=read(root/'current_gt_annotations.json');conditions=read(root/'scene_condition_manifest.json')['frames']
    frames=[];events=[];feature=[];raws=defaultdict(list);worlds=[];association_rows=[];severe=[];lane_rows=[]
    observations=Counter();m2_camera=defaultdict(Counter);position=defaultdict(Counter)
    modules=defaultdict(Counter);coverage=defaultdict(Counter);pairs_count=0
    with (destination/'per_frame_metrics.jsonl').open('w') as ff,(destination/'per_question_metrics.jsonl').open('w') as qf:
        for paired in rows(source):
            pairs_count+=1;assert paired['fair_inputs_verified']
            feature.append(dict(log_id=paired['log_id'],workload_id=paired['workload_id'],frame_id=paired['frame_id'],**paired['feature_error']))
            for arm in ('A','C'):
                r=paired[arm];assert r['config_sha256']==protocol['config_sha256']
                assert r['execution_completed'] and r['module_execution']['completed']
                assert sum(q['observed_forward_calls'] for q in r['requests'])==r['observed_llm_forward_calls']
                obs=objects[r['frame_id']];raw=raw_anchors(r);committed=current_anchors(r)
                raw_d=detection_frame(r,raw,obs);world_d=detection_frame(r,committed,obs) if arm=='C' else None
                raws[(r['workload_id'],arm)].append(raw_d)
                if r['workload_id']=='S1':
                    lane_rows.append(dict(arm=arm,frame_id=r['frame_id'],lane_predictions=r['visual_quality']['lane_predictions']))
                    if arm=='C':worlds.append(world_d);association_rows.append(r)
                visual=r['stages'].get('eva_native' if arm=='A' else 'm2_total',{})
                frame=dict(arm=arm,workload_id=r['workload_id'],log_id=r['log_id'],scene_id=r['scene_id'],
                    frame_id=r['frame_id'],frame_index=r['frame_index'],phase=r['phase'],questions=len(r['requests']),
                    e2e_ms=r['e2e_wall_ms'],model_ms=r['model_e2e_wall_ms'],input_ms=r['input_preparation_ms'],
                    M1_ms=r['m1_including_adapters_ms'],M3_ms=r['m3_wall_ms'],visual_CUDA_ms=visual.get('gpu_span_ms'),
                    M1_commit_host_ms=r['stages'].get('m1_commit',{}).get('host_wall_ms',0.),
                    M1_precheck_host_ms=r['stages'].get('m1_precheck',{}).get('host_wall_ms',0.),
                    feature_compute_CUDA_ms=(sum(v['gpu_span_ms'] for k,v in r['stages'].items()
                        if k=='m2_patch_embedding' or k.startswith(('m2_window_block_','m2_suffix_block_'))) if arm=='C' else visual.get('gpu_span_ms')),
                    visual_host_ms=visual.get('host_wall_ms'),peak_gpu_bytes=r['peak_gpu_bytes'],
                    prefill_CUDA_ms=sum(q['prefill_cuda_ms'] for q in r['requests']),
                    decode_CUDA_ms=sum(q['decode_cuda_ms'] for q in r['requests']),
                    LLM_forwards=r['observed_llm_forward_calls'],FAST=sum(q['llm_skipped'] for q in r['requests']),
                    raw_visible_GT=raw_d['groups']['visible_GT'],committed_visible_GT=world_d['groups']['visible_GT'] if world_d else None,
                    state_objects=len(committed),all_state_anchors=len((r.get('committed_world') or {}).get('anchors',[])),
                    M1_store_anchor_count=r['m1_commit'].get('anchor_count') if arm=='C' else None,
                    M1_store_feature_count=r['m1_commit'].get('feature_count') if arm=='C' else None,
                    M1_state_update_ms=r['m1_commit'].get('state_update_ms') if arm=='C' else None,
                    M1_association_ms=r['m1_commit'].get('association_ms') if arm=='C' else None,
                    config_sha256=r['config_sha256'],effective_config_sha256=r['effective_config_sha256'])
                frame.update({k:conditions[r['frame_id']][k] for k in ('dynamics','occlusion','density','targets','target_change','environment')})
                frame['visible_camera_count']=len(conditions[r['frame_id']]['cameras_with_visible_GT'])
                if arm=='C':
                    d=r['m2'];frame.update(executed_windows=d.get('executed_windows'),reused_windows=d.get('reused_windows'),
                        suffix_skipped_blocks=d.get('global_suffix_skipped_blocks'),
                        suffix_executed_blocks=d.get('global_suffix_executed_blocks'))
                    cameras=d.get('per_camera',[])
                    camera_items=cameras.items() if isinstance(cameras,dict) else ((str(info.get('camera_id',info.get('camera',i))),info) for i,info in enumerate(cameras))
                    for camera,info in camera_items:
                        if isinstance(info,dict):m2_camera[camera].update({k:v for k,v in info.items() if isinstance(v,(int,float)) and not isinstance(v,bool)})
                    region=d.get('region_refresh',{})
                    observations['ordinary_refresh_windows']+=len(region.get('ordinary_window_ids',[]))
                    observations['mandatory_refresh_windows']+=len(region.get('mandatory_window_ids',[]))
                    observations['actual_mandatory_window_computations']+=d.get('mandatory_windows',0)
                    observations['actual_ordinary_window_computations']+=d.get('ordinary_windows',0)
                frames.append(frame);append(ff,frame)
                for qi,(request,answer) in enumerate(zip(r['requests'],r['answers'])):
                    q=request['question'];target=question_target(q,obs)
                    quality=grade(q,answer['A'][0],r['frame_id'],objects)
                    allowed=set(target.get('acceptable_annotation_ids',[]));eligible_anchors=eligible(committed,q) if arm=='C' else []
                    eligible_ids={a['anchor_id'] for a in eligible_anchors}
                    def retained(matches,ids=None):
                        return any(m['known'] and m['class_correct'] and m['visible'] and m['annotation_id'] in allowed
                            for aid,m in matches.items() if ids is None or aid in ids)
                    target_known=target['known'];cov=coverage[(r['workload_id'],arm)]
                    cov['requests']+=1;cov['known_targets']+=int(target_known)
                    raw_recalled=retained(raw_d['matches']) if target_known else None
                    committed_recalled=retained(world_d['matches']) if target_known and world_d else None
                    eligible_recalled=retained(world_d['matches'],eligible_ids) if target_known and world_d else None
                    if target_known:
                        cov['raw_recalled']+=int(raw_recalled)
                        if arm=='C':cov['committed_recalled']+=int(committed_recalled);cov['eligible_recalled']+=int(eligible_recalled)
                    target_status={}
                    for stage_name,candidates,recalled in (('raw',raw,raw_recalled),('committed',committed,committed_recalled),('eligible',eligible_anchors,eligible_recalled)):
                        if arm=='A' and stage_name!='raw':
                            target_status[stage_name]='NOT_APPLICABLE_NATIVE';continue
                        potential=any(a['semantic_class']==g['semantic_class'] and math.dist(a['center_ego'],g['center_ego'])<2.
                            for a in candidates for g in obs if g['annotation_id'] in allowed and g['visible'])
                        state=('GT_UNKNOWN' if not target_known else 'CONFIRMED_RECALLED' if recalled else
                            'PREDICTION_MATCHING_UNKNOWN' if potential else 'NO_CORRECT_CLASS_CANDIDATE_WITHIN_2M')
                        target_status[stage_name]=state
                        cov[stage_name+'_'+state]+=1
                    proposal=request.get('fast_answer') or {}
                    identity=request_identity(request.get('fast_answer'),target,world_d['matches'] if world_d else {})
                    e=dict(arm=arm,workload_id=r['workload_id'],log_id=r['log_id'],scene_id=r['scene_id'],
                        frame_id=r['frame_id'],frame_index=r['frame_index'],phase=r['phase'],request_index=qi,
                        arrival_timestamp=r['timestamp'],
                        question=q,task=question_facts(q).intent,answer=answer['A'][0],quality=quality,target=target,
                        raw_target_recalled=raw_recalled,committed_target_recalled=committed_recalled,eligible_target_recalled=eligible_recalled,
                        target_coverage_status=target_status,proposal_identity=identity,
                        llm_skipped=request['llm_skipped'],observed_forward_calls=request['observed_forward_calls'],
                        llm_forward_calls=request['llm_forward_calls'],request_wall_ms=request['end_to_end_ms'],
                        prefill_CUDA_ms=request['prefill_cuda_ms'],decode_CUDA_ms=request['decode_cuda_ms'],
                        output_tokens=len(request['output_token_ids']),reused_tokens=request.get('reused_tokens',0),
                        draft_candidate_tokens=request.get('draft_candidate_tokens',0),draft_accepted_tokens=request.get('draft_accepted_tokens',0),
                        request_seed=request['request_seed'],sampling_first_forward_rng=request['sampling_first_forward_rng'],
                        proposal=request.get('fast_answer'),answer_source=request.get('answer_source'),
                        question_fact_extraction_wall_ms=proposal.get('question_fact_extraction_wall_ms'),
                        answer_validity_check_wall_ms=proposal.get('answer_validity_check_wall_ms'),
                        current_fact_extraction_wall_ms=proposal.get('current_fact_extraction_wall_ms'),
                        fast_answer_construction_wall_ms=proposal.get('fast_answer_construction_wall_ms'),
                        question_processing_host_ms=max(0.,request['end_to_end_ms']-request.get('prefill_including_cache_wall_ms',request['prefill_cuda_ms'])-
                            request.get('decode_loop_gpu_span_ms',request['decode_cuda_ms'])) if arm=='C' else None,
                        question_processing_measurement='descriptive host remainder; not exclusive GPU compute; Native has no separate question-processing observer')
                    e.update({k:frame[k] for k in ('dynamics','occlusion','density','targets','target_change','environment','visible_camera_count')})
                    if request['llm_skipped']:
                        assert request['observed_forward_calls']==0 and request['llm_forward_calls']==0
                        assert not request['prefill_completed'] and not request['decode_completed']
                    if e['task']=='position' and quality['known']:
                        expected=set(target.get('directions',[]));actual=set((quality.get('extracted') or {}).get('directions',[]))
                        fb=any(a in actual and b in expected for a,b in (('front','rear'),('rear','front')))
                        lr=any(a in actual and b in expected for a,b in (('left','right'),('right','left')))
                        pos=position[arm];pos['scored']+=1;pos['front_back_reversal']+=int(fb);pos['left_right_reversal']+=int(lr)
                        pos['diagonal_mismatch']+=int(len(expected)==2 and actual!=expected)
                        classes=(quality.get('extracted') or {}).get('classes',[])
                        pos['explicit_wrong_target_class']+=int(bool(classes) and target.get('semantic_class') not in classes)
                        pos['known_proposal_identity']+=int(identity['target']['known'])
                        pos['wrong_proposal_nearest_identity']+=int(identity['target']['known'] and not identity['target'].get('correct'))
                    if quality['known'] and quality['severe_error']:severe.append(e)
                    events.append(e);append(qf,e)
                    modules[arm].update(dict(LLM_forwards=request['llm_forward_calls'],requests=1,
                        prefill_skips=int(request['llm_skipped']),decode_skips=int(request['llm_skipped']),
                        complete_skipped_requests=int(request['llm_skipped']),fallback=int(not request['llm_skipped']),
                        output_tokens=len(request['output_token_ids']),
                        generated_tokens=len(request['output_token_ids'])*int(not request['llm_skipped']),
                        FAST_constructed_tokens=len(request['output_token_ids'])*int(request['llm_skipped']),
                        reused_KV_tokens=request.get('reused_tokens',0),
                        draft_candidates=request.get('draft_candidate_tokens',0),draft_accepted=request.get('draft_accepted_tokens',0)))
            if pairs_count%250==0:print('REPORT_PROGRESS',pairs_count,flush=True)
    assert pairs_count*2==status['full_model_calls']
    if not pilot:assert status['full_model_calls']==protocol['planned_main_model_calls']
    workloads={}
    for wid,name,_,_ in WORKLOADS:
        part=[r for r in frames if r['workload_id']==wid]
        if part:
            workloads[wid]=dict(name=name,**pair_summary(part,[e for e in events if e['workload_id']==wid]))
            q=workloads[wid]['quality_pairs'];budget=read(root/'authoritative_R0_policy.json')['quality_budget']
            q['frozen_average_quality_budget']=budget
            q['observed_loss_within_budget']=q['loss_A_minus_C']<=budget if q['loss_A_minus_C'] is not None else None
            q['separate_observed_zero_new_severe']=q['new_severe']==0
            q['strict_risk_certified']=False
    temporal={p:pair_summary([r for r in frames if r['phase']==p],[e for e in events if e['phase']==p]) for p in ('cold','transition','steady')}
    task_groups={t:pair_summary(frames,[e for e in events if e['task']==t]) for t in ('category','existence','position','unsupported','motion')}
    # Task frame E2E is shared by questions; only per-question timers are task-specific.
    for t,g in task_groups.items():
        g.pop('A');g.pop('C');g.pop('speedup');g.pop('latency_reduction');g.pop('bootstrap')
        g['service_latency_ms']={arm:distribution([e['request_wall_ms'] for e in events if e['task']==t and e['arm']==arm]) for arm in ('A','C')}
        g['Prefill_CUDA_ms']={arm:distribution([e['prefill_CUDA_ms'] for e in events if e['task']==t and e['arm']==arm]) for arm in ('A','C')}
        g['Decode_CUDA_ms']={arm:distribution([e['decode_CUDA_ms'] for e in events if e['task']==t and e['arm']==arm]) for arm in ('A','C')}
    strata={}
    for dimension in ('dynamics','occlusion','density','targets','target_change','environment','visible_camera_count'):
        strata[dimension]={str(value):pair_summary([r for r in frames if r[dimension]==value],
            [e for e in events if e[dimension]==value]) for value in sorted({r[dimension] for r in frames},key=str)}
    coverage_out={wid:{arm:dict(c,raw_recall=ratio(c['raw_recalled'],c['known_targets']),
        committed_recall=ratio(c['committed_recalled'],c['known_targets']) if arm=='C' else None,
        eligible_recall=ratio(c['eligible_recalled'],c['known_targets']) if arm=='C' else None,
        UNKNOWN_targets=c['requests']-c['known_targets']) for (w,arm),c in coverage.items() if w==wid} for wid in workloads}
    detection={wid:{arm:aggregate_detection(part) for (w,arm),part in raws.items() if w==wid} for wid in workloads}
    m1=dict(status='MEASURED',target_coverage=coverage_out,raw_detection=detection,
        unique_S1_committed_detection=aggregate_detection(worlds) if worlds else None,
        association=association(association_rows,objects) if association_rows else None,
        state_anchors=distribution([r['all_state_anchors'] for r in frames if r['arm']=='C']),
        current_objects=distribution([r['state_objects'] for r in frames if r['arm']=='C']),
        stored_anchors=distribution([r['M1_store_anchor_count'] for r in frames if r['arm']=='C']),
        stored_features=distribution([r['M1_store_feature_count'] for r in frames if r['arm']=='C']),
        state_update_ms=distribution([r['M1_state_update_ms'] for r in frames if r['arm']=='C']),
        association_ms=distribution([r['M1_association_ms'] for r in frames if r['arm']=='C']),
        cost_ms=distribution([r['M1_ms'] for r in frames if r['arm']=='C']),
        state_commit_host_ms=distribution([r['M1_commit_host_ms'] for r in frames if r['arm']=='C']),
        state_precheck_host_ms=distribution([r['M1_precheck_host_ms'] for r in frames if r['arm']=='C']),
        temporal_coverage={p:{arm:dict(GT=sum(r['raw_visible_GT']['GT'] for r in frames if r['phase']==p and r['arm']==arm),
            TP=sum(r['raw_visible_GT']['TP'] for r in frames if r['phase']==p and r['arm']==arm)) for arm in ('A','C')} for p in temporal},
        causal_claim='A/C compares the complete system; it does not isolate a causal M1 coverage improvement. Native has no committed/eligible state; its corresponding cells are N/A.')
    m1['recall_uncertainty']='Proven recall uses all known GT targets as denominator; ambiguous prediction matches remain explicitly PREDICTION_MATCHING_UNKNOWN and are not certified misses or true positives. Both counts and conservative lower bounds are retained.'
    if m1['association']:
        m1['association']['scope']='Partial unique-GT matching over complete selected Scenes. Observed spans are censored by Scene boundaries and UNKNOWN gaps; not true lifetime or standard MOTA/IDF1/HOTA.'
    m2=dict(status='MEASURED',Native_Full_visual_CUDA_ms=distribution([r['visual_CUDA_ms'] for r in frames if r['arm']=='A']),
        R0_incremental_visual_CUDA_ms=distribution([r['visual_CUDA_ms'] for r in frames if r['arm']=='C']),
        feature_compute_CUDA_ms={arm:distribution([r['feature_compute_CUDA_ms'] for r in frames if r['arm']==arm]) for arm in ('A','C')},
        feature_error=dict(relative_l2=distribution([e['relative_l2'] for e in feature]),rmse=distribution([e['rmse'] for e in feature]),
            max_abs=distribution([e['max_abs'] for e in feature]),per_frame=feature,
            by_workload={wid:dict(relative_l2=distribution([e['relative_l2'] for e in feature if e['workload_id']==wid]),
                rmse=distribution([e['rmse'] for e in feature if e['workload_id']==wid])) for wid in workloads}),
        actual_window_execution={k:sum(r[k] or 0 for r in frames if r['arm']=='C') for k in ('executed_windows','reused_windows','suffix_skipped_blocks','suffix_executed_blocks')},
        refresh_reasons=dict(observations),per_camera=dict(m2_camera),
        window_count_scope='executed/reused/actual_mandatory/actual_ordinary count window-block computations; refresh_windows counts geometric region IDs per frame before per-block expansion. Suffix blocks are counted separately.',
        detection=detection,official_full_val_detection=dict(status='NOT_CERTIFIED',reason='One complete scene per Log, not the full official validation split. Conservative current-GT recall is a separate subset metric.'),
        timing_note='Native full original backbone versus M2 sole executor, both CUDA events around actual execution; not a Full-M2 control or theoretical window estimate.')
    m3=dict(status='MEASURED',arms={arm:dict(c,
        speculative_token_acceptance=ratio(c['draft_accepted'],c['draft_candidates']),
        mean_generated_tokens_per_LLM_request=ratio(c['generated_tokens'],c['fallback']),
        mean_output_tokens_per_all_requests=ratio(c['output_tokens'],c['requests']),
        Prefill_CUDA_ms=distribution([e['prefill_CUDA_ms'] for e in events if e['arm']==arm]),
        Decode_CUDA_ms=distribution([e['decode_CUDA_ms'] for e in events if e['arm']==arm]),
        Question_Processing_host_remainder_ms=distribution([e['question_processing_host_ms'] for e in events if e['arm']==arm]),
        Question_Fact_Extraction_host_ms=distribution([e['question_fact_extraction_wall_ms'] for e in events if e['arm']==arm]),
        Answer_Validity_Check_host_ms=distribution([e['answer_validity_check_wall_ms'] for e in events if e['arm']==arm]),
        Current_Fact_Extraction_host_ms=distribution([e['current_fact_extraction_wall_ms'] for e in events if e['arm']==arm]),
        Fast_Answer_Construction_host_ms=distribution([e['fast_answer_construction_wall_ms'] for e in events if e['arm']==arm]),
        M3_total_ms_per_frame=distribution([r['M3_ms'] for r in frames if r['arm']==arm]),
        FAST_quality=quality_summary([e for e in events if e['arm']==arm and e['llm_skipped']]),
        LLM_fallback_quality=quality_summary([e for e in events if e['arm']==arm and not e['llm_skipped']])) for arm,c in modules.items()},
        counting='Prefill and Decode simultaneous skips are the same complete skipped request; never summed as two skips.',
        prefix_note='Frozen world_guided branch retains existing exact-prefix code but explicitly does not run legacy prefix planner or persistent prefill KV; report measured KV reuse, including zero. No module code is removed or enabled outside frozen policy.')
    fast=[e for e in events if e['arm']=='C' and e['llm_skipped']]
    identity_known=[e for e in fast if e['proposal_identity']['target']['known']]
    m3['FAST_physical_target_identity']=dict(requests=len(fast),scored=len(identity_known),UNKNOWN=len(fast)-len(identity_known),
        correct=sum(bool(e['proposal_identity']['target'].get('correct')) for e in identity_known),
        accuracy=mean([float(e['proposal_identity']['target'].get('correct',False)) for e in identity_known]),
        scope='Separate physical identity metric. Ambiguous or unmatched detector identities remain UNKNOWN even if the answer text has known common task facts.')
    write(destination/'m1_metrics.json',m1);write(destination/'m2_metrics.json',m2);write(destination/'m3_metrics.json',m3)
    write(destination/'workload_results.json',workloads);write(destination/'temporal_results.json',temporal)
    write(destination/'task_results.json',task_groups);write(destination/'condition_results.json',strata)
    write(destination/'position_error_breakdown.json',dict(counts=position,
        nearest_target_scope='Proposal identity only when both GT nearest identity and detector match are uniquely known; free text alone cannot certify a physical nearest identity.'))
    native_es={(e['workload_id'],e['frame_id'],e['request_index']):e for e in events if e['arm']=='A'}
    for e in severe:
        if e['arm']=='C':
            native=native_es[(e['workload_id'],e['frame_id'],e['request_index'])]
            e['new_relative_to_Native']=native['quality']['known'] and not native['quality']['severe_error']
            e['Native_answer']=native['answer'];e['Native_quality']=native['quality']
            identity=e['proposal_identity']['target']
            e['causal_scope']=('Both methods show a severe error on this request; not new relative to Native' if not e['new_relative_to_Native'] else
                'FAST physical-target mismatch observed; A/C alone cannot isolate an individual M1/M2/M3 cause' if e['llm_skipped'] and identity['known'] and not identity.get('correct') else
                'FAST answer contradicts known task facts; physical target identity remains UNKNOWN' if e['llm_skipped'] and not identity['known'] else
                'Known answer-fact severe error; full raw prediction and generation evidence retained, no causal intervention or policy change')
    write(destination/'severe_error_cases.json',dict(absolute_cases=severe,
        unique_cases={arm:len({(e['frame_id'],e['question']) for e in severe if e['arm']==arm}) for arm in ('A','C')},
        note='Repeated workload observations and unique frame/question cases are both disclosed; no error-based exclusions or policy changes.'))
    certified=[w for w,g in workloads.items() if g['bootstrap'].get('speedup_CI95',[0])[0]>1]
    slower=[w for w,g in workloads.items() if g['speedup']<=1]
    quality_loss=[w for w,g in workloads.items() if (g['quality_pairs']['loss_A_minus_C'] or 0)>0]
    new_severe=sum(g['quality_pairs']['new_severe'] for g in workloads.values())
    total=pair_summary(frames,events)
    verdict=dict(status='PILOT_VALIDATED' if pilot else 'COMPLETED_REAL_PAIRED_BENCHMARK',configuration='R0',
        frozen_config_sha256=protocol['config_sha256'],actual_new_full_model_calls=status['full_model_calls'],
        main_calls=status['full_model_calls'] if not pilot else 0,pilot_excluded_from_main=True,
        logs=len({r['log_id'] for r in frames}),scenes=len({r['scene_id'] for r in frames}),
        GPU='RTX3090',workloads=workloads,paired_speedup_CI_above_one=certified,point_estimate_no_E2E_benefit=slower,
        observed_known_answer_quality_losses=quality_loss,new_severe_relative_to_Native=new_severe,
        new_unique_severe_frame_questions=len({(e['frame_id'],e['question']) for e in severe if e['arm']=='C' and e.get('new_relative_to_Native')}),
        C_unique_severe_frame_questions=len({(e['frame_id'],e['question']) for e in severe if e['arm']=='C'}),
        A_unique_severe_frame_questions=len({(e['frame_id'],e['question']) for e in severe if e['arm']=='A'}),
        C_absolute_severe=total['C_quality']['absolute_severe'],A_absolute_severe=total['A_quality']['absolute_severe'],
        production_deployment_allowed=False,independent_TEST=False,
        original_authority_risk=read(root/'frozen_main_config.json')['known_risk'],
        paper_main_configuration=('CONDITIONAL: valid fixed full-R0 research comparison; whole-system performance claims restricted to supported workloads, with all measured quality/coverage losses and severe errors disclosed. Not strict safety, independent generalization, or publication acceptance certification.'),
        unsupported_claims=['Full 150-scene official val performance','Independent TEST generalization','Causal per-module accuracy improvement',
            'Exhaustive visual absence/FP labels','Queueing SLA','Module-level saved time equals E2E saved time'],
        algorithm_parameters_changed=False,Native_source_modified=False,weights_modified=False)
    write(destination/'final_research_verdict.json',verdict)
    if not pilot:
        lane_evaluate(root,destination,lane_rows)
    render(destination,root,workloads,temporal,task_groups,strata,m1,m2,m3,verdict,total)
    write(destination/'report_source_attestation.json',dict(source=__file__,sha256=sha256(__file__),
        paired_receipts_sha256=sha256(source),paired_frames=pairs_count,configuration_sha256=protocol['config_sha256'],
        recomputation='Deterministic receipt-only computation; GT cached before GPU execution, no policy selection or inference.'))
    print('REPORT_COMPLETE',status['full_model_calls'],len(events),'QA',flush=True)


def verify_receipts(root):
    """Recheck actual pair/config/forward evidence independently of prose tables."""
    protocol=validate_frozen(root);frozen=read(root/'frozen_main_config.json')
    assert (root/'frozen_main_config.sha256').read_text().strip()==protocol['config_sha256']
    runtime={}
    for record in rows(root/'main/effective_runtime_configs.jsonl'):
        pair_id=record.pop('pair_id');arm=record['arm']
        if arm=='C':
            for name in ('M1','M2','M3'):assert record[name]==frozen[name],(pair_id,name)
        else:assert all(record[name] is None for name in ('M1','M2','M3'))
        runtime[(pair_id,arm)]=content_hash(record)
    count=Counter();seen=set();pairs_expected={p['pair_id']:p for p in read(root/'workload_manifest.json')['pairs']}
    for p in rows(root/'paired_gpu_receipts.jsonl'):
        a,c=p['A'],p['C'];key=(p['pair_id'],p['frame_id']);assert key not in seen;seen.add(key)
        planned=pairs_expected[p['pair_id']]
        assert a['frame_id']==planned['scene']['tokens'][a['frame_index']]
        assert a['arm_order']==planned['order']==c['arm_order']
        for field in ('frame_id','frame_index','timestamp','camera_ids','questions','prompt_token_ids','request_seeds','input_digest'):
            assert a[field]==c[field],(key,field)
        assert a['m1_commit_count']==0 and a['committed_world'] is None and a['m2'] is None
        assert c['module_execution']['m1']['commit_completed'] and c['m2']['executor']=='WindowExecutor'
        for arm,r in (('A',a),('C',c)):
            assert r['config_sha256']==protocol['config_sha256']
            assert r['effective_config_sha256']==runtime[(p['pair_id'],arm)]
            assert abs(r['e2e_wall_ms']-r['input_preparation_ms']-r['model_e2e_wall_ms'])<1e-6
            assert sum(q['llm_forward_calls'] for q in r['requests'])==r['observed_llm_forward_calls']
            assert len(r['requests'])==len(r['answers'])==len(r['questions'])
            count[arm+'_frames']+=1;count[arm+'_questions']+=len(r['requests'])
            count[arm+'_actual_LLM_forwards']+=r['observed_llm_forward_calls']
            count[arm+'_idle_frames']+=int(not r['requests'])
            for q in r['requests']:
                assert q['observed_forward_calls']==q['llm_forward_calls']
                if q['llm_skipped']:
                    assert arm=='C' and q['observed_forward_calls']==0
                    assert not q['prefill_completed'] and not q['decode_completed'] and q['layout_preparation_ms']==0
                    count['complete_zero_forward_skips']+=1
        for aq,cq in zip(a['requests'],c['requests']):
            if not cq['llm_skipped']:assert aq['sampling_first_forward_rng']==cq['sampling_first_forward_rng']
    assert 2*len(seen)==protocol['planned_main_model_calls']
    assert sum(len(p['scene']['tokens']) for p in pairs_expected.values())==len(seen)
    for p,h in read(root/'historical_output_hashes.json').items():assert sha256(p)==h,p
    from mmcv import Config
    baseline=Config.fromfile('projects/configs/OmniDrive/mask_eva_lane_det_vlm.py')
    assert all(not baseline.model.get(k,False) for k in ('m1_enable','m2_enable','m3_enable','worldstream_require_m123'))
    assert not read(AUTHORITY/'final_research_policy.json')['production_deployment_allowed']
    write(root/'integrity_validation.json',dict(status='PASS',counts=count,paired_frames=len(seen),
        config_sha256=protocol['config_sha256'],historical_files_verified=len(read(root/'historical_output_hashes.json')),
        runtime_parameters_match_frozen=True,Native_M123_disabled=True,C_full_M123_enabled=True,
        pair_inputs_and_first_forward_RNG_verified=True,all_idle_frames_retained=True,
        actual_zero_forward_skips_verified=True,production_default_off=True,
        no_historical_GPU_receipt_reuse=True,algorithm_sources_unchanged=True))
    print('PAIRED_RECEIPT_INTEGRITY_PASS',dict(count),flush=True)


def fmt(value,digits=3):
    return 'N/A' if value is None else f'{value:.{digits}f}' if isinstance(value,(float,int)) else str(value)


def md_table(headers,items):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in items)


def figures(dest,workloads,m1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    folder=dest/'figures';folder.mkdir(exist_ok=True)
    value_or_nan=lambda value:np.nan if value is None else value
    keys=list(workloads);x=np.arange(len(keys));labels=[workloads[k]['name'].replace('-','-\n',1) for k in keys]
    fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    for arm,offset,color in [('A',-.18,'#777777'),('C',.18,'#176DAB')]:
        axs[0].bar(x+offset,[workloads[k][arm]['E2E_ms']['mean'] for k in keys],.36,label='Native' if arm=='A' else 'M123 R0',color=color)
    axs[0].set_ylabel('Complete visual-frame E2E (ms)');axs[0].legend(frameon=False)
    speeds=np.array([workloads[k]['speedup'] for k in keys]);intervals=[workloads[k]['bootstrap'].get('speedup_CI95') for k in keys]
    axs[1].scatter(x,speeds,color='#176DAB',zorder=3)
    for i,interval in enumerate(intervals):
        if interval:axs[1].vlines(i,*interval,color='#176DAB',lw=2)
    axs[1].axhline(1,color='#888888',ls='--',lw=1);axs[1].set_ylabel('Native / M123 R0 speedup')
    axs[1].set_title('95% paired Log bootstrap intervals',fontsize=10)
    for ax in axs:ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for suffix in ('pdf','svg'):fig.savefig(folder/('workload_performance.'+suffix),bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    for arm,offset,color in [('A',-.18,'#777777'),('C',.18,'#176DAB')]:
        axs[0].bar(x+offset,[value_or_nan(workloads[k]['quality_pairs'][arm+'_accuracy']) for k in keys],.36,
            label='Native' if arm=='A' else 'M123 R0',color=color)
        axs[1].bar(x+offset,[value_or_nan(m1['raw_detection'][k][arm]['groups']['visible_GT']['proven_class_recall']) for k in keys],.36,color=color)
    axs[0].set_ylabel('Known paired answer fact accuracy');axs[0].legend(frameon=False)
    axs[1].set_ylabel('Proven raw visible-object recall')
    for ax in axs:ax.set_xticks(x,labels);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for suffix in ('pdf','svg'):fig.savefig(folder/('quality_coverage.'+suffix),bbox_inches='tight')
    plt.close(fig)


def render(dest,root,workloads,temporal,tasks,strata,m1,m2,m3,verdict,total):
    t1=[];csv_rows=[]
    for wid,g in workloads.items():
        q=g['quality_pairs'];ci=g['bootstrap'].get('speedup_CI95')
        t1.append([g['name'],fmt(g['A']['E2E_ms']['mean']),fmt(g['C']['E2E_ms']['mean']),fmt(g['speedup']),
            fmt(100*g['latency_reduction'])+'%',fmt(q['A_accuracy'])+' → '+fmt(q['C_accuracy']),
            str(q['scored'])+'/'+str(q['all_requests']),str(q['new_severe'])])
        csv_rows.append(dict(workload=g['name'],Native_E2E_ms=g['A']['E2E_ms']['mean'],M123_E2E_ms=g['C']['E2E_ms']['mean'],
            speedup=g['speedup'],latency_reduction=g['latency_reduction'],speedup_CI95_low=ci[0] if ci else None,
            speedup_CI95_high=ci[1] if ci else None,Native_accuracy=q['A_accuracy'],M123_accuracy=q['C_accuracy'],
            scored=q['scored'],all_requests=q['all_requests'],new_severe=q['new_severe']))
    table1=md_table(['Workload','Native E2E ms','M123 E2E ms','Speedup A/C','Latency Reduction','Answer fact accuracy A → C','Scored / All','New severe'],t1)
    t2=[]
    for phase,g in temporal.items():
        t2.append([phase,fmt(g['A']['E2E_ms']['mean']),fmt(g['C']['E2E_ms']['mean']),fmt(g['speedup']),
            fmt(g['C']['M1_ms']['mean']),fmt(g['C']['visual_CUDA_ms']['mean']),fmt(g['C']['M3_ms']['mean'])])
    table2=md_table(['State','Native E2E','M123 E2E','Speedup','M1 host ms','M2 CUDA ms','M3 host ms'],t2)
    t3=[]
    for task in ('category','existence','position'):
        g=tasks[task];known=[e for e in rows(dest/'per_question_metrics.jsonl') if e['task']==task and e['arm']=='C' and e['target']['known']]
        recalled=mean([e['eligible_target_recalled'] for e in known])
        t3.append([task,fmt(g['A_quality']['scored_answer_accuracy']),fmt(g['C_quality']['scored_answer_accuracy']),
            fmt(recalled),fmt(g['C_quality']['FAST_coverage']),str(g['A_quality']['absolute_severe'])+' / '+str(g['C_quality']['absolute_severe']),
            fmt(g['service_latency_ms']['A']['mean'])+' / '+fmt(g['service_latency_ms']['C']['mean'])])
    table3=md_table(['Task','Native Accuracy','M123 Accuracy','C Eligible Target Recall','FAST Coverage','Severe A/C','Question service ms A/C'],t3)
    table4=md_table(['Module','Native mean ms/frame','R0 mean ms/frame','Scope'],[
        ['M1 host','0',fmt(total['C']['M1_ms']['mean']),'Additional state maintenance incl adapters'],
        ['Vision CUDA',fmt(total['A']['visual_CUDA_ms']['mean']),fmt(total['C']['visual_CUDA_ms']['mean']),'Native full backbone vs sole incremental executor'],
        ['M3 host',fmt(total['A']['M3_ms']['mean']),fmt(total['C']['M3_ms']['mean']),'Native generate sum vs complete M3 runtime'],
        ['Prefill CUDA',fmt(total['A']['prefill_CUDA_ms_per_frame']['mean']),fmt(total['C']['prefill_CUDA_ms_per_frame']['mean']),'Actual forwards, averaged over all visual frames'],
        ['Decode CUDA',fmt(total['A']['decode_CUDA_ms_per_frame']['mean']),fmt(total['C']['decode_CUDA_ms_per_frame']['mean']),'Actual forwards, averaged over all visual frames']])
    t5=[]
    for dim,groups in strata.items():
        for value,g in groups.items():
            t5.append([dim,value,str(g['A']['frames']),str(g['A']['logs']),fmt(g['speedup']),
                fmt(g['A_quality']['scored_answer_accuracy'])+' / '+fmt(g['C_quality']['scored_answer_accuracy']),
                fmt(g['target_coverage']['C']['eligible_target_recall']),
                fmt(g['target_coverage']['A']['raw_visible_object_recall'])+' / '+fmt(g['target_coverage']['C']['raw_visible_object_recall']),
                str(g['quality_pairs']['new_severe'])])
    table5=md_table(['Dimension','Condition','Frames/arm','Logs','Speedup','Accuracy A/C','C Eligible Recall','Object Recall A/C','New severe'],t5)
    t6=[]
    for wid,g in workloads.items():
        cov=m1['target_coverage'][wid];det=m1['raw_detection'][wid]
        t6.append([g['name'],fmt(g['speedup']),fmt(100*g['latency_reduction'])+'%',
            fmt(g['quality_pairs']['loss_A_minus_C']),fmt(det['A']['groups']['visible_GT']['proven_class_recall'])+' / '+fmt(det['C']['groups']['visible_GT']['proven_class_recall']),
            fmt(cov['C']['eligible_recall']),fmt(m2['feature_error']['by_workload'][wid]['relative_l2']['mean']),str(g['quality_pairs']['new_severe'])])
    table6=md_table(['Workload','A/C','Reduction','Answer loss A-C','Raw visible Recall A/C','C Eligible Recall','Feature RelL2','New severe'],t6)
    target_rows=[]
    for wid,g in workloads.items():
        cov=m1['target_coverage'][wid];c=cov['C'];a=cov['A']
        target_rows.append([g['name'],str(c['known_targets'])+'/'+str(c['requests']),
            fmt(a['raw_recall']),fmt(c['raw_recall']),fmt(c['committed_recall']),fmt(c['eligible_recall']),
            str(c.get('eligible_PREDICTION_MATCHING_UNKNOWN',0)),str(c['UNKNOWN_targets'])])
    target_table=md_table(['Workload','Known identity targets / All','Native raw target recall','C raw target recall',
        'C committed recall','C eligible recall','C eligible matching UNKNOWN','GT identity UNKNOWN'],target_rows)
    (dest/'target_coverage_comparison.md').write_text(target_table+'\n'+m1['causal_claim']+'\n\n'+m1['recall_uncertainty']+'\n')
    intro=(f'# Frozen R0 versus Original Native\n\n{verdict["logs"]} exposed validation Logs, {verdict["scenes"]} complete Scenes; '
        f'{verdict["actual_new_full_model_calls"]} new full-model calls. No historical inference reused. '
        f'Frozen configuration SHA256 `{verdict["frozen_config_sha256"]}`.\n\n'
        'R0 M1=.3, M2 budget=3, unchanged complete C-basic/World-Guided Decode. A has M1/M2/M3 disabled and invokes original full vision/native generation. '
        'Base/source audit, full configuration, effective runtime configs, unchanged weights and identical paired image/prompt/RNG receipts are retained. '
        'Production remains off. The historical residual-severe verdict is retained.\n\n'
        'Primary E2E includes input loading/preprocessing/H2D/question encoding plus synchronized complete model call and answer export. '
        'Offline hashing, feature transfer/error computation, GT scoring and JSON receipt serialization are outside E2E. '
        'Idle frames are included. FPS/QPS use the same summed frame E2E; no queue-delay SLA is claimed. '
        'The model-only synchronized latency is separately retained for historical comparisons. '
        'Question-service timers exclude shared vision and must not be substituted for frame E2E. Nested CUDA and host timers are not additive.\n\n'
        'Statistics: paired Log-cluster bootstrap (2000 resamples), all selected frames; 18 Log clusters are sampling units, not independent TEST certification. '
        'Only one timing repeat per arm; no warmup frames discarded. Cold starts remain in all summaries.\n\n')
    witness=read(root/('pilot' if verdict['status']=='PILOT_VALIDATED' else 'main')/'kernel_witness.json')
    environment=read(root/'environment_versions.json')
    intro+=(f'Environment: {witness["GPU"]}, Torch {witness["torch"]}, CUDA {witness["CUDA"]}, {environment["python"]}. '
        f'Existing environment reused without installs or rebuilding. Base Git commit `{read(root/"frozen_main_config.json")["source_commit"]}`. '
        'Complete packages, driver/hardware snapshot and sampled GPU telemetry are retained in environment_versions.json, hardware_environment.json and gpu_telemetry.jsonl. '
        'Read-only observer overhead remains inside measured model time; no heavy profiler or theoretical cost subtraction is used.\n\n')
    limitation=('\nAccuracy is conservative current-visible GT task-fact accuracy on scored requests; unsupported natural questions use the existing limited exact-current-QA fact recall. '
        'The report retains UNKNOWN and full denominators; it does not certify exhaustive free-form answer accuracy, negative existence, unmatched FP, or physical nearest identity from text. '
        'Severe direction mistakes are reported separately and never offset by mean accuracy gains. '
        'All validation Logs have historical exposure; no independent generalization or strict safety claim. '
        'A/C cannot isolate a causal M1/M2/M3 quality contribution.\n')
    all_tables=intro+'## Table 1: Workload performance\n\n'+table1+'\n## Table 2: Temporal state\n\n'+table2+'\n## Table 3: Question tasks\n\n'+table3+'\n## Table 4: Module costs\n\n'+table4+'\n## Table 5: Conditions\n\n'+table5+'\n## Table 6: Quality and latency\n\n'+table6+limitation
    (dest/'main_results_tables.md').write_text(all_tables)
    (dest/'workload_comparison.md').write_text(intro+table1+limitation)
    (dest/'condition_comparison.md').write_text(intro+table2+'\n'+table5+limitation)
    (dest/'module_efficiency_report.md').write_text(intro+table4+'\n'+m3['counting']+'\n\n'+m3['prefix_note']+limitation)
    identity=m3['FAST_physical_target_identity']
    (dest/'quality_evaluation_report.md').write_text(intro+table3+'\n'+table6+'\n'+target_table+'\n'+
        f'FAST physical target identity: {identity["correct"]}/{identity["scored"]} confirmed correct, '
        f'{identity["UNKNOWN"]}/{identity["requests"]} UNKNOWN. '
        'This identity metric remains separate from answer-level common task facts. Ambiguous prediction matches are retained as PREDICTION_MATCHING_UNKNOWN in per-question coverage status; target recalls are conservative proven-recall bounds. '
        'Native has no committed M1 or eligible M3 targets, so those Native cells are N/A.\n\n'+
        f'Severe observations: Native {verdict["A_absolute_severe"]}, C {verdict["C_absolute_severe"]}, new C relative to Native {verdict["new_severe_relative_to_Native"]}. '
        f'Unique severe frame/question cases: Native {verdict["A_unique_severe_frame_questions"]}, C {verdict["C_unique_severe_frame_questions"]}, new C {verdict["new_unique_severe_frame_questions"]}. '
        'Workload repeats are disclosed and never counted as independent physical cases. See position_error_breakdown.json and severe_error_cases.json for front/rear, lateral, diagonal, class and uniquely matched nearest-target errors.\n'+limitation)
    conclusions='\n'.join(f'- {g["name"]}: A/C {g["speedup"]:.4f}, E2E reduction {100*g["latency_reduction"]:.2f}%, known answer loss A-C {fmt(g["quality_pairs"]["loss_A_minus_C"])}, new severe {g["quality_pairs"]["new_severe"]}.' for g in workloads.values())
    (dest/'e2e_performance_report.md').write_text(intro+table1+'\n'+table2+'\n'+conclusions+'\n'+limitation+
        '\nLow-frequency state-maintenance costs must be assessed from each full-frame workload row. '
        'A positive module saving is not proof of positive E2E. The highest speedup workload is '+max(workloads,key=lambda w:workloads[w]['speedup'])+'.\n')
    discussion=[]
    for wid,g in workloads.items():
        discussion.append(dict(workload=g['name'],complete_E2E_saving_ms=g['A']['E2E_ms']['mean']-g['C']['E2E_ms']['mean'],
            added_M1_host_ms=g['C']['M1_ms']['mean'],M3_host_saving_ms=g['A']['M3_ms']['mean']-g['C']['M3_ms']['mean'],
            visual_CUDA_saving_ms=g['A']['visual_CUDA_ms']['mean']-g['C']['visual_CUDA_ms']['mean'],
            whole_system_positive=g['speedup']>1,
            low_frequency_cost_interpretation='Use complete E2E directly; host and CUDA differences are descriptive components, not an additive causal decomposition.'))
    write(dest/'state_maintenance_tradeoff.json',discussion)
    (dest/'research_conclusions.md').write_text(intro+'## Workload conclusions\n\n'+conclusions+'\n\n'+
        f'Largest observed speedup: {workloads[max(workloads,key=lambda w:workloads[w]["speedup"])]["name"]}. '
        f'Point-estimate no-E2E-benefit workloads: {verdict["point_estimate_no_E2E_benefit"]}. '
        f'Paired Log-bootstrap intervals entirely above 1: {verdict["paired_speedup_CI_above_one"]}.\n\n'+
        '## State maintenance and workload suitability\n\n'+table4+'\n'+table2+'\n'+
        'Full-system savings and added M1 cost are recorded separately in state_maintenance_tradeoff.json. '
        'Any low-frequency row at or below speedup 1 has no measured whole-system benefit, regardless of visual or language module savings. '
        'Cold, transition and steady labels use the same pre-frozen GT rule in both arms.\n\n'+
        '## Quality, targets and scene conditions\n\n'+table3+'\n'+table6+'\n'+target_table+'\n'+
        f'New severe observations relative to Native: {verdict["new_severe_relative_to_Native"]}; '
        f'absolute Native/M123 severe observations: {verdict["A_absolute_severe"]}/{verdict["C_absolute_severe"]}. '
        'See severe_error_cases.json for unique physical frame/question counts and complete Native/C answers. '
        'Answer, raw detector, committed state and M3-eligible target denominators are distinct. '
        'Condition-specific speed, eligible target coverage and object recall are reported in condition_comparison.md; small groups do not certify robustness. '
        'Observed A/C coverage differences cannot establish an isolated causal M1 benefit.\n\n'+
        '## Research configuration decision\n\n'+verdict['paper_main_configuration']+'. '+limitation)
    with (dest/'main_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    write(dest/'main_results.json',dict(workloads=workloads,temporal=temporal,tasks=tasks,conditions=strata))
    flat_metrics=[]
    for wid,g in workloads.items():
        for arm in ('A','C'):
            p=g[arm];q=g[arm+'_quality']
            flat_metrics.append(dict(workload=g['name'],arm=arm,frames=p['frames'],logs=p['logs'],
                mean_ms=p['E2E_ms']['mean'],p50_ms=p['E2E_ms']['p50'],p95_ms=p['E2E_ms']['p95'],p99_ms=p['E2E_ms']['p99'],
                model_mean_ms=p['model_E2E_ms']['mean'],QA_frame_mean_ms=p['QA_frame_E2E_ms']['mean'],
                idle_frame_mean_ms=p['idle_frame_E2E_ms']['mean'],FPS=p['FPS'],QPS=p['QPS'],peak_GPU_bytes=p['peak_GPU_bytes'],
                answer_accuracy=q['scored_answer_accuracy'],scored=q['scored'],UNKNOWN=q['UNKNOWN'],all_requests=q['requests']))
    with (dest/'workload_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat_metrics[0]));writer.writeheader();writer.writerows(flat_metrics)
    py='taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python'
    (dest/'REPRODUCE.md').write_text(f'''# Reproduce frozen R0 A/C evaluation

Use the existing OmniDrive conda environment and an idle RTX3090. No installs, training, weight changes or tuning. Every command excludes CPUs4/5. GPU stages require host CUDA access (the filesystem sandbox may hide CUDA), acquire /tmp/omnidrive_m123_latency_gpu.lock, and wait without stopping other processes.

Read-only configuration/source/history verification:

```sh
{py} -m tools.worldstream_m123.frozen_main_benchmark verify --root {root}
```

Receipt-only offline reconstruction (writes only generated analysis reports, not frozen inputs, original paired receipts or any historical directory):

```sh
{py} -m tools.worldstream_m123.frozen_main_report --root {root}
```

For a NEW actual GPU repetition, first create a nonexisting output directory with prepare, then run pilot, pilot analysis, and main in order. The new run uses its own config SHA because its absolute policy-file binding changes; all semantic R0 parameters remain the same. Keep runs separate; repeated exposed data is not independent TEST.

```sh
{py} -m tools.worldstream_m123.frozen_main_benchmark prepare --root /tmp/omnidrive-r0-ac-repeat
{py} -m tools.worldstream_m123.frozen_main_benchmark pilot --root /tmp/omnidrive-r0-ac-repeat
{py} -m tools.worldstream_m123.frozen_main_report --root /tmp/omnidrive-r0-ac-repeat --pilot
{py} -m tools.worldstream_m123.frozen_main_benchmark main --root /tmp/omnidrive-r0-ac-repeat
{py} -m tools.worldstream_m123.frozen_main_report --root /tmp/omnidrive-r0-ac-repeat
```

Scene selection is fixed before inference: first complete readable scene per available Log, official first to last sample, all five unchanged schedules. A/C use the same per-request initial RNG state; source/runtime/config/weight hashes and actual forward counts accompany each receipt. A/C arm order alternates by scene/workload. Both task heads and all M1/M2/M3 state reset before each arm. Primary statistics include idle frames and cold starts.

R0 is the only FINAL_MAIN_CONFIGURATION. Original authority SHA and all full dataclass parameters are retained. Explicit benchmark activation preserves the complete M123 serving route; production defaults remain off. Historical severe-risk and exposure restrictions remain. Raw Native source is archived in baseline_sources; serving modules are unchanged. See experiment_protocol.json for timing, bootstrap, scoring and scope definitions.

After execution, paired_gpu_receipts.jsonl may be a small chronological index into receipt_chunks/*.jsonl.gz. Each index row pins the original complete A/C record bytes with SHA256; receipt_archive.json pins each compressed chunk, original chunk bytes, and the concatenated original full JSONL SHA256. The report reader resolves either inline or indexed receipts automatically. Lossless compression affects only this new run; historical directories remain byte-identical. Raw arm-execution order is retained separately in main/raw_receipt_chunks and the same manifest. This prevents redundant multi-gigabyte working copies and oversized Git blobs without dropping raw requests.
''')
    figures(dest,workloads,m1)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--pilot',action='store_true')
    p.add_argument('--verify',action='store_true')
    a=p.parse_args()
    if a.verify:verify_receipts(a.root)
    else:report(a.root,a.pilot)


if __name__=='__main__':main()
