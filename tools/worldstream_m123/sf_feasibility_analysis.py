"""Observed window/event distributions, legal enrichment, and gated estimates."""
from collections import Counter,defaultdict
import json
from pathlib import Path
from tools.worldstream_m123.benchmark.common import write,sha256
from tools.worldstream_m123.sf_feasibility import rows,PRIOR


def stats(rr):
    lives=Counter(r['anchor']['anchor_id'] for r in rr if r['anchor']['stable_state_valid'])
    per_scene=Counter(r['scene_id'] for r in rr)
    adjacent={r['frame_id'] for old,r in zip(rr,rr[1:]) if old['scene_id']==r['scene_id'] and r['ordinal']==old['ordinal']+1}
    hits=[r for r in rr if r['anchor_hit'] and r['frame_id'] in adjacent]
    checks=Counter(c['status'] for r in hits for c in r['core_fact_checks_offline'])
    return dict(frames=len(rr),scenes=len(per_scene),logs=len({r['log_id'] for r in rr}),
        valid_frames=sum(lives.values()),anchor_extensions=len(hits),
        transitions=len(adjacent),anchor_hit_rate=len(hits)/len(adjacent) if adjacent else None,
        valid_sequences=len(lives),positive_logs=len({r['log_id'] for r in hits}),
        sustained_sequences=len({r['anchor']['anchor_id'] for r in hits}),
        mean_valid_anchor_lifetime=sum(lives.values())/len(lives) if lives else 0.,lifetimes=list(lives.values()),
        retained_fact_offline_checks=dict(checks),core_unknown_fraction=checks['unknown']/sum(checks.values()) if checks else None,
        first_rejection_counts=dict(Counter(reason for r in rr for reason in r['selection']['rejected'].values())),
        core_update_reasons=dict(Counter(r['anchor']['update_reason'] for r in rr if not r['anchor_hit'])))


def analyze_data(root):
    plan=json.loads((root/'experiment_plan.json').read_text());seq=json.loads((root/'audit_sequence.json').read_text())
    if json.loads((root/'state_audit/status.json').read_text())['status']!='COMPLETED':raise ValueError('Incomplete causal audit')
    rr=list(rows(root/'state_audit/frames.jsonl'));assert [r['frame_id'] for r in rr]==seq['tokens']
    groups=defaultdict(list)
    for r in rr:groups[r['scene_id']].append(r)
    original=set(plan['original_train_scenes']);training=[r for r in rr if r['split']=='train'];validation=[r for r in rr if r['split']=='validation']
    historical=json.loads((PRIOR/'historical_data_audit.json').read_text())
    tables=json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text());log_by_scene={r['token']:r['log_token'] for r in tables}
    write(root/'training_historical_exposure.json',dict(source_index=str(PRIOR/'historical_data_audit.json'),
        source_sha256=sha256(PRIOR/'historical_data_audit.json'),
        logs={seq['scene_logs'][s]:dict(selected_scene=s,role=seq['scene_splits'][s],
            previous_sources=[dict(path=v['path'],sha256=v.get('sha256'),scenes_in_same_log=[x for x in v.get('scenes',[]) if log_by_scene.get(x)==seq['scene_logs'][s]])
                for v in historical['sources'] if any(log_by_scene.get(x)==seq['scene_logs'][s] for x in v.get('scenes',[]))]) for s in groups},
        recent_exposure_index=str(root/'official_train_pool_audit.json'),
        limitation='Available archived historical index plus explicit recent manifests; no selected training log is advertised as globally unseen.'))
    old_snapshots={r['frame_id']:r for r in rows(PRIOR/'adaptation_capture/stable_world_snapshots.jsonl') if r['scene_id'] in original}
    old_classes={}
    for r in rows(PRIOR/'adaptation_capture/m1_state.jsonl'):
        state=r['committed_state'];fid=state['frame_id']
        if fid in old_snapshots:old_classes[fid]={a['anchor_id']:a['semantic_class'] for a in state['anchors']}
    old_rejections=Counter();old_car=Counter()
    for fid,r in old_snapshots.items():
        p=json.loads(r['provenance_json'])
        old_rejections.update(p['rejected'].values())
        old_car.update(reason for aid,reason in p['rejected'].items() if old_classes[fid].get(aid)=='car')
    write(root/'old_48_diagnosis.json',dict(frames=len(old_snapshots),first_rejections_all=dict(old_rejections),first_rejections_car=dict(old_car),
        no_qualified_facts_frames=sum(not r['facts'] for r in old_snapshots.values()),
        limitation='First-failure counts from preserved true old snapshots, not disjoint all-filter causal effects. Physical occlusion versus detector miss versus ID fragmentation cannot be resolved from absent observations alone.'))
    windows={str(n):{name:stats([r for s,gg in groups.items() if predicate(s,gg[0]) for r in gg[:n]])
        for name,predicate in [('original_train',lambda s,r:s in original),('all_train',lambda s,r:r['split']=='train'),
            ('expanded_train',lambda s,r:r['split']=='train' and s not in original),('validation',lambda s,r:r['split']=='validation')]}
        for n in plan['windows']}
    detail=Counter();events=Counter();previous={}
    for r in rr:
        cars=[d for d in r['filter_details'] if d['semantic_class']=='car']
        detail['frames_no_current_car_candidates']+=not bool(cars)
        rule=json.loads((PRIOR/'semantic_qualification.json').read_text())['rules']['car']
        for d in cars:
            detail['car_observations']+=1
            detail['score_below_threshold']+=d['score'] is None or d['score']<rule['min_score']
            detail['residual_above_threshold']+=d['residual_m'] is not None and d['residual_m']>rule['max_residual_m']
            detail['no_previous_frame_association']+=not d['source_is_previous_frame']
            detail['no_current_observation_support']+=not d['current_observation_support']
        old=previous.get(r['scene_id']);valid=r['anchor']['stable_state_valid']
        if valid and not r['anchor_hit']:events['core_created_or_replaced']+=1
        if r['anchor_hit']:events['core_continued']+=1
        if not valid:events['no_core_frames']+=1
        if old and old['anchor']['stable_state_valid'] and not valid:events['core_lost']+=1
        if old and old['anchor']['stable_state_valid'] and valid and not r['anchor_hit']:events['core_replaced']+=1
        if r['anchor_hit'] and r['new_current_ids']:events['new_object_but_core_kept']+=1
        if old:
            previous_core={f[0] for f in old['anchor']['facts']}
            current_ids={d['anchor_id'] for d in r['filter_details']}
            detail['previous_core_id_not_currently_observed']+=bool(previous_core-current_ids)
        previous[r['scene_id']]=r
    # Positive-event enrichment retains real contiguous +/-2 frame windows.
    # No duplicated frame or modified snapshot; validation keeps ALL 32 frames.
    chosen=[];selection={}
    for scene,gg in groups.items():
        indices=set(range(len(gg))) if gg[0]['split']=='validation' else {0,1,len(gg)-1}
        if gg[0]['split']=='train':
            for i,r in enumerate(gg):
                if r['anchor']['stable_state_valid']:indices.update(range(max(0,i-2),min(len(gg),i+3)))
            indices.update(range(0,len(gg),8))
        selection[scene]=sorted(indices)
        chosen.extend(gg[i] for i in sorted(indices))
    enriched=[r for r in chosen if r['split']=='train'];train_stats=stats(enriched);val_stats=stats(validation)
    lim=plan['enrichment']
    data_go=(train_stats['positive_logs']>=lim['min_training_positive_logs'] and train_stats['sustained_sequences']>=lim['min_training_core_sequences']
        and train_stats['valid_frames']>=lim['min_training_valid_frames'] and train_stats['anchor_extensions']>=lim['min_training_extensions']
        and val_stats['positive_logs']>=lim['min_validation_positive_logs'] and val_stats['anchor_extensions']>=lim['min_validation_extensions']
        and train_stats['retained_fact_offline_checks'].get('correct',0)>=lim['min_training_extensions']
        and not train_stats['retained_fact_offline_checks'].get('wrong',0) and not val_stats['retained_fact_offline_checks'].get('wrong',0)
        and (train_stats['core_unknown_fraction'] is not None and train_stats['core_unknown_fraction']<=.05)
        and (val_stats['core_unknown_fraction'] is not None and val_stats['core_unknown_fraction']<=.05))
    write(root/'core_event_distribution.json',dict(windows=windows,per_scene={s:stats(gg) for s,gg in groups.items()},
        event_counts=dict(events),independent_filter_counts=dict(detail),raw_train=stats(training),enriched_train=train_stats,validation=val_stats,
        enriched_frame_indices=selection,data_go=data_go,limits=lim,
        natural_coverage_warning='Additional TRAIN logs were selected using historical positive evidence; even raw32 distribution is a candidate-pool audit, not deployment coverage. Enriched rate is never deployment rate.',
        unknown_occlusion='A missing current ID cannot distinguish physical occlusion, detector miss, or association-ID change without an independent visibility/association oracle. No such physical cause is invented.'))
    with (root/'core_event_timeline.jsonl').open('w') as f:
        for r in rr:
            f.write(json.dumps({k:r[k] for k in ('ordinal','scene_id','frame_id','log_id','split','timestamp','anchor_hit','anchor','core_fact_checks_offline')})+'\n')
    samples=[]
    for r in chosen:
        t=r['frame_id'];path=root/'state_audit/m3_inputs'/(t+'.pt');qsrc=Path('data/nuscenes/conv/train')/(t+'.json')
        pairs=json.loads(qsrc.read_text())[:3]
        samples.append(dict(scene_id=r['scene_id'],frame_id=t,capture=str(path),capture_sha256=sha256(path),
            stable_snapshot=r['snapshot'],question_source=str(qsrc),question_source_sha256=sha256(qsrc),
            pairs=[dict(question_index=i,question=p['question'],answer=p['answer']) for i,p in enumerate(pairs)]))
    target=json.loads((PRIOR/'target_sequence.json').read_text())
    manifest=dict(scene_splits=dict(seq['scene_splits'],**target['scene_splits']),scene_logs=dict(seq['scene_logs'],**target['scene_logs']),
        nuscenes_scene_table='data/nuscenes/v1.0-trainval/scene.json',semantic_qualification=plan['qualification_path'],
        base_weights_sha256='6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d',samples=samples,
        source='Actual sequential M1/M2 capture, full32 causal context. Train event windows sampled offline, validation full32; no future information enters snapshots.',
        training_admission=data_go,raw_and_enriched_distribution=str(root/'core_event_distribution.json'))
    write(root/'training_dataset_manifest.json',manifest)
    # Existing trainer takes the same schema; no second training implementation.
    write(root/'training_manifest.json',manifest)
    print(json.dumps(dict(windows=windows,data_go=data_go,enriched=train_stats,validation=val_stats),indent=2))


def estimate(root):
    import numpy as np
    micro=json.loads((root/'controlled_micro/summary.json').read_text())
    comp=json.loads((root/'controlled_micro/composer_cost.json').read_text())
    timeline=list(rows(PRIOR/'stable_anchor_timeline.jsonl'))
    plan=json.loads((PRIOR/'experiment_plan.json').read_text())
    independent=[r for r in timeline if r['split']=='qualification_validation' and plan['scene_datasets'][r['scene_id']]=='val']
    # One-entry state replay: first question on each frame refreshes the entry;
    # later same-frame matches do not replace it. Layout changes reset it.
    workloads={}
    for questions in (1,3):
        cache=None;count=Counter();lives=[]
        for r in independent:
            mode='stable_core_visual' if r['anchor']['stable_state_valid'] else 'native_full_visual'
            scope=(r['scene_id'],mode)
            if cache and cache['scope']!=scope:cache=None;count['scope_or_layout_resets']+=1
            for qi in range(questions):
                count['requests']+=1;count['valid_core_requests']+=r['anchor']['stable_state_valid']
                if cache:
                    if cache['frame']==r['frame_id']:count['same_frame_visual_candidate_requests']+=1
                    else:
                        count['crossframe_text_candidate_requests']+=1
                        if r['anchor_hit'] and cache['anchor']==r['anchor']['anchor_id'] and mode=='stable_core_visual':
                            count['anchor_hit_requests']+=1;count['crossframe_stable64_exact_candidates']+=1
                if not cache or cache['frame']!=r['frame_id']:
                    cache=dict(scope=scope,frame=r['frame_id'],anchor=r['anchor']['anchor_id'])
        count['current_SF_scheduler_admissions']=0
        count['formal_executor_reuse_requests']=0
        p=count['crossframe_stable64_exact_candidates']/count['requests']
        q=count['valid_core_requests']/count['requests']
        full= micro['by_kind']['full'];plus=micro['by_kind']['text_plus64'];text=micro['by_kind']['text']
        cuda_save=micro['summary']['full_minus_text64_model_cuda_ms']['mean']
        marginal=micro['summary']['text_minus_text64_model_cuda_ms']['mean']
        executor_save=micro['summary']['full_minus_text64_executor_wall_ms']['mean']
        warm=comp['summary']['warm']['wall_ms']['mean']
        planner=plus['planner_wall_ms']['mean']
        native=micro['by_kind']['native']
        native_wall=native['executor_wall_ms']['mean']
        full_cache_host=full['executor_wall_ms']['mean']-native_wall
        # Independent-domain CPU state replay includes no model/GT fitting.
        core_cost=sum(r['selection_wall_ms']+r['anchor_wall_ms'] for r in independent)/len(independent)/questions
        p_same=count['same_frame_visual_candidate_requests']/count['requests']
        p_text=(count['crossframe_text_candidate_requests']-count['crossframe_stable64_exact_candidates'])/count['requests']
        p_full=1-p-p_same-p_text
        expected_saved=p*cuda_save
        crossframe_model_saved=expected_saved+p_text*(full['model_prefill_cuda_ms']['mean']-text['model_prefill_cuda_ms']['mean'])
        sameframe_model_saved=p_same*(full['model_prefill_cuda_ms']['mean']-micro['by_kind']['same_frame_visual']['model_prefill_cuda_ms']['mean'])
        composer_per_request=q*warm/questions  # runtime composes once per frame, outside its question loop
        extra=composer_per_request+planner+core_cost
        costs={kind:micro['by_kind'][kind]['executor_wall_ms']['mean'] for kind in ('full','text','text_plus64','same_frame_visual')}
        expected_executor=p_full*costs['full']+p_text*costs['text']+p*costs['text_plus64']+p_same*costs['same_frame_visual']
        net=native_wall-expected_executor-extra
        # Incremental world-prefix benefit beyond text/same-frame reuse.
        world_marginal=p*(costs['text']-costs['text_plus64'])-composer_per_request-core_cost
        measured=list(rows(root/'controlled_micro/records.jsonl'))
        repeats=sorted({r['repeat'] for r in measured})
        def series(kind,key):
            return np.asarray([np.mean([r[key] for r in measured if r['repeat']==repeat and r['kind']==kind]) for repeat in repeats])
        generator=np.random.default_rng(20261028)
        draws=generator.integers(0,len(repeats),size=(10000,len(repeats)))
        composer_values=np.asarray([r['warm']['wall_ms'] for r in comp['records']])
        composer_draws=generator.integers(0,len(composer_values),size=draws.shape)
        cboot=composer_values[composer_draws].mean(1)
        s={kind:series(kind,'executor_wall_ms')[draws].mean(1) for kind in ('native','full','text','text_plus64','same_frame_visual')}
        pboot=series('text_plus64','planner_wall_ms')[draws].mean(1)
        total_samples=s['native']-(p_full*s['full']+p_text*s['text']+p*s['text_plus64']+p_same*s['same_frame_visual'])-q*cboot/questions-pboot-core_cost
        marginal_samples=p*(s['text']-s['text_plus64'])-q*cboot/questions-core_cost
        workloads[str(questions)]=dict(counters=dict(count),observed_eligible_probability=p,valid_core_probability=q,
            optimistic_prefill_saved_ms_per_request=expected_saved,
            optimistic_all_crossframe_model_prefill_saved_ms_per_request=crossframe_model_saved,
            optimistic_sameframe_model_prefill_saved_ms_per_request=sameframe_model_saved,
            optimistic_total_model_prefill_saved_ms_per_request=crossframe_model_saved+sameframe_model_saved,
            optimistic_stable64_marginal_saved_ms_per_request=p*marginal,
            optimistic_executor_wall_saved_ms_per_request=p*executor_save,
            estimated_extra_ms_per_request=extra,estimated_net_ms_per_request=net,
            incremental_world_prefix_net_ms_per_request=world_marginal,
            timing_bootstrap_ci95=dict(total_net_ms=np.quantile(total_samples,[.025,.975]).tolist(),
                marginal_world_net_ms=np.quantile(marginal_samples,[.025,.975]).tolist(),
                scope='Nine micro repetitions, questions grouped within repetition; Composer resampled separately. Measurement uncertainty only, not scene/population E2E CI.'),
            components=dict(composer_wall=composer_per_request,planner_wall=planner,full_executor_minus_native_wall=full_cache_host,core_anchor_CPU=core_cost,
                native_generation_wall=native_wall,expected_executor_wall=expected_executor,probabilities=dict(full=p_full,text=p_text,text64=p,same_frame=p_same)),
            optimistic_unmeasured_costs_assumed_zero=['Current M1 candidate extraction/protocol hashing','M3 execution signature and deployment binding',
                'SF expand_input assembly','Additional allocator/serialization overhead beyond the timed executor'],
            conservative_scope='Warm Composer cost used even for new anchors; unmeasured extra costs are zero. This favors the proposed method; predicted net is not measured end-to-end net.',
            limitation='Optimistic one-entry replay with calibrated exact-match admission assumed only for eligible matches. Same-frame gains reported separately; they are not crossframe world-prefix benefit. Current SF scheduler has no trained-layout policy, so real admissions0. Event replay is not matcher or service execution.',
            independent_frames=len(independent),independent_scenes=len({r['scene_id'] for r in independent}))
    assert abs(workloads['3']['components']['composer_wall']*3-workloads['1']['components']['composer_wall'])<1e-10
    data=json.loads((root/'core_event_distribution.json').read_text())
    independent_scene_ids={r['scene_id'] for r in independent}
    lengths=[r['input_tokens'] for r in rows(PRIOR/'qualification_val_capture/per_question_metrics.jsonl') if r['scene_id'] in independent_scene_ids]
    controls=[r['input_tokens'] for r in rows(root/'controlled_micro/records.jsonl')]
    length_scope=dict(independent_original_input_tokens=dict(minimum=min(lengths),maximum=max(lengths),mean=float(np.mean(lengths)),requests=len(lengths)),
        controlled_input_tokens=sorted(set(controls)),
        limitation='Three original questions from one development frame pair; weighting assumes their measured costs transfer to the independent input-length distribution. Not a calibrated global service cost model.')
    # Retain sensitivity to the earlier timing-complete run whose subsequent
    # diagnostic JSON serialization failed. Never hide or cherry-pick it.
    previous_path=root/'controlled_micro_failed02/records.jsonl'
    previous_sensitivity={}
    if previous_path.exists():
        prior_rows=list(rows(previous_path))
        for label,w in workloads.items():
            c={kind:float(np.mean([r['executor_wall_ms'] for r in prior_rows if r['kind']==kind])) for kind in ('native','full','text','text_plus64','same_frame_visual')}
            pp=w['components']['probabilities'];overhead=w['estimated_extra_ms_per_request']
            previous_sensitivity[label]=dict(control_rows=len(prior_rows),
                total_net_ms=c['native']-sum(pp[k]*c[{'text64':'text_plus64','same_frame':'same_frame_visual'}.get(k,k)] for k in pp)-overhead,
                marginal_world_net_ms=pp['text64']*(c['text']-c['text_plus64'])-w['components']['composer_wall']-w['components']['core_anchor_CPU'],
                limitation='Earlier complete timing rows retained. Final measured Composer/state/Planner overhead reused only for sensitivity; not an independent complete trial.')
    speed_go=(workloads['1']['timing_bootstrap_ci95']['total_net_ms'][0]>0
        and workloads['1']['timing_bootstrap_ci95']['marginal_world_net_ms'][0]>0)
    decision=dict(data_go=data['data_go'],speed_go=speed_go,training_go=data['data_go'] and speed_go,
        current_car_semantic_novelty='All valid one-fact states encode the same object/car bytes; learned prompt utility not established.',
        reason='Positive optimistic total AND marginal world-prefix net are not supported under timing uncertainty' if not speed_go else 'Positive bounded hardware opportunity; trained representation quality still required',
        formal_deployment_go=False,formal_SF_service_executed=False)
    write(root/'kv_speedup_upper_bound.json',dict(controlled_micro=str(root/'controlled_micro/summary.json'),workloads=workloads,
        input_length_scope=length_scope,
        earlier_timing_run_sensitivity=previous_sensitivity,
        inference='Model-estimated opportunity on old independent qualification VAL, not trained service speedup or confidence interval.',
        optimistic_gate_rule='Continue only if one-question optimistic total AND incremental world-prefix timing-bootstrap lower bounds are positive; multi-question same-frame savings cannot rescue a crossframe world-prefix claim.'))
    write(root/'feasibility_decision.json',decision)
    print(json.dumps(decision,indent=2))


def main():
    import argparse,os
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--stage',choices=['data','estimate'],required=True);a=p.parse_args()
    if {4,5}&set(os.sched_getaffinity(0)):p.error('Use taskset -c 0-3,6-31')
    (analyze_data if a.stage=='data' else estimate)(Path(a.root))


if __name__=='__main__':main()
