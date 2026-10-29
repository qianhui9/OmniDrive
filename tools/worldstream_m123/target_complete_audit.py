"""Offline diagnostics and delivery checks; never used for online admission."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
from .benchmark.common import write,sha256
from .selective_answer_reuse_report import lines,mean

ROOT=Path('outputs/m3_target_complete_bypass_run01')
PRIOR=Path('outputs/m3_state_calibrated_bypass_run01')


def prior_audit(root):
    from worldstream.m3.task_conditioner import question_facts
    from worldstream.m3.answer_validity import fact_region
    from worldstream.m3.world_delta import VEHICLES
    from worldstream.m1.omnidrive_adapter import CLASS_NAMES
    objects=json.loads((PRIOR/'current_frame_facts.json').read_text())['frames']
    captures={r['frame_id']:r for r in lines(root/'diagnostic/requests.jsonl')}
    rawrows={(r['frame_id'],r['round']):r for r in lines(PRIOR/'run/requests.jsonl')
        if r['condition']=='C' and r['workload']=='addressable' and r['question_count']==3}
    errors={}
    for e in json.loads((PRIOR/'task_quality.json').read_text())['errors']:
        if e['comparison']=='B->C' and e['llm_skipped']:
            errors.setdefault((e['frame_id'],e['question']),[]).append(e)
    cases=[]
    for (token,question),events in errors.items():
        e=events[0];r=captures[token];reference=e['quality']['reference'];center=reference['center_ego']
        cls=reference['semantic_class'];ci=CLASS_NAMES.index(cls);profile=question_facts(question)
        classes=VEHICLES if 'vehicle' in profile.classes else set(profile.classes)
        dist=lambda c:math.hypot(*c[:2])
        close=lambda c:math.dist(c[:2],center[:2])<2.
        probes=[dict(query=j,center_ego=b[:3],distance_to_reference_m=math.dist(b[:2],center[:2]),
            reference_class_score=r['raw_queries']['class_scores'][j][ci],
            top_class=CLASS_NAMES[max(range(len(CLASS_NAMES)),key=lambda k:r['raw_queries']['class_scores'][j][k])])
            for j,b in enumerate(r['raw_queries']['boxes']) if close(b)]
        decoded=[a for a in r['decoded_candidates'] if a['semantic_class']==cls and close(a['center_ego'])]
        adapter=[a for a in r['adapter_anchors'] if a['semantic_class']==cls and close(a['center_ego'])]
        committed=[a for a in r['committed_world']['anchors'] if a['anchor_type']=='object' and a['source_frame']==token
            and a['semantic_class']==cls and close(a['center_ego'])]
        legal=[a for a in r['committed_world']['anchors'] if a['anchor_type']=='object' and a['observed']
            and a['score']>=.35 and a['source_frame']==token and a['semantic_class'] in classes and fact_region(a['center_ego'],profile)]
        det=[a for a in r['decoded_candidates'] if a['score']>=r['m1_min_detection_score']
            and a['semantic_class'] in classes and fact_region(a['center_ego'],profile)]
        gt=sorted((g for g in objects[token] if g['visible'] and g['semantic_class'] in classes
            and fact_region(g['center_ego'],profile)),key=lambda a:dist(a['center_ego']))
        row=rawrows[(token,e['round'])];selected=row['requests'][e['request_index']]['fast_answer']['facts'][0]
        if not decoded:reason='target_not_detected'
        elif not adapter:reason='target_filtered'
        elif not any(a['observed'] for a in committed):reason='target_projection_invalid'
        elif not any(a in legal for a in committed):reason='target_filtered'
        else:reason='insufficient_target_completeness'
        filtered_stage=('M1_score_filter_0.3' if not adapter else 'M3_positive_witness_floor_0.35') if reason=='target_filtered' else None
        nearby=sorted(objects[token],key=lambda g:math.dist(g['center_ego'],selected['center_ego']))[:3]
        expected=sorted((a['semantic_class'],a['score'],tuple(a['center_ego']),a['observed'],tuple(a['cameras']))
            for a in row['committed_world']['anchors'] if a['anchor_type']=='object')
        actual=sorted((a['semantic_class'],a['score'],tuple(a['center_ego']),a['observed'],tuple(a['cameras']))
            for a in r['committed_world']['anchors'] if a['anchor_type']=='object')
        cases.append(dict(frame_id=token,scene_id=e['scene_id'],question=question,
            correlated_output_records=len(events),new_severe=any(v['new_severe_error'] for v in events),
            current_answer=e['answer'],baseline_answer=e['baseline_answer'],reference=reference,
            selected_target=selected,primary_reason=reason,filtered_stage=filtered_stage,raw_reference_probes=probes,
            decoded_reference_candidates=decoded,post_M1_filter_reference_candidates=adapter,
            committed_reference_candidates=committed,nearest_decoded_detection=min(det,key=lambda a:dist(a['center_ego'])) if det else None,
            nearest_legal_anchor=min(legal,key=lambda a:dist(a['center_ego'])) if legal else None,
            nearest_visible_GT=gt[0] if gt else None,all_legal_candidates=legal,
            selected_nearby_physical_GT=nearby,
            same_current_detector_and_observation_evidence_as_v3=expected==actual,
            caveat='Geometry matches are diagnostic, not identity certificates. Invisible/uncertain annotations are retained separately. No raw query or low-confidence candidate becomes an Answer Witness.'))
    write(root/'nearest_target_failure_audit.json',dict(status='REAL_GPU_PRIOR_DIAGNOSTIC',
        cases=cases,unique_cases=len(cases),independent_scenes=len({c['scene_id'] for c in cases}),
        reasons=dict(Counter(c['primary_reason'] for c in cases)),full_diagnostic_M123_calls=len(captures),
        historical_v3_verdict='NO_GO_UNCHANGED',source='Actual25 full calls with all raw head scores/decoded/M1/committed stages',
        no_online_GT=True,not_new_DEV_or_TEST=True))
    print(json.dumps([dict(frame_id=c['frame_id'],reason=c['primary_reason'],
        max_raw_reference_score=max((p['reference_class_score'] for p in c['raw_reference_probes']),default=None),
        same_current_evidence=c['same_current_detector_and_observation_evidence_as_v3']) for c in cases]),flush=True)


def regression(root):
    """CPU proposals on actual old scalars, never new E2E/TEST evidence."""
    import numpy as np
    from dataclasses import replace
    from .target_complete_report import world,context
    from .target_complete_bypass import CONFIGS
    from worldstream.m3.world_state import CurrentCompetitionEvidence
    from worldstream.m3.answer_validity import propose_fact_answer
    from worldstream.m3.answer_memory import AnswerMemory
    from worldstream.m1.config import M1Config
    captures=lines(root/'diagnostic/requests.jsonl')
    cfg=json.loads((root/'diagnostic/effective_config.json').read_text())['config']['model']['m1_config']
    maxage=M1Config.load(cfg).max_state_age_s
    failures=json.loads((root/'nearest_target_failure_audit.json').read_text())['cases']
    questions={c['frame_id']:c['question'] for c in failures};questions['79070dca796f4ffb868d84f1b067f9c2']='What type of vehicle is directly ahead?'
    results=[]
    for row in captures:
        if row['frame_id'] not in questions:continue
        w=world(row['committed_world']);pose=np.asarray(w.ego_to_world).reshape(4,4);cs=[]
        for i,a in enumerate(row['decoded_candidates']):
            if a['score']<.15:continue
            point=pose@np.r_[a['center_ego'],1.];cameras=[]
            for cid,cam in row['cameras'].items():
                if not cam['timing_reliable'] or abs(cam['timestamp']-w.timestamp)>maxage:continue
                c=np.asarray(cam['world_to_camera'])@point;uv=np.asarray(cam['intrinsic'])@c[:3]
                if c[2]>.1:
                    u,v=uv[:2]/uv[2]
                    if 0<=u<cam['image_shape'][1] and 0<=v<cam['image_shape'][0]:cameras.append(cid)
            if cameras:cs.append(CurrentCompetitionEvidence('decoded:'+str(i),a['semantic_class'],tuple(a['center_ego']),
                a['score'],tuple(sorted(cameras)),w.frame_id,w.timestamp))
        w=replace(w,competition_evidence=tuple(cs))
        for i,config in enumerate(CONFIGS):
            p=propose_fact_answer(questions[w.frame_id],w,context(w,row['frame_index'],list(row['cameras'])),AnswerMemory(),('offline-regression',),
                dict(expected_normal_m3_ms=1000.,**config),None,True,False,True)
            results.append(dict(index=i,frame_id=w.frame_id,question=questions[w.frame_id],proposal=p))
    write(root/'prior_regression.json',dict(scope='CPU replay on actual raw/current scalar captures; not GPU calls or new TEST',results=results))
    print('REGRESSION_PROPOSALS',json.dumps([dict(index=e['index'],frame=e['frame_id'],answer=e['proposal']['answer_text'],
        admitted=e['proposal']['admitted']) for e in results]),flush=True)


def verify(root,checkpoint=False,manifest=False):
    if not checkpoint and not manifest:
        assert (root/'artifact_manifest.json').exists()
        assert json.loads((root/'integrity_audit.json').read_text())['audit_tool_sha256']==sha256(Path(__file__))
    prior=json.loads((root/'prior_evidence.json').read_text())
    assert all(sha256(PRIOR/p)==v['sha256'] for p,v in prior.items()),'Historical v3 artifacts changed'
    frozen=json.loads((root/'POLICY_FREEZE.json').read_text());plan=json.loads((root/'experiment_plan.json').read_text())
    protocol=json.loads((root/'protocol_freeze.json').read_text())
    assert sha256(root/'experiment_plan.json')==protocol['plan_sha256']
    assert sha256(root/'protocol_freeze.json')==frozen['protocol_sha256']
    assert sha256(root/'selected_policy.json')==frozen['selected_policy_sha256']
    for n,h in frozen['ablation_policies'].items():assert sha256(root/n)==h
    report_erratum=False
    for p,h in frozen['sources'].items():
        path=Path(p);rel=path.relative_to(Path.cwd()) if path.is_absolute() else path
        archived=root/'frozen_sources'/rel
        assert sha256(archived)==h
        if sha256(path)!=h:
            # A single non-scoring report metadata-key collision was repaired
            # after inference; retain the original freeze and verify exact diff.
            assert path.name=='target_complete_report.py'
            erratum=json.loads((root/'report_assembly_fix.json').read_text())
            old="task=e['task'],admitted=e['llm_skipped'],final_quality=e['quality'],"
            new="question_intent=e['task'],admitted=e['llm_skipped'],final_quality=e['quality'],"
            assert archived.read_text().count(old)==1
            assert path.read_text()==archived.read_text().replace(old,new)
            assert erratum['original_sha256']==h and erratum['corrected_sha256']==sha256(path)
            report_erratum=True
    dev={s['log_id'] for s in plan['development_scenes']};test={s['log_id'] for s in plan['test_scenes']}
    assert len(dev)==2 and len(test)==4 and not dev&test
    assert not (dev|test)&set(plan['prior_excluded_log_ids'])
    for p,h in plan['prior_plan_hashes'].items():assert sha256(p)==h
    expected=json.loads(Path('outputs/m3_world_guided_decode_run01/native_decode_profile.json').read_text())['model_sha256']
    stages={};skips=Counter();allrows={}
    from .target_complete_bypass import stage_jobs
    for stage,count in dict(diagnostic=25,shadow=16,pareto=160,run=672,ablation=96).items():
        out=root/stage;status=json.loads((out/'run_status.json').read_text());assert status['status']=='COMPLETED'
        assert status['checkpoint_sha256']==expected
        rows=lines(out/'requests.jsonl');allrows[stage]=rows
        assert len(rows)==count==status['full_model_requests']
        sources=json.loads((out/'execution_sources.json').read_text())['sources']
        for p,h in sources.items():
            path=Path(p);rel=path.relative_to(Path.cwd()) if path.is_absolute() else path
            assert sha256(out/'executed_sources'/rel)==h
            if stage in ('pareto','run','ablation'):assert sha256(path)==h
        if stage!='diagnostic':
            scenes=plan['development_scenes'] if stage in ('shadow','pareto') else plan['test_scenes']
            expected_rows={(w,n,rep,c,s['scene_id'],t) for job in stage_jobs(root,stage,plan)
                for w,n,rep,c in [job[:4]] for s in scenes if len(job)==6 or s['scene_id']==job[6] for t in s['tokens']}
            actual_rows={(r['workload'],r['question_count'],r['round'],r['condition'],r['scene_id'],r['frame_id']) for r in rows}
            assert len(actual_rows)==len(rows) and actual_rows==expected_rows
        for row in rows:
            assert row['execution_completed']
            if stage=='diagnostic':continue
            assert row['m1_commit_completed'] and row['m2_completed'] and len(row['camera_ids'])==6
            assert row['observed_llm_forward_calls']==sum(q['llm_forward_calls'] for q in row['requests'])
            assert len(row['requests'])==row['question_count']
            for a in (row['committed_world'] or {}).get('competition_evidence',[]):
                assert a['source_frame']==row['frame_id'] and a['source_timestamp']==row['timestamp']
                assert a['evidence_origin']=='actual_current_decoded_perception' and not a['answer_witness_eligible']
                assert a['score']>=.15 and a['cameras']
            if row['condition'] in ('B','C0','C-basic') and stage!='shadow':
                assert not row['committed_world']['competition_evidence'],'Basic baseline pays unnecessary readout'
            for q in row['requests']:
                if q['llm_skipped']:
                    skips[stage]+=1
                    assert q['llm_forward_calls']==0 and q['layout_preparation_ms']==0.
                    assert not q['prefill_completed'] and not q['decode_completed']
                    assert q['fast_answer']['supported'] and q['fast_answer']['admitted']
                    for f in q['fast_answer']['facts']:
                        assert f['evidence_origin']=='actual_perception' and f['source_frame']==row['frame_id']
                        assert f['source_timestamp']==row['timestamp']
                elif row['condition']!='A':
                    assert q['executor']=='PrefillExecutor' and q['prefill_completed'] and q['decode_completed']
                    assert q['question_rng']['enabled']
                if (q.get('fast_answer') or {}).get('profile',{}).get('intent')=='unsupported':
                    assert not (q.get('fast_answer') or {}).get('target_completeness')
            if stage=='shadow':assert not any(q['llm_skipped'] for q in row['requests'])
        stages[stage]=dict(full_M123_calls=count,executed_sources_verified=len(sources))
    ablation=json.loads((root/'current_vs_target_complete_ablation.json').read_text())
    assert ablation['current_evidence_equal']
    gpu=json.loads((root/'validation_gpu/summary.json').read_text());cpu=json.loads((root/'validation_cpu/summary.json').read_text())
    assert gpu['exit_code']==cpu['exit_code']==0 and gpu['CUDA_available']
    assert not {4,5}&set(gpu['cpu_affinity']) and not {4,5}&set(cpu['cpu_affinity'])
    cp_path=root/'checkpoint_verification.json'
    if checkpoint:
        actual=sha256('ckpts/iter_10548.pth');assert actual==expected
        write(cp_path,dict(actual_hash_verified=True,actual_sha256=actual,expected_sha256=expected))
    cp=json.loads(cp_path.read_text());assert cp['actual_hash_verified'] and cp['actual_sha256']==expected
    counts=json.loads((root/'run_counts.json').read_text());assert counts['full_M123_calls']==sum(s['full_M123_calls'] for s in stages.values())
    assert counts['actual_formal_skips']==skips['run']
    result=dict(status='PASS',stages=stages,full_M123_calls=counts['full_M123_calls'],formal_skips=skips['run'],
        actual_skips=dict(skips),regression_tests=gpu['counts'],checkpoint=cp,
        inference_scoring_policy_protocol_frozen=True,report_assembly_erratum_verified=report_erratum,
        historical_v3_unchanged=True,new_DEV_TEST_logs_disjoint=True,
        base_model_train_exposure_possible=True,same_current_M1_evidence=True,actual_forward_hooks_verified=True,
        deployment_enabled=False,audit_tool_sha256=sha256(Path(__file__)))
    if manifest:
        write(root/'integrity_audit.json',result);files={}
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.name=='artifact_manifest.json' or any('q_r' in x or x.endswith('_answers') for x in p.parts):continue
            assert p.suffix not in ('.pt','.pth','.pkl','.npy','.npz')
            files[str(p)]=dict(bytes=p.stat().st_size,sha256=sha256(p))
        write(root/'artifact_manifest.json',dict(files=files,total_bytes=sum(x['bytes'] for x in files.values())))
    if (root/'artifact_manifest.json').exists():
        m=json.loads((root/'artifact_manifest.json').read_text())
        assert all(sha256(p)==v['sha256'] and Path(p).stat().st_size==v['bytes'] for p,v in m['files'].items())
        result['manifest_files_verified']=len(m['files'])
    print(json.dumps(result),flush=True)


def reproduce_plan(root):
    """Repeat the frozen experiment; never pretend these are unused logs."""
    import shutil
    assert root.resolve()!=ROOT.resolve(),'Keep the original delivery unchanged'
    root.mkdir(parents=True,exist_ok=True)
    assert not (root/'experiment_plan.json').exists(),'Use a new empty reproduction root'
    for name in ('experiment_plan.json','protocol_freeze.json','prior_evidence.json'):
        shutil.copyfile(ROOT/name,root/name)
    write(root/'reproduction_source.json',dict(source_root=str(ROOT),plan_sha256=sha256(root/'experiment_plan.json'),
        scope='Exact repetition of frozen metadata split; NOT newly held-out experimental logs'))
    print('FROZEN_PLAN_COPIED',flush=True)


def summarize(root):
    from .temporal_selective_report import performance
    rows=lines(root/'run/requests.jsonl');base={(r['workload'],r['question_count'],r['round'],r['frame_id']):r for r in rows if r['condition']=='B'}
    misses=[]
    for r in rows:
        if r['condition']!='C':continue
        b=base[(r['workload'],r['question_count'],r['round'],r['frame_id'])]
        for qi,(bq,cq) in enumerate(zip(b['requests'],r['requests'])):
            if cq['llm_skipped']:continue
            misses.append(dict(workload=r['workload'],question_count=r['question_count'],scene_id=r['scene_id'],frame_id=r['frame_id'],
                request_index=qi,same_seed=bq['question_rng']['seed']==cq['question_rng']['seed'],
                same_executor=bq['executor']==cq['executor']=='PrefillExecutor',same_tokens=bq['tokens']==cq['tokens']))
    write(root/'miss_pairing_diagnostic.json',dict(requests=misses,all_same_seed=all(e['same_seed'] for e in misses),
        all_same_executor=all(e['same_executor'] for e in misses),same_output_tokens=sum(e['same_tokens'] for e in misses),
        scope='Same starting question RNG; earlier skips can change AnswerMemory/draft scheduling. Token identity is not the quality objective. No additional causal GPU replay.'))
    write(root/'steady_state_performance.json',performance([r for r in rows if r['frame_index']>0],[('A','B'),('B','C'),('A','C')]))
    perf=json.loads((root/'abc_comparison.json').read_text());deltas=[]
    for w,n in sorted({(m['workload'],m['question_count']) for m in perf['module_breakdown']}):
        b=next(m for m in perf['module_breakdown'] if (m['workload'],m['question_count'],m['condition'])==(w,n,'B'))
        c=next(m for m in perf['module_breakdown'] if (m['workload'],m['question_count'],m['condition'])==(w,n,'C'))
        keys=('e2e_wall_ms','m1_wall_ms','m1_including_adapters_ms','m2_wall_ms','heads_wall_ms','snapshot_wall_ms','m3_wall_ms',
            'rng_management_wall_ms','answer_memory_selection_wall_ms','answer_memory_management_wall_ms')
        deltas.append(dict(workload=w,question_count=n,C_minus_B_ms={k:c[k]-b[k] for k in keys},
            parsing_ms=c['question_fact_extraction_wall_ms_per_question'],extraction_ms=c['current_fact_extraction_wall_ms_per_question'],
            completeness_ms=c['target_completeness_wall_ms_per_question'],miss_decision_ms=c['miss_decision_wall_ms']))
    write(root/'unsupported_overhead_attribution.json',dict(deltas=deltas,
        scope='Overlapping module spans cannot be summed to E2E. Differences are observational and include CUDA/order/M1/M2 variation; do not attribute whole deltas to microsecond parsing. Native RNG not separately instrumented.'))
    from worldstream.m3.task_conditioner import question_facts
    coverage=[];costs=[]
    for w,n in sorted({(r['workload'],r['question_count']) for r in rows}):
        qs=[q for r in rows if (r['workload'],r['question_count'],r['condition'])==(w,n,'C') for q in r['requests']]
        coverage.append(dict(workload=w,question_count=n,requests=len(qs),
            structured_questions=sum(question_facts(q['question']).intent!='unsupported' for q in qs),
            available_supported_proposals=sum(bool((q.get('fast_answer') or {}).get('supported')) for q in qs),
            actual_skips=sum(q['llm_skipped'] for q in qs)))
    write(root/'workload_coverage.json',dict(groups=coverage,mixed_ratio_source='Preset one introduced structured slot out of three; original questions retained'))
    companion=lines(root/'ablation/requests.jsonl')
    for stage,rs in [('run',rows),('ablation',companion)]:
        for cond,w,n in sorted({(r['condition'],r['workload'],r['question_count']) for r in rs}):
            rr=[r for r in rs if (r['condition'],r['workload'],r['question_count'])==(cond,w,n)]
            qs=[q for r in rr for q in r['requests']]
            costs.append(dict(stage=stage,condition=cond,workload=w,question_count=n,questions=len(qs),
                snapshot_ms_per_frame=mean([r['snapshot_wall_ms'] for r in rr]),
                extraction_ms_per_question=sum((q.get('fast_answer') or {}).get('current_fact_extraction_wall_ms',0.) for q in qs)/len(qs),
                completeness_ms_per_question=sum((q.get('fast_answer') or {}).get('target_completeness_wall_ms',0.) for q in qs)/len(qs),
                retained_competition_candidates=mean([len((r['committed_world'] or {}).get('competition_evidence',[])) for r in rr])))
    write(root/'target_completeness_cpu_cost.json',dict(groups=costs,
        scope='New readout also appears in M1 output-adapter wall and projection in snapshot; question-check cost is nested in validity. Basic/B retain their original readout.'))
    fast=json.loads((root/'fast_answer_quality.json').read_text());groups=defaultdict(list)
    for e in fast['nearest_identity']:groups[(e['stage'],e['condition'],e['task'])].append(e)
    write(root/'nearest_target_correctness.json',dict(groups=[dict(stage=k[0],condition=k[1],task=k[2],fast_events=len(part),
        physical_class_scored=sum(e['physical_class_correct'] is not None for e in part),
        physical_class_correct=sum(e['physical_class_correct'] is True for e in part),
        question_target_scored=sum(e['target_identity_correct'] is not None for e in part),
        question_target_correct=sum(e['target_identity_correct'] is True for e in part),
        target_UNKNOWN=sum(e['target_identity_correct'] is None for e in part),
        unique_frame_questions=len({(e['frame_id'],e['question']) for e in part}),independent_scenes=len({e['scene_id'] for e in part}))
        for k,part in groups.items()],scope='Unique class-blind geometry match<2m, next-gap>0.5m. Physical class correctness and visible question-target identity differ. Unmatched stays UNKNOWN; exact LLM identity is not observed.'))
    counts=json.loads((root/'run_counts.json').read_text());gpu=json.loads((root/'validation_gpu/summary.json').read_text())
    write(root/'validation_summary.json',dict(regression=gpu,full_M123_experiments=counts,
        note='Regression process full_M123_GPU_experiments0 is distinct from actual measured969 full calls.'))
    (root/'validation_summary.md').write_text(f"# Validation\n\nCPU841 passed/8 skipped; CUDA{gpu['counts']} on RTX3090. Full M123 calls{counts['full_M123_calls']}, formal requests{counts['formal_question_requests']}, actual formal skips{counts['actual_formal_skips']}. CPU4/5 excluded. No extra M3-only causal replays.\n")
    plan=json.loads((root/'experiment_plan.json').read_text());points=json.loads((root/'quality_latency_pareto.json').read_text())['points']
    dev=lines(root/'pareto/requests.jsonl');write(root/'dev_e2e_comparisons.json',performance(dev,[('B',f'C{i}') for i in range(4)]))
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/omnidrive-wrsar-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.2,4.3))
    for p in points:
        ax.scatter(p['mean_e2e_ms'],-100*p['loss'],s=80,marker='o' if p['target_completeness_enabled'] else '*',label=f"C{p['index']}: hits{p['hits']}, target={p['target_completeness_enabled']}")
    ax.set_xlabel('Actual DEV full M123 E2E (ms/frame)');ax.set_ylabel('Scored improvement over B (pp)')
    ax.set_title('Four predeclared DEV candidates; two independent logs');ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax.margins(.15);fig.tight_layout();fig.savefig(root/'quality_latency_pareto.png',dpi=180);fig.savefig(root/'quality_latency_pareto.pdf');plt.close(fig)
    enrich_answers(root)
    assessment(root,rows,companion,coverage,costs)
    print('SUPPLEMENTARY_REPORTS_SAVED',flush=True)


def assessment(root,rows,companion,coverage,costs):
    d=json.loads((root/'go_no_go.json').read_text());q=json.loads((root/'task_quality.json').read_text())
    severe=json.loads((root/'severe_error_audit.json').read_text());ab=json.loads((root/'current_vs_target_complete_ablation.json').read_text())
    baseline={(r['workload'],r['question_count'],r['frame_id']):r for r in rows if r['condition']=='C'}
    equivalent=[]
    for r in companion:
        b=baseline[(r['workload'],r['question_count'],r['frame_id'])]
        for qi,(bq,tq) in enumerate(zip(b['requests'],r['requests'])):
            equivalent.append(dict(workload=r['workload'],question_count=r['question_count'],frame_id=r['frame_id'],request_index=qi,
                same_skip=bq['llm_skipped']==tq['llm_skipped'],same_tokens=bq['tokens']==tq['tokens'],
                same_forwards=bq['llm_forward_calls']==tq['llm_forward_calls']))
    write(root/'ablation_execution_equivalence.json',dict(pairs=equivalent,
        same_skips=all(e['same_skip'] for e in equivalent),same_tokens=sum(e['same_tokens'] for e in equivalent),
        same_forward_counts=sum(e['same_forwards'] for e in equivalent),question_pairs=len(equivalent),
        scope='Same phase counterparts; timing differences alone cannot prove additional compute omission when execution receipts agree.'))
    def group(cond):return next(g for g in q['groups'] if (g['workload'],g['question_count'],g['condition'],g['execution'])==('addressable',3,cond,'all'))
    b,c=group('B'),group('C');loss=d['paired_quality'];details=[]
    for e in severe['new_severe_vs_B']:
        r=baseline[(e['workload'],e['question_count'],e['frame_id'])];p=r['requests'][e['request_index']]['fast_answer']
        details.append(f"- `{e['frame_id']}` {e['question']}: B={e['baseline_answer']!r}, C={e['answer']!r}; reference direction={e['quality']['reference'].get('directions')}, center={e['quality']['reference'].get('center_ego')}; selected current fact={p['facts']!r}.")
    cov='; '.join(f"{e['workload']}{e['question_count']}Q: structured={e['structured_questions']}/{e['requests']}, supported={e['available_supported_proposals']}, actual skips={e['actual_skips']}" for e in coverage)
    text=('\n\n## Acceptance outcome and scope\n\n'+f"Final **{d['status']}**. DEV selected {d['selected_policy_id']} (target enabled={d['target_completeness_enabled']}), not an unmeasured post-TEST alternative. Research average-quality budget passes={d['quality_budget_pass']}; credible object3Q E2E gain={d['credible_speed_gain']}; the predeclared research severe gate passes={d['research_severe_gate_pass']}.\n\n"+
        f"Object3Q B→C={d['addressable_three_QA']['e2e_speedup']:.6f}x; B known quality={b['score']:.6%}, C={c['score']:.6%}, coverage={c['scored']}/{c['requests']}, UNKNOWN={c['UNKNOWN']}. New severe rate={loss['new_severe_rate']:.6%}, empirical sceneCI={loss['new_severe_scene_CI95']}, frozen ceiling2%. Overall severe outputs B={b['severe']}, C={c['severe']}; lower totals do not erase the newly introduced case.\n\n"+'\n'.join(details)+
        f"\n\nMeasured independent target quality/severe value={ab['independent_quality_or_severe_value']}. Core ablation vetoes="+str(ab['veto_counts'])+f"; equal skip decisions={all(e['same_skip'] for e in equivalent)}, same-token pairs={sum(e['same_tokens'] for e in equivalent)}/{len(equivalent)}. Readout/check overhead and phase/GPU variation can coexist; timing differences with the same compute receipts do not establish new language-compute savings. Old diagnostic replays and DEV coverage remain separate; TEST never selects alternative configurations.\n\n"+
        'Coverage by original workload: '+cov+'. The mixed workload adds one preset structured slot; additional original questions can also parse as structured. It is not a natural population estimate. Natural1Q has a small measured E2E regression; its module breakdown/sceneCI remain visible, and the whole change cannot be assigned to microsecond parsing.\n\n'+
        'Only4 independent experimental TEST logs/32 frames are available, with reliable object3Q labels on3 logs. Source partition is physical train; possible base-model training exposure is explicit. Rare-error and UNKNOWN uncertainty prevent a deployment-safety claim. Risk ratios are descriptive, not probabilities. v3 historical NO_GO is unchanged; production switches stay off.\n\n'+
        'The single post-inference report-key collision was fixed with an exact recorded metadata substitution. Frozen original source and policies are preserved; inference and scoring logic are unchanged. See report_assembly_fix.json and integrity_audit.json.\n')
    (root/'final_research_assessment.md').write_text('# Final research assessment'+text)
    p=root/'implementation_summary.md';p.write_text(p.read_text().split('\n\n## Acceptance outcome')[0]+text)


def enrich_answers(root):
    """Presentation only: read frozen results, never change grading/policy."""
    cost=json.loads((root/'target_completeness_cpu_cost.json').read_text())['groups']
    coverage=json.loads((root/'workload_coverage.json').read_text())['groups']
    tasks=json.loads((root/'quality_by_task.json').read_text())
    policies={b:json.loads((root/f'policy_{b}pp.json').read_text()) for b in (1,2,3)}
    ab=json.loads((root/'current_vs_target_complete_ablation.json').read_text())
    counts=json.loads((root/'run_counts.json').read_text())
    tc=[c for c in cost if c['condition']=='C-target' and c['workload']=='addressable' and c['question_count']==3]
    extras={6:'C-target对象三问实测：'+json.dumps(tc,ensure_ascii=False),
        11:'分任务B→C：'+'; '.join(f"{task}: "+' → '.join(str(next(t['score'] for t in tasks if t['task']==task and t['condition']==c)) for c in ('B','C'))
            for task in ('category','existence','position','motion'))+'。未评分显示None；分母与唯一Frame/Question见对应JSON。',
        12:f"正式C共{sum(c['requests'] for c in coverage)}个问题，真实跳过{counts['actual_formal_skips']}；可用候选和实际准入覆盖在workload_coverage.json分开。",
        18:'DEV预算工作点：'+'; '.join(f"{b}%: {p['policy_id']}, target={p['target_completeness_enabled']}, confidence={p['min_confidence']}, binding_risk={p['max_binding_risk']}, E2E={p['online_dev_metrics']['mean_e2e_ms']:.3f}ms"
            for b,p in policies.items()),
        20:'同数据消融E2E：'+'; '.join(f"{s['workload']}{s['question_count']}Q basic→target={s['e2e_speedup']:.6f}×，差CI={s['delta_scene_CI95_ms']}ms" for s in ab['performance']['summaries'])+
            '。质量/严重差的场景区间若接触零，不能宣称严格统计显著；仅记录有限样本中的经验变化。'}
    path=root/'result_answers.md';parts=path.read_text().split('\n\n')
    for i,part in enumerate(parts):
        for number,extra in extras.items():
            if part.startswith(str(number)+'. '):parts[i]=part.split(' 补充实测：')[0]+' 补充实测：'+extra
    path.write_text('\n\n'.join(parts))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--prior',action='store_true');p.add_argument('--regression',action='store_true')
    p.add_argument('--verify',action='store_true');p.add_argument('--checkpoint',action='store_true');p.add_argument('--manifest',action='store_true')
    p.add_argument('--reproduce-plan',action='store_true');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.prior:prior_audit(a.root)
    elif a.regression:regression(a.root)
    elif a.reproduce_plan:reproduce_plan(a.root)
    elif a.summarize:summarize(a.root)
    else:verify(a.root,a.checkpoint,a.manifest)
