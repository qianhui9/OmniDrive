"""Finite historical diagnostics and explicit independent-validation gaps."""
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import subprocess
from time import perf_counter

from .benchmark.common import write,sha256
from .final_fact_bypass import PRIOR,lines,mean,world,context,load_objects
from .fact_bypass_identity import question_target,evaluate,match_instances,request_identity
from .world_guided_decode_v2_report import scene_ci


def replay(rows,objects,policy):
    from worldstream.m3.answer_memory import AnswerMemory
    from worldstream.m3.answer_validity import propose_fact_answer
    out=[]
    for row in rows:
        w=world(row['committed_world']);ctx=context(w,row['frame_index'],row['camera_ids'])
        for i,(req,ans) in enumerate(zip(row['requests'],row['answers'])):
            q=req['question'];p=propose_fact_answer(q,w,ctx,AnswerMemory(),('offline',),policy,
                target_binding_enabled=True)
            target=question_target(q,objects[w.frame_id]);b=evaluate(q,ans['A'][0],objects[w.frame_id],target,w.frame_id)
            cq=evaluate(q,p['answer_text'],objects[w.frame_id],target,w.frame_id) if p['admitted'] else b
            out.append(dict(scene_id=row['scene_id'],log_id=row['log_id'],frame_id=w.frame_id,question=q,
                task=p['profile']['intent'],proposal=p,baseline_quality=b,candidate_quality=cq,
                fallback_uses_recorded_B=True,estimated_avoided_request_ms=req['end_to_end_ms'] if p['admitted'] else 0.))
    return out


def compare(part,policy_id):
    known=[r for r in part if r['baseline_quality']['known'] and r['candidate_quality']['known']]
    paired=[dict(scene_id=r['log_id'],loss=r['baseline_quality']['score']-r['candidate_quality']['score'],
        new_severe=float(bool(r['candidate_quality']['severe_error']) and not r['baseline_quality']['severe_error'])) for r in known]
    fast=[r for r in part if r['proposal']['admitted']]
    return dict(policy_id=policy_id,requests=len(part),hypothetical_hits=len(fast),actual_skips=0,
        known=len(known),UNKNOWN=len(part)-len(known),
        B_score=mean([r['baseline_quality']['score'] for r in known]),
        candidate_score=mean([r['candidate_quality']['score'] for r in known]),
        paired_loss=mean([r['loss'] for r in paired]),scene_CI95=scene_ci(paired,'loss') if paired else None,
        new_severe=sum(r['new_severe'] for r in paired),
        new_severe_scene_CI95=scene_ci(paired,'new_severe') if paired else None,
        independent_logs=len({r['log_id'] for r in part}),
        hypothetical_avoided_request_ms=sum(r['estimated_avoided_request_ms'] for r in part),
        measured_M123_E2E_ms=None,
        tasks=[dict(task=t,requests=sum(r['task']==t for r in part),
            hypothetical_hits=sum(r['task']==t and r['proposal']['admitted'] for r in part),
            paired_known=sum(r['task']==t for r in known),
            B_score=mean([r['baseline_quality']['score'] for r in known if r['task']==t]),
            candidate_score=mean([r['candidate_quality']['score'] for r in known if r['task']==t]))
            for t in ('category','existence','position')])


def policy_summaries(parts,ids):
    summaries=[compare(part,pid) for pid,part in zip(ids,parts)]
    basic=parts[0]
    for summary,part in zip(summaries,parts):
        removed=[(b,r) for b,r in zip(basic,part) if b['task']=='position' and b['proposal']['admitted'] and not r['proposal']['admitted']]
        scored=[(b,r) for b,r in removed if b['candidate_quality']['known'] and r['candidate_quality']['known']]
        summary['position_fallback_vs_basic']=dict(additional_hypothetical_LLM_requests=len(removed),
            paired_known=len(scored),UNKNOWN=len(removed)-len(scored),
            paired_quality_delta_vs_basic=mean([r['candidate_quality']['score']-b['candidate_quality']['score'] for b,r in scored]),
            causal_GPU_E2E_measured=False)
    return summaries


def diagnose(root):
    rows=lines(PRIOR/'shadow/requests.jsonl');objects=load_objects({r['frame_id'] for r in rows})
    base=json.loads((PRIOR/'selected_policy.json').read_text())
    base.update(status='DIAGNOSTIC_ONLY_NOT_FROZEN',main_path='current_only',policy_id='C-basic',
        position_mode='basic',target_completeness_enabled=False,temporal_risk_enabled=False)
    basic=replay(rows,objects,base)
    # Fit a SINGLE detector-support floor from historical DEV only. No old TEST
    # errors enter fitting. If no positive/error separation exists, do not
    # invent a threshold or declare Position solved.
    errors=[r['proposal']['support_score'] for r in basic if r['task']=='position' and
        r['proposal']['admitted'] and r['candidate_quality']['known'] and r['candidate_quality']['score']<1]
    floor=math.nextafter(max(errors),1.) if errors else base['min_confidence']
    restricted=dict(base,policy_id='C-position-restricted',position_mode='restricted',position_min_confidence=floor)
    fallback=dict(base,policy_id='C-position-fallback',position_mode='fallback')
    policies=[base,restricted,fallback];allparts=[basic]+[replay(rows,objects,p) for p in policies[1:]]
    summaries=policy_summaries(allparts,[p['policy_id'] for p in policies])
    for p in policies:write(root/(p['policy_id']+'.json'),p)
    with (root/'historical_dev_counterfactuals.jsonl').open('w') as f:
        for p,part in zip(policies,allparts):
            for r in part:f.write(json.dumps(dict(policy_id=p['policy_id'],**r))+'\n')
    write(root/'dev_policy_comparison.json',dict(status='HISTORICAL_DEV_CPU_COUNTERFACTUAL_ONLY',
        source=str(PRIOR/'shadow/requests.jsonl'),source_sha256=sha256(PRIOR/'shadow/requests.jsonl'),
        policies=summaries,threshold_fit=dict(method='Just above maximum known incorrect admitted Position support on historical DEV; no TEST input',
            position_errors=len(errors),position_min_confidence=floor,
            status='NO_EVIDENCE_FOR_STRICTER_FLOOR' if not errors else 'HISTORICAL_DEV_DIAGNOSTIC_FIT'),
        new_dev_scenes=0,qualified_best_policy=None,actual_full_M123_calls=0,actual_skips=0,
        reason='No fresh independent evaluation pool; old DEV replay may diagnose but cannot freeze a new quality-E2E optimum',
        shadow_is_not_e2e=True))
    write(root/'quality_latency_pareto.json',dict(status='NOT_MEASURED_NEW_POLICY_E2E',
        quality_budgets=[.01,.02,.03],default_budget=.02,points=summaries,
        best_new_DEV_policy=None,reason='All latency savings are counterfactual; no new full M123 E2E or independent TEST',
        historical_best_measured_DEV='target_dev_0',historical_result_unchanged=True))
    write(root/'selected_policy.json',dict(status='INCOMPLETE_INDEPENDENT_VALIDATION',
        main_path='current_only',selected_policy_id=None,automatic_deployment=False,
        target_completeness_enabled=False,temporal_risk_enabled=False,
        historical_best_measured_DEV_policy='target_dev_0',
        reason='Cannot select/freeze a new optimal policy without new DEV quality-E2E evidence. Diagnostic candidates are rejected by online runtime policy validation.'))
    # CPU microbenchmark: no model, no artificial latency or GPU extrapolation.
    from worldstream.m3.answer_memory import AnswerMemory
    from worldstream.m3.answer_validity import propose_fact_answer
    samples=[];warm=rows[0];w=world(warm['committed_world']);ctx=context(w,warm['frame_index'],warm['camera_ids'])
    for p in policies:
        for _ in range(5):replay(rows[:1],objects,p)
        admission=[];validity=[];total=[];parsing=[]
        for _ in range(10):
            for row in rows:
                w=world(row['committed_world']);ctx=context(w,row['frame_index'],row['camera_ids'])
                for req in row['requests']:
                    tick=perf_counter();r=propose_fact_answer(req['question'],w,ctx,AnswerMemory(),('cost-only',),p,target_binding_enabled=True)
                    total.append((perf_counter()-tick)*1000);validity.append(r['answer_validity_check_wall_ms'])
                    parsing.append(r['question_fact_extraction_wall_ms'])
                    if r['profile']['intent']=='position' and 'position_admission' in r:admission.append(r['position_admission_wall_ms'])
        samples.append(dict(policy_id=p['policy_id'],calls=len(total),
            mean_total_ms=mean(total),median_total_ms=statistics.median(total),
            mean_validity_ms=mean(validity),mean_parse_ms=mean(parsing),
            position_gate_calls=len(admission),mean_position_admission_ms=mean(admission),
            position_gate_nested_in_validity=True,cpu_affinity=sorted(__import__('os').sched_getaffinity(0))))
    write(root/'cpu_cost.json',dict(status='MEASURED_CPU_ONLY',warmup_passes=5,repeats=10,groups=samples,
        model_called=False,E2E_inference_gain_measured=False))
    position_audit(root,policies)


def position_audit(root,policies):
    old=json.loads((PRIOR/'nearest_target_failure_audit.json').read_text())
    old_objects=json.loads(Path('outputs/m3_state_calibrated_bypass_run01/current_frame_facts.json').read_text())['frames']
    captures={r['frame_id']:r for r in lines(PRIOR/'diagnostic/requests.jsonl')}
    cases=[]
    for c in old['cases']:
        cases.append({k:c[k] for k in ('frame_id','scene_id','question','new_severe','current_answer','baseline_answer',
            'selected_target','primary_reason','filtered_stage','decoded_reference_candidates',
            'post_M1_filter_reference_candidates','committed_reference_candidates','nearest_visible_GT','selected_nearby_physical_GT')})
        cases[-1].update(source=str(PRIOR/'nearest_target_failure_audit.json'),
            interpretation='Raw decoded/adapter/commit receipts establish the filter stage; geometry match is diagnostic, not an identity certificate.',new_test=False)
        capture=captures[c['frame_id']];w=capture['committed_world'];gs=old_objects[c['frame_id']]
        matched=match_instances(w['anchors'],gs,w['frame_id'],w['timestamp'])
        cases[-1]['new_offline_identity_evaluation']=request_identity(dict(facts=[c['selected_target']]),
            question_target(c['question'],gs),matched)
    # v4 error: no new GPU replay. Record exactly which causal stages were
    # retained. Do not invent raw detections or assert filtering from absence.
    records=lines(root/'historical_quality_requests.jsonl')
    bad=[r for r in records if r['condition']=='C' and r['task']=='position' and (
        (r['quality']['known'] and r['quality']['score']<1) or
        (r['original_quality']['known'] and r['original_quality']['score']<1))]
    severe=json.loads((PRIOR/'severe_error_audit.json').read_text())
    source_cases=severe['new_severe_vs_B']
    if isinstance(source_cases,dict):source_cases=source_cases.get('cases',[])
    old_severe={(r['frame_id'],r['question']):r for r in source_cases}
    included={(r['frame_id'],r['question']) for r in bad}
    bad += [r for r in records if r['condition']=='C' and (r['frame_id'],r['question']) in old_severe
        and (r['frame_id'],r['question']) not in included]
    raw_rows=lines(PRIOR/'run/requests.jsonl')
    for r in bad:
        row=next(x for x in raw_rows if x['condition']=='C' and x['workload']=='addressable' and x['question_count']==3 and x['frame_id']==r['frame_id'])
        w=row['committed_world'];facts=(r['proposal'] or {}).get('facts',[])
        target=r['target_reference']
        historical_reference=old_severe.get((r['frame_id'],r['question']),{}).get('quality',{}).get('reference',{})
        ref=target.get('center_ego') or historical_reference.get('center_ego')
        close=[a for a in w['anchors'] if ref and a['anchor_type']=='object' and math.dist(a['center_ego'],ref)<2.]
        cases.append(dict(frame_id=r['frame_id'],scene_id=r['scene_id'],question=r['question'],
            generation='v4',source=str(PRIOR/'run/requests.jsonl'),new_test=False,
            selected_facts=facts,target_reference=target,physical_object=r['object'],question_target=r['target'],
            historical_frozen_reference=historical_reference,
            historical_new_severe=(r['frame_id'],r['question']) in old_severe,
            historical_severe_verdict='UNCHANGED even when stricter visibility labels are UNKNOWN',
            committed_anchors_near_reference=close,
            geometry_evidence='Ego axes and current GT use the same frame; a sector disagreement alone does not prove a coordinate transform bug.',
            raw_M1_detected_true_target='UNDETERMINED: pre-filter raw queries/decoded candidates were not captured for this formal v4 frame',
            filtered_at_M1_or_M3='UNDETERMINED without pre-filter capture',
            primary_reason='Frozen v4 reference shows wrong answer sector; physical-instance/visibility attribution remains unconfirmed; pre-commit raw stage missing',
            current_answer=r['answer'],baseline_answer=next(x['answer'] for x in records if x['condition']=='B' and x['frame_id']==r['frame_id'] and x['question']==r['question']),
            evidence_limit='Visibility is annotation/projection proxy, not manual image adjudication'))
    # Original Car/Truck fix is a separate regression, never threshold input.
    prior=json.loads((PRIOR/'prior_regression.json').read_text())
    car=[x for x in prior['results'] if x['frame_id']=='79070dca796f4ffb868d84f1b067f9c2' and x['index']==0]
    car_token='79070dca796f4ffb868d84f1b067f9c2'
    if car and car_token in captures and car_token in old_objects:
        w=captures[car_token]['committed_world'];gs=old_objects[car_token]
        car[0]['new_offline_identity_evaluation']=request_identity(car[0]['proposal'],
            question_target(car[0]['question'],gs),match_instances(w['anchors'],gs,w['frame_id'],w['timestamp']))
    write(root/'position_risk_audit.json',dict(status='HISTORICAL_INSTANCE_DIAGNOSTIC',cases=cases,
        original_car_truck_regression=car,
        original_car_truck_root_cause='Historical v3 interface audit: M2 whole-box reuse qualification had filtered current detection evidence; current detector-center projection is retained. No new runtime filter reinstates M2 eligibility.',
        historical_v3_source='outputs/m3_state_calibrated_bypass_run01',
        current_detection_support_preserved=True,new_raw_GPU_calls=0,
        policies='Only historical DEV fits the diagnostic floor; old v3/v4 errors are regression cases only',
        unresolved_cases=sum('UNDETERMINED' in str(c.get('raw_M1_detected_true_target','')) for c in cases)))


def report(root):
    audit=json.loads((root/'dataset_split_provenance.json').read_text())
    train_logs={s['log_id'] for s in audit['partitions']['train']['scene_manifest']}
    val_logs={s['log_id'] for s in audit['partitions']['val']['scene_manifest']}
    audit['official_train_val_log_overlap']=sorted(train_logs&val_logs)
    write(root/'dataset_split_provenance.json',audit)
    dev=json.loads((root/'dev_policy_comparison.json').read_text())
    retained=lines(root/'historical_dev_counterfactuals.jsonl')
    ids=['C-basic','C-position-restricted','C-position-fallback']
    dev['policies']=policy_summaries([[r for r in retained if r['policy_id']==pid] for pid in ids],ids)
    write(root/'dev_policy_comparison.json',dev)
    pareto=json.loads((root/'quality_latency_pareto.json').read_text());pareto['points']=dev['policies']
    write(root/'quality_latency_pareto.json',pareto)
    cov=json.loads((root/'quality_label_coverage.json').read_text())
    identity=json.loads((root/'nearest_target_identity_audit.json').read_text())
    cost=json.loads((root/'cpu_cost.json').read_text())
    test={k:json.loads((root/k/'summary.json').read_text()) for k in ('validation_cpu','validation_gpu') if (root/k/'summary.json').exists()}
    write(root/'validation_summary.json',test)
    checkpoint=json.loads((root/'checkpoint_metadata.json').read_text())
    expected=json.loads((PRIOR/'checkpoint_verification.json').read_text())['expected_sha256']
    assert checkpoint['sha256']==expected,'Checkpoint changed since historical v4'
    write(root/'checkpoint_verification.json',dict(actual_sha256=checkpoint['sha256'],expected_sha256=expected,
        verified_by='Read-only SHA256 during prepare',unchanged=True,
        tensor_data_loaded=False))
    sources=[Path(p) for p in subprocess.check_output(['git','ls-files','worldstream','projects','tools/worldstream_m123','tests'],text=True).splitlines() if p.endswith('.py')]
    sources += [Path(__file__),Path('tools/worldstream_m123/final_fact_bypass.py'),
        Path('tools/worldstream_m123/fact_bypass_identity.py'),Path('tests/worldstream_m3/test_final_fact_bypass.py')]
    write(root/'execution_sources.json',dict(source_hashes={str(p):sha256(p) for p in sorted(set(sources))},
        diagnostic_input_hashes={str(p):sha256(p) for p in [PRIOR/'run/requests.jsonl',PRIOR/'shadow/requests.jsonl',
            PRIOR/'selected_policy.json',PRIOR/'nearest_target_failure_audit.json',PRIOR/'severe_error_audit.json']},
        no_new_test_execution=True))
    write(root/'protocol_status.json',dict(status='NO_NEW_TEST_PROTOCOL_OR_POLICY_FROZEN',
        default_quality_budget=.02,quality_budgets=[.01,.02,.03],
        historical_v4_severe_criterion='New severe rate and empirical scene-CI upper <= 2%; unchanged failed gate',
        required_before_new_TEST=['Unused model-held-out scene/log pool','Fresh DEV measured full M123 policy comparison',
            'Pre-TEST policy, scoring, quality/severe criteria and source freeze'],
        tests_are_regressions_not_quality_experiments=True))
    write(root/'abc_comparison.json',dict(status='NOT_RUN_NO_INDEPENDENT_TEST',new_run_summaries=[],
        full_M123_calls=0,actual_LLM_skips=0,
        historical_reference=dict(path=str(PRIOR/'abc_comparison.json'),sha256=sha256(PRIOR/'abc_comparison.json'),
            object3Q_B_to_C=1.344391,mixed3Q_B_to_C=1.116378),
        no_new_FPS_QPS_memory_or_phase_timings=True))
    write(root/'workload_coverage.json',dict(status='NO_NEW_FORMAL_REQUESTS',new_requests=0,
        historical_v4_reference=json.loads((PRIOR/'workload_coverage.json').read_text()),
        source=str(PRIOR/'workload_coverage.json'),
        scope='Historical opportunities only; preset mixed fraction is not a natural QA distribution'))
    write(root/'go_no_go.json',dict(status='INCOMPLETE_INDEPENDENT_VALIDATION',research_status='INCOMPLETE',
        reasons=['No new compatible official validation log isolated from all historical M3 use',
            'Physical train WRSAR scenes overlap the original model training partition',
            'No new DEV policy with measured full M123 E2E qualification','Nearest-instance and visibility labels remain incomplete'],
        historical_v4_verdict='NO_GO_UNCHANGED',independent_test_qualified=False,
        current_full_M123_GPU_calls=0,current_actual_LLM_skips=0,
        new_TEST_e2e_speedup=None,new_TEST_new_severe_errors=None,new_TEST_severe_CI95=None,
        quality_budgets=[.01,.02,.03],default_quality_budget=.02,strict_severe_risk_qualified=False,
        deployment_enabled=False,tests=test))
    write(root/'run_counts.json',dict(full_M123_GPU_calls=0,actual_LLM_skips=0,
        CPU_counterfactuals=sum(p['requests'] for p in dev['policies']),
        scope='CUDA toy-kernel regression is not full M123 or real task QA'))
    (root/'main_path_cleanup.md').write_text(
        '# Current-only path\n\nQuestion parsing → committed current M1 facts → existing binding gate plus one Position support floor → fact rendering and zero LLM forwards, otherwise the original Target-Aware K8 History-Draft. '
        'There is one M3Runtime. `main_path=current_only` disables full-answer-history lookup and refuses Temporal Risk/Target Completeness flags. '
        'The legacy path remains available for source-frozen historical reproduction. Default selective_answer_mode remains off.\n\n'
        'Question-level RNG isolation, current observation interfaces, GPU timing, missing-witness UNKNOWN behavior, and the M2 reuse/current detection distinction remain unchanged. '
        'The Position confidence floor is detector support, not target correctness probability. Category and positive Existence do not pay this gate; absence never yields No. '
        'No new competing-object score, runtime, visual model, weight, M1/M2 algorithm or sampling change is introduced. '
        'Runtime admission additionally requires a FROZEN_RESEARCH_POLICY; this run delivers only diagnostic policy candidates.\n')
    table='\n'.join(f"| {p['policy_id']} | {p['hypothetical_hits']}/{p['requests']} | {p['known']} | {p['candidate_score']} | {p['new_severe']} | not measured |" for p in dev['policies'])
    (root/'runtime_cost_report.md').write_text('# CPU cost\n\nMeasured CPU-only over retained historical DEV snapshots, 5 warmup passes and 10 repeats. '
        'Parsing and Position admission are nested in validity; do not sum nested spans or extrapolate them into full M123 GPU gains.\n\n'+
        '\n'.join(f"- {g['policy_id']}: total {g['mean_total_ms']:.6f} ms/question; parsing {g['mean_parse_ms']:.6f}; validity {g['mean_validity_ms']:.6f}; Position gate {g['mean_position_admission_ms']} ms per evaluated Position gate." for g in cost['groups'])+
        '\n\nNo new M1/M2/Heads/snapshot/prefill/decode/FPS/QPS/memory measurement is available. Historical synchronized spans stay in the original v4 report.\n')
    (root/'e2e_performance_report.md').write_text('# E2E status\n\nINCOMPLETE_INDEPENDENT_VALIDATION. New full M123 GPU calls: 0; real skips: 0. '
        'The metadata audit found no eligible fresh official validation log. No new formal GPU baselines were run. '
        'Historical object3Q B→C 1.344391× and preset mixed3Q 1.116378× remain v4-only evidence; they are not measurements of this cleanup.\n\n'
        '| Historical DEV CPU policy | Hypothetical skips | Scored | Candidate answer score | New severe | New E2E |\n|---|---:|---:|---:|---:|---|\n'+table+
        '\n\nFallback uses recorded paired B answers only as a counterfactual. New B/C generation, latency and risk require a fresh frozen experiment. '
        'Natural, fixed, addressable and preset mixed workloads are not interchangeable; no natural distribution claim is made.\n')
    (root/'independent_validation_gap.md').write_text('# Independent validation gap\n\n'+
        f"Official local validation: {audit['partitions']['val']['scenes']} scenes / {audit['partitions']['val']['logs']} logs; " +
        f"compatible first8 segments: {audit['partitions']['val']['compatible_scenes']}; unused compatible validation logs: {len(audit['unused_compatible_validation_logs'])}.\n\n"+
        'The checkpoint embedded training configuration directly identifies nuScenes train and train QA. v4 physical-train scenes are therefore exposed or potentially exposed; changing a round/question order does not create independence. '
        'All historical M3/M123 metadata and request records exclude entire logs. Alternate local trainval indexes represent the same physical scenes. '
        'A new independently labelled compatible six-camera dataset/log pool and a pre-TEST DEV/protocol freeze are required. '
        'No downloads, re-training or old TEST relabelling as new TEST were performed.\n')
    answers=[
        '是。单一Runtime保留解析、当前事实、已有绑定门槛、一个Position支持度门槛、事实回答或原K8生成；历史复现分支隔离保留。',
        '是。current_only路径拒绝这两个历史风险开关；历史实验入口保留。',
        '是。current_detection_support及中心投影接口未改变，没有恢复M2全框复用资格对检测事实的过滤。',
        'v3新增左右错误dcdbd110...：真实Front-Left目标的已解码候选最高0.286888，被M1 .3过滤，当前旁路选Front-Right。另一个73c335...目标0.309866通过M1但低于M3 .35。v4的2ea0bbd8...在冻结标签下方位错误成立，但新保守实例/可见性评价为UNKNOWN，且正式帧无过滤前raw捕获，无法断言M1漏检、过滤或具体身份误判。逐例新匹配和旧全链路证据见position_risk_audit.json。',
        '当前观测、ROI、几何扇区、已有绑定风险和DEV置信度门槛通过者可候选快速回答；这仍不是可靠最近实例认证，本轮没有新独立TEST证明。',
        '新TEST未运行，不能宣称减少。历史DEV三个CPU反事实的新增严重错误见dev_policy_comparison.json；不作为新实验结论。',
        '历史DEV全部回退反事实多调用2次Position LLM，但这2条配对事实标签均UNKNOWN，因此不能判断方位事实质量提高或下降；不能把更多回退视为改善。新GPU E2E未测。',
        '实现保留；历史DEV每种策略Category候选3/16、Existence候选8/16相同。新完整E2E收益未测。',
        '本轮真实跳过0，因未启动完整模型；v4正式为77/480。CPU反事实命中不算跳过率提升。',
        '见runtime_cost_report.md及cpu_cost.json：真实CPU微测，Position gate嵌套于validity；不能外推GPU E2E。',
        f"历史v4对象三问C提案Object标签 {cov['object']['scored']}/{cov['requests']}；真正快速回答的标签分母另见nearest_target_identity_audit.json。无新独立TEST标签。",
        f"历史C提案Question Target标签 {cov['question_target']['scored']}/{cov['requests']}；这是实例正确性，不能与回答方位正确率混合。",
        f"历史回答标签原{cov['old_answer_scored']}/96，现{cov['answer_scored']}/96；新增同扇区/同类别回答标签不建立唯一实例身份，同时更严格地拒绝可见性歧义。UNKNOWN原因及获得/失去标签逐条列出，未放宽2m/.5m实例匹配标准。",
        '是。权重内嵌config和本地原始config都使用nuScenes train及train Conv/VQA。没有逐scene训练执行清单，不声称精确训练曝光可排除。',
        f"未找到：合格且历史未使用的官方val日志 {len(audit['unused_compatible_validation_logs'])}。",
        '只能支持当前事实旁路的工程实现、历史受限对象QA性能和诊断；不能支持新泛化GO或2%罕见严重风险认证。',
        '没有合格的新DEV最优配置。历史实测最佳target_dev_0未改变；三个新策略仅CPU诊断，不冻结为在线策略。',
        '新正式值null；历史v4对象三问1.344391×，不能归给本轮代码或Target Completeness。',
        '新正式值null；历史v4预设混合1.116378×，不代表真实自然QA分布。',
        '新TEST新增严重错误和CI均null，不伪报零。历史标签的配对新增/净变化、场景CI与唯一请求见severe_error_audit.json；历史v4原验收1/41和NO_GO保持不变。',
        '尚不能判定更优，没有新配置真实质量—E2E配对。历史重评分和CPU成本不能证明新Pareto优势。',
        'World-Fact-Grounded Selective Language Bypass：利用M1当前已提交结构化事实选择性省略Prefill和Decode；此前真实收益来自Current-Only机制。',
        '缺少独立模型未见TEST、充分最近目标身份标签、严格罕见严重风险样本、新DEV最佳策略及本轮完整E2E测量；不证明Temporal Risk/Completeness独立贡献。',
        '已有工程与限定对象QA的历史性能依据，尚不足作为MobiCom等会议M3核心研究结论的完整证据；需补独立TEST、任务覆盖及场景级质量—时延/严重风险证据。']
    (root/'result_answers.md').write_text('# 24项结果回答\n\n结论：INCOMPLETE_INDEPENDENT_VALIDATION；历史v4 NO_GO不变。\n\n'+
        '\n\n'.join(f'{i}. {s}' for i,s in enumerate(answers,1))+'\n')
    (root/'implementation_summary.md').write_text('# Delivery\n\nCurrent-only runtime cleanup, a single Position confidence gate, class-independent one-to-one offline matching, separate object/target/answer denominators, and checkpoint/split provenance audits are implemented. '
        'Result: INCOMPLETE_INDEPENDENT_VALIDATION. No eligible fresh validation logs; no new full M123 GPU run, zero real language skips. Historical v3/v4 NO_GO unchanged.\n\n'+
        '\n'.join(f"- {k}: {v['counts']} (exit {v['exit_code']})" for k,v in test.items())+
        '\n\nThe checkpoint metadata was read without loading tensor storages and the original file was SHA256 verified. Diagnostic policy JSONs intentionally cannot activate online bypass. '
        'CPU4/5 excluded; GPU regression uses the existing lock and waits for idleness. See result_answers.md and independent_validation_gap.md.\n')
    cmd='taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python'
    (root/'REPRODUCE.md').write_text('# Reproduce\n\nUse the existing OmniDrive environment; no install/download/training. Every command excludes faulty CPU4/5. '
        'This delivery contains no fresh formal TEST and no claim of new full-model speedup.\n\nRead-only historical integrity verification:\n\n```sh\n'+
        cmd+' -m tools.worldstream_m123.final_fact_bypass verify\n```\n\nTo regenerate diagnostics, use a NEW root. The original historical outputs must remain in place:\n\n```sh\n'+
        '\n'.join(cmd+f' -m tools.worldstream_m123.final_fact_bypass {s} --root /tmp/omnidrive-final-fact-replay' for s in ('prepare','diagnose'))+'\n'+
        cmd+' -m tools.worldstream_m123.answer_validity_validate --output /tmp/omnidrive-final-fact-replay/validation_cpu\n'+
        cmd+' -m tools.worldstream_m123.answer_validity_validate --gpu --output /tmp/omnidrive-final-fact-replay/validation_gpu\n'+
        cmd+' -m tools.worldstream_m123.final_fact_bypass report --root /tmp/omnidrive-final-fact-replay\n```\n\n'
        'CUDA regression requires host GPU access, acquires /tmp/omnidrive_m123_latency_gpu.lock, waits for other users, and never kills their processes. '
        'A dataset audit is metadata/annotation-only. Historical DEV threshold fitting is diagnostic; report CPU timings vary by system load. '
        'Fresh independent DEV/TEST and a pre-TEST measured policy freeze are required before any new formal model run. '
        'Historical v4 exact repetition still uses its frozen_sources and original REPRODUCE.md; changing current code does not overwrite those sources.\n')
