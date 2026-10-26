"""Rebuild independent validation statistics from paired, unfiltered raw logs."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np

from .common import write
from .metrics import distribution
from .quality_calibration import paired_quality


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def save_rows(path, values):
    Path(path).write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False)+'\n' for x in values))


def cluster_interval(records, native='native', wsdr='wsdr', seed=0, draws=5000):
    """Resample whole scenes, retaining correlated frames/questions and repeats.

    Positive change = WSDR slower / greater score. Mean is observation weighted;
    repeated measurements never increase the number of independent clusters.
    """
    groups = defaultdict(list)
    for r in records: groups[r['scene_id']].append((r[native], r[wsdr]))
    if not groups:
        return dict(n=0, scenes=0, delta=None, change_percent=None, ci95_delta=None, ci95_percent=None)
    sums = np.asarray([[sum(a for a,b in v), sum(b for a,b in v), len(v)] for v in groups.values()])
    a,b,n = sums.sum(axis=0)
    result = dict(n=int(n), scenes=len(groups), native_mean=float(a/n), wsdr_mean=float(b/n),
        delta=float((b-a)/n), change_percent=float((b/a-1)*100) if a else None,
        ci95_delta=None, ci95_percent=None, method='Paired scene cluster percentile bootstrap; 5000 draws',
        sign='wsdr - native; negative latency change is an improvement')
    if len(groups) >= 2:
        rng = np.random.RandomState(seed)
        samples = sums[rng.randint(len(sums), size=(draws, len(sums)))].sum(axis=1)
        delta = (samples[:,1]-samples[:,0])/samples[:,2]
        result['ci95_delta'] = np.percentile(delta, [2.5,97.5]).tolist()
        if np.all(samples[:,0] > 0):
            result['ci95_percent'] = np.percentile((samples[:,1]/samples[:,0]-1)*100, [2.5,97.5]).tolist()
    return result


def unique_index(records, keys):
    result = {}
    for r in records:
        key = tuple(r[k] for k in keys)
        if key in result: raise ValueError('Duplicate raw measurement: '+repr(key))
        result[key] = r
    return result


def runtime(frame, questions):
    if frame.get('m3'):
        return frame['m3']['runtime_host_elapsed_ms']
    return sum(q['host_generate_call_ms'] for q in questions if q['frame_id'] == frame['frame_id'])


def validate_actual(q):
    d = q.get('reuse_diagnostics') or {}
    if not d: return
    actual = d['reused_tokens']
    if actual + d['effective_prefill_tokens'] != d['input_tokens']:
        raise ValueError('Actual prefill token accounting mismatch')
    if sum(d['forward_prefill_lengths']) != d['effective_prefill_tokens']:
        raise ValueError('Actual model forward lengths disagree')
    if not d['prefill_completed'] or not d['decode_completed']:
        raise ValueError('Request did not complete prefill/decode')
    if d['reuse_type'] == 'bounded_approx' and actual <= 0:
        raise ValueError('Approximate decision skipped no actual computation')


def run_statistics(frames, questions):
    details = [q['reuse_diagnostics'] for q in questions if q.get('reuse_diagnostics')]
    for q in questions: validate_actual(q)
    skipped = sum(d['reused_tokens'] for d in details)
    return dict(frames=len(frames), question_measurements=len(questions),
        prefill_ms=distribution(q['prefill_ms'] for q in questions),
        decode_ms=distribution(q['decode_ms'] for q in questions),
        m3_runtime_ms=distribution(runtime(f, questions) for f in frames),
        request_ms=distribution(f['request_ms'] for f in frames),
        throughput_frames_per_second=1000*len(frames)/sum(f['request_ms'] for f in frames),
        input_tokens=sum(q['input_tokens'] for q in questions),
        actual_prefill_tokens=sum(q['effective_prefill_tokens'] for q in questions),
        actual_skipped_tokens=skipped,
        actual_reuse_requests=sum(d['reused_tokens'] > 0 for d in details),
        actual_reuse_request_fraction=sum(d['reused_tokens'] > 0 for d in details)/len(questions),
        reuse_by_type={kind:dict(requests=sum(d['reuse_type']==kind and d['reused_tokens']>0 for d in details),
            tokens=sum(d['reused_tokens'] for d in details if d['reuse_type']==kind))
            for kind in ('exact','bounded_approx')},
        cross_frame_approx_requests=sum(d['reuse_type']=='bounded_approx' and d['reused_tokens']>0
            and d['cache_source_frame']!=q['frame_id'] for q in questions
            for d in [q.get('reuse_diagnostics') or {}] if d),
        candidate_requests=sum(bool(d.get('candidate_prefix_tokens')) for d in details),
        tokens_beyond_exact_match=sum(max(0,d['reused_tokens']-d['matched_prefix_tokens']) for d in details),
        planning_matching_ms=distribution(d['planning_ms'] for d in details),
        cache_branch_gpu_ms=distribution(d.get('cache_branch_gpu_span_ms') or 0 for d in details),
        cache_update_cpu_enqueue_ms=distribution(d.get('cache_update_cpu_enqueue_ms',0) for d in details),
        cache_update_scope='CPU enqueue only; GPU clone/update included in measured prefill, not separately isolated',
        cache_bytes_max=max((d['cache_bytes'] for d in details),default=0),
        peak_allocated_bytes=max(f['peak_gpu_allocated_bytes'] for f in frames),
        peak_reserved_bytes=max(f['peak_gpu_reserved_bytes'] for f in frames))


def quality_statistics(items):
    if not items: return dict(unique_questions=0, f1_loss=None)
    records=[dict(scene_id=i['scene_id'], native=i['quality']['baseline_f1'], wsdr=i['quality']['f1']) for i in items]
    ci=cluster_interval(records)
    metrics={}
    keys=('numeric_exact','numeric_value_precision','numeric_value_recall',
        'spatial_label_match','spatial_label_f1','motion_label_match','motion_label_f1',
        'safety_label_match','safety_label_f1')
    for key in keys:
        applicable=[i for i in items if i['quality']['baseline_task_metrics'].get(key) is not None]
        metric_records=[dict(scene_id=i['scene_id'], native=float(i['quality']['baseline_task_metrics'][key]),
            wsdr=float(i['quality']['task_metrics'][key])) for i in applicable]
        result=cluster_interval(metric_records)
        metrics[key]=dict(native_errors=sum(i['quality']['baseline_task_metrics'][key] is False for i in applicable),
            new_correct_to_incorrect=sum(i['quality']['baseline_task_metrics'][key] is True
                and i['quality']['task_metrics'][key] is False for i in applicable), **result)
    loss_ci=[-ci['ci95_delta'][1],-ci['ci95_delta'][0]] if ci['ci95_delta'] else None
    return dict(unique_questions=len(items), scenes=ci['scenes'],
        native_gt_token_f1=ci['native_mean'], wsdr_gt_token_f1=ci['wsdr_mean'],
        f1_loss=-ci['delta'], f1_loss_ci95=loss_ci,
        answer_equal=sum(i['quality']['answer_equal'] for i in items),
        answer_equal_rate=sum(i['quality']['answer_equal'] for i in items)/len(items),
        output_token_f1=float(np.mean([i['quality']['output_token_f1'] for i in items])),
        token_ids_equal=sum(i['token_equal'] for i in items),
        critical_regression_requests=sum(i['quality']['critical_regressions']>0 for i in items),
        critical_regression_checks=sum(i['quality']['critical_regressions'] for i in items),
        task_metrics=metrics, scope='GT lexical/number extraction proxies; agreement is not absolute semantic correctness')


def analyze(root):
    from .independent_validation import verify_freeze
    from .decision_replay import compare_trace
    root=Path(root)
    frozen=verify_freeze(root)
    sequence=json.loads((root/'sequence.json').read_text())
    plan=json.loads((root/'run_plan.json').read_text())
    manifest=json.loads((root/'data_manifest.json').read_text())
    expected_questions=[(f,q) for f in sequence['tokens'] for q in sequence['questions'][f]]
    runs={}; traces=[]
    for job in plan['jobs']:
        folder=root/'runs'/job['name']
        status=json.loads((folder/'status.json').read_text())
        if status.get('status')!='COMPLETED' or status.get('integrity')!='PASS':
            raise ValueError('Incomplete or invalid worker retained: '+job['name'])
        f=rows(folder/'per_frame_metrics.jsonl'); q=rows(folder/'per_question_metrics.jsonl')
        if [r['frame_id'] for r in f]!=sequence['tokens']:
            raise ValueError('Missing/reordered frames')
        if [(r['frame_id'],r['question']) for r in q]!=expected_questions:
            raise ValueError('Missing/reordered questions')
        if any(r['reference'] is None for r in q): raise ValueError('Missing GT')
        if not all(r['native_generation_executed'] and not r['prefill_executor_executed'] for r in q) and job['strategy']=='native_full':
            raise ValueError('Native reference did not use native full generation')
        if job['strategy']=='wsdr_kv':
            if not all(r['prefill_executor_executed'] and not r['native_generation_executed'] for r in q):
                raise ValueError('WSDR bypassed PrefillExecutor')
            t=rows(folder/'decision_traces.jsonl')
            if len(t)!=len(q): raise ValueError('Missing per-request planner traces')
            traces.extend(dict(ti, repeat=job['repeat']) for ti in t)
        runs[job['name']]=dict(frames=f, questions=q, statistics=run_statistics(f,q))
    all_quality=[]; pairs=[]; repeat_summary=[]
    for repeat in range(plan['repeats']):
        native=runs[f'r{repeat}_native_full']; wsdr=runs[f'r{repeat}_wsdr_kv']
        frame_records=[]; question_records=[]
        for a,b in zip(native['frames'],wsdr['frames']):
            if a['seed']!=b['seed']: raise ValueError('Unpaired seeds')
            rec=dict(scene_id=a['scene_id'],frame_id=a['frame_id'],repeat=repeat,
                native_request_ms=a['request_ms'],wsdr_request_ms=b['request_ms'],
                native_m3_ms=runtime(a,native['questions']),wsdr_m3_ms=runtime(b,wsdr['questions']),
                perception_digest_equal=a['output_digest']==b['output_digest'])
            frame_records.append(rec)
        for a,b in zip(native['questions'],wsdr['questions']):
            if a['reference']!=b['reference'] or a['input_witness']!=b['input_witness']:
                raise ValueError('Paired native/WSDR inputs or generation differ: '+a['frame_id'])
            d=b['reuse_diagnostics']
            item=dict(scene_id=a['scene_id'],frame_id=a['frame_id'],question=a['question'],repeat=repeat,
                profile=d['task_profile'], reuse_type=d['reuse_type'],
                native_answer=a['answer'],wsdr_answer=b['answer'], reference=a['reference'],
                quality=paired_quality(b['answer'],a['answer'],a['reference']),
                token_equal=a['token_ids']==b['token_ids'],actual_reuse_length=d['reused_tokens'],
                cache_age_frames=d['cache_age_frames'],approx_reuse_depth=d['approx_reuse_depth'],
                cache_source_frame=d['cache_source_frame'], risk=d['estimated_quality_risk'],
                native_prefill_ms=a['prefill_ms'],wsdr_prefill_ms=b['prefill_ms'])
            question_records.append(item)
        all_quality.extend(question_records);pairs.extend(frame_records)
        comparisons=dict(prefill=cluster_interval(question_records,'native_prefill_ms','wsdr_prefill_ms'),
            m3=cluster_interval(frame_records,'native_m3_ms','wsdr_m3_ms'),
            end_to_end=cluster_interval(frame_records,'native_request_ms','wsdr_request_ms'))
        repeat_summary.append(dict(repeat=repeat,native=native['statistics'],wsdr=wsdr['statistics'],
            paired_latency=comparisons, quality=quality_statistics(question_records)))
    save_rows(root/'paired_questions.jsonl',all_quality)
    save_rows(root/'paired_frames.jsonl',pairs)
    primary=[i for i in all_quality if i['repeat']==0]
    quality=quality_statistics(primary)
    quality['by_profile']={p:quality_statistics([i for i in primary if i['profile']==p]) for p in sorted({i['profile'] for i in primary})}
    quality['by_scene']={s:quality_statistics([i for i in primary if i['scene_id']==s]) for s in sorted(sequence['scene_splits'])}
    road=[i for i in primary if re.search(r'\b(road|lane|traffic|driv|safe|collision|intersection)',i['question']+' '+i['reference'],re.I)]
    quality['road_driving_subset']=quality_statistics(road)
    quality['road_driving_subset']['semantic_correctness']='NOT_HUMAN_ADJUDICATED; token/lexical proxies only'
    quality['repeated_quality_scope']='Round 0 = primary unique quality set. Repeats are paired stability checks, not additional independent questions.'
    by_question=defaultdict(list)
    for i in all_quality: by_question[(i['frame_id'],i['question'])].append(i)
    quality['across_repeat_native_output_changes']=sum(len({i['native_answer'] for i in v})>1 for v in by_question.values())
    quality['across_repeat_wsdr_output_changes']=sum(len({i['wsdr_answer'] for i in v})>1 for v in by_question.values())
    errors=[i for i in all_quality if i['quality']['critical_regressions'] or i['quality']['f1_loss']>0 or not i['quality']['answer_equal']]
    save_rows(root/'new_error_cases.jsonl',errors)
    latency=dict(prefill=cluster_interval(all_quality,'native_prefill_ms','wsdr_prefill_ms'),
        m3=cluster_interval(pairs,'native_m3_ms','wsdr_m3_ms'),
        end_to_end=cluster_interval(pairs,'native_request_ms','wsdr_request_ms'))
    replays=[dict(compare_trace(t,root/'frozen/m3_reuse_policy.json'),repeat=t['repeat']) for t in traces]
    save_rows(root/'decision_replays.jsonl',replays)
    details=[q['reuse_diagnostics'] for job in plan['jobs'] if job['strategy']=='wsdr_kv'
        for q in runs[job['name']]['questions']]
    primary_details=[q['reuse_diagnostics'] for q in runs['r0_wsdr_kv']['questions']]
    signals=dict(measurements=len(details), unique_questions=len(primary_details),
        m1_validity=dict(Counter(str(d['m1_state_valid']) for d in primary_details)),
        m2_validity=dict(Counter(str(d['m2_context_valid']) for d in primary_details)),
        m2_refresh_ratio=distribution(d['m2_refresh_ratio'] for d in primary_details),
        m2_cache_age_frames=distribution(d['m2_cache_age_frames'] for d in primary_details),
        m2_mapping_available=dict(Counter(str(d['m2_mapping_available']) for d in primary_details)),
        actual_decision_reasons=dict(Counter(d['decision_reason'] for d in primary_details)),
        replay_baseline_mismatches=sum(not r['baseline_reproduced'] for r in replays),
        by_profile={p:dict(risk=distribution(d['estimated_quality_risk'] for d in primary_details if d['task_profile']==p),
            actual_reuse_length=distribution(d['reused_tokens'] for d in primary_details if d['task_profile']==p))
            for p in sorted({d['task_profile'] for d in primary_details})},
        variants={})
    for variant in ('without_m1','without_m2'):
        valid=[r for r in replays if r['baseline_reproduced']]
        signals['variants'][variant]=dict(
            semantics='Neutralize M1 validity gate' if variant=='without_m1' else 'Neutral M2 validity, zero refresh/age',
            measurements=len(valid),
            changed_decisions=sum(r['actual']['decision']['reuse_length']!=r[variant]['decision']['reuse_length'] for r in valid),
            accepted_to_rejected=sum(r['actual']['decision']['reuse_length']>0 and r[variant]['decision']['reuse_length']==0 for r in valid),
            rejected_to_accepted=sum(r['actual']['decision']['reuse_length']==0 and r[variant]['decision']['reuse_length']>0 for r in valid),
            token_delta=sum(r[variant]['decision']['reuse_length']-r['actual']['decision']['reuse_length'] for r in valid),
            reason_changes=sum(r['actual']['decision']['decision_reason']!=r[variant]['decision']['decision_reason'] for r in valid),
            candidate_count_changes=sum(len(r['actual']['candidate_prefix_tokens'])!=len(r[variant]['candidate_prefix_tokens']) for r in valid),
            by_profile={p:dict(n=sum(r['profile']==p for r in valid),
                changed=sum(r['profile']==p and r['actual']['decision']['reuse_length']!=r[variant]['decision']['reuse_length'] for r in valid))
                for p in sorted({r['profile'] for r in valid})})
    save_rows(root/'signal_request_rows.jsonl',[dict(frame_id=q['frame_id'],question=q['question'],
        **{k:d.get(k) for k in ('task_profile','m1_state_valid','m2_context_valid','m2_refresh_ratio',
            'm2_cache_age_frames','m2_mapping_available','estimated_quality_risk','reused_tokens','decision_reason')})
        for q in runs['r0_wsdr_kv']['questions'] for d in [q['reuse_diagnostics']]])
    write(root/'m1_m2_diagnostics.json',signals)
    first=runs['r0_wsdr_kv']['statistics']
    quality_gate=(quality['f1_loss_ci95'] is not None and quality['f1_loss_ci95'][1]<=frozen['quality_budget']
        and all(p['f1_loss_ci95'] is not None and p['f1_loss_ci95'][1]<=frozen['quality_budget'] for p in quality['by_profile'].values())
        and not any(i['quality']['critical_regressions'] for i in all_quality))
    ci=latency['end_to_end']['ci95_delta']
    stable=all(r['paired_latency']['end_to_end']['delta']<0 for r in repeat_summary)
    gates=dict(independent_scenes=manifest['actual_scenes']>=2, frozen_policy=True,
        actual_cross_frame_approximation=first['cross_frame_approx_requests']>0,
        empirical_quality_budget=quality_gate,
        positive_end_to_end_interval=ci is not None and ci[1]<0,
        every_repeat_end_to_end_improved=stable,
        exact_generated_token_agreement=all(i['token_equal'] for i in all_quality if i['reuse_type']=='exact'),
        planner_replay_reproduced=signals['replay_baseline_mismatches']==0)
    summary=dict(status='VALIDATED_ON_THIS_SAMPLE' if all(gates.values()) else 'NOT_YET_VALIDATED',
        gates=gates, unique_frames=len(sequence['tokens']),unique_questions=len(primary),
        independent_scenes=manifest['actual_scenes'], repeats=plan['repeats'],
        repeats_summary=repeat_summary,paired_latency=latency,quality=quality,signals=signals,
        primary_reuse_statistics=first, all_input_witnesses_equal=True,
        perception_digest_mismatches=sum(not p['perception_digest_equal'] for p in pairs),
        limitations=['Few scene clusters; percentile intervals are descriptive and do not establish population safety.',
            'Only short natural consecutive scene segments; no claim of complete-scene evaluation.',
            'Quality labels are lexical proxies, not human semantic correctness.',
            'M3 native timing is summed native generation wall per frame; WSDR timing is whole runtime wall.',
            'Read-only tracing adds scalar metadata bookkeeping to WSDR measurement; no extra model forwards.',
            'Offline counterfactuals change planner signals only, with fixed observed overhead; not accuracy/speed evidence.',
            'Non-reuse requests and every repeat retained; repeated questions do not multiply quality N.'])
    write(root/'comparison_summary.json',summary)
    render_report(root,summary)
    print(summary['status'], 'E2E change %',latency['end_to_end']['change_percent'],flush=True)
    return summary


def fmt(v):
    return 'N/A' if v is None else f'{v:.3f}'


def render_report(root,s):
    first=s['repeats_summary'][0]; native=first['native']; wsdr=first['wsdr']
    lines=['# 冻结 WSDR-KV 独立场景验证', '',
        f"结论：**{s['status']}**。{s['independent_scenes']} 个独立 M3 场景、{s['unique_frames']} 帧、"
        f"{s['unique_questions']} 个唯一问题，两个正式路径各 {s['repeats']} 轮。重复测量不扩大质量样本量。",'',
        f"首轮实际复用 {wsdr['actual_reuse_requests']} 请求、{wsdr['actual_skipped_tokens']} token；"
        f"精确 {wsdr['reuse_by_type']['exact']['requests']} 请求 / {wsdr['reuse_by_type']['exact']['tokens']} token；"
        f"近似 {wsdr['reuse_by_type']['bounded_approx']['requests']} 请求 / {wsdr['reuse_by_type']['bounded_approx']['tokens']} token；"
        f"跨帧近似 {wsdr['cross_frame_approx_requests']} 次。候选长度不计入实际收益。",'',
        '## 配对重复测量', '',
        '变化定义为 WSDR − native，负时延变化表示改善。Prefill/Decode 单位为每问题；M3 与完整请求为每帧。', '',
        '| 轮次 | Prefill 变化 % | M3 变化 % | 端到端变化 % | native / WSDR 帧时延均值 ms |',
        '|---|---:|---:|---:|---:|']
    for r in s['repeats_summary']:
        p=r['paired_latency']
        lines.append(f"| {r['repeat']+1} | {fmt(p['prefill']['change_percent'])} | {fmt(p['m3']['change_percent'])} | "
            f"{fmt(p['end_to_end']['change_percent'])} | {fmt(r['native']['request_ms']['mean'])} / {fmt(r['wsdr']['request_ms']['mean'])} |")
    lines+=['','场景聚类 Bootstrap（保留同场景所有连续帧及重复测量，5000 次）：','']
    for key,label in [('prefill','Prefill/问'),('m3','M3/帧'),('end_to_end','端到端/帧')]:
        p=s['paired_latency'][key]
        lines.append(f"- {label}：配对均值差 {fmt(p['delta'])} ms，变化 {fmt(p['change_percent'])}%；95% CI ms {p['ci95_delta']}，百分比 {p['ci95_percent']}。")
    lines+=['','| 轮次 / 路径 | Prefill mean/P50/P95 ms | M3 mean/P50/P95 ms | 完整请求 mean/P50/P95 ms | 帧/s | 峰值 allocated GiB |',
        '|---|---|---|---|---:|---:|']
    for r in s['repeats_summary']:
        for name in ('native','wsdr'):
            a=r[name]
            spans=[' / '.join(fmt(a[k][x]) for x in ('mean','p50','p95')) for k in ('prefill_ms','m3_runtime_ms','request_ms')]
            lines.append(f"| {r['repeat']+1} {name} | {' | '.join(spans)} | {fmt(a['throughput_frames_per_second'])} | {fmt(a['peak_allocated_bytes']/1024**3)} |")
    q=s['quality']
    lines+=['','## 独立质量与新增错误','',
        f"质量损失定义 native_score − wsdr_score；首轮 GT Token F1 {fmt(q['native_gt_token_f1'])} / {fmt(q['wsdr_gt_token_f1'])}，"
        f"损失 {fmt(q['f1_loss'])}，场景聚类 95% CI {q['f1_loss_ci95']}。答案一致 {q['answer_equal']}/{q['unique_questions']}。"
        f"新增关键退化请求 {q['critical_regression_requests']}；各轮具体输出及错误见 paired_questions.jsonl、new_error_cases.jsonl。",'',
        '| profile | 唯一问题 | F1 损失 | 95% CI | 新增关键退化请求 |','|---|---:|---:|---|---:|']
    for p,a in q['by_profile'].items():
        lines.append(f"| {p} | {a['unique_questions']} | {fmt(a['f1_loss'])} | {a['f1_loss_ci95']} | {a['critical_regression_requests']} |")
    lines+=['','| 任务代理 | n | native / WSDR 均值 | native 已有错误 | 新增 correct→incorrect |','|---|---:|---|---:|---:|']
    for key,a in q['task_metrics'].items():
        lines.append(f"| {key} | {a['n']} | {fmt(a.get('native_mean'))} / {fmt(a.get('wsdr_mean'))} | {a['native_errors']} | {a['new_correct_to_incorrect']} |")
    lines+=['','这些数字/空间/运动/安全标签是明确的词法代理。输出一致不代表任务绝对正确，空分母不是零误差。'
        '道路驾驶子集指标及逐场景质量差见 comparison_summary.json；未进行人工完整语义判定。', '',
        '## M1/M2 信号与离线回放','',
        f"首轮 M1 有效性 {s['signals']['m1_validity']}；M2 有效性 {s['signals']['m2_validity']}；"
        f"M2 刷新比例 {s['signals']['m2_refresh_ratio']}；mapping_available {s['signals']['m2_mapping_available']}。",'']
    for name,a in s['signals']['variants'].items():
        lines.append(f"- {name}：{a['measurements']} 个重复测量，改变决策 {a['changed_decisions']}，"
            f"拒绝→接受 {a['rejected_to_accepted']}，接受→拒绝 {a['accepted_to_rejected']}，规划 token 差 {a['token_delta']}。")
    lines+=['',f"真实决策回放不一致 {s['signals']['replay_baseline_mismatches']}。离线回放仅说明规划敏感性，不能解释成真实精度或加速收益。"
        '刷新长期为 1 或缺少区域映射时，不能宣称 M2 区域映射改善了 M3。','',
        '## 冻结、可复现性与验收','',
        '策略、模型 SHA-256、Git commit、原配置和代码版本见 freeze_initial.json、freeze_execution.json；'
        '数据来源和场景隔离见 historical_data_audit.json、data_manifest.json、sequence.json。'
        '每轮 GPU 状态、环境、顺序及真实命令见 commands.json、run_plan.json，原始失败日志保留在 raw_logs/。','',
        '| 验收项 | 结果 |','|---|---|']
    lines += [f'| {k} | {v} |' for k,v in s['gates'].items()]
    lines+=['','不能以部分 Prefill 节省替代稳定端到端净收益。未通过项应作为下一阶段问题，禁止在本独立集上调阈值再宣布通过。'
        '未校准/被禁用的任务与超出成本输入范围的请求继续完整计算；各 profile 实际复用数量见原始日志。','',
        '复现（必须使用新的目录准备和运行；analyze 只重建派生统计）：','',
        '```bash',
        'taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.benchmark.independent_validation prepare --run-dir outputs/NEW_UNUSED_RUN',
        'taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.benchmark.independent_validation run --run-dir outputs/NEW_UNUSED_RUN',
        f'taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.benchmark.independent_validation analyze --run-dir {root}',
        '```','', '限制：','']+[f'- {x}' for x in s['limitations']]
    (root/'experiment_report.md').write_text('\n'.join(lines)+'\n')
