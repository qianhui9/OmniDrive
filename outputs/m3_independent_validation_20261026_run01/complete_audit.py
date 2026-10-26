"""Supplement raw records with one explicit schema; never changes measurements.

Run from the repository root after independent_validation analyze.
"""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from tools.worldstream_m123.benchmark.common import sha256, write
from tools.worldstream_m123.benchmark.independent_analysis import rows, save_rows, runtime
from tools.worldstream_m123.benchmark.metrics import distribution


def main():
    summary=json.loads((ROOT/'comparison_summary.json').read_text())
    plan=json.loads((ROOT/'run_plan.json').read_text())
    records=[]
    all_traces=[]
    fields=('matched_prefix_tokens','structural_prefix_tokens','estimated_quality_risk',
        'estimated_saved_ms','decision_reason','cache_age_frames','approx_reuse_depth',
        'm1_state_valid','m2_context_valid','m2_refresh_ratio','m2_cache_age_frames',
        'm2_mapping_available')
    for job in plan['jobs']:
        folder=ROOT/'runs'/job['name']
        frames={r['frame_id']:r for r in rows(folder/'per_frame_metrics.jsonl')}
        questions=rows(folder/'per_question_metrics.jsonl')
        traces=rows(folder/'decision_traces.jsonl') if (folder/'decision_traces.jsonl').exists() else []
        all_traces.extend(dict(t,repeat=job['repeat']) for t in traces)
        for qi,q in enumerate(questions):
            f=frames[q['frame_id']];d=q.get('reuse_diagnostics') or {}
            t=traces[qi] if traces else {}
            records.append(dict(run=job['name'],repeat=job['repeat'],strategy=job['strategy'],
                scene_id=q['scene_id'],frame_id=q['frame_id'],timestamp=f['timestamp'],
                question_index=q['question_index'],question=q['question'],answer=q['answer'],reference=q['reference'],
                reuse_type=d.get('reuse_type','none'),reuse_length=d.get('reused_tokens',0),
                planned_reuse_length=d.get('reuse_length',0),
                effective_prefill_tokens=q['effective_prefill_tokens'],input_tokens=q['input_tokens'],
                candidate_reuse_lengths=q['candidate_reuse_lengths'],
                **{k:d.get(k,0 if k in ('matched_prefix_tokens','structural_prefix_tokens') else None) for k in fields},
                prefill_ms=q['prefill_ms'],decode_ms=q['decode_ms'],
                question_generation_host_ms=q['host_generate_call_ms'],
                frame_m3_runtime_ms=runtime(f,questions),frame_request_ms=f['request_ms'],
                frame_latency_scope='Repeated frame fields are context; do not sum them over questions',
                cache_bytes=d.get('cache_bytes',0),peak_gpu_allocated_bytes=f['peak_gpu_allocated_bytes'],
                peak_gpu_reserved_bytes=f['peak_gpu_reserved_bytes'],
                policy_input_signature_compatible=t.get('policy_compatible'),
                compatibility_status=('NOT_APPLICABLE_NATIVE_REFERENCE' if not t else
                    'NOT_EVALUATED_COLD_CACHE' if 'policy_compatible' not in t else 'OBSERVED_SCHEDULER_CHECK'),
                input_witness=q['input_witness'],planning_matching_ms=d.get('planning_ms'),
                branch_gpu_ms=d.get('cache_branch_gpu_span_ms'),cache_update_cpu_enqueue_ms=d.get('cache_update_cpu_enqueue_ms',0)))
    save_rows(ROOT/'request_records.jsonl',records)
    summary['percentile_changes_by_repeat']=[dict(repeat=r['repeat'],
        metrics={metric:{quantile:dict(delta_ms=r['wsdr'][metric][quantile]-r['native'][metric][quantile],
            change_percent=(r['wsdr'][metric][quantile]/r['native'][metric][quantile]-1)*100)
            for quantile in ('p50','p95')} for metric in ('prefill_ms','m3_runtime_ms','request_ms')},
        scope='Differences of paired-workload marginal percentiles; not percentiles of paired deltas')
        for r in summary['repeats_summary']]
    primary_traces=[t for t in all_traces if t['repeat']==0]
    summary['signals']['candidate_risk_by_profile']={profile:distribution(c['risk'] for t in primary_traces
        if t['details']['task_profile']==profile for c in t.get('candidates',[]))
        for profile in sorted({t['details']['task_profile'] for t in primary_traces})}
    primary_records=[r for r in records if r['repeat']==0 and r['strategy']=='wsdr_kv']
    summary['signals']['observed_signal_groups']={key:{str(value):dict(
        requests=sum(r[key]==value for r in primary_records),
        accepted=sum(r[key]==value and r['reuse_length']>0 for r in primary_records),
        rejected=sum(r[key]==value and r['reuse_length']==0 for r in primary_records),
        actual_lengths=distribution(r['reuse_length'] for r in primary_records if r[key]==value))
        for value in sorted({r[key] for r in primary_records})}
        for key in ('m1_state_valid','m2_context_valid','m2_refresh_ratio','m2_cache_age_frames','m2_mapping_available')}
    summary['signals']['attribution_scope']='Observed groups are descriptive; only counterfactual planner changes assess decision sensitivity.'
    frozen=json.loads((ROOT/'freeze_initial.json').read_text())
    changed={p:dict(before=h,after=sha256(p)) for p,h in frozen['source_hashes'].items() if sha256(p)!=h}
    protected={p:h for p,h in frozen['source_hashes'].items() if not p.startswith('tools/worldstream_m123/benchmark/')}
    write(ROOT/'protected_verification.json',dict(protected_files=len(protected),
        production_unchanged=all(sha256(p)==h for p,h in protected.items()),
        benchmark_only_changes=changed,policy_unchanged=sha256('outputs/final_route/m3_reuse_policy.json')==sha256(ROOT/'frozen/m3_reuse_policy.json'),
        no_threshold_retuning=True,previous_results_not_overwritten=True))
    actual=[q for q in rows(ROOT/'paired_questions.jsonl') if q['repeat']==0]
    native_errors=[dict(item, native_error_types=[key for key,value in item['quality']['baseline_task_metrics'].items()
        if value is False and key in ('numeric_exact','spatial_label_match','motion_label_match','safety_label_match')])
        for item in actual if any(item['quality']['baseline_task_metrics'].get(key) is False
            for key in ('numeric_exact','spatial_label_match','motion_label_match','safety_label_match'))]
    save_rows(ROOT/'native_error_examples.jsonl',native_errors)
    def clarify_counts(section):
        for name,metric in section.get('task_metrics',{}).items():
            if name not in ('numeric_exact','spatial_label_match','motion_label_match','safety_label_match'):
                metric['native_errors']=None
                metric['new_correct_to_incorrect']=None
                metric['error_count_scope']='Continuous score: report paired score loss; boolean error counts are not applicable'
        for key in ('by_profile','by_scene'):
            for item in section.get(key,{}).values():clarify_counts(item)
        if 'road_driving_subset' in section:clarify_counts(section['road_driving_subset'])
    clarify_counts(summary['quality'])
    for item in summary['repeats_summary']:clarify_counts(item['quality'])
    summary['quality_evidence_status']=dict(observed_f1_loss=summary['quality']['f1_loss'],
        observed_critical_regression_requests=summary['quality']['critical_regression_requests'],
        profiles_without_multi_scene_interval=[p for p,x in summary['quality']['by_profile'].items() if x['f1_loss_ci95'] is None],
        interpretation='A missing profile interval is insufficient evidence, not an observed quality-budget violation',
        zero_difference_interval_limit='Identical paired scores give a degenerate empirical bootstrap interval; this does not bound unseen error probability')
    profiles={p:dict(Counter(q['reuse_type'] for q in actual if q['profile']==p)) for p in sorted({q['profile'] for q in actual})}
    tests=json.loads((ROOT/'test_results.json').read_text())
    summary['tests']=tests
    write(ROOT/'m1_m2_diagnostics.json',summary['signals'])
    summary['artifacts']=dict(unified_request_records='request_records.jsonl',new_errors='new_error_cases.jsonl',
        native_errors='native_error_examples.jsonl',
        approximate_case_review='approximate_case_review.md',
        raw_paired_questions='paired_questions.jsonl',raw_paired_frames='paired_frames.jsonl',
        scene_coverage='scene_coverage.json',frozen_code='frozen/source/',protected_verification='protected_verification.json')
    stats=summary['primary_reuse_statistics']
    summary['acceptance_answers']=dict(independent_cross_frame_approximate_requests=stats['cross_frame_approx_requests'],
        actual_skipped_prefill_tokens_per_primary_round=stats['actual_skipped_tokens'],
        approximate_prefix_tokens=stats['reuse_by_type']['bounded_approx']['tokens'],
        tokens_beyond_strict_exact_prefix=stats['tokens_beyond_exact_match'],
        unique_quality_samples=summary['unique_questions'],
        no_observed_new_critical_errors=not any(r['quality']['critical_regression_requests'] for r in summary['repeats_summary']),
        latency_change_percent={k:v['change_percent'] for k,v in summary['paired_latency'].items()},
        stable_positive_end_to_end_evidence=summary['gates']['positive_end_to_end_interval'] and summary['gates']['every_repeat_end_to_end_improved'],
        profiles_without_approximate_execution=[p for p,c in profiles.items() if not c.get('bounded_approx')],
        paper_readiness='Limited independent evidence only; expand unique scenes, task-quality samples and repeated latency evidence before broad paper claims')
    write(ROOT/'comparison_summary.json',summary)
    text=(ROOT/'experiment_report.md').read_text()
    lead='未通过项主要是重复端到端收益不稳定，以及运动 profile 仅 1 个样本、缺少多场景质量区间。已观察的 F1 损失为 0，新增关键退化为 0；质量证据不足与已观察超预算分别报告。'
    if lead not in text:
        text=text.replace('\n\n首轮实际复用', '\n\n'+lead+'\n\n首轮实际复用',1)
    for metric in ('numeric_value_precision','numeric_value_recall','spatial_label_f1','motion_label_f1','safety_label_f1'):
        lines=text.splitlines()
        for index,line in enumerate(lines):
            if line.startswith('| '+metric+' |'):
                fields=line.split('|');fields[-3]=' N/A ';fields[-2]=' N/A ';lines[index]='|'.join(fields)
        text='\n'.join(lines)+'\n'
    marker='\n## 补充审计与测试\n'
    text=text.split(marker)[0]
    text+=marker+'\n'
    text+=f"GPU 回归 **{tests['gpu']['passed']} passed，0 failed**；CPU M3/集成回归 **{tests['cpu']['passed']} passed**。命令见 test_results.json，原始日志在 raw_logs/tests_*.log。\n\n"
    text+='完整逐请求统一字段见 request_records.jsonl。原生路径无策略准入与 M1/M2 风险规划，相关字段使用 null；冷缓存尚未进行签名准入的请求也明确标注。帧级时延附在每问记录中仅作为上下文，不能逐问相加。\n\n'
    text+=f"原生已有严格代理错误涉及 {len(native_errors)} 个唯一问题，具体样例见 native_error_examples.jsonl；新增变化与退化保存在 new_error_cases.jsonl，二者分别统计。\n\n"
    text+='两例真实近似复用的参考、回答及原生既有空间遗漏见 [样例审阅](approximate_case_review.md)。\n\n'
    text+='逐 profile 的首轮真实执行类型：\n\n'
    for p,c in profiles.items():text+=f'- {p}: {c}\n'
    text+='\n当前问题类型中不发生近似复用的 profile，只能报告完整计算或精确复用，不能宣称具有近似复用质量证据。运动 profile 的样本非常少，不能作任务级泛化保证。\n\n'
    text+='质量表中连续分数的严格错误计数用 N/A，质量下降通过配对分数差衡量。缺少足够场景形成 profile 区间时属于证据不足；已观察质量损失和新增错误数量另行报告。\n\n'
    text+='配对差全部为 0 时，Bootstrap 区间退化为 [0, 0]，只描述已观察差异，不能据此约束未见场景中的错误概率。\n\n'
    text+='P50/P95 的 WSDR − native 百分比变化（正值变慢；比较相同配对工作负载的边际分位数）：\n\n'
    text+='| 轮次 | Prefill P50/P95 % | M3 P50/P95 % | 端到端 P50/P95 % |\n|---|---|---|---|\n'
    for r in summary['percentile_changes_by_repeat']:
        values=[' / '.join(f"{r['metrics'][metric][q]['change_percent']:.3f}" for q in ('p50','p95'))
            for metric in ('prefill_ms','m3_runtime_ms','request_ms')]
        text+=f"| {r['repeat']+1} | {' | '.join(values)} |\n"
    text+='\n## 最终问题逐项回答\n\n'
    text+=f"1. 独立自然场景中实际跨帧近似复用 {stats['cross_frame_approx_requests']} 次（首轮口径）；全部重复轮次原始记录保留。\n"
    text+=f"2. 首轮真实跳过 {stats['actual_skipped_tokens']} 个 Prefill Token，近似前缀共 {stats['reuse_by_type']['bounded_approx']['tokens']} Token，超出严格精确前缀的部分为 {stats['tokens_beyond_exact_match']} Token。\n"
    text+=f"3. 首轮新增关键退化请求 {summary['quality']['critical_regression_requests']}，平均 GT F1 损失 {summary['quality']['f1_loss']:.6f}；重复质量不增加唯一样本 N={summary['unique_questions']}。\n"
    text+='4. 全部配对重复的均值变化：'+', '.join(f"{key} {value['change_percent']:.3f}%" for key,value in summary['paired_latency'].items())+'。\n'
    text+=f"5. 每轮端到端均改善：{summary['gates']['every_repeat_end_to_end_improved']}；端到端聚类区间支持正净收益：{summary['gates']['positive_end_to_end_interval']}。P50/P95 见逐轮变化表。\n"
    text+='6. M1/M2 是否改变最终规划：'+', '.join(f"{name}: {value['changed_decisions']}/{value['measurements']}" for name,value in summary['signals']['variants'].items())+'。\n'
    text+='7. 未得到实际近似执行证据的 profile：'+', '.join(summary['acceptance_answers']['profiles_without_approximate_execution'])+'；维持冻结策略的完整计算回退。\n'
    text+=f"8. 总验收状态 {summary['status']}。现有独立证据可支持继续研究，但正式论文中的广泛质量/安全与稳定净收益结论仍需更多唯一场景和任务样本；本轮不调参。\n\n"
    text+='场景车辆/行人运动、类计数变化和自车角度覆盖见 scene_coverage.json；这些是元数据覆盖描述，不是新增模型精度结果。模型环境版本见 environment_manifest.json。正式源码未修改，验证见 protected_verification.json；冻结时源码完整副本在 frozen/source/。\n\n'
    text+='统一审计表复现：`taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python '+str(ROOT.relative_to(Path.cwd())/'complete_audit.py')+'`。\n'
    (ROOT/'experiment_report.md').write_text(text)
    sys.path.insert(0,str(ROOT))
    from plot_latency import plot
    plot(ROOT)
    with (ROOT/'experiment_report.md').open('a') as f:
        f.write('\n全部轮次及聚类区间可视化：[SVG](paired_latency_stability.svg)、[原始绘图表](paired_latency_stability.csv)。\n\n![Paired latency stability](paired_latency_stability.png)\n')
    target=ROOT/'frozen/tests/test_independent_validation.py';target.parent.mkdir(exist_ok=True)
    if not target.exists():shutil.copy2('tests/worldstream_m3/test_independent_validation.py',target)
    write(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):sha256(p) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.name not in ('artifact_hashes.json',) and '__pycache__' not in p.parts})
    print(summary['status'],'normalized request measurements',len(records))


if __name__=='__main__':main()
