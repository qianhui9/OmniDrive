"""Report measured v5 results, preserving failed requests in quality counts."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .audit import write
from .streaming_replay import distribution
from .v3_costs import upper95
from .v5 import ROOT, sha256


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def mean(values):
    values=[x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def fmt(value, digits=2):
    return f'{value:.{digits}f}' if value is not None else 'n/a'


def paired_scenes(pairs,reference_ms,candidate_ms):
    groups=defaultdict(list)
    for reference,candidate in pairs:groups[candidate.get('scene',candidate.get('dataset_identity',{}).get('scene_token'))].append((reference,candidate))
    scene_rows=[]
    for scene,values in sorted(groups.items()):
        a=mean([reference_ms(x) for x,y in values]);b=mean([candidate_ms(y) for x,y in values])
        scene_rows.append(dict(scene=scene,frames=len(values),reference_ms=a,candidate_ms=b,
                               increment_ms=b-a,speedup=a/b if b else None))
    deltas=[r['increment_ms'] for r in scene_rows]
    return dict(scene_pairs=scene_rows,scenes=len(scene_rows),paired_frames=len(pairs),
        scene_mean_increment_ms=mean(deltas),scene_increment_upper95_ms=upper95(deltas),
        mean_scene_speedup=mean([r['speedup'] for r in scene_rows]))


def compact_report():
    rows=read_rows(ROOT/'traces/compact_frames.jsonl')
    protocol=json.loads((ROOT/'experiment_protocol.json').read_text())
    expected=sum(len(s['tokens']) for specs in protocol['selections'].values() for s in specs)*3
    if len(rows)!=expected:raise ValueError(f'Incomplete compact experiment: {len(rows)} / {expected}')
    reference={(r['split'],r['sample_token']):r for r in rows if r['mode']=='full'}
    summaries={};quality={};pair_results={}
    for split in protocol['selections']:
        summaries[split]={};quality[split]={};pair_results[split]={}
        for mode in protocol['modes']:
            values=[r for r in rows if r['split']==split and r['mode']==mode]
            complete=[r for r in values if r['complete_vlm_timing']]
            parsed=[r for r in values if r['planning']['status']=='parsed']
            def stages(row,stage,field):return sum(s[field] for s in row['stages'] if s['stage']==stage)
            summaries[split][mode]=dict(requests=len(values),complete_vlm_requests=len(complete),
                failed_requests=len(values)-len(complete),
                input_tokens=distribution([r['input_tokens'] for r in values]),
                effective_multimodal_tokens=distribution([r['effective_multimodal_tokens'] for r in values]),
                context_construction_ms=distribution([r['context_construction_ms'] for r in values]),
                e2e_complete_only_ms=distribution([r['e2e_wall_ms'] for r in complete]),
                prefill_wall_ms=distribution([stages(r,'prefill','wall_ms') for r in complete]),
                prefill_gpu_ms=distribution([stages(r,'prefill','gpu_ms') for r in complete]),
                decode_wall_ms=distribution([stages(r,'decode','wall_ms') for r in complete]),
                decode_gpu_ms=distribution([stages(r,'decode','gpu_ms') for r in complete]),
                generated_tokens=distribution([len(r['generated_ids'][0]) for r in complete]),
                peak_gpu_bytes=distribution([r['peak_gpu_bytes'] for r in values]),
                cpu_rss_highwater_bytes=distribution([r['cpu_rss_bytes'] for r in values]),
                cpu_memory_scope='Process lifetime ru_maxrss highwater, not per-method incremental memory',
                planning_parsed=len(parsed),invalid_trajectory_rate=(len(values)-len(parsed))/len(values),
                planning_ade_m=mean([r['planning'].get('ade_m') for r in parsed]),
                planning_fde_m=mean([r['planning'].get('fde_m') for r in parsed]),
                failure_reasons=dict(Counter(r['failure']['type'] for r in values if r['failure'])),
                timing_scope='Real GPU instrumented offline full VLM; conditional latency excludes failed requests explicitly counted above')
            if mode=='full':continue
            pairs=[(reference[(split,r['sample_token'])],r) for r in values]
            scene_quality=[]
            for scene in sorted({r['scene'] for r in values}):
                subset=[(a,b) for a,b in pairs if b['scene']==scene]
                scored=[(a,b) for a,b in subset if a['planning'].get('ade_m') is not None and b['planning'].get('ade_m') is not None]
                fde_pairs=[(a,b) for a,b in subset if a['planning'].get('fde_m') is not None and b['planning'].get('fde_m') is not None]
                ade=mean([b['planning']['ade_m']-a['planning']['ade_m'] for a,b in scored])
                fde=mean([b['planning']['fde_m']-a['planning']['fde_m'] for a,b in fde_pairs])
                baseline_invalid=sum(a['planning']['status']!='parsed' for a,b in subset)
                candidate_invalid=sum(b['planning']['status']!='parsed' for a,b in subset)
                failures=sum(b['failure'] is not None for a,b in subset)
                passed=bool(ade is not None and fde is not None and ade<=0 and fde<=0 and
                    candidate_invalid<=baseline_invalid and failures==0)
                scene_quality.append(dict(scene=scene,requests=len(subset),paired_ade_scorable=len(scored),
                    paired_fde_scorable=len(fde_pairs),mean_ade_delta_m=ade,mean_fde_delta_m=fde,
                    baseline_invalid=baseline_invalid,candidate_invalid=candidate_invalid,failures=failures,passed=passed))
            quality[split][mode]=dict(quality_pass=all(r['passed'] for r in scene_quality),
                scenes=scene_quality,token_equality=sum(a['generated_ids']==b['generated_ids'] for a,b in pairs),
                token_equality_scope='diagnostic only; approximate inputs may produce different valid answers')
            complete_pairs=[(a,b) for a,b in pairs if a['complete_vlm_timing'] and b['complete_vlm_timing']]
            result=paired_scenes(complete_pairs,lambda r:r['e2e_wall_ms'],lambda r:r['e2e_wall_ms'])
            result['effective_token_reduction']=distribution([a['effective_multimodal_tokens']-b['effective_multimodal_tokens'] for a,b in pairs])
            result['prefill_gpu_delta_ms']=distribution([stages(b,'prefill','gpu_ms')-stages(a,'prefill','gpu_ms') for a,b in complete_pairs])
            result['decode_gpu_delta_ms']=distribution([stages(b,'decode','gpu_ms')-stages(a,'decode','gpu_ms') for a,b in complete_pairs])
            result['all_requests_complete']=len(complete_pairs)==len(pairs)
            result['quality_constrained_speedup_established']=bool(quality[split][mode]['quality_pass'] and
                result['all_requests_complete'] and result['scene_increment_upper95_ms'] is not None and
                result['scene_increment_upper95_ms']<0 and all(a['effective_multimodal_tokens']>b['effective_multimodal_tokens'] for a,b in pairs))
            result['speedup_requires_quality_pass']=True
            pair_results[split][mode]=result
    write(ROOT/'compact_llm_latency.json',dict(summary=summaries,paired=pair_results,
        latency_not_inferred_from_token_ratios=True,qa_quality=None,
        prior_detailed_format=dict(status='actual_gpu_oom_first_calibration_candidate' if (ROOT/'logs/compact_gpu_attempt02.log').exists() else 'Not Run in this reproduction',
            evidence='logs/compact_gpu_attempt02.log' if (ROOT/'logs/compact_gpu_attempt02.log').exists() else None,online_approved=False),
        representation_amendment='compact_representation_amendment.json'))
    write(ROOT/'compact_planning_quality.json',dict(by_split=quality,frozen_quality=protocol['compact_quality'],
        no_gt_in_inputs=True,invalid_outputs_remain_in_denominator=True,trajectory_parser='original tools.worldstream_m3.qa_analysis.planning_quality'))
    passed=all(pair_results[s]['current_compact']['quality_constrained_speedup_established'] for s in ('heldout','final_test'))
    write(ROOT/'compact_input_feasibility.json',dict(status='completed_real_gpu_independent_scene_probe',
        frames=expected//3,scenes=18,methods=protocol['modes'],online_approved=False,
        current_compact_quality_and_net_cost_pass=passed,
        decision='Feasibility positive; deployment requires separate serving certificate' if passed else 'No-Go for frozen current compact representation',
        input_audit=dict(object_carriers='256 learned VLM queries, distinct from detection object queries; no guaranteed object-to-token mapping',
            ego_carrier='one can_bus embedding appended by detection head',map_carriers='256 learned VLM queries, distinct from decoded lane queries',
            full_carriers=513,insertion='IMAGE_TOKEN_INDEX=-200 at original prompt image placeholder',
            effective_tokens='Original text length minus one sentinel plus 513; compact uses native text positions, compact+full expands the retained sentinel',
            attention='Native causal attention/position generation; tokenizer and weights unchanged',
            map_fields='Decoded lane control points and scores, not complete topology, markings, sign or traffic light state',
            current_compact='Lossy fixed class/xy/lane-first-mid-last/ego format; new format comprehension not assumed'),
        oracle_probe=dict(status='Not Run',reason='No GT capability probe needed to decide measured online-input candidate; GT excluded from runtime and formal quality decisions')))
    failures=[]
    for r in rows:
        if r['mode']=='full':continue
        ref=reference[(r['split'],r['sample_token'])]
        if r['planning']['status']!='parsed' or (r['planning'].get('ade_m') is not None and ref['planning'].get('ade_m') is not None and r['planning']['ade_m']>ref['planning']['ade_m']):
            failures.append(dict(split=r['split'],scene=r['scene'],sample_token=r['sample_token'],mode=r['mode'],
                baseline_answer=ref['answer'],candidate_answer=r['answer'],baseline_planning=ref['planning'],
                candidate_planning=r['planning'],failure=r['failure'],context=r['context']))
    write(ROOT/'compact_failure_cases.json',dict(scope='All observed failures; not additional independent samples',cases=failures))
    return summaries,quality,pair_results,passed


def system_report():
    if json.loads((ROOT/'system_status.json').read_text())['status']!='completed':raise ValueError('Incomplete system run')
    rows=[]
    for path in sorted(ROOT.glob('final_test_*/frames.jsonl')):rows.extend(read_rows(path))
    refs={r['dataset_identity']['sample_token']:r for r in rows if r['mode']=='m1'}
    summary={};paired={}
    for mode in ('original','m1','overlap','unified_auto'):
        values=[r for r in rows if r['mode']==mode]
        summary[mode]=dict(frames=len(values),service='current_output' if mode=='original' else 'persistent_state',
            e2e_ms=distribution([r['frame']['e2e_wall_ms'] for r in values]),
            lookup_ms=distribution([r['frame']['admission']['lookup_wall_ms'] for r in values]),
            overlap_frames=sum(bool(r['frame']['m1'].get('overlap_active')) for r in values),
            admission_reasons=dict(Counter(r['frame']['admission']['reason'] for r in values)),
            tracked_cache_bytes=distribution([r['frame']['memory']['tracked_cache_bytes'] for r in values]),
            peak_gpu_allocated_bytes=distribution([r['frame']['memory']['gpu_peak_allocated'] for r in values]),
            cpu_rss_highwater_bytes=distribution([r['frame']['memory']['cpu_rss_highwater_bytes'] for r in values]))
        if mode in ('overlap','unified_auto'):
            pairs=[(refs[r['dataset_identity']['sample_token']],r) for r in values]
            paired[mode]=paired_scenes(pairs,lambda r:r['frame']['e2e_wall_ms'],lambda r:r['frame']['e2e_wall_ms'])
    auto=[r for r in rows if r['mode']=='unified_auto']
    checks=[json.loads(p.read_text()) for p in ROOT.glob('final_test_*/strict_quality.json')]
    quality=bool(checks) and all(x['passed'] for x in checks)
    cost=json.loads((ROOT/'cost_calibration.json').read_text())
    value=dict(status='completed_real_gpu_uninstrumented_stages',summary=summary,paired_same_service=paired,
        strict_quality_pass=quality,auto_admitted=sum(r['frame']['admission']['allowed'] for r in auto),
        auto_requests=len(auto),auto_coverage=sum(r['frame']['admission']['allowed'] for r in auto)/len(auto),
        cost_calibration=cost,certificate_scope='Fresh v5 source/model/CPU/resource/generation/trace contract; profile_stages=False; no v4 certificates reused',
        fully_uninstrumented_deployment_certified=False,
        deployment_limit='Stage timers are disabled; generated-ID capture, observer references, audit trace and post-timer snapshots remain. This certificate does not authorize a different production implementation.',
        includes_state_update_and_maintenance=True,
        remaining_overhead='Input/digests/raw-capture bookkeeping follows inherited runner timing scope; raw snapshot serialization occurs outside measured intervals',
        m2_exact_camera_skips=0,m2_exact_window_skips=0,
        m2_scope='M2 reuse disabled in this frozen serving comparison; zero actual skips, no new exact-reuse opportunity claimed; prior moving-scene evidence retained',
        attribution='Different scene support and new implementation: do not attribute v4-v5 coverage changes solely to code')
    write(ROOT/'auto_admission_results.json',value)
    # Compare true native Full output against the offline passthrough on matching
    # independent final frames, with identical fixed question/RNG contract.
    compact={r['sample_token']:r for r in read_rows(ROOT/'traces/compact_frames.jsonl') if r['split']=='final_test' and r['mode']=='full'}
    native=[r for r in rows if r['mode']=='original']
    native_check=dict(pairs=len(native),
        tokens_equal=all(r['generated_ids'][0]==compact[r['dataset_identity']['sample_token']]['generated_ids'] for r in native),
        scope='Real original single Planning native generator vs offline Full passthrough, independent final scenes')
    write(ROOT/'full_carrier_native_regression.json',native_check)
    if not native_check['tokens_equal']:raise AssertionError('Full Carrier native passthrough output changed')
    return value


def main():
    summaries,quality,paired,compact_pass=compact_report()
    system=system_report()
    facts=json.loads((ROOT/'relative_direction_candidate_results.json').read_text())
    opportunity=json.loads((ROOT/'task_opportunity_results.json').read_text())
    identity_probe=json.loads((ROOT/'identity_signature_microbenchmark.json').read_text())
    history=dict(status='Not Run / No-Go',relative_direction_independent_quality_pass=False,
        grounded_input_complete_original_tasks=0,actual_m1_to_m3_state_consumption=0,
        historical_vs_current_only_incremental_savings=None,net_benefit_established=False,
        reason='No independent identity/foreground-supported facts or grounded input-complete original tasks; no runtime retrieval/heuristic prompt insertion developed')
    write(ROOT/'historical_incremental_value.json',history)
    actual=dict(status='measured_compute_and_quality_separated',current_only=paired,
        historical_actual_skipped_operations=0,m2_exact_skipped_operations=0,
        approved_stateful_llm_token_reduction=0,
        quality_constrained_compact_gain_established=compact_pass,
        same_service_overlap=system['paired_same_service'],
        original_vs_persistent_services_are_distinct=True)
    write(ROOT/'actual_computation_savings.json',actual)
    decisions=dict(H1=dict(decision='No-Go under current evidence interface',
        certified=facts['overall']['certified_count'],independent_quality_pass=False,
        reason='No independent identity/foreground producer; finite 95% scene-calibrated bound unsupported'),
        B=dict(decision='Conditional text dependencies found, grounding unresolved; no online consumption',
            lexical=4881,reviewed=opportunity['stratified_review_count'],input_complete=0),
        H2=dict(decision='Feasibility Positive' if compact_pass else 'No-Go for tested frozen compact representation',
            quality_and_net_compute_pass=compact_pass,not_a_proof_against_all_possible_formats=True),
        D=dict(decision='Not Run / No-Go',reason=history['reason']),
        E=dict(decision='Go within measured audited contract with stage timers off' if system['strict_quality_pass'] and
            system['paired_same_service']['unified_auto']['scene_increment_upper95_ms']<0 else 'No-Go for new Auto net-benefit claim',
            strict_quality=system['strict_quality_pass']),
        research_direction='Dependency-Safe VLM Runtime',
        stop_unvalidated_state_and_feature_cache_stacking=True,online_new_fact_or_compact_enabled=False)
    write(ROOT/'go_nogo_decisions.json',decisions)
    ablations=[dict(method=m,status='Run',evidence='auto_admission_results.json') for m in ('Original OmniDrive','Persistent Serial','Persistent Overlap','Unified Auto')]
    ablations.extend(dict(method=m,status='Run offline',evidence=file) for m,file in
        [('Original Task Fact Candidate','relative_direction_candidate_results.json'),
         ('Calibrated Relative Direction Fact','relative_direction_risk_coverage.json'),
         ('Full Carrier','compact_llm_latency.json'),('Current-Perception Compact','compact_llm_latency.json'),
         ('Current Compact + Full Carrier','compact_llm_latency.json')])
    ablations.extend(dict(method=m,status='Not Run / No-Go',reason=history['reason']) for m in ('Historical Fact + Compact','Verified Direct Query'))
    for row in ablations:
        if row['method']=='Calibrated Relative Direction Fact':
            row.update(status='Offline calibration diagnostic / No-Go',reason='No finite qualified interval or identity/foreground producer; no deployed calibration')
    write(ROOT/'system_ablation.json',dict(methods=ablations))
    full=summaries['final_test']['full'];compact=summaries['final_test']['current_compact']
    calibration_history=('初次包装器错误无完成帧；详细数值文本在第一校准候选触发真实 GPU OOM。随后短格式在任何独立结果前冻结。短格式首次重跑因原生旧输出文件触发跳过生成，无完成帧；最终使用全新输出命名空间。修订与全部失败日志单列。'
        if (ROOT/'logs/compact_gpu_attempt02.log').exists() else
        '本次直接复现定型短格式；原始 v5 的包装器、详细格式 OOM 与旧输出文件失败保存在原始实验目录，不计入本次结果。')
    lines=['# OmniDrive 第五轮实验报告','',
        f'**结论：H1 当前证据接口 No-Go；H2 冻结短格式 {decisions["H2"]["decision"]}。未启用历史状态消费或紧凑输入线上替代。**','',
        '18 个未用于既有 v1-v4 协议的场景，校准/独立质量/最终测试各 6 场景，每场景 4 个连续真实 Keyframe。',
        '原始 Checkpoint/内嵌 LoRA/Tokenizer/单问题采样温度 0.1、top-p 0.75、max_new_tokens 320、问题文本与轨迹解析器不变。',
        'CPU 亲和性排除 4/5，PyTorch/OpenCV 各 1 线程；环境、权重 SHA、源代码指纹、原始轨迹与失败日志均保存。',
        calibration_history,'',
        '| 最终独立组 | 请求/完成 | 有效输入 Token P50 | E2E P50/P95 ms（完成请求） | 无效轨迹率 | ADE/FDE m（可解析） |',
        '|---|---:|---:|---:|---:|---:|']
    for mode,row in summaries['final_test'].items():
        time=row['e2e_complete_only_ms'];tokens=row['effective_multimodal_tokens']
        lines.append(f'| {mode} | {row["requests"]}/{row["complete_vlm_requests"]} | {fmt(tokens.get("p50"),1)} | {fmt(time.get("p50"))}/{fmt(time.get("p95"))} | {row["invalid_trajectory_rate"]:.2%} | {fmt(row["planning_ade_m"],3)}/{fmt(row["planning_fde_m"],3)} |')
    lines+=['','失败请求始终留在质量分母。可解析轨迹的 ADE/FDE 与无效比例共同判断；不能将短答案、解析失败或容量拒绝当成质量约束加速。',
        'Prefill/Decode 实际 GPU/Wall 时间、上下文构造成本、峰值显存和逐场景配对见 compact_llm_latency.json；未按 Token 比例外推加速。','',
        '| 最终系统方法 | 服务 | E2E P50/P95 ms | 实际 Overlap |','|---|---|---:|---:|']
    for mode,row in system['summary'].items():
        time=row['e2e_ms'];lines.append(f'| {mode} | {row["service"]} | {fmt(time.get("p50"))}/{fmt(time.get("p95"))} | {row["overlap_frames"]}/{row["frames"]} |')
    lines+=['',f'新 Auto 批准 {system["auto_admitted"]}/{system["auto_requests"]}，严格 Token/感知/时序/持久状态一致性：{system["strict_quality_pass"]}。',
        f'同服务 Auto 场景平均时间比 {fmt(system["paired_same_service"]["unified_auto"]["mean_scene_speedup"],4)}，场景平均增量单侧 95% Bootstrap 上界 {fmt(system["paired_same_service"]["unified_auto"]["scene_increment_upper95_ms"])} ms。',
        '该性能结论限实际执行与证书工作负载，非硬实时保证。新证书未沿用旧插桩证书；仍含原生输出写入、历史维护、准入和 trace 成本。',
        '本轮关闭阶段计时插桩，但保留生成 ID 捕获、observer 引用、核验 trace 与计时外原始快照；没有为去掉这些核验代码的其他生产实现颁发证书。','',
        '1. 实现在线证据认证函数的真实接受/拒绝分支，候选与认证结果分离。当前上游缺身份/前景证据，未接入线上，不把接口实现等同于可靠覆盖。',
        f'2. 新场景方向候选 {facts["overall"]["candidate_count"]} 条，可评分 {facts["overall"]["scorable_candidates"]}，Oracle 正确候选 {facts["overall"]["oracle_correct_candidates"]}，认证 {facts["overall"]["certified_count"]}。零接受的 FAR 和上界未定义，独立接受场景为零，部署 No-Go。',
        '3. RGB、相机标定/时间 Pose、历史检测、当前 Head、关联和双向几何均来自真实接口；同一 ROI 背景也能满足纹理/极线检查。未提供独立实例前景或视觉指称映射，不能证明目标身份。',
        '4. 原始 VQA 共 72,189，词法候选 4,881。48 条分层逐题文本复核加 3 条定向复核发现 bus 相对位置的条件机会；未完成图像唯一指称与答案粒度确认。已验证输入完备任务为零，其余保持 Unknown；没有声称人工标注已完成。',
        f'5. Compact 原始 Planning 独立质量门结果见 compact_planning_quality.json，最终 current_compact 通过：{quality["final_test"]["current_compact"]["quality_pass"]}。该结论限被冻结格式，不证明所有可能格式均不可行。',
        f'6. 有效 Token 和 Prefill/Decode GPU 均实际记录。最终质量与净计算收益联合通过：{paired["final_test"]["current_compact"]["quality_constrained_speedup_established"]}。C 增加文本同时保留全部 513 Carrier，不作为压缩成功证据。',
        '7. 历史状态相对 Current-Only 的额外计算收益未建立；前置门未通过，不执行不合法组合。',
        '8. 新增 M1→M3 历史状态消费为零。没有仅按关键词插入历史对象，也没有绕过准入直接回答。',
        f'9. 未建立状态化质量约束 E2E 加速。持续状态服务的 Auto 结论为：{decisions["E"]["decision"]}；当前输出服务与持续状态服务分别报告。',
        f'10. v4 的 19 个回退是 6 个冷启动无成本证据与 13 个实测 gap 超证书范围。本轮只减少完整 tensor/version witness 未变时的重复字符串哈希，仍逐请求检查全部参数/缓冲区版本。实际加载模型上的受控 CPU 元数据微测 P50：旧 {fmt(identity_probe["latency_ms"]["v4_identity"]["p50"],3)} ms、新 {fmt(identity_probe["latency_ms"]["v5_identity"]["p50"],3)} ms，签名逐次相等 {identity_probe["repeated_signatures_equal"]}。微测不是 E2E 加速；源码、资源、服务合同变化使证书失配。',
        '11. 保留 M1 持久服务与 EVA Overlap、严格依赖/证书与回退。本轮服务对照禁用 M2 复用，实际 Camera/Window 跳算为零；不冒充新的 Exact 机会证明，既有移动场景零机会证据保留。不继续堆叠未验证世界状态/特征缓存，也不默认训练或微调。',
        '12. 建议论文主线采用 Dependency-Safe VLM Runtime。Task-Stateful VLM 和 Compact Semantic Serving 尚未有足够独立质量与净计算证据，保留为受明确前提约束的研究方向。','',
        '局限：本地验证集场景留出，不是官方隐藏测试；每场景 4 帧、6 场景最终样本仅支持冻结协议下的有限结论。',
        '6 场景不足以得到 95% 场景单位 Conformal 有限界，2m 离线源关联存在截断，不能作为全目标在线误差保证。',
        '短格式丢弃对象尺寸/速度/高度/外观及细地图形状，完整 Carrier 所编码的语义也不等价于 decoded Head 字段；质量失败不能仅归因于文本理解能力。',
        '复现命令见 commands.sh；所有旧版本报告、协议和证书及模型权重保持不变，完整性核对见 delivery_manifest.json。']
    (ROOT/'experiment_report.md').write_text('\n'.join(lines)+'\n')
    # Standalone scientific plot, with evidence scope directly in the title.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    modes=list(summaries['final_test'])
    labels=['Full','Compact','Compact + Full'];x=np.arange(len(modes))
    axes[0].bar(labels,[summaries['final_test'][m]['effective_multimodal_tokens'].get('p50',0) for m in modes])
    axes[0].set_ylabel('Effective multimodal input tokens (P50)')
    axes[1].bar(x-.18,[summaries['final_test'][m]['planning_ade_m'] or 0 for m in modes],.36,label='ADE')
    axes[1].bar(x+.18,[summaries['final_test'][m]['planning_fde_m'] or 0 for m in modes],.36,label='FDE')
    axes[1].set_xticks(x);axes[1].set_xticklabels(labels);axes[1].set_ylabel('Parsed Planning error (metres)');axes[1].legend()
    axes[2].bar(labels,[(summaries['final_test'][m]['e2e_complete_only_ms'].get('p50') or 0)/1000 for m in modes])
    axes[2].set_ylabel('Complete-request E2E P50 (seconds)')
    for axis in axes:axis.tick_params(axis='x',rotation=15)
    fig.suptitle('Real GPU offline probe, independent final scenes; quality gate '+('PASS' if compact_pass else 'FAIL')+'; NOT online approved')
    fig.tight_layout();fig.savefig(ROOT/'figures/compact_tokens_quality.png',dpi=160);plt.close(fig)
    protected=json.loads((ROOT/'preserved_artifacts.json').read_text())
    changed=[p for p,h in protected.items() if sha256(p)!=h]
    annotation='data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl'
    annotation_hash=sha256(annotation)
    previous_annotation=json.loads(Path('outputs/system_optimization_v4/task_state_feasibility.json').read_text())['annotation_sha256']
    write(ROOT/'dataset_tokenizer_manifest.json',dict(annotation=annotation,annotation_sha256=annotation_hash,
        annotation_matches_preserved_v4=annotation_hash==previous_annotation,
        tokenizer_and_config_sha256={str(p):sha256(p) for p in Path('ckpts/pretrain_qformer').glob('*')
            if p.name in ('tokenizer.model','tokenizer_config.json','special_tokens_map.json','config.json','generation_config.json')}))
    if annotation_hash!=previous_annotation:raise AssertionError('Annotation differs from preserved original corpus')
    write(ROOT/'delivery_manifest.json',dict(old_artifact_count=len(protected),old_artifacts_unchanged=not changed,
        changed_old_files=changed,weights=json.loads((ROOT/'system_weights_preservation.json').read_text()),
        regression_log_tail=(ROOT/'logs/regression_gpu.log').read_text().splitlines()[-1],
        source_files={str(p):sha256(p) for p in [Path('worldstream/runtime.py'),Path('worldstream/m1/fact_certifier.py'),*Path('tools/system_optimization').glob('v5*.py'),Path('tools/system_optimization/compact_context.py')]}))
    if changed:raise AssertionError('Protected old artifacts changed')


if __name__=='__main__':main()
