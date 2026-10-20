"""Rebuild findings from completed P0 evidence, without promoting old results."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .common import counts, manifest, protected_files, sha256, source_evidence, stats, write


def read(root, name):
    return json.loads((root/name).read_text())


def validate_run(root):
    status = read(root, 'status.json')
    if status.get('status') != 'completed':
        raise ValueError('Incomplete/failed audit cannot support a final census')
    for key in ('protected_source_unchanged', 'weights_versions_unchanged', 'hooks_restored', 'source_unchanged'):
        if status.get(key) is not True:
            raise ValueError('Integrity check failed: '+key)
    protocol = read(root, 'protocol.json')
    expected = [(g['scene'], t) for g in protocol['groups'] for t in g['tokens']]
    frames, states = read(root, 'frames.json'), read(root, 'states.json')
    if [(f['scene'], f['frame']) for f in frames] != expected:
        raise ValueError('Missing, reordered or duplicate frame records')
    if [(f['scene'], f['frame']) for f in states] != expected:
        raise ValueError('M1 state population differs from native frames')
    questions = read(root, 'questions.json')
    frame_keys = set(expected)
    if not questions or any((q['scene'], q['frame']) not in frame_keys for q in questions):
        raise ValueError('Question provenance mismatch')
    if any(len(s['camera_timestamps']) != 6 for s in states):
        raise ValueError('Not six-camera data')
    if any(f['actual_skips'] != 0 for f in frames):
        raise ValueError('P0 observer must not execute reuse')
    if len({(q['scene'], q['frame']) for q in questions}) != len(expected):
        raise ValueError('Some frames did not execute their native questions')
    if sum(f['native_prefill_calls'] for f in frames) != len(questions):
        raise ValueError('Native prefill count does not match question population')
    return protocol, frames, states, questions


def candidate_profitability(temporal, kv, questions, frames):
    """Microbenchmarks reject unprofitable plans; never certify service speed."""
    temporal_groups = {}
    for p in temporal:
        temporal_groups.setdefault((p['scene'], p['branch']), []).append(p)
    per_branch = {}
    for (scene, branch), rows in temporal_groups.items():
        per_branch.setdefault(branch, []).append(dict(scene=scene, measured_layers=len(rows),
            net_wall_ms=sum(p['measured_net_wall_ms'] for p in rows),
            gpu_interval_delta_ms=sum(p['measured_gpu_interval_delta_ms'] for p in rows),
            numerical_equal=all(p['bitwise_equal'] for p in rows)))
    qindex = {(q['scene'], q['frame']): q for q in questions}
    findex = {(f['scene'], f['frame']): f for f in frames}
    prefix_rows = []
    for p in kv:
        key = (p['scene'], p['frame'])
        q, f = qindex[key], findex[key]
        overhead = q['signature_wall_ms'] + q['match_wall_ms'] + sum(f['cache_capture_wall_ms'])
        prefix_rows.append(dict(scene=p['scene'], frame=p['frame'],
            prefix_tokens=p['exact_input_prefix_tokens'],
            optimistic_prefill_wall_delta_ms=p['optimistic_prefill_wall_delta_ms'],
            measured_validation_publication_wall_ms=overhead,
            optimistic_net_wall_ms=p['optimistic_prefill_wall_delta_ms']-overhead,
            last_logits_bitwise_equal=p['last_logits_bitwise_equal'],
            last_logits_max_abs=p['last_logits_max_abs']))
    # Stability across measured scene units, not a favourable single noisy timing.
    temporal_go = {b: bool(rows) and all(r['net_wall_ms'] > 0 for r in rows)
                   for b, rows in per_branch.items()}
    kv_go = bool(prefix_rows) and all(r['optimistic_net_wall_ms'] > 0 for r in prefix_rows)
    return dict(temporal_by_scene=per_branch, prefix_by_scene=prefix_rows,
        temporal_positive_in_all_measured_scenes=temporal_go,
        prefix_positive_in_all_measured_scenes=kv_go,
        any_profitable_candidate=any(temporal_go.values()) or kv_go,
        interpretation='Diagnostic serial CPU wall. CUDA event intervals include launch gaps, are not kernel sums. Summed layer deltas describe the same candidate plan, not additive E2E speedup. No production qualification.')


def opportunity_gate(evidence):
    """Research admission, never a production certificate or a local cost vote.

    Unknown/missing evidence is a rejection. State evidence must match the
    candidate's claim (tensor computation provenance is distinct from physical
    object identity). A positive microbenchmark alone cannot admit research.
    """
    required = ('historical_sources_legal', 'mathematical_closure',
        'state_evidence_qualified', 'avoids_native_repeated_work',
        'net_cost_probe_positive', 'whole_service_upper_bound_worthwhile',
        'validation_plan_executable')
    missing = [name for name in required if evidence.get(name) is not True]
    values = [evidence.get(name) for name in
              ('optimistic_gpu_savings_upper_ms','minimum_required_savings_ms')]
    valid = all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0 for v in values)
    scope_positive = valid and values[0] > values[1]
    if not scope_positive:
        missing.append('insufficient_compute_scope_or_unmeasured_service_cost')
    return dict(mathematical_closure=evidence.get('mathematical_closure') is True,
        state_evidence_qualified=evidence.get('state_evidence_qualified') is True,
        performance_opportunity_go=bool(scope_positive and evidence.get('net_cost_probe_positive') is True
            and evidence.get('whole_service_upper_bound_worthwhile') is True),
        research_go=not missing, production_go=False, stop_reasons=missing,
        active_task_quality='Not Run', active_e2e='Not Run', sampling_distribution='Not Certified')


def build(root, output, regression=None):
    if output.resolve() == root.resolve() or output.resolve() == root.parent.resolve():
        raise ValueError('Preserve historical experiments: choose a new analysis output directory')
    protocol, frames, states, questions = validate_run(root)
    output.mkdir(parents=True, exist_ok=True)
    temporal_rows = read(root, 'temporal_rows.json')
    temporal = read(root, 'temporal_cost_probes.json')
    kv = read(root, 'kv_cost_probes.json')
    profit = candidate_profitability(temporal, kv, questions, frames)
    # Local positive timing is only a cost diagnostic. Previous evidence has
    # neither qualified execution nor a complete-service opportunity contract.
    admission = opportunity_gate(dict(mathematical_closure=bool(temporal or kv),
        historical_sources_legal=bool(temporal or kv), state_evidence_qualified=False,
        avoids_native_repeated_work=False, net_cost_probe_positive=profit['any_profitable_candidate']))
    p0_go = admission['research_go']
    stage_reason = ('P0 research prerequisites passed; numerical/task qualification remains required.' if p0_go else
        'P0 No-Go: qualified state, dependency-safe work avoidance and complete-service opportunity contract are required; local timings alone do not admit execution.')
    archived_paths = [
        'outputs/module1_final_validation_refinement/formal_go_nogo.json',
        'outputs/module2_state_aware_incremental_reuse/formal_go_nogo.json',
        'outputs/module2_state_aware_incremental_reuse/population_census.json',
        'outputs/module2_state_aware_incremental_reuse/source_aligned_feature_analysis.json',
        'outputs/module2_state_aware_incremental_reuse/reuse_feasibility_upper_bound.json',
        'outputs/module3_refinement/state_feasibility.json',
        'outputs/module3_refinement/numerical_summary.json',
        'outputs/module3_refinement/cost_table.json',
        'outputs/module3_refinement/quality_certificate.json',
        'outputs/system_optimization_v5/go_nogo_decisions.json']
    archive = [source_evidence(p) for p in archived_paths]
    write(output/'archived_evidence.json', dict(role='Previously exposed evidence, not new experiments', sources=archive))
    old_eva = next(a['value'] for a in archive if a['path'].endswith('source_aligned_feature_analysis.json'))
    if old_eva is None or old_eva['complete_source_windows'] != 0:
        raise ValueError('Recheck EVA boundary: archived zero-opportunity evidence changed or is missing')
    old_camera_pairs = (old_eva['frames']-old_eva['scenes'])*6
    nframes, nscenes = len(frames), len(protocol['groups'])
    transitions = nframes - nscenes
    cross = [q for q in questions if q['cross_frame_source']]
    candidates = {
        'eva_window': dict(status='No-Go (preserved)', population='Archived Exploratory 8 scenes/96 frames/576 camera observations',
            counts=counts(old_camera_pairs, old_camera_pairs), unit='historical camera observation; complete legal source windows=0',
            opportunity_source='outputs/module2_state_aware_incremental_reuse/source_aligned_feature_analysis.json',
            reason='No complete natural source windows. Do not rerun contextual window approximations.'),
        'temporal_full_head_or_attention': dict(status='No-Go',
            counts=counts(transitions*2, transitions*2), unit='frame transition × head',
            reason='Current query, current visual features, pose/time-dependent key positions, and updated FIFO memory change the computation closure. Native temporal memory is already part of the baseline.'),
        'temporal_value_projection': dict(status='Local cost candidate only' if any(profit['temporal_positive_in_all_measured_scenes'].values()) else 'No-Go on measured net cost',
            counts=counts(len(temporal_rows), len(temporal_rows)), unit='head layer invocation',
            potential_retained_rows=sum(r['identical_value_input_rows'] for r in temporal_rows),
            complete_v_input_rows=sum(r['retained_source_rows'] for r in temporal_rows),
            equal_k_input_rows=sum(r['identical_key_input_rows'] for r in temporal_rows),
            reason='Only Wv(value) has a row-local closure; its source equality, gather and concatenation cost is measured. No Q/K/attention/head skipped.'),
        'cross_frame_prefix_kv': dict(status='Local cost candidate only' if profit['prefix_positive_in_all_measured_scenes'] else 'No-Go on measured net cost',
            counts=counts(len(questions), len(cross)), unit='native question',
            nonzero_cross_frame_matches=sum(q['actual_exact_lcp'] > 0 for q in cross),
            minimum32_matches=sum(q['actual_exact_lcp'] >= 32 for q in cross),
            exact_lcp_tokens=stats(q['actual_exact_lcp'] for q in cross),
            exact_visual_prefix_tokens=sum(q['exact_visual_prefix_tokens'] for q in cross),
            mathematically_reusable_tokens=sum(q['actual_exact_lcp'] for q in cross),
            lawful_executable_kv_tokens=0, actual_cross_frame_runtime_hits=0,
            reason='True matches of native historical expanded inputs measured. cross_frame=True alone is not a hit. Input equality does not certify shape-dependent FP16 KV/logits or original sampling distribution.'),
        'task_state_replacement': dict(status='No-Go', counts=counts(len(questions), len(questions)),
            unit='native question', historical_text_candidates=sum(q['state_query']['admitted'] for q in questions),
            task_complete_inputs=0, reason='Partial M1 history is not complete planning/scene evidence; physical validity cannot replace full multimodal input.')}
    census = dict(schema_version=1, stage='P0', status='completed', population='Exploratory',
        original_task='planning', task_classifier_label='global_safety',
        native_multi_cross_frame_requalification='Not Run',
        provenance=str(root), scenes=nscenes, frames=nframes, transitions=transitions,
        camera_observations=6*nframes, questions=len(questions), tasks=dict(Counter(q['task'] for q in questions)),
        m1=dict(anchor_observations=sum(len(s['anchors']) for s in states),
            observation_valid_anchors=sum(s['observation_valid_anchors'] for s in states),
            physical_certified_anchors=sum(s['physically_certified_anchors'] for s in states),
            feature_observations=sum(len(s['features']) for s in states),
            feature_observations_semantics='Metadata records, not reusable captured tensors',
            committed_tensor_bytes=stats(f['m1']['tensor_bytes'] for f in frames),
            captured_feature_tensors_available=False if all(f['m1']['tensor_bytes']==0 for f in frames) else None,
            task_rejections=dict(Counter(q['state_query']['reason'] for q in questions))),
        candidates=candidates, funnel_semantics='eligible=audited population; attempted=inspected history; admitted=authorized execution; actually_skipped=real execution. Observation-only rejection/fallback is not a failed task.',
        unseen_heldout_scenes=0, online_decisions_use_gt=False)
    write(output/'opportunity_census.json', census)
    dependencies = dict(schema_version=1, default_enabled=False, candidate_boundaries=[
        dict(name='EVA', dependencies='RGB receptive field + historical ABS/RoPE/attention context; block 2 global', closure=False,
             source=['worldstream/m2/dependency_planner.py','worldstream/m2/source_alignment.py','worldstream/m2/conditional_prefix.py']),
        dict(name='Detection/Map full head', dependencies='current img_feats, pos_embed, queries, temporal memory, pose/time, masks', closure=False,
             source=['projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py','projects/mmdet3d_plugin/models/dense_heads/petr_head_map.py']),
        dict(name='Temporal V-only linear', dependencies='retained native row + per-layer Wv/bias + dtype/backend', mathematical_closure=True,
             original_memory_already_used=True, state_origin='native historical head output, FIFO position with cold padding excluded',
             not_covered=['Q','K with changing key_pos','softmax','attention output','current queries','full head','task sufficiency'],
             source=['projects/mmdet3d_plugin/models/utils/petr_transformer.py','projects/mmdet3d_plugin/models/utils/attention.py']),
        dict(name='Causal text prefix KV', dependencies='exact expanded embeddings/token IDs/positions/mask, model/LoRA/backend, source lifecycle',
             mathematical_closure=True, sampling_distribution_equivalence=False,
             source=['worldstream/m3/prefix_matcher.py','worldstream/m3/prefix_kv_cache.py','worldstream/m3/prefill_executor.py','worldstream/m3/runtime.py']),
        dict(name='Task history replacement', dependencies='complete necessary current native task evidence + frozen task qualification', closure=False,
             source=['worldstream/m3/world_state.py','worldstream/m3/task_conditioner.py','worldstream/m1/physical_certifier.py'])],
        measured_cost=profit, dependency_proof_is_not_numerical_or_quality_approval=True)
    write(output/'dependency_analysis.json', dependencies)
    write(output/'cross_frame_compute_savings.json', dict(status='P0 observation only',
        actual_skipped_operators=[], actual_skipped_eva_windows=0, actual_reused_kv_tokens=0,
        actual_avoided_prefill_forwards=0, actual_avoided_decode_forwards=0,
        measured_gpu_net_savings_ms=None, claimed_algorithmic_work_savings=0,
        native_prefill_forwards=sum(f['native_prefill_calls'] for f in frames),
        native_decode_forwards=sum(f['native_decode_calls'] for f in frames),
        added_diagnostic_llm_prefill_forwards=sum(1+p['full']['warmup']+p['full']['repeats']+
            p['suffix']['warmup']+p['suffix']['repeats'] for p in kv),
        added_diagnostic_value_linear_calls=sum(2+p['full']['warmup']+p['full']['repeats']+
            p['candidate']['warmup']+p['candidate']['repeats'] for p in temporal),
        diagnostic_probes_not_counted_as_savings=True, p0_cost_evidence=profit,
        theoretical_bounds='Natural EVA admissible upper bound=0. V-only dense linear is an optimistic bound; prefix full-minus-suffix is a microprobe, not linear token extrapolation.',
        extra_probe_forwards='Retained in raw cost probes, never included in native forward counters or savings.'))
    regression_value = read(regression.parent, regression.name) if regression and regression.is_file() else dict(status='Not Run')
    write(output/'correctness_results.json', dict(status='P0 instrumentation checked; P2 Not Run',
        reason=stage_reason, regression=regression_value,
        protected_runtime_files_unchanged=read(root,'status.json')['protected_source_unchanged'],
        weights_versions_unchanged=read(root,'status.json')['weights_versions_unchanged'],
        temporal_projection_diagnostics=dict(probes=len(temporal), bitwise_equal=sum(p['bitwise_equal'] for p in temporal),
            max_abs=max((p['max_abs'] for p in temporal), default=None),
            dense_vs_native_projection_max_abs=max((p['dense_vs_native_projection_max_abs'] for p in temporal), default=None),
            inherited_source_rows_max_abs=max((p['reused_rows_projection_max_abs'] for p in temporal), default=None),
            freshly_recomputed_rows_max_abs=max((p['fresh_rows_projection_max_abs'] for p in temporal), default=None),
            first_difference_boundary='Wv projection of freshly recomputed rows after changing GEMM shape. Dense diagnostic matches native; inherited history matches dense. No downstream propagation claimed or repair attempted.'),
        prefix_prefill_diagnostics=dict(probes=len(kv), last_logits_bitwise_equal=sum(p['last_logits_bitwise_equal'] for p in kv),
            max_abs=max((p['last_logits_max_abs'] for p in kv), default=None),
            original_sampling_distribution_certified=False),
        sampling_distribution_status='Not Certified', active_task_quality_status='Not Run',
        active_e2e_status='Not Run', numerical_local_differences_measured=True,
        active_mechanism_tests=dict(status='Not Run', reason='No new execution mechanism implemented before P0 gate.'),
        note='Equal diagnostic rows or passed existing regression do not certify an active strategy.'))
    write(output/'downstream_task_quality.json', dict(status='Not Run', reason=stage_reason,
        frozen_thresholds_modified=False, original_native_tasks_executed=len(questions),
        actual_quality_comparisons=0, active_strategy_quality_pass=None,
        native_outputs=str(root/'frames.json'), native_token_ids=str(root/'questions.json'),
        detection_map_temporal_digests=str(root/'frames.json'),
        note='No active-vs-native pair: native output presence is not task quality approval.'))
    write(output/'end_to_end_comparison.json', dict(status='Not Run', reason=stage_reason,
        modes={name:dict(status='Not Run', p50_ms=None,p95_ms=None) for name in
               ['Native Original','Persistent Full','State Shadow','Stateful Active']},
        disable_history_ablation=dict(status='Not Run', reason=stage_reason),
        p0_audit_only=dict(instrumented_wall_ms=stats(f['audit_wall_ms'] for f in frames),
            peak_gpu_allocated_bytes=max(f['peak_allocated_bytes'] for f in frames),
            not_a_full_service_performance_comparison=True),
        e2e_speedup=None, scheduling_overlap_credited=0, kernel_acceleration_credited=0))
    formal = dict(state_evidence_go=False, dependency_closure_go=bool(temporal or kv),
        actual_cross_frame_reuse_go=False, numerical_correctness_go=False,
        frozen_task_quality_go=False, gpu_net_savings_go=False, e2e_speedup_go=False, production_go=False,
        p0_go=p0_go, P1='Required' if p0_go else 'Not Implemented / No-Go',
        P2='Not Run', P3='Not Run', reason=stage_reason,
        scope='state_evidence_go refers to formally certified M1 task/physical state. dependency_closure_go is limited to mathematical V-linear/text-prefix boundaries; never a full head or sampling guarantee.',
        production_status='Disabled', research_qualification='Exploratory only')
    formal.update(admission=admission, mathematical_closure=admission['mathematical_closure'],
        state_evidence_qualified=admission['state_evidence_qualified'],
        performance_opportunity_go=admission['performance_opportunity_go'],
        active_task_quality_status='Not Run', active_e2e_status='Not Run',
        numerical_qualification_status='Not Certified; local differences measured')
    write(output/'formal_go_nogo.json', formal)
    write(output/'delivery_manifest.json', dict(raw_artifacts=manifest(root.glob('*.json')),
        protected_current_matches_run=manifest(protected_files()) == read(root,'protected_before.json'),
        audit_tool_source=manifest(Path(__file__).parent.glob('*.py')),
        executed_runtime_source_current_matches={name:sha256(name)==digest for name,digest in
            read(root,'source_manifest.json').items() if Path(name).name in ('gpu_audit.py','common.py')},
        analysis_revision='Reporting additions after GPU completion do not change the executed audit; per-run original reporting source is archived.',
        prior_untracked_directory='outputs/m3_state_guided_verified_decode/ (preserved, not consumed)',
        unavailable_ide_tab='outputs/m2_freeze_m3_task_serving/experiment_report.md does not exist in this checkout'))
    prefix_stats = candidates['cross_frame_prefix_kv']['exact_lcp_tokens']
    cost_table = '\n'.join(f"| {branch} V-only | {min(-r['net_wall_ms'] for r in rows):.3f}–{max(-r['net_wall_ms'] for r in rows):.3f} | {len(rows)} 场景 |"
                           for branch, rows in profit['temporal_by_scene'].items())
    if profit['prefix_by_scene']:
        cost_table += f"\n| Prefix KV（含签名/匹配/发布） | {min(-r['optimistic_net_wall_ms'] for r in profit['prefix_by_scene']):.3f}–{max(-r['optimistic_net_wall_ms'] for r in profit['prefix_by_scene']):.3f} | {len(profit['prefix_by_scene'])} 场景 |"
    regression_text = (f"{regression_value['tests']} 项通过，失败 {regression_value['failures']}，错误 {regression_value['errors']}，跳过 {regression_value['skipped']}"
                       if regression_value.get('status') == 'passed' else regression_value.get('status', 'Not Run'))
    text = f'''# OmniDrive 跨帧计算复用机会审计

本轮 P0 {'Go（须进入 P1，尚未授权生产）' if p0_go else 'No-Go；按前置停止条件冻结 P1 新执行机制'}。原生模型、权重、LoRA、任务、Sampling 参数和冻结阈值不修改。新增工具不接入 Production/Auto。

实际人口：{nscenes} 个已曝光场景、{nframes} 个自然连续帧、{transitions} 次跨帧转换、{6*nframes} 路六相机观测、{len(questions)} 个原生问题，全部为 Exploratory。原配置原生任务为 Planning；本轮不删减问题或更换输入。旧 EVA 96 帧的失败人口单独引用，绝不与新人口相加。

M1 观测有效 Anchor 累计 {census['m1']['observation_valid_anchors']}，正式物理认证累计 {census['m1']['physical_certified_anchors']}。每帧来源 Frame/Camera/Timestamp/Epoch/版本、Dirty/Unknown 和提交前历史快照保存在 GPU 原始记录。观测有效、物理身份、Transformer 等价和任务输入充分性分别判断。完整 Planning 证据没有由部分历史覆盖。

M1 Feature 数量是带版本的元数据记录；{nframes} 帧实际提交的 Tensor 字节 P50/P95 为 {census['m1']['committed_tensor_bytes']['p50']:.0f}/{census['m1']['committed_tensor_bytes']['p95']:.0f}，不能把未捕获的 Feature 当作可执行缓存。

Detection/Map 原有 Memory 已经是原模型基线的一部分。当前视觉、查询和位姿/时间相关 Key 变化，不能复用整个 Head 或 Attention。本轮发现真实保留的历史 Value 输入行累计 {sum(r['identical_value_input_rows'] for r in temporal_rows)}，只对 Wv 线性投影存在行级数学依赖闭包。测量完整投影与含来源核验、Gather、部分 GEMM、拼接的候选，按场景与分支汇总见 dependency_analysis.json；不能把原生 Memory 再记一次节省。

局部数值根因探针：全量诊断投影与原生投影逐位一致，继承的历史行也一致；首次差异位于缩小 GEMM Shape 后重新计算的新行。实际数值和首个差异索引保留在 temporal_cost_probes.json，不把它推演成已经测得的下游任务失败，也不为零净收益机制追加修复。

Expanded Input 的真实跨帧 LCP P50/P95 为 {prefix_stats['p50'] if prefix_stats else None}/{prefix_stats['p95'] if prefix_stats else None} Token。历史源实际匹配 {len(cross)} 个问题；其中匹配视觉前缀 Token 总数 {sum(q['exact_visual_prefix_tokens'] for q in cross)}。这是实际张量匹配而非仅允许跨 Frame ID。合法可执行 KV Token、实际执行命中和跳算均为 0；改变 Prefill Shape 的数值路径及原始 Sampling 分布未获认证。输入相同不能推出 FP16 输出逐位相同。{len(kv)} 次真实前缀探针中，仅 {sum(p['last_logits_bitwise_equal'] for p in kv)} 次最后位置 logits 逐位相同，最大绝对差 {max((p['last_logits_max_abs'] for p in kv), default=None)}。

P0 成本探针保留 Native 全量执行。V-only 候选和前缀候选均有原始 GPU Event/CPU Wall 样本，验证与缓存成本单列。CUDA Event 区间包含主机发射空隙，不称为 Kernel 时间总和。没有将审计时延作为完整服务性能、将 Token 比例外推为 GPU 节省或把同帧 Prefix/Batch/Overlap 记为跨帧收益。

| 被测候选 | 相对完整计算的净增耗时 ms | 成本覆盖 |
|---|---:|---|
{cost_table}

上表是局部串行计划的实测成本范围，不能相加成为 E2E 增量。Detection 仅第 0 层满足历史 Value 闭包，Map 覆盖六层。没有少执行一次完整 Prefill 或任何 Decode。

本轮基准只覆盖逐层来源核验、Gather/Concat 的最小 V-only 计划和现有 PrefixKVCache 的匹配/签名/发布计划；No-Go 限这些实测计划。没有声称所有未来的缓存或专用 Kernel 都数学上不可能获益。新增专用 Kernel 不属于本轮任务范围。

本轮实际减少 GPU 操作：0；实际复用 KV Token：0；完整 VLM 净加速：未建立。P2 活跃机制质量与 P3 四模式配对/关闭历史消融均为 Not Run，原因见 formal_go_nogo.json。原生回答、Token ID、感知和 Memory 摘要已保存，但不冒充新策略通过冻结质量。

原 M1/M2/M3 加新增审计回归：{regression_text}。这属于回归核验，不替代新执行机制的完整任务质量认证。

被否定的假设限本轮边界：Stable 观测可直接跳过 Transformer；已有 Temporal Memory 自动带来新增跳算；相同文本前缀自动获得有利净成本及严格 Sampling 等价。没有证明所有可能跨帧算法都不可行。建议停止继续投入当前 EVA Window、整 Head、原有 Prefix 边界的缓存复杂度；只有新增可观测且收益足够的依赖边界，才重新进入 P0。

运行证据：{root}。环境、Commit、完整 Checkpoint SHA（含 LoRA）、Tokenizer 和数据文件 SHA、随机种子、Sampling 协议及源码前后哈希均已记录。审计用独立临时钩子，退出后恢复，原 M1/M2/M3 和模型文件未修改。已有未跟踪 SGVD 目录保留且没有读取或恢复。

复现见 commands.sh；交付代码边界见 implementation_changes.md。未运行阶段的 null 不是零时延或质量通过。
'''
    (output/'experiment_report.md').write_text(text)
    return formal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh analysis directory; historical output is protected')
    parser.add_argument('--regression', type=Path)
    args = parser.parse_args()
    formal = build(args.run, args.output, args.regression)
    print(json.dumps(formal, ensure_ascii=False, indent=2))
    if formal['p0_go']:
        raise SystemExit('P0 passed; finish authorized P1/P2/P3 before declaring the task complete.')


if __name__ == '__main__':
    main()
