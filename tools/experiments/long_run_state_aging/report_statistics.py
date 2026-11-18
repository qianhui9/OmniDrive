"""Sample-size and uncertainty tables; report real zero/undefined rates honestly."""


def append_statistics(out, manifest, summary):
    from .protocol import STAGES
    from .report import csv_replace
    fmt = lambda v: 'N/A' if v is None else f'{v:.2f}'
    interval = lambda r, m: ('N/A' if r.get(m + '_ci95_low') is None else
        f"[{r[m + '_ci95_low']:.2f}, {r[m + '_ci95_high']:.2f}]")
    rows = summary['stage_summary']
    names = {'native': 'Native VLM', 'worldstream': 'WorldStream'}
    columns = ('factual_qa_accuracy_percent', 'stale_answer_rate_percent', 'average_e2e_ms',
               'p95_e2e_ms', 'peak_gpu_allocated_mb')
    table = ['# 表1：不同运行阶段的系统性能变化', '',
        f"运行状态：{summary['status']}；已完成配对重复：{len(summary['completed_paired_repeats'])}/{manifest['effective_repeats']}。", '',
        '| Stage | Method | Factual QA Accuracy (%) | Stale Answer Rate (%) | Mean E2E (ms) | P95 E2E (ms) | Peak GPU Allocated (MiB) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        table.append('| ' + ' | '.join([row['stage'], names[row['method']]] + [fmt(row[k]) for k in columns]) + ' |')
    table += ['', '表注：按日志内部时间顺序三等分，每帧保留三个固定事实问题与两个原始 QA。主表 Accuracy 为 known 事实分，E2E 包含完整五问帧；P95 合并实际帧样本，Peak 取观测最大值。N/A 表示无有效分母，0% 表示存在分母但未观测到错误。样本量与场景 bootstrap 区间见 report.md。', '']
    (out / '表1.md').write_text('\n'.join(table))
    csv_replace(out / 'paper_table.csv', [{k: r[k] for k in ('stage', 'method', *columns)} for r in rows])

    lines = ['', '## 扩样范围和独立样本量', '',
        '协议 v3：扩展真实场景覆盖，不通过循环视频、串接场景或重复查询来增加独立样本数。所有日志内部持续保留状态；只在真实日志边界重置。', '',
        '该集合来自既有主实验，不是新增独立测试集。每个场景仍约 20 秒；全部场景时长之和不构成一条连续长日志，无法据此证明小时级稳态。', '']
    coverage = summary.get('unique_GT_coverage')
    if coverage:
        lines += [f"正式选择完整日志 {coverage['selected_full_scene_count']} 条，本次测量 {coverage['selected_scene_count']} 条；每方法每次重复 {coverage['per_method_frames_per_repeat']} 帧、{coverage['per_method_queries_per_repeat']} 个查询。最长连续测量时长 {coverage['longest_continuous_seconds']:.3f} 秒。", '',
            '| Stage | Unique Frames | Unique Queries | Known GT | UNKNOWN GT | Unique Changed Query Transitions | Scenes with Changed Queries |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for row in coverage['rows']:
            if row['scene_id'] == 'ALL':
                lines.append('| ' + ' | '.join([row['stage']] + [str(row[k]) for k in
                    ('unique_frames', 'unique_queries', 'known_GT_queries', 'unknown_GT_queries',
                     'unique_changed_query_transitions', 'scenes_with_changed_query_transitions')]) + ' |')
        lines += ['', '上表在模型运行前按 GT 计数，每个 scene/frame/question 只计一次，不乘重复次数。变化单位是采样问题事实转变，不声称是彼此独立的物理事件；真正的统计聚类单位是完整场景。实际可评分分母可能低于预检 GT 可用量。', '']
    lines += ['| Stage | Method | Scene Clusters | Unique Scorable Queries | Unique Stale-Eligible Transitions | Repeated Stale Observations | Stale Errors |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| ' + ' | '.join([row['stage'], names[row['method']]] + [str(row[k]) for k in
            ('independent_scenes', 'unique_scorable_queries', 'unique_stale_eligible_query_transitions',
             'stale_eligible_queries', 'stale_answers')]) + ' |')
    lines += ['', '## 场景聚类 Bootstrap 95% 区间', '',
        f"Bootstrap draws={manifest.get('config', {}).get('bootstrap_samples', 2000)}。每次抽样完整场景，保留其中全部时间帧、五问和三次运行；两个方法使用相同的场景重采样，因此比较是配对的。该区间反映既有场景集合上的经验不确定性，不能把单场景里的帧或重复查询视为独立样本。", '',
        '| Stage | Method | Accuracy 95% CI (%) | Stale 95% CI (%) | Mean E2E 95% CI (ms) | P95 E2E 95% CI (ms) | Stale CI Status |',
        '|---|---|---|---|---|---|---|']
    for row in rows:
        lines.append('| ' + ' | '.join([row['stage'], names[row['method']]] + [interval(row, k) for k in columns[:4]] +
            [row['stale_answer_rate_percent_ci_status']]) + ' |')
    lines += ['', '少于两个有效场景聚类时不报告区间；bootstrap 中有效分母样本少于 80% 时不报告区间。零错误或全错误的率不会凭非参数 bootstrap 声称零宽度置信区间，点估计仍按真实数据写为 0%/100%，区间标 N/A 并说明边界原因。没有加伪计数或人为补样。对部分重采样无分母的指标，区间明确条件于非零分母，valid_draws/total_draws 在 CSV 中保留。', '',
        '配对差值及区间见 paired_comparison.csv：Accuracy/Stale 为 WorldStream−Native 的百分点差，E2E/P95 为 ms 差；e2e_latency_reduction_percent 为相对时延降低百分比。', '',
        '## 独立运行间 Mean ± SD', '',
        '| Stage | Method | Accuracy Mean ± SD (%) | Mean E2E Mean ± SD (ms) | Per-Repeat P95 Mean ± SD (ms) |',
        '|---|---|---:|---:|---:|']
    for row in rows:
        cells = []
        for key in ('factual_qa_accuracy_percent', 'average_e2e_ms', 'p95_e2e_ms'):
            cells.append(fmt(row[key + '_repeat_mean']) + ' ± ' + fmt(row[key + '_repeat_sd']))
        lines.append('| ' + ' | '.join([row['stage'], names[row['method']], *cells]) + ' |')
    lines += ['', 'SD 是同一整组日志的重复运行间样本标准差（ddof=1），不是场景 bootstrap 区间。只有一次完整重复时 SD 为 N/A。每次 P95 的均值与主表 pooled P95 是不同统计量，不混用。峰值显存为实际观测最大值，不把它的抽样变动包装成内存上界。', '',
        '## 输出索引', '',
        '- 表1.md / paper_table.csv：论文七列表格。',
        '- stage_summary.csv / overall_summary.csv：阶段和总体指标、样本数、重复 SD、场景 bootstrap 区间及状态。',
        '- scene_summary.csv / repeat_summary.csv：逐日志和逐重复统计，便于检查异常场景。',
        '- sample_coverage.json / sample_coverage.csv：预检 GT 样本覆盖，不乘重复次数。',
        '- paired_comparison.csv：方法配对差值与不确定性。',
        '- query_group_summary.csv：固定三问和原始两问的有限事实评分，完整生成质量仍为 N/A。', '']
    with (out / 'report.md').open('a') as handle:
        handle.write('\n'.join(lines))
