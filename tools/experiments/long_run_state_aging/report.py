"""Offline scoring and publication plots from durable real inference receipts."""
from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import numpy as np

from .protocol import (METHODS, STAGES, read, write, read_journal, validate_rows)
from .instrumentation import distribution


def csv_replace(path, rows, fields=()):
    fields = list(dict.fromkeys(list(fields) + [k for r in rows for k in r]))
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: 'N/A' if v is None else json.dumps(v, ensure_ascii=False)
                if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})
    tmp.replace(path)


def ratio(n, d, scale=1):
    return scale * n / d if d else None


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def maximum(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def memory_aggregate(frames):
    rows = [r.get('memory', {}) for r in frames]
    return dict(peak_gpu_allocated_mb=maximum([r.get('gpu_peak_allocated_mb') for r in rows]),
        peak_gpu_reserved_mb=maximum([r.get('gpu_peak_reserved_mb') for r in rows]),
        max_observed_host_rss_mb=maximum([r.get(key) for r in rows
                                        for key in ('host_rss_before_mb', 'host_rss_after_mb')]),
        host_process_lifetime_peak_rss_mb=maximum([r.get('host_process_lifetime_peak_rss_mb') for r in rows]))


def aggregate(frames, events):
    from tools.environment_stability_eval.metrics import accuracy
    known = accuracy(events)
    values = [r['e2e_ms'] for r in frames]
    return dict(frames=len(frames), scored_queries=known['scorable_queries'],
        unknown_queries=known['unknown_queries'], factual_qa_accuracy_percent=known['overall_accuracy_percent'],
        fully_correct_accuracy_percent=known['fully_correct_accuracy_percent'],
        stale_answer_rate_percent=known['stale_answer_rate_percent'], stale_answers=known['stale_answers'],
        stale_eligible_queries=known['queries_after_changes'],
        average_e2e_ms=mean(values), p95_e2e_ms=float(np.percentile(values, 95)) if values else None,
        sustained_fps=ratio(len(frames), sum(values) / 1000),
        auxiliary_processing_wall_fps=ratio(len(frames), sum(r['processing_wall_ms'] for r in frames) / 1000),
        average_instrumentation_ms=mean([r['instrumentation_ms'] for r in frames]),
        **memory_aggregate(frames))


def aging_aggregate(frames):
    if not frames:
        return dict(ogsm_state_size_mb=None, davp_cache_size_mb=None, average_window_reuse_age_frames=None,
                    qcse_cache_size_mb=None, token_acceptance_percent=None,
                    ogsm_peak_state_size_mb=None, davp_peak_cache_size_mb=None,
                    qcse_peak_cache_size_mb=None, peak_combined_retained_payload_mb=None)
    ogsm = [r['aging']['ogsm'] for r in frames]
    davp = [r['aging']['davp'] for r in frames]
    qcse = [r['aging']['qcse'] for r in frames]
    concat = lambda rows, key: [v for r in rows for v in r[key]]
    video_seconds = sum(r['video_interval_s'] for r in frames)
    proposed, accepted = sum(r['proposed_tokens'] for r in qcse), sum(r['accepted_tokens'] for r in qcse)
    reuse = concat(davp, 'reused_window_age_frames')
    result = dict(ogsm_state_size_mb=mean([r['state_size_mb'] for r in ogsm]),
        ogsm_peak_state_size_mb=max(r['state_size_mb'] for r in ogsm),
        object_anchors=mean([r['object_anchors'] for r in ogsm]), map_anchors=mean([r['map_anchors'] for r in ogsm]),
        state_storage_age_distribution_s=distribution(concat(ogsm, 'record_storage_age_s')),
        observation_age_distribution_s=distribution(concat(ogsm, 'source_observation_age_s')),
        state_commit_events=len(ogsm), video_exposure_s=video_seconds,
        state_update_frequency_hz=ratio(len(ogsm), video_seconds),
        state_association_success_percent=ratio(sum(r['association_pairs'] for r in ogsm),
                                                sum(r['association_predictions'] for r in ogsm), 100),
        invalidated_state_ratio=ratio(sum(r['invalidated_anchors'] for r in ogsm), sum(r['considered_anchors'] for r in ogsm)),
        unmatched_predictions=sum(r['unmatched_predictions'] for r in ogsm), ambiguous_association_count=None,
        davp_cache_size_mb=mean([r['cache_size_mb'] for r in davp]),
        davp_tensor_cache_size_mb=mean([r['tensor_cache_size_mb'] for r in davp]),
        davp_peak_cache_size_mb=max(r['cache_size_mb'] for r in davp),
        cached_window_count=mean([r['cached_window_count'] for r in davp]),
        average_window_reuse_age_frames=mean(reuse),
        p95_window_reuse_age_frames=float(np.percentile(reuse, 95)) if reuse else None,
        window_age_distribution_frames=distribution(concat(davp, 'window_age_frames')),
        window_reuse_rate_percent=ratio(sum(r['reused_windows'] for r in davp), sum(r['total_windows'] for r in davp), 100),
        windows_at_or_over_reuse_limit=sum(r['windows_at_or_over_reuse_limit'] for r in davp),
        average_global_anchor_age_frames=mean([r['global_anchor_age_frames'] for r in davp]),
        global_anchor_refresh_count=sum(r['global_anchor_refreshed'] for r in davp),
        global_anchor_refresh_frequency_hz=ratio(sum(r['global_anchor_refreshed'] for r in davp), video_seconds),
        full_global_refresh_frequency_percent=ratio(sum(r['full_global_refresh'] for r in davp), len(davp), 100),
        average_global_feature_drift=mean([r['global_feature_drift'] for r in davp]),
        incremental_visual_execution_ratio=ratio(sum(r['incremental_visual_execution'] for r in davp), len(davp)),
        average_visual_ms=mean([r['visual_ms'] for r in davp]),
        qcse_cache_size_mb=mean([r['cache_size_mb'] for r in qcse]),
        qcse_peak_cache_size_mb=max(r['cache_size_mb'] for r in qcse),
        # Peak simultaneous sum, not sum of three unrelated per-module maxima.
        # Roots may share immutable objects; this is not exclusive system RAM.
        peak_combined_retained_payload_mb=max(r['aging']['ogsm']['state_size_mb'] +
            r['aging']['davp']['cache_size_mb'] + r['aging']['qcse']['cache_size_mb'] for r in frames),
        candidate_cache_entries=mean([r['cache_entries'] for r in qcse]),
        candidate_token_count=mean([r['candidate_token_count'] for r in qcse]),
        candidate_age_distribution_s=distribution(concat(qcse, 'candidate_age_s')),
        proposed_tokens=proposed, accepted_tokens=accepted,
        token_acceptance_percent=ratio(accepted, proposed, 100),
        token_validation_failure_percent=ratio(proposed - accepted, proposed, 100),
        candidate_entry_validation_failure_percent=None,
        direct_answer_rate_percent=ratio(sum(r['direct_answers'] for r in qcse), sum(r['queries'] for r in qcse), 100),
        average_language_ms=mean([r['language_ms'] for r in qcse]))
    # Age-stratified acceptance uses actual selected source frames, not resident ages.
    requests = concat(qcse, 'request_candidate_ages')
    result['acceptance_by_source_age_s'] = [dict(age_interval=[lo, hi],
        proposed_tokens=sum(r['proposed_tokens'] for r in requests if r['source_age_s'] is not None and lo < r['source_age_s'] <= hi),
        accepted_tokens=sum(r['accepted_tokens'] for r in requests if r['source_age_s'] is not None and lo < r['source_age_s'] <= hi))
        for lo, hi in ((0, .5), (.5, 1), (1, 2), (2, 4), (4, 8))]
    for row in result['acceptance_by_source_age_s']:
        row['token_acceptance_percent'] = ratio(row['accepted_tokens'], row['proposed_tokens'], 100)
    return result


def report(out):
    from tools.environment_stability_eval.common import jsonl
    from tools.environment_stability_eval.metrics import score_rows
    from tools.worldstream_m123.r0_task_evaluation import sha
    manifest = read(out / 'manifest.json')
    from .protocol import PROTOCOL_VERSION
    if manifest.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('Legacy protocol is immutable; use a new v3 output directory')
    gt = list(jsonl(out / 'query_ground_truth.jsonl'))
    objects = {r['frame_id']: r['objects'] for r in jsonl(out / 'evaluation_objects.jsonl')}
    attempts, selected, all_rows, all_events, per_attempt = [], {}, [], [], []
    runtime_by_run = {}
    for repeat in range(manifest['effective_repeats']):
        for method in METHODS:
            for directory in sorted((out / 'runs' / f'repeat_{repeat:03d}' / method).glob('attempt_*')):
                rows = read_journal(directory / 'receipts.jsonl')
                complete = validate_rows(manifest, rows, method, repeat, allow_partial=True)
                marker = directory / 'complete.json'
                complete = complete and marker.exists()
                if marker.exists():
                    status = read(marker)
                    if status['receipts_sha256'] != sha(directory / 'receipts.jsonl') or status['frames'] != len(rows):
                        raise ValueError('Modified or incomplete completed journal')
                    if not complete:
                        raise ValueError('Completion marker does not cover a complete sequence')
                attempt = str(directory.relative_to(out))
                for row in rows:
                    row.update(attempt=attempt, attempt_complete=complete)
                    scene = next(s for s in manifest['scenes'] if s['scene_id'] == row['scene_id'])
                    i = row['frame_index']
                    row['video_interval_s'] = scene['timestamps'][i] - scene['timestamps'][i - 1] if i else 0
                events = score_rows(rows, gt, objects) if rows else []
                stage_by_frame = {r['frame_id']: r['stage'] for r in rows}
                for event in events:
                    event.update(stage=stage_by_frame[event['frame_id']], attempt=attempt, attempt_complete=complete)
                attempts.append(dict(attempt=attempt, method=method, repeat=repeat, frames=len(rows), complete=complete,
                                     contributes_to_paired_results=False))
                all_rows.extend(rows)
                all_events.extend(events)
                for stage in STAGES:
                    fs = [r for r in rows if r['stage'] == stage]
                    es = [e for e in events if e['stage'] == stage]
                    per_attempt.append(dict(scope='attempt', attempt=attempt, attempt_complete=complete,
                        method=method, repeat=repeat, stage=stage, **aggregate(fs, es),
                        **(aging_aggregate(fs) if method == 'worldstream' else {})))
                if complete:
                    if (repeat, method) in selected:
                        raise ValueError('Multiple completed attempts for one method/repeat')
                    selected[repeat, method] = (rows, events)
                    runtime_by_run[repeat, method] = read(directory / 'runtime_identity.json')
    paired_repeats = [i for i in range(manifest['effective_repeats']) if all((i, m) in selected for m in METHODS)]
    for repeat in paired_repeats:
        native, world = selected[repeat, 'native'][0], selected[repeat, 'worldstream'][0]
        for a, b in zip(native, world):
            for key in ('frame_id', 'scene_id', 'timestamp', 'questions', 'groups', 'fairness'):
                if a[key] != b[key]:
                    raise ValueError('Unpaired actual input: ' + key)
        for key in ('checkpoint_sha256', 'device', 'precision_by_dtype', 'generation', 'gpu_uuid'):
            if runtime_by_run[repeat, 'native'][key] != runtime_by_run[repeat, 'worldstream'][key]:
                raise ValueError('Unpaired runtime: ' + key)
    paired_rows = [r for i in paired_repeats for m in METHODS for r in selected[i, m][0]]
    paired_events = [r for i in paired_repeats for m in METHODS for r in selected[i, m][1]]
    for attempt in attempts:
        attempt['contributes_to_paired_results'] = attempt['complete'] and attempt['repeat'] in paired_repeats
    from .statistics import sample_counts, repeat_statistics, paired_intervals, METRICS
    stages, overall_rows, comparisons = [], [], []
    for stage in (*STAGES, 'ALL'):
        parts = {method: [r for r in paired_rows if r['method'] == method and
                          (stage == 'ALL' or r['stage'] == stage)] for method in METHODS}
        query_parts = {method: [r for r in paired_events if r['method'] == method and
                                (stage == 'ALL' or r['stage'] == stage)] for method in METHODS}
        config = manifest.get('config', {})
        intervals, deltas = paired_intervals(parts, query_parts,
            samples=config.get('bootstrap_samples', 2000), seed=config.get('bootstrap_seed', 20261118))
        stage_points = {}
        for method in METHODS:
            fs, es = parts[method], query_parts[method]
            row = dict(scope='paired_pooled', stage=stage, method=method, paired_repeats=len(paired_repeats),
                **aggregate(fs, es), **(aging_aggregate(fs) if method == 'worldstream' else {}),
                **sample_counts(fs, es), **repeat_statistics(fs, es), **intervals[method])
            stage_points[method] = row
            (overall_rows if stage == 'ALL' else stages).append(row)
        for metric in METRICS:
            a, b = stage_points['native'][metric], stage_points['worldstream'][metric]
            comparisons.append(dict(stage=stage, metric=metric + '_worldstream_minus_native',
                point_estimate=b - a if a is not None and b is not None else None,
                **deltas[metric + '_worldstream_minus_native']))
        a, b = stage_points['native']['average_e2e_ms'], stage_points['worldstream']['average_e2e_ms']
        comparisons.append(dict(stage=stage, metric='e2e_latency_reduction_percent',
            point_estimate=100 * (1 - b / a) if a and b else None,
            **deltas['e2e_latency_reduction_percent']))
    csv_replace(out / 'stage_summary.csv', stages + per_attempt, ('scope', 'stage', 'method'))
    csv_replace(out / 'overall_summary.csv', overall_rows)
    csv_replace(out / 'paired_comparison.csv', comparisons)
    scene_rows, repeat_rows = [], []
    for stage in (*STAGES, 'ALL'):
        for method in METHODS:
            for scene in manifest['scenes']:
                fs = [r for r in paired_rows if r['method'] == method and r['scene_id'] == scene['scene_id'] and
                      (stage == 'ALL' or r['stage'] == stage)]
                es = [r for r in paired_events if r['method'] == method and r['scene_id'] == scene['scene_id'] and
                      (stage == 'ALL' or r['stage'] == stage)]
                scene_rows.append(dict(scene_id=scene['scene_id'], stage=stage, method=method,
                    **aggregate(fs, es), **sample_counts(fs, es), **repeat_statistics(fs, es)))
            for repeat in paired_repeats:
                fs = [r for r in paired_rows if r['method'] == method and r['repeat'] == repeat and
                      (stage == 'ALL' or r['stage'] == stage)]
                es = [r for r in paired_events if r['method'] == method and r['repeat'] == repeat and
                      (stage == 'ALL' or r['stage'] == stage)]
                repeat_rows.append(dict(repeat=repeat, stage=stage, method=method, **aggregate(fs, es)))
    csv_replace(out / 'scene_summary.csv', scene_rows)
    csv_replace(out / 'repeat_summary.csv', repeat_rows, ('repeat', 'stage', 'method'))
    query_groups = []
    for stage in STAGES:
        for method in METHODS:
            for label, predicate in (
                ('fixed_facts', lambda e: e['query_index'] < 3),
                ('original_qa_limited_facts', lambda e: e['query_index'] >= 3)):
                es = [e for e in paired_events if e['stage'] == stage and
                      e['method'] == method and predicate(e)]
                stats = aggregate([], es)
                # All five queries execute in one model call; no fabricated
                # per-subset E2E latency, memory or FPS measurements.
                query_groups.append(dict(stage=stage, method=method, query_subset=label,
                    queries=len(es), **{k: stats[k] for k in ('scored_queries', 'unknown_queries',
                    'factual_qa_accuracy_percent', 'fully_correct_accuracy_percent',
                    'stale_answer_rate_percent', 'stale_answers', 'stale_eligible_queries')},
                    **sample_counts([], es), holistic_generation_quality=None))
    csv_replace(out / 'query_group_summary.csv', query_groups)
    frames_export = []
    event_by_frame = defaultdict(list)
    for event in all_events:
        event_by_frame[event['attempt'], event['frame_id']].append(event)
    identity = ('attempt', 'attempt_complete', 'method', 'repeat', 'scene_id', 'frame_id', 'frame_index', 'timestamp', 'stage')
    for row in all_rows:
        item = {k: row[k] for k in identity}
        item.update(aggregate([row], event_by_frame[row['attempt'], row['frame_id']]),
                    e2e_ms=row['e2e_ms'], processing_wall_ms=row['processing_wall_ms'],
                    instrumentation_ms=row['instrumentation_ms'], state_before=row['state_before'], state_after=row['state_after'])
        item.update(row.get('memory', {}))
        frames_export.append(item)
    csv_replace(out / 'per_frame_metrics.csv', frames_export, identity + ('e2e_ms', 'state_before', 'state_after'))
    for module, filename in (('ogsm', 'ogsm_state_aging.csv'), ('davp', 'davp_cache_aging.csv'), ('qcse', 'qcse_cache_aging.csv')):
        csv_replace(out / filename, [dict(**{k: r[k] for k in identity}, **r['aging'][module])
            for r in all_rows if r['method'] == 'worldstream'], identity + ('state_size_mb' if module == 'ogsm' else 'cache_size_mb',))
    csv_replace(out / 'scored_queries.csv', all_events, ('method', 'repeat', 'frame_id', 'question', 'scorable', 'score', 'stale_answer'))
    status = ('COMPLETED' if len(paired_repeats) == manifest['effective_repeats'] else
              'PARTIAL' if all_rows else 'PREPARED_NO_INFERENCE')
    summary = dict(protocol_version=PROTOCOL_VERSION, status=status,
        report_time=datetime.now().astimezone().isoformat(), smoke=manifest['smoke'],
        manifest='manifest.json', scenes=manifest['scenes'], completed_paired_repeats=paired_repeats,
        effective_repeats=manifest['effective_repeats'], attempts=attempts, stage_summary=stages,
        overall_summary=overall_rows, paired_comparison=comparisons,
        uncertainty_definition=manifest.get('statistics_definition', 'Paired scene-cluster bootstrap; repeats stay inside each scene.'),
        unique_GT_coverage=read(out / 'sample_coverage.json') if (out / 'sample_coverage.json').exists() else None,
        metrics_available=bool(paired_rows), partial_journals_excluded_from_pooled=True,
        cross_stage_state_continuity_verified=bool(paired_rows),
        missing_metrics=dict(exact_change_response_delay=manifest['response_delay_reason'],
            association_identity_error_rate='No reliable correspondence between predicted anchor identity and GT instance; geometric matching is only a diagnostic.',
            stale_response_duration='Exact change timestamp and continuously observed answer are unavailable.',
            ambiguous_association_count='Commit does not publish an aggregate ambiguity counter.',
            candidate_entry_validation_failure='No complete per-entry rejection counter; token verifier rejection rate is separately reported.'),
        memory_definition='MiB logical retained payload estimate: Python graph + deduplicated tensor/array storage; not process RSS/allocator allocation.',
        query_group_summary=query_groups,
        primary_accuracy_scope='All five queries with reliable known factual GT, unchanged main evaluator; fixed three and original QA separated in query_group_summary.csv.',
        original_qa_generation_quality=None,
        generation_quality_reason='Existing scorer checks limited factual content, not holistic generation quality; no unvalidated quality score invented.',
        horizon_limitation=manifest['horizon_limitation'])
    write(out / 'summary.json', summary)
    render_report(out, manifest, summary, stages)
    if paired_rows:
        plots(out, paired_rows, paired_events, stages)


def render_report(out, manifest, summary, stages):
    fmt = lambda v: 'N/A' if v is None else f'{v:.3f}'
    lines = ['# WorldStream 连续运行与状态老化', '', f"状态：{summary['status']}；Smoke：{manifest['smoke']}；已完成配对重复：{len(summary['completed_paired_repeats'])}/{manifest['effective_repeats']}。", '',
        '## 连续序列', '', '| Scene | 视频时长 (s) | 测量帧数 | Early / Middle / Late 帧数 |', '|---|---:|---:|---|']
    for s in manifest['scenes']:
        lines.append(f"| {s['name']} / {s['scene_id']} | {s['measured_duration_seconds']:.3f} | {len(s['tokens'])} | {list(s['stage_counts'].values())} |")
    lines += ['', manifest['horizon_limitation'], '阶段仅用于离线统计；不在阶段边界重置。每次独立运行、每个日志起点初始化；Native 每帧清空感知时序记忆。', '',
        '## 表一：不同运行阶段的系统性能', '',
        '| Stage | Method | Factual QA Accuracy (%) | Stale Answer Rate (%) | Avg E2E (ms) | P95 E2E (ms) | FPS | Scored Queries | Stale Denominator |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in stages:
        lines.append('| ' + ' | '.join([r['stage'], r['method']] + [fmt(r[k]) for k in
            ('factual_qa_accuracy_percent', 'stale_answer_rate_percent', 'average_e2e_ms', 'p95_e2e_ms', 'sustained_fps')]
            + [str(r['scored_queries']), str(r['stale_eligible_queries'])]) + ' |')
    lines += ['', '## Peak Memory / Cache Size', '',
        '| Stage | Method | Peak GPU Allocated (MiB) | Peak GPU Reserved (MiB) | Peak OGSM (MiB) | Peak DAVP (MiB) | Peak QCSE (MiB) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in stages:
        lines.append('| ' + ' | '.join([r['stage'], r['method']] + [fmt(r.get(k)) for k in
            ('peak_gpu_allocated_mb', 'peak_gpu_reserved_mb', 'ogsm_peak_state_size_mb',
             'davp_peak_cache_size_mb', 'qcse_peak_cache_size_mb')]) + ' |')
    lines += ['', '## 原始 QA 补充结果', '',
        '每帧严格执行三个固定事实问题和两个原始 QA，缺少任一原问题即拒绝运行。主表沿用全部五问中 known 事实评分；固定三问与两个原始 QA 的独立评分和分母见 query_group_summary.csv。原始 QA 只提供现有评价器支持的有限事实分，完整生成质量为 N/A，不将其宣称为生成质量。', '']
    lines += ['', '## 表二：WorldStream 状态及缓存老化', '',
        '| Stage | OGSM State (MiB) | DAVP Cache (MiB) | Avg Reused Window Age (frames) | QCSE Cache (MiB) | Token Acceptance (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in stages:
        if r['method'] == 'worldstream':
            lines.append('| ' + ' | '.join([r['stage']] + [fmt(r[k]) for k in
                ('ogsm_state_size_mb', 'davp_cache_size_mb', 'average_window_reuse_age_frames', 'qcse_cache_size_mb', 'token_acceptance_percent')]) + ' |')
    lines += ['', '## 统计口径及可用性', '',
        '- Accuracy 复用主实验 grade，对 known 查询平均事实分，保留原自然 QA 部分分；UNKNOWN 排除并单列，完全正确率另存 CSV。',
        '- Stale 只依据同一问题的真实前次可比 GT 与当前 GT；旧 GT 满分、当前 GT 未满分才算过期回答。没有变化分母时为 N/A。',
        '- 原始 QA、三条主实验固定事实问题及 GT 均复用，不向模型传入 GT；问答收据与评分保存以供审计。',
        '- E2E 包含原始读图/预处理、H2D、问题编码、完整模型与回答导出；冷首帧计入。FPS = 帧数 / 累计 E2E 秒，沿用 main_protocol_v1。',
        '- 只读 instrumentation 在计时之后；其耗时与包含审计的辅助处理墙钟吞吐另存。',
        '- GPU Peak 是每帧五问 E2E 区间的 CUDA allocator 高水位（包含驻留模型），阶段取最大值；reset_peak_memory_stats 只重置计数，不清缓存。Allocated 与 Reserved 分列，不等于设备全部显存。',
        '- Host RSS 只在帧端点采样，max_observed_host_rss 不是阶段真实峰值；ru_maxrss 明确记录为进程生命周期高水位，可能包含加载阶段。缓存峰值是逐帧结束后保留载荷的最大值。',
        '- 存储单位 MiB；状态和候选缓存为逻辑对象图大小估算，计 Python 元数据和去重后的数组/张量存储。DAVP 另存准确 tensor bytes。并非显存分配器占用或进程 RSS。',
        '- 表二采用逐帧平均，峰值另存 CSV。年龄分布按锚点/窗口/条目观测汇总，比例按分子分母合并，不平均百分比。',
        '- OGSM 的 track_birth_timestamp 存储年龄、source_timestamp 观测年龄、last_verified_timestamp 验证年龄分别报告；长存储年龄不代表当前观测有效。关联成功率仅表示几何一对一匹配，并非真实身份准确率。',
        '- DAVP 缓存窗口数按 prefix block × camera × window 计；复用年龄仅统计本帧未计算窗口，0 个复用窗口时为 N/A。全局年龄与原生刷新事件直接读实现。',
        '- QCSE 读取 answer_memory.records/acceptance/performance；运行时 cache_bytes=0 指没有持久 KV，不能当作候选缓存大小。超龄条目可能仍驻留，但检索会拒绝，不假定会立即淘汰。',
        '- QCSE 接受率为实际验证接受 Token / 实际提出 Token，拒绝率为其补数；无候选 Token 时为 N/A。按实际候选来源年龄分桶另存 CSV。',
        '- 更新频率采用视频时间域，首帧 interval=0，其余为当前与前帧时间差；不能解释为真实实时部署吞吐。',
        '- 精确 Change Response Delay、Stale Response Duration：N/A。关键帧标注不能定位精确物理变化时间，因此不生成 change_response.csv。',
        '- Ambiguous Association Count、完整的候选条目验证失败率：N/A，现有接口未发布完整计数；不从不相关字段猜测。',
        '- OGSM 保存真实关联对供追溯，几何匹配率并非错误身份关联率；尚无预测锚点与 GT identity 的可靠对应，因此不虚构 identity error rate。',
        '- 每个方法/重复为独立子进程，CUDA 上下文隔离；相机、位姿、问题、RNG 与模型配置配对核验。',
        '- 中断保留 fsync 后逐帧记录；报表可读取完整行，忽略最后半行。部分 attempt 单列，不进入配对汇总。恢复跳过完整运行，失败运行由新进程从首帧重跑，绝不冒充完整状态恢复。',
        f"- Git Commit：{manifest['git_commit'] or 'N/A（当前分发目录无法通过 Git 读取提交；manifest 保留源码 SHA256）'}。", '',
        '## 实际观测', '']
    if summary['metrics_available']:
        for method in METHODS:
            early = next(r for r in stages if r['method'] == method and r['stage'] == 'Early')
            late = next(r for r in stages if r['method'] == method and r['stage'] == 'Late')
            lines.append(f"- {method}：Early/Late 平均 E2E = {fmt(early['average_e2e_ms'])}/{fmt(late['average_e2e_ms'])} ms；事实准确率 = {fmt(early['factual_qa_accuracy_percent'])}/{fmt(late['factual_qa_accuracy_percent'])}%。")
        lines.append('- 已检查完整配对收据中的跨阶段状态版本连续。Smoke/短场景的结果不足以证明小时级稳定性。')
    else:
        lines.append('尚无完整配对 GPU 推理记录；表中的 N/A 不是 0。未从其他实验导入缓存统计或构造数值，未生成无数据图。')
    tmp = out / 'report.md.tmp'
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(out / 'report.md')
    from .report_statistics import append_statistics
    append_statistics(out, manifest, summary)
    # Keep a stable Chinese entry point for the full automatically generated report.
    (out / '结果.md').write_text((out / 'report.md').read_text())


def plots(out, frames, events, stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    folder = out / 'figures'
    folder.mkdir(exist_ok=True)
    colors = dict(native='#526777', worldstream='#157F83')

    def save(fig, name):
        fig.tight_layout()
        for suffix in ('pdf', 'png'):
            fig.savefig(folder / (name + '.' + suffix), dpi=300, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for method in METHODS:
        # Align by original index within each independent log. Moving bins pool
        # repeated observations; they never concatenate logs into one trajectory.
        part = [r for r in frames if r['method'] == method]
        maximum = max(r['frame_index'] for r in part)
        width = max(1, (maximum + 1) // 12)
        xs, latencies, accuracies = [], [], []
        for start in range(0, maximum + 1, width):
            fs = [r for r in part if start <= r['frame_index'] < start + width]
            frame_ids = {r['frame_id'] for r in fs}
            es = [e for e in events if e['method'] == method and e['scorable'] and
                  e['frame_id'] in frame_ids]
            xs.append(mean([r['frame_index'] + 1 for r in fs]))
            latencies.append(mean([r['e2e_ms'] for r in fs]))
            accuracies.append(mean([100 * e['score'] for e in es]))
        for ax, ys in zip(axes, (latencies, accuracies)):
            ax.plot(xs, [np.nan if v is None else v for v in ys], marker='o', markersize=3,
                    label=method, color=colors[method])
            ax.set_xlabel('Chronological frame index within log')
            ax.grid(alpha=.2)
    axes[0].set_ylabel('Average E2E latency (ms)')
    axes[1].set_ylabel('Factual QA accuracy (%)')
    axes[1].set_ylim(0, 102)
    axes[0].legend(frameon=False)
    save(fig, 'figure1_long_run_performance')
    world = [r for r in stages if r['method'] == 'worldstream']
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.7))
    for ax, key, label in zip(axes, ('ogsm_peak_state_size_mb', 'davp_peak_cache_size_mb', 'qcse_peak_cache_size_mb'),
                              ('OGSM world state', 'DAVP visual cache', 'QCSE candidate cache')):
        ys = [r.get(key) for r in world]
        ax.bar(range(len(STAGES)), [np.nan if v is None else v for v in ys], color='#157F83', width=.6)
        ax.set_xticks(range(len(STAGES)), STAGES)
        ax.set_title(label)
        ax.set_ylabel('Peak retained payload (MiB)')
        ax.grid(axis='y', alpha=.2)
    save(fig, 'figure2_state_cache_size')
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.7))
    for ax, key, label in zip(axes, ('average_window_reuse_age_frames', 'average_global_anchor_age_frames', 'token_acceptance_percent'),
                              ('DAVP reused window age (frames)', 'DAVP global anchor age (frames)', 'QCSE Token acceptance (%)')):
        ys = [r.get(key) for r in world]
        ax.plot(range(len(STAGES)), [np.nan if v is None else v for v in ys], marker='o', color='#157F83')
        ax.set_xticks(range(len(STAGES)), STAGES)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        if all(v is None for v in ys):
            ax.text(.5, .5, 'N/A: no proposed/reused tokens', transform=ax.transAxes, ha='center', fontsize=7)
        ax.grid(alpha=.2)
    save(fig, 'figure3_cache_aging_reuse')
