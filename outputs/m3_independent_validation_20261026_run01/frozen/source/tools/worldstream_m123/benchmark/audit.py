"""Offline reconciliation of actual rows, model calls and protected resources."""
import argparse
from collections import Counter
from pathlib import Path

from .common import sha256, write
from .report import read_json, read_rows
from .metrics import distribution, ratio, quality_summary


def validate_execution_row(row, expected_questions=1):
    """Reconcile completed flags with observed executor work, never infer speedup."""
    trace = row.get('module_execution') or {}
    m1, m2, m3 = (trace.get(k, {}) for k in ('m1', 'm2', 'm3'))
    visual, language = row.get('m2') or {}, row.get('m3') or {}
    checks = dict(request=trace.get('completed') is True,
        identity=all(trace.get(k) == row.get(k) for k in ('scene_id', 'frame_id', 'timestamp')),
        m1=all(m1.get(k) for k in ('precheck_entered', 'precheck_completed', 'commit_entered', 'commit_completed'))
            and m1.get('transaction_status') in ('committed', 'rejected'),
        m2=m2.get('runtime_entered') and m2.get('execution_completed') and visual.get('executor') == 'WindowExecutor'
            and visual.get('runtime_entered') and visual.get('execution_completed'),
        m3=m3.get('runtime_entered') and m3.get('execution_completed')
            and language.get('runtime_entered') and language.get('execution_completed')
            and len(language.get('requests', [])) == expected_questions and expected_questions > 0,
        no_native_generate=not row.get('stages', {}).get('native_generation', {}).get('calls', 0))
    identity = row.get('identity') or {}
    checks['run_identity'] = (trace.get('identity') == identity and bool(identity.get('config'))
        and bool(identity.get('checkpoint')) and all(isinstance(identity.get(k), str) and len(identity[k]) == 64
            for k in ('config_sha256', 'checkpoint_sha256', 'source_manifest_sha256')))
    stages = [s for s in visual.get('stages', []) if 'total_windows' in s]
    reused = sum(s['reused_windows'] for s in stages)
    checks['window_counts'] = bool(stages) and all(s['executed_windows'] + s['reused_windows'] == s['total_windows'] for s in stages)
    checks['observed_attention'] = sum(s['executed_windows'] for s in stages) == sum(row.get('observed_window_attention_batches', {}).values())
    checks['actual_reuse'] = (visual.get('actual_skipped_windows') == reused
        and visual.get('actual_window_reuse') == bool(reused) and visual.get('full_recompute') == (reused == 0))
    for i, request in enumerate(language.get('requests', [])):
        checks[f'question_{i}'] = (request.get('executor') == 'PrefillExecutor'
            and request.get('prefill_entered') and request.get('prefill_completed') and request.get('decode_completed')
            and request.get('full_prefill') == (request.get('reused_tokens', 0) == 0)
            and request.get('reused_tokens', 0) <= request.get('matched_prefix', 0)
            and (not request.get('historical_text_consumed') or request.get('state_retrieval', {}).get('admitted')))
    return [key for key, passed in checks.items() if not passed]


def execution_summary(dest):
    """Summarize only persisted evidence from the existing worker, including failure."""
    dest = Path(dest)
    rows = read_rows(dest/'per_frame_metrics.jsonl')
    failed = read_rows(dest/'request_failures.jsonl')
    questions = read_rows(dest/'per_question_metrics.jsonl')
    traces = [(r.get('module_execution') or {}) for r in rows + failed]
    language = [q for r in rows for q in (r.get('m3') or {}).get('requests', [])]
    visuals = [r.get('m2') or {} for r in rows]
    reasons = lambda values: dict(Counter(str(x) for x in values if x))
    rate = lambda n, d: dict(count=n, denominator=d, rate=ratio(n, d))
    timings = lambda records: {k: distribution(r[k] for r in records if isinstance(r.get(k), (int, float)))
        for k in sorted({k for r in records for k in r if k.endswith('_ms') or k in ('precheck_total', 'observation_validation')})}
    m1 = [t.get('m1', {}) for t in traces]
    totals = distribution(r['request_ms'] for r in rows)
    problems = {r['frame_id']: validate_execution_row(r, len((r.get('m3') or {}).get('requests', []))) for r in rows}
    problems = {k:v for k,v in problems.items() if v}
    stage_times = {name: dict(gpu_stream_span_ms=distribution(r['stages'][name]['gpu_span_ms'] for r in rows if name in r['stages']),
        host_enqueue_ms=distribution(r['stages'][name]['host_wall_ms'] for r in rows if name in r['stages']))
        for name in sorted({k for r in rows for k in r.get('stages', {})})}
    per_frame = [dict(frame_id=r['frame_id'], total_windows=sum(s.get('total_windows', 0) for s in v.get('stages', [])),
        executed_windows=sum(s.get('executed_windows', 0) for s in v.get('stages', [])),
        reused_windows=sum(s.get('reused_windows', 0) for s in v.get('stages', []))) for r,v in zip(rows,visuals)]
    stage_reasons = [s.get('full_update_reason') for v in visuals for s in v.get('stages', [])]
    result = dict(status='PASS' if rows and not failed and not problems and read_json(dest/'status.json', {}).get('status') == 'COMPLETED' else 'FAIL',
        frames=len(rows), requests_attempted=len(traces), failures=rate(len(failed), len(traces)),
        failure_reasons=reasons((t.get('error') or {}).get('message') for t in traces),
        failure_stages=reasons((t.get('error') or {}).get('stage') for t in traces), contract_violations=problems,
        sequence=read_json(dest/'sequence.json', {}), checkpoint_validation=read_json(dest/'checkpoint_validation.json', {}).get('status'),
        warmup_frames=read_json(dest/'metrics.json', {}).get('warmup_frames'),
        latency_ms=totals, throughput_frames_s=1000*len(rows)/totals['total'] if totals['total'] else None,
        peak_gpu_allocated_bytes=max((r['peak_gpu_allocated_bytes'] for r in rows), default=None),
        peak_gpu_reserved_bytes=max((r['peak_gpu_reserved_bytes'] for r in rows), default=None),
        stage_timings=stage_times,
        m1=dict(precheck=rate(sum(bool(t.get('precheck_completed')) for t in m1), len(m1)),
            commit=rate(sum(bool(t.get('commit_completed')) for t in m1), len(m1)),
            transactions=reasons(t.get('transaction_status') for t in m1),
            epochs=reasons(str(t.get('state_epoch')) for t in m1),
            valid_states=distribution(t['valid_states'] for t in m1 if 'valid_states' in t),
            expired_states_total=sum(t.get('expired_states', 0) for t in m1),
            frames_with_available_history=sum(t.get('valid_states', 0) > 0 for t in m1),
            refresh_reasons=reasons(reason for t in m1 for reason in t.get('full_refresh_reasons', [])),
            precheck_timings=timings([r.get('m1_cpu_timings_ms', {}) for r in rows]),
            commit_timings=timings([r.get('m1_commit_timings_ms', {}) for r in rows]),
            semantics='valid states are observation support only, never physical identity certification'),
        m2=dict(actual_window_reuse=rate(sum(bool(v.get('actual_window_reuse')) for v in visuals), len(rows)),
            runtime_completed=rate(sum(bool(t.get('m2', {}).get('runtime_entered') and t.get('m2', {}).get('execution_completed')) for t in traces), len(traces)),
            full_recompute=rate(sum(bool(v.get('full_recompute')) for v in visuals), len(rows)),
            per_frame=per_frame, actual_skipped_attention_windows=sum(v.get('actual_skipped_windows', 0) for v in visuals),
            full_recompute_reasons=reasons(v.get('full_recompute_reason') for v in visuals),
            cache_published_frames=sum(bool(v.get('cache_saved')) for v in visuals),
            publication_failures=reasons(v.get('cache_publication_error') for v in visuals),
            cache_miss_or_invalidation_reasons=reasons(v.get('full_update_reason') for v in visuals),
            cost_table_status=reasons(v.get('cost_table_status') for v in visuals),
            stage_admission_reasons=reasons(stage_reasons),
            cost_admission_rejection_stages=sum(x in ('unprofiled_cost', 'estimated_cost_not_beneficial', 'cost_schema_mismatch', 'invalid_cost_table') for x in stage_reasons),
            quality_or_approval_rejection_stages=sum(bool(x and ('quality' in x or 'evidence' in x)) for x in stage_reasons),
            admission_caveat='Counts cover reached stage decisions only; a cache miss can prevent reaching cost/quality admission. cost_table_status=ready means parsed, not production approval.',
            numerical_checks=[c for r in rows for c in r.get('m2_reuse_checks', [])],
            host_diagnostics=timings([{k:v for k,v in d.items() if k.startswith('M2_') and k.endswith('_ms')} for d in visuals]),
            timing_caveat='M2_*_ms are unsynchronized host diagnostics, not GPU kernel times; GPU spans in stage_timings'),
        m3=dict(requests=len(language), runtime_completed=rate(sum(bool(t.get('m3', {}).get('runtime_entered') and t.get('m3', {}).get('execution_completed')) for t in traces), len(traces)),
            prefill_completed=sum(bool(q.get('prefill_completed')) for q in language),
            decode_completed=sum(bool(q.get('decode_completed')) for q in language),
            historical_text_consumed=rate(sum(bool(q.get('historical_text_consumed')) for q in language), len(language)),
            history_reasons=reasons(q.get('state_retrieval', {}).get('reason') for q in language),
            kv_reuse_requests=sum(q.get('reused_tokens', 0) > 0 for q in language),
            reused_tokens_total=sum(q.get('reused_tokens', 0) for q in language),
            reused_tokens_per_request=distribution(q.get('reused_tokens', 0) for q in language),
            matched_prefix_requests=sum(q.get('matched_prefix', 0) > 0 for q in language),
            prefix_miss_requests=sum(q.get('matched_prefix', 0) == 0 for q in language),
            full_prefill=sum(bool(q.get('full_prefill')) for q in language),
            full_prefill_reasons=reasons(q.get('full_prefill_reason') for q in language),
            prefix_admission_reasons=reasons(q.get('prefix_admission') for q in language),
            combined_quality_cost_rejections=sum(q.get('prefix_admission') == 'prefix_evidence_unavailable' for q in language),
            historical_quality_rejections=sum(q.get('state_retrieval', {}).get('reason') == 'historical_context_quality_unverified' for q in language),
            admission_caveat='Existing prefix scheduler reports combined quality/cost rejection; these causes cannot be counted independently from that reason.',
            cache_writes=sum(q.get('cache_stored') is True for q in language),
            cache_write_failures=sum(q.get('cache_stored') is False for q in language),
            cache_bytes=distribution(q['cache_bytes'] for q in language if 'cache_bytes' in q),
            cache_budget_bytes=distribution(q['cache_budget_bytes'] for q in language if 'cache_budget_bytes' in q),
            timings=timings(language)),
        semantic_quality=quality_summary(questions), official_detection=None, official_map=None,
        limits=['Short sequence reference only; no baseline speedup measurement, no production approval.',
            'Detection/Map outputs saved in predictions.pkl; official full-split accuracy NOT_RUN.',
            'CUDA events measure stream spans including host gaps; nested spans must not be added.',
            'Warmup excluded; analysis hooks and profiling synchronization affect latency.'])
    write(dest/'execution_summary.json', result)
    lines = ['# M1 + M2 + M3 真实 GPU 短序列执行审计', '',
        f"状态：{result['status']}；有效帧：{len(rows)}；问题：{len(language)}；失败请求：{len(failed)}。", '',
        '本表由同目录 JSONL 自动汇总；短序列参考值不代表完整集精度、加速收益或生产复用批准。', '',
        '| 项目 | 本轮实测 |', '|---|---|',
        f"| 端到端均值 / P50 / P95（ms） | {totals['mean']} / {totals['p50']} / {totals['p95']} |",
        f"| 顺序吞吐（frame/s） | {result['throughput_frames_s']} |",
        f"| M1 Precheck / Commit 完成 | {result['m1']['precheck']['count']} / {result['m1']['commit']['count']} |",
        f"| M2 实际复用帧 / 全量帧 | {result['m2']['actual_window_reuse']['count']} / {result['m2']['full_recompute']['count']} |",
        f"| M3 历史消费 / KV 复用请求 | {result['m3']['historical_text_consumed']['count']} / {result['m3']['kv_reuse_requests']} |", '',
        '完整阶段耗时、原因分布、窗口计数、缓存写入/占用、问答评分见 [execution_summary.json](execution_summary.json)。', '',
        '| 阶段（CUDA stream span） | 次数 | 均值 ms | P50 ms | P95 ms |', '|---|---:|---:|---:|---:|']
    for name, value in stage_times.items():
        d = value['gpu_stream_span_ms']
        lines.append(f"| {name} | {d['n']} | {d['mean']:.3f} | {d['p50']:.3f} | {d['p95']:.3f} |")
    lines += ['', 'CUDA Event 在请求末同步后读取，包含 CPU 提交间隙；嵌套跨度不可相加。M2 原有 host 分段不是 GPU 耗时。', '',
        'M1 有效性只表示普通观测支持。窗口输出复用、跳过 Attention、端到端时间节省是不同事件；历史文本消费、KV 复用、生成成功也是不同事件。', '',
        '未运行官方完整集 Detection/Map 指标、配对基线、历史文本独立质量认证或新增成本认证。无实际复用时没有复用数值差异样本。']
    (dest/'execution_report.md').write_text('\n'.join(lines)+'\n')
    return result


def audit(root, hash_checkpoint=False):
    root = Path(root)
    manifest = read_json(root/'run_manifest.json', {})
    workers, failures = {}, []
    for dest in sorted((root/'experiments').glob('*')):
        if not dest.is_dir():
            continue
        status = read_json(dest/'status.json', {})
        rows = read_rows(dest/'per_frame_metrics.jsonl')
        questions = read_rows(dest/'per_question_metrics.jsonl')
        sequence = read_json(dest/'sequence.json', {})
        validation = read_json(dest/'checkpoint_validation.json', {})
        checks = dict(completed=status.get('status') == 'COMPLETED',
            strict_weights=validation.get('status') == 'PASS' and not validation.get('missing_keys')
                and not validation.get('unexpected_keys') and not validation.get('shape_mismatches'),
            unique_frames=len({r['frame_id'] for r in rows}) == len(rows),
            sequence_exact=[r['frame_id'] for r in rows] == sequence.get('tokens'),
            native_baseline_camera_accounting=all(r['native_eva_cameras']==6 for r in rows if not r.get('m2')),
            window_execution_accounting=all(sum(r.get('observed_window_attention_batches', {}).values()) ==
                sum(s.get('executed_windows',0) for s in r['m2'].get('stages',[]))
                for r in rows if (r.get('m2') or {}).get('cache_boundary') == 'window'),
            matched_question_frames=all(q['frame_id'] in {r['frame_id'] for r in rows} for q in questions),
            one_real_question_per_frame=len(questions) == len(rows),
            gt_coverage=all(q['quality'] is not None for q in questions if q['workload'] == 'language'),
            same_executor_accounting=all(q['native_generation_executed'] or q.get('prefill_executor_executed',False) for q in questions))
        if (dest/'m1_scene_integrity.json').is_file():
            scenes = read_json(dest/'m1_scene_integrity.json')
            checks['m1_scene_ledger'] = all(x.get('ledger', {}).get('status') == 'PASS' and 'metrics' in x for x in scenes.values())
        for name, passed in checks.items():
            if not passed:
                failures.append(dest.name+':'+name)
        workers[dest.name] = dict(checks=checks, frames=len(rows), questions=len(questions),
            scenes=len({r['scene_id'] for r in rows}),
            native_generations=sum(q['native_generation_executed'] for q in questions),
            m3_reasons=dict(Counter(q.get('m3_reason') for q in questions)))
    protected = {p: Path(p).is_file() and sha256(p) == expected
                 for p, expected in manifest.get('preflight', {}).get('protected_hashes', {}).items()}
    failures += ['protected_changed:'+p for p, passed in protected.items() if not passed]
    checkpoint = None
    if hash_checkpoint:
        argv = manifest.get('preflight', {}).get('resources', {})
        checkpoint_path = next((p for p in argv if p.endswith('.pth') and 'iter_' in p), None)
        if checkpoint_path:
            expected = manifest['preflight']['checkpoint']['sha256']
            actual = sha256(checkpoint_path)
            checkpoint = dict(path=checkpoint_path, expected=expected, actual=actual, unchanged=expected == actual)
            if expected != actual:
                failures.append('checkpoint_changed')
    value = dict(status='PASS' if workers and not failures else 'FAIL', failures=failures,
        experiments=workers, protected_unchanged=protected, checkpoint=checkpoint,
        official_full_validation='NOT_RUN' if not read_json(root/'status.json', {}).get('official_detection_available') else 'See experiment statuses',
        scope='PASS reconciles the experiments actually run; it never means blocked full validation completed')
    write(root/'audit.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--hash-checkpoint', action='store_true')
    args = parser.parse_args()
    print(audit(args.run_dir, args.hash_checkpoint)['status'])
