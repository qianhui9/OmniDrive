"""Regenerate all reports from this run's measured rows, including partial runs."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from tools.worldstream_m123.stateful_opportunity.common import sha256, write
from .metrics import compare, distribution, paired_scene_ci, quality_summary, ratio


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def read_rows(path):
    rows = []
    if path.is_file():
        with path.open() as handle:
            for line in handle:
                if not line.endswith('\n'):
                    # A live worker may currently be writing the final record.
                    # Completed malformed records still raise; never hide them.
                    break
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def group_summary(rows, questions):
    times = distribution(r['request_ms'] for r in rows)
    stages = {name: dict(host_wall_ms=distribution(r['stages'][name]['host_wall_ms'] for r in rows if name in r['stages']),
                        gpu_stream_span_ms=distribution(r['stages'][name]['gpu_span_ms'] for r in rows if name in r['stages']))
              for name in sorted({s for r in rows for s in r['stages']})}
    return dict(frames=len(rows), unique_frames=len({r['frame_id'] for r in rows}),
        scenes=len({r['scene_id'] for r in rows}), repeats=len({r['repeat'] for r in rows}),
        request_ms=times, model_ms=distribution(r['model_ms'] for r in rows),
        preprocessing_ms=distribution(r['preprocessing_ms'] for r in rows),
        throughput_frames_s=1000*len(rows)/times['total'] if times['total'] else None,
        peak_gpu_allocated_bytes=max((r['peak_gpu_allocated_bytes'] for r in rows), default=None),
        quality=quality_summary(questions),
        stage_latency=stages,
        full_validation=all(r['full_validation'] for r in rows) if rows else False)


def module_summary(rows, questions, workers):
    result = {}
    for condition in ('B1', 'B2', 'B3'):
        rr = [r for r in rows if r['condition'] == condition]
        qq = [q for q in questions if q['condition'] == condition]
        if not rr:
            result[condition] = dict(status='NOT_RUN', m1=None, m2=None, m3=None)
            continue
        counts, resets, invalidations, physical_rejections = Counter(), Counter(), Counter(), Counter()
        latency = defaultdict(list)
        for r in rr:
            c, p = r.get('m1_commit', {}), r.get('m1_precheck', {})
            counts['committed'] += len(c.get('committed_anchor_ids', []))
            counts['associated'] += len(c.get('association_pairs', []))
            counts['created'] += len(c.get('committed_anchor_ids', []))-len(c.get('association_pairs', []))
            tracks = c.get('association_tracks') or {}
            for key in ('legal_edges', 'ambiguous_edges', 'admissible_edges', 'matched', 'dormant_candidates'):
                if key in tracks:
                    counts['association_'+key] += tracks[key]
            counts['expired'] += len(p.get('expired_anchor_ids', []))
            counts['valid_history'] += r.get('m1_valid_anchors', 0)
            counts['history_candidates'] += r.get('m1_historical_candidates', 0)
            counts['transactions_committed'] += p.get('transaction_status') == 'committed'
            if p.get('reset_reason'):
                resets[p['reset_reason']] += 1
            for value in p.get('anchors', {}).values():
                invalidations.update(value.get('reasons', []))
                cert = value.get('physical_certification')
                if cert is not None:
                    counts['precheck_physical_evaluations'] += 1
                    counts['precheck_physical_producer_available'] += bool(cert.get('producer_available'))
                    physical_rejections.update(cert.get('rejection_reasons', []))
            certifications = c.get('post_perception_physical_certifications', [])
            counts['physical_candidates'] += len(certifications)
            for cert in certifications:
                counts['physical_passed'] += bool(cert.get('certified', False))
                physical_rejections.update(cert.get('rejection_reasons', []))
            for key, value in c.items():
                if key.endswith('_ms') and isinstance(value, (float, int)):
                    latency[key].append(value)
            for key, value in p.get('timings_ms', {}).items():
                if isinstance(value, (float, int)):
                    latency['precheck_'+key].append(value)
        identities = {name: x.get('metrics', {}).get('m1') for name, x in workers.items()
                      if x.get('args', {}).get('condition') == condition and x.get('metrics', {}).get('m1')}
        m1 = dict(status='MEASURED', event_counts=dict(counts), resets=dict(resets),
            state_age_s=distribution(v for r in rr for v in r.get('m1_state_ages_s', [])),
            retained_state_events=sum(r.get('m1_retained_states', 0) for r in rr),
            physical_phase_counts=dict(Counter(str(r.get('m1_physical_phase')) for r in rr)),
            valid_history_fraction=ratio(counts['valid_history'], counts['history_candidates']),
            physical_rejection_reasons=dict(physical_rejections), invalidation_reasons=dict(invalidations),
            historical_age_s=distribution(a['age_s'] for r in rr for a in r.get('m1_precheck', {}).get('anchors', {}).values() if a.get('age_s') is not None),
            physical_zero_explanation=('Independent physical producer/evidence missing in precheck; post-perception certification had no qualified candidates'
                if counts['precheck_physical_evaluations'] and not counts['precheck_physical_producer_available'] else
                'No candidate reached the separate physical certification phase' if not counts['physical_candidates']
                else 'Physical certification ran; independent evidence/qualification rejections are recorded') if not counts['physical_passed'] else None,
            latency_ms={k: distribution(v) for k, v in latency.items()}, association_by_experiment=identities,
            transaction_failures='See each worker status; only returned successful commits appear in these rows',
            cuda_separable_status='NOT_AVAILABLE: state management runs on CPU; transfer timing retained separately')
        m2rows = [r for r in rr if r.get('m2') is not None]
        m2 = dict(status='NOT_RUN', frames=0)
        if m2rows:
            reused = sum(r['m2'].get('actual_reused_cameras', 0) for r in m2rows)
            native = sum(r['native_eva_cameras'] for r in m2rows)
            checks = [v for r in m2rows for v in r.get('m2_reuse_checks', [])]
            m2 = dict(status='MEASURED', frames=len(m2rows), camera_requests=6*len(m2rows),
                candidate_cameras=sum(r['m2'].get('candidate_cameras', 0) for r in m2rows),
                exact_verified_cameras=sum(r['m2'].get('verified_exact_cameras', 0) for r in m2rows),
                eligible_exact_cameras=sum(r['m2'].get('eligible_exact_cameras', 0) for r in m2rows),
                admitted_frames=sum(r['m2'].get('mode') == 'camera_reuse' for r in m2rows),
                actual_reused_cameras=reused, observed_native_eva_cameras=native,
                camera_skip_audit='PASS' if reused+native == 6*len(m2rows) else 'FAIL',
                reuse_rate=ratio(reused, 6*len(m2rows)),
                fallback_reasons=dict(Counter(r['m2'].get('fallback_reason') or 'admitted' for r in m2rows)),
                cache_creations=sum(bool(r['m2'].get('cache_saved')) for r in m2rows),
                cache_reset_reasons=dict(Counter(r['m2']['global_cache_reset_reason'] for r in m2rows if r['m2'].get('global_cache_reset_reason'))),
                peak_cache_bytes=max(r['m2'].get('cache_bytes', 0) for r in m2rows),
                quality_comparisons=len(checks), exact_equal=sum(v['exact'] for v in checks),
                quality_status='MEASURED' if checks else 'NOT_MEASURED: no actual reuse',
                quality_errors=dict(maximum=max((v['max_abs'] for v in checks), default=None),
                    sample_mean=distribution(v['mean_abs'] for v in checks),
                    sample_p95=distribution(v['p95_abs'] for v in checks)),
                validation_ms=distribution(r['m2']['M2_validation_ms'] for r in m2rows if 'M2_validation_ms' in r['m2']),
                skipped_compute_ms=0. if reused == 0 else None,
                saved_compute_status='MEASURED_ZERO: no EVA cameras skipped' if reused == 0 else 'NOT_MEASURED: requires paired native counterfactual timing',
                net_overhead='Measured by paired B2 minus B1 request times below; historical calibration is excluded')
            m2['boundary']='camera'
            if all(r['m2'].get('cache_boundary')=='window' for r in m2rows):
                window_stages=[s for r in m2rows for s in r['m2'].get('stages',[]) if 'total_windows' in s]
                total=sum(s['total_windows'] for s in window_stages)
                executed=sum(s['executed_windows'] for s in window_stages)
                skipped=sum(s['reused_windows'] for s in window_stages)
                observed=sum(sum(r.get('observed_window_attention_batches',{}).values()) for r in m2rows)
                m2.update(boundary='window',actual_reused_cameras=None,camera_skip_audit='NOT_APPLICABLE: window boundary',
                    exact_verified_cameras=None,eligible_exact_cameras=None,
                    candidate_windows=sum(r['m2'].get('candidate_windows') or 0 for r in m2rows),
                    verified_exact_windows=sum(r['m2'].get('verified_exact_windows',0) for r in m2rows),
                    admitted_frames=sum(r['m2'].get('actual_skipped_windows',0)>0 for r in m2rows),
                    prefix_window_requests=total,executed_prefix_windows=executed,actual_reused_windows=skipped,
                    observed_attention_window_batches=observed,
                    window_skip_audit='PASS' if executed+skipped==total and observed==executed else 'FAIL',
                    reuse_rate=ratio(skipped,total),
                    quality_status='NOT_MEASURED: native feature counterfactual requires separate window validation',
                    skipped_compute_ms=0. if not skipped else None,
                    saved_compute_status='MEASURED_ZERO' if not skipped else 'NOT_MEASURED: use paired native timing')
        m3rows = [r for r in rr if r.get('m3') is not None]
        m3 = dict(status='NOT_RUN', requests=None)
        if m3rows:
            details = [q for r in m3rows for q in r['m3'].get('requests', [])]
            m3 = dict(status='MEASURED', requests=len(qq),
                cache_candidates=sum(d.get('matched_prefix', 0) > 0 for d in details),
                admitted=sum(d.get('reused_tokens', 0) > 0 or d.get('selection', {}).get('admitted', False) for d in details),
                cache_hits=sum(d.get('reused_tokens', 0) > 0 for d in details),
                native_generations=sum(q['native_generation_executed'] for q in qq),
                actual_optimized_requests=sum(not q['native_generation_executed'] for q in qq),
                fallback_reasons=dict(Counter(q.get('m3_reason') or 'per_request_diagnostics' for q in qq)),
                native_count_audit='PASS' if len(qq) == sum(q['native_generation_executed'] for q in qq)+sum(not q['native_generation_executed'] for q in qq) else 'FAIL',
                reused_prefix_tokens=sum(d.get('reused_tokens', 0) for d in details),
                actual_input_tokens=sum(q.get('input_tokens') or 0 for q in qq),
                generated_tokens=sum(q.get('generated_tokens') or 0 for q in qq),
                generated_tokens_per_second=ratio(sum(q.get('generated_tokens') or 0 for q in qq)*1000,
                    sum(q.get('host_generate_call_ms') or 0 for q in qq)),
                generate_call_ms=distribution(q['host_generate_call_ms'] for q in qq if q.get('host_generate_call_ms') is not None),
                quality=quality_summary(qq),
                by_task={t: quality_summary([q for q in qq if q['task_type'] == t]) for t in sorted({q['task_type'] for q in qq})},
                scheduler_host_overhead_ms=distribution(
                    r['stages']['m3_auto_total']['host_wall_ms']-r['stages'].get('native_generation', {}).get('host_wall_ms', 0.)
                    for r in m3rows if 'm3_auto_total' in r['stages']),
                scheduler_scope='M3 runtime host wall minus nested native generate host wall; no CUDA times subtracted')
        result[condition] = dict(status='MEASURED', m1=m1, m2=m2, m3=m3)
    return result


def generate(root):
    root = Path(root)
    manifest = read_json(root/'run_manifest.json', {})
    frames, questions, workers = [], [], {}
    worker_root = root/'experiments'
    if worker_root.exists():
        for dest in sorted(p for p in worker_root.iterdir() if p.is_dir()):
            workers[dest.name] = dict(status=read_json(dest/'status.json', {'status': 'NOT_RUN'}),
                args=read_json(dest/'invocation.json', {}), metrics=read_json(dest/'metrics.json', {}),
                checkpoint_validation=read_json(dest/'checkpoint_validation.json', {}))
            frames += read_rows(dest/'per_frame_metrics.jsonl')
            questions += read_rows(dest/'per_question_metrics.jsonl')
    for name, rows in [('per_frame_metrics.jsonl', frames), ('per_question_metrics.jsonl', questions)]:
        with (root/name).open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    groups = {}
    csv_rows = []
    for condition, workload, full, measurement in sorted({(r['condition'], r['workload'], r['full_validation'], r.get('measurement_set', 'smoke')) for r in frames}):
        rr = [r for r in frames if (r['condition'], r['workload'], r['full_validation'], r.get('measurement_set', 'smoke')) == (condition, workload, full, measurement)]
        qq = [q for q in questions if q['condition'] == condition and q['workload'] == workload and q.get('measurement_set', 'smoke') == measurement]
        expected_full = manifest.get('preflight', {}).get('dataset', {}).get('frames')
        full_coverage = full and len({r['frame_id'] for r in rr}) == expected_full
        scope = 'full_validation' if full_coverage else 'partial_full_validation' if full else 'subset'
        key = f'{condition}/{workload}/{measurement}/'+scope
        s = groups[key] = group_summary(rr, qq)
        s['full_validation'] = full_coverage
        csv_rows.append(dict(condition=condition, workload=workload, measurement_set=measurement, scope=scope,
            frames=s['frames'], unique_frames=s['unique_frames'], scenes=s['scenes'], repeats=s['repeats'],
            request_mean_ms=s['request_ms']['mean'], request_p50_ms=s['request_ms']['p50'],
            request_p95_ms=s['request_ms']['p95'], model_mean_ms=s['model_ms']['mean'],
            throughput_frames_s=s['throughput_frames_s'], questions=s['quality']['questions'],
            valid_gt=s['quality']['valid_gt'], answer_exact_match=s['quality']['answer_exact_match'],
            token_f1=s['quality']['token_f1']))
    fields = list(csv_rows[0]) if csv_rows else ['condition', 'workload', 'scope', 'status']
    with (root/'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    with (root/'stage_latency.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['condition', 'workload', 'measurement_set', 'repeat', 'scene_id', 'frame_id', 'stage', 'calls', 'host_wall_ms', 'gpu_span_ms'])
        writer.writeheader()
        for r in frames:
            for name, stage in r['stages'].items():
                writer.writerow({**{k: r.get(k) for k in ('condition', 'workload', 'measurement_set', 'repeat', 'scene_id', 'frame_id')}, 'stage': name, **stage})
    comparisons = {}
    for workload, measurement in sorted({(r['workload'], r.get('measurement_set', 'smoke')) for r in frames}):
        for base, final in [('B0', 'B3'), ('B0', 'B1'), ('B1', 'B2'), ('B2', 'B3')]:
            # Pair exact frame, scene, repeat, workload AND full/subset scope.
            matching = defaultdict(dict)
            for r in frames:
                if r['workload'] == workload and r.get('measurement_set', 'smoke') == measurement and r['condition'] in (base, final):
                    matching[(r['frame_id'], r['repeat'], r['full_validation'])][r['condition']] = r
            paired = [v for v in matching.values() if base in v and final in v]
            if not paired:
                continue
            b = sum(v[base]['request_ms'] for v in paired)/len(paired)
            f = sum(v[final]['request_ms'] for v in paired)/len(paired)
            comparisons[f'{workload}/{measurement}/{base}->{final}'] = dict(paired_requests=len(paired),
                baseline_mean_ms=b, final_mean_ms=f, incremental_ms=f-b, **compare(b, f),
                ci=paired_scene_ci([v[c] for v in paired for c in (base, final)], base, final))
    native = {(q['workload'], q.get('measurement_set'), q['repeat'], q['frame_id'], q['question']): q for q in questions if q['condition'] == 'B0'}
    paired_questions = [(native[k], q) for q in questions if q['condition'] == 'B3'
                        and (k := (q['workload'], q.get('measurement_set'), q['repeat'], q['frame_id'], q['question'])) in native]
    token_pairs = [(a, b) for a, b in paired_questions if a.get('token_ids') is not None and b.get('token_ids') is not None]
    output_map = {(r['workload'], r.get('measurement_set'), r['repeat'], r['frame_id']): r for r in frames if r['condition'] == 'B0'}
    perception_pairs = [(output_map[k], r) for r in frames if r['condition'] == 'B3'
                        and (k := (r['workload'], r.get('measurement_set'), r['repeat'], r['frame_id'])) in output_map]
    overall = dict(groups=groups, paired_latency_comparisons=comparisons,
        gpu_active_kernel_time_status='NOT_MEASURED: event spans include host gaps; no claim of summed active kernel time',
        official_map_comparison=dict(B0=None, B3=None, delta=None, status='NOT_RUN: no completed full-split map evaluation'),
        official_detection_comparison={name: dict(B0=None, B3=None, delta=None,
            status='NOT_AVAILABLE' if not manifest.get('preflight', {}).get('dataset', {}).get('tables', {}).get('validation_annotation_integrity', {}).get('status') == 'PASS' else 'NOT_RUN')
            for name in ('mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE')},
        official_by_experiment={k: v['metrics'] for k, v in workers.items() if v['metrics'].get('full_validation')},
        paired_language=dict(questions=len(paired_questions), token_comparisons=len(token_pairs),
            token_equal=sum(a['token_ids'] == b['token_ids'] for a, b in token_pairs),
            token_equal_rate=ratio(sum(a['token_ids'] == b['token_ids'] for a, b in token_pairs), len(token_pairs))),
        paired_perception=dict(comparisons=len(perception_pairs),
            exact_equal=sum(a['output_digest'] == b['output_digest'] for a, b in perception_pairs),
            exact_equal_rate=ratio(sum(a['output_digest'] == b['output_digest'] for a, b in perception_pairs), len(perception_pairs))),
        scope_note='All counts refer only to new run rows; repeated requests are labelled. No historic microbenchmark is a system result.')
    modules = module_summary(frames, questions, workers)
    # Populate official values only from completed full-split evaluators.
    official = {}
    for worker in workers.values():
        if worker['metrics'].get('full_validation') and worker['metrics'].get('official_detection'):
            official[worker['args'].get('condition')] = worker['metrics']['official_detection']
    for name, value in overall['official_detection_comparison'].items():
        for condition in ('B0', 'B3'):
            value[condition] = next((v for k, v in official.get(condition, {}).items() if k.endswith('/'+name)), None)
        if value['B0'] is not None and value['B3'] is not None:
            value.update(delta=value['B3']-value['B0'], status='MEASURED')
    official_maps = {v['args'].get('condition'): v['metrics']['official_map'] for v in workers.values()
                     if v['metrics'].get('full_validation') and v['metrics'].get('official_map') is not None}
    if 'B0' in official_maps and 'B3' in official_maps:
        overall['official_map_comparison'].update(B0=official_maps['B0'], B3=official_maps['B3'],
                                                  status='MEASURED: see raw protocol metrics')
    protected = manifest.get('preflight', {}).get('protected_hashes', {})
    unchanged = {p: Path(p).is_file() and sha256(p) == h for p, h in protected.items()}
    status = dict(experiments={k: v['status'] for k, v in workers.items()},
        plan=read_json(root/'plan.json', {}), protected_unchanged=unchanged,
        measured_frames=len(frames), measured_questions=len(questions),
        complete=read_json(root/'driver_status.json', {}).get('status') == 'COMPLETED',
        official_detection_available=manifest.get('preflight', {}).get('dataset', {}).get('tables', {}).get('validation_annotation_integrity', {}).get('status') == 'PASS')
    write(root/'overall_metrics.json', overall)
    write(root/'module_metrics.json', modules)
    write(root/'status.json', status)
    integrity = manifest.get('preflight', {}).get('dataset', {}).get('tables', {}).get('validation_annotation_integrity', {})
    lines = ['# OmniDrive 正式路线实际评测报告', '',
        f'运行目录：`{root}`。本报告可由 `python -m tools.worldstream_m123.benchmark.report --run-dir {root}` 重新生成。', '',
        '## 完成范围与限制', '',
        f'- 本次真实测量 {len(frames)} 帧请求、{len(questions)} 个语言请求（包括标注的重复测量）。',
        f'- 官方检测 GT 完整性：`{integrity.get("status", "NOT_AVAILABLE")}`；受影响帧 {integrity.get("affected_frames")}，sample.anns 链接 {integrity.get("linked_annotations")}，反向标注 {integrity.get("reverse_annotations")}。',
        '- GT 链接不一致时不发布官方 mAP/NDS/误差指标；不修改原数据。子集预测一致性不能替代全量官方精度。',
        '- M1 采用原有类别、尺寸、1m 唯一匹配的部分 Oracle 离线评分；不报告未实现的 IDF1/MOTA。',
        '- 生成参数沿用生产设置；以帧 token 确定相同随机种子。模型执行包含原生答案写盘；审计 JSON 序列化和反事实正确性检查在计时之外。',
        '- CUDA 数字是默认流事件跨度，包含 CPU 提交间隙；嵌套阶段不能相加充当 GPU 活跃计算时间。', '',
        '## 本次环境与权重', '',
        f'- Python：`{manifest.get("preflight", {}).get("environment", {}).get("python")}`。',
        f'- GPU：`{manifest.get("preflight", {}).get("environment", {}).get("gpu")}`。',
        f'- Checkpoint SHA256：`{manifest.get("preflight", {}).get("checkpoint", {}).get("sha256")}`。',
        f'- 正式配置、成本表、证书及既有 summary/verification 保持不变：`{all(unchanged.values()) if unchanged else "NOT_VERIFIED"}`。', '',
        '## 实际任务与请求延迟', '',
        '| 条件/任务/范围 | 帧请求/独立帧 | 场景 | 重复 | 平均/P50/P95 ms | 帧/s | 有效GT | EM/F1 |',
        '|---|---:|---:|---:|---|---:|---:|---|']
    def fmt(x):
        return 'null' if x is None else f'{x:.4f}'
    for key, s in groups.items():
        t, q = s['request_ms'], s['quality']
        lines.append(f'| {key} | {s["frames"]}/{s["unique_frames"]} | {s["scenes"]} | {s["repeats"]} | {fmt(t["mean"])}/{fmt(t["p50"])}/{fmt(t["p95"])} | {fmt(s["throughput_frames_s"])} | {q["valid_gt"]} | {fmt(q["answer_exact_match"])}/{fmt(q["token_f1"])} |')
    if not groups:
        lines.append('| NOT_RUN | 0/0 | 0 | 0 | null | null | 0 | null |')
    lines += ['', '## 全量官方任务精度', '',
        '| 指标 | B0 | B3 | B3-B0 | 状态 |', '|---|---:|---:|---:|---|']
    for name, value in overall['official_detection_comparison'].items():
        lines.append(f'| {name} | {fmt(value["B0"])} | {fmt(value["B3"])} | {fmt(value["delta"])} | {value["status"]} |')
    lines += ['', 'mAP/NDS 的百分点变化为上述原始差值 ×100。地图使用现有 OpenLaneV2 指标，未实现的 IoU/mIoU 不填数。',
              '', '地图官方结果：`'+json.dumps(overall['official_map_comparison'], ensure_ascii=False)+'`。']
    lines += ['', '## 配对比较与模块增量', '']
    for key, c in comparisons.items():
        lines.append(f'- {key}：{c["paired_requests"]} 对请求，增量 {fmt(c["incremental_ms"])} ms，延迟变化 {fmt(c["latency_change_percent"])}%，加速比 {fmt(c["speedup"])}。场景置信区间见 JSON。')
    lines += ['', f'检测及地图输出哈希完全一致：{overall["paired_perception"]}。',
              f'同问题原生/Auto token 完全一致：{overall["paired_language"]}。', '',
              '## M1、M2、M3 实际行为', '']
    for cond, value in modules.items():
        lines += [f'### {cond}', '']
        if value['status'] == 'NOT_RUN':
            lines.append('NOT_RUN：本次尚无该条件的真实帧记录。')
            continue
        m1, m2, m3 = (value[k] for k in ('m1', 'm2', 'm3'))
        lines += [f'- M1 事件计数：`{m1["event_counts"]}`。有效历史状态比例 {fmt(m1["valid_history_fraction"])}。',
                  f'- 物理认证：{m1["physical_zero_explanation"] or "见独立认证计数"}；普通关联从不计入物理认证。']
        for name, identity in m1['association_by_experiment'].items():
            lines.append(f'- {name} 关联精度 {fmt(identity["association_precision"])}，错误率 {fmt(identity["wrong_association_rate"])}，有效召回 {fmt(identity["effective_recall"])}，覆盖 {fmt(identity["association_coverage"])}；分子/分母 `{identity["counts"]}`；连续性 `{identity["continuity"]}`。')
        if m2['status'] == 'MEASURED':
            if m2.get('boundary')=='window':
                lines.append(f'- M2 窗口级：实际复用 {m2["actual_reused_windows"]}/{m2["prefix_window_requests"]} 个 block-window，执行 {m2["executed_prefix_windows"]} 个，实际 attention batch 计数 {m2["observed_attention_window_batches"]}，审计 {m2["window_skip_audit"]}。拒绝原因 `{m2["fallback_reasons"]}`。')
            else:
                lines.append(f'- M2：实际复用 {m2["actual_reused_cameras"]}/{m2["camera_requests"]} 相机，原生 EVA {m2["observed_native_eva_cameras"]} 相机，跳过审计 {m2["camera_skip_audit"]}；真实复用质量比较 {m2["quality_comparisons"]} 例。拒绝原因 `{m2["fallback_reasons"]}`。')
        if m3['status'] == 'MEASURED':
            lines.append(f'- M3：请求 {m3["requests"]}，候选 {m3["cache_candidates"]}，准入 {m3["admitted"]}，命中 {m3["cache_hits"]}，原生生成 {m3["native_generations"]}，实际优化 {m3["actual_optimized_requests"]}；原因 `{m3["fallback_reasons"]}`。调度额外 host wall 平均 {fmt(m3["scheduler_host_overhead_ms"]["mean"])} ms。')
    lines += ['', '## 阶段瓶颈（独立阶段跨度，不相加推断端到端）', '']
    for name, s in groups.items():
        stages = [(label, value['gpu_stream_span_ms']['mean']) for label, value in s['stage_latency'].items()
                  if label in ('eva_native', 'detection', 'map', 'prefill', 'decode')]
        lines.append(f'- {name}：'+', '.join(f'{label} {fmt(value)} ms' for label, value in sorted(stages, key=lambda x: -x[1])))
    lines += ['', '## 四个问题的证据边界', '',
        '1. 整体精度：只有全量官方评测完成且 GT 完整时才能判断；本次可用范围和配对效率数值如上。',
        '2. M1：上列部分 Oracle 关联与连续性来自真实提交；独立物理认证、对齐分母及标准 MOT 指标分别报告，不互相替代。',
        '3. M2：只将真实跳过的 EVA 相机计入复用。零复用时跳过计算时间为零；净影响使用配对 B2/B1 实测差值。',
        '4. M3：空批准表走正式 Auto 原生回退。语言质量只对有效 GT 评分；净影响使用配对 B3/B2，联合影响使用 B3/B0。',
        '5. 瓶颈：stage_latency.csv 提供本次原生 EVA、检测、地图、prefill/decode 和模块跨度；未取得数据的阶段标记缺失，不使用历史微基准补齐。', '',
        '## 失败和未完成记录', '',
        '详细状态见 status.json、plan.json、各实验 failure.log 和 raw_logs；失败实验的部分测量保留。', '',
        '```json', json.dumps(status, ensure_ascii=False, indent=2), '```', '']
    (root/'report.md').write_text('\n'.join(lines))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    generate(parser.parse_args().run_dir)
