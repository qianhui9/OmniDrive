"""Identical offline scoring and paired, sequence-clustered statistics."""
from collections import Counter, defaultdict
import math
from pathlib import Path
import numpy as np

from .common import LEVELS, METHODS, csv_write, dump_jsonl, jsonl, read, write


def ratio(numerator, denominator, scale=1.):
    return scale * numerator / denominator if denominator else None


def validate_pair(manifest, windows, rows, runtimes=None):
    expected = [(r, s['scene_id'], t) for r in range(manifest['config']['repeats'])
                for s in manifest['scenes'] for t in s['tokens']]
    by_frame = {t: w for w in windows for t in w['frame_ids']}
    for method in METHODS:
        part = rows[method]
        if [(r['repeat'], r['scene_id'], r['frame_id']) for r in part] != expected:
            raise ValueError('Missing/duplicated/reordered frames: ' + method)
        for row in part:
            if row['execution_status'] != 'COMPLETED' or not math.isfinite(row['e2e_ms']) or row['e2e_ms'] <= 0:
                raise ValueError('Invalid execution/timing receipt')
            if not len(row['questions']) == len(row['groups']) == len(row['answers']) == len(row['requests']):
                raise ValueError('Missing question/answer/request')
            if len(row['questions']) != len(set(row['questions'])) or not all(isinstance(a, str) for a in row['answers']):
                raise ValueError('Duplicate query or invalid prediction')
            w = by_frame[row['frame_id']]
            if row['window_id'] != w['window_id'] or row['stability_level'] != w['stability_level']:
                raise ValueError('GT stability assignment changed in model receipts')
            if not w['start_timestamp'] <= row['timestamp'] < w['end_timestamp']:
                raise ValueError('Invalid timestamp/window assignment')
    for a, b in zip(rows['native'], rows['worldstream']):
        for key in ('frame_id', 'scene_id', 'repeat', 'timestamp', 'questions', 'groups', 'fairness'):
            if a[key] != b[key]:
                raise ValueError('Unpaired input: ' + key)
    if runtimes:
        for key in ('checkpoint_sha256', 'device', 'gpu_uuid', 'fp16', 'batch_size', 'generation', 'precision_by_dtype'):
            if runtimes['native'][key] != runtimes['worldstream'][key]:
                raise ValueError('Unpaired runtime: ' + key)


def score_rows(rows, gt_rows, objects, grade_fn=None):
    if grade_fn is None:
        from tools.worldstream_m123.continuous_stream_report import grade as grade_fn
    truth = {(r['frame_id'], r['question']): r for r in gt_rows}
    result = []
    for row in rows:
        token = row['frame_id']
        for qi, (question, answer) in enumerate(zip(row['questions'], row['answers'])):
            gt = truth[token, question]
            if row['timestamp'] != gt['timestamp']:
                raise ValueError('Receipt timestamp differs from authoritative frame metadata')
            quality = grade_fn(question, answer, token, objects)
            known = bool(gt['ground_truth']['known'] and quality.get('known'))
            score = quality.get('score') if known else None
            if known and (not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError('Invalid factual score')
            changed = gt['fact_status'] == 'Changed'
            previous_quality = (grade_fn(question, answer, gt['previous_frame_id'], objects)
                                if changed else None)
            stale_eligible = bool(changed and known and previous_quality.get('known'))
            stale = bool(stale_eligible and score < 1 and previous_quality['score'] == 1)
            result.append(dict(method=row['method'], repeat=row['repeat'], scene_id=row['scene_id'],
                frame_id=token, timestamp=row['timestamp'], window_id=row['window_id'],
                stability_level=row['stability_level'], query_id=gt['query_id'], query_index=qi,
                question=question, group=gt['group'], ground_truth=gt['ground_truth'], prediction=answer,
                previous_frame_id=gt['previous_frame_id'], previous_timestamp=gt['previous_timestamp'],
                previous_ground_truth=gt['previous_ground_truth'], fact_status=gt['fact_status'],
                change_type=gt['change_type'], scorable=known, score=score,
                correct=(score == 1 if known else None), stale_eligible=stale_eligible,
                stale_answer=stale if stale_eligible else None,
                previous_GT_score=previous_quality.get('score') if previous_quality else None,
                quality=quality))
    return result


def accuracy(events):
    known = [q for q in events if q['scorable']]
    stable = [q for q in known if q['fact_status'] == 'Stable']
    changed = [q for q in known if q['fact_status'] == 'Changed']
    stale = [q for q in events if q['stale_eligible']]
    return dict(queries=len(events), scorable_queries=len(known), unknown_queries=len(events) - len(known),
        overall_accuracy_percent=ratio(sum(q['score'] for q in known), len(known), 100),
        fully_correct_accuracy_percent=ratio(sum(q['correct'] for q in known), len(known), 100),
        stable_fact_accuracy_percent=ratio(sum(q['score'] for q in stable), len(stable), 100),
        changed_fact_accuracy_percent=ratio(sum(q['score'] for q in changed), len(changed), 100),
        stable_queries=len(stable), changed_queries=len(changed),
        unclassified_scorable_queries=sum(q['fact_status'] == 'Unclassified' for q in known),
        changed_GT_queries=sum(q['fact_status'] == 'Changed' for q in events),
        queries_after_changes=len(stale), stale_answers=sum(bool(q['stale_answer']) for q in stale),
        stale_answer_rate_percent=ratio(sum(bool(q['stale_answer']) for q in stale), len(stale), 100))


def latency(rows, segments):
    values = [r['e2e_ms'] for r in rows]
    seconds = sum(s['wall_seconds'] for s in segments)
    if sum(s['frames'] for s in segments) != len(rows):
        raise ValueError('Throughput segments do not cover all measured frames')
    if any(not math.isfinite(s['wall_seconds']) or s['wall_seconds'] <= 0 for s in segments):
        raise ValueError('Invalid processing-segment elapsed time')
    return dict(frames=len(rows), average_e2e_ms=float(np.mean(values)) if values else None,
        p95_e2e_ms=float(np.percentile(values, 95)) if values else None,
        sustained_fps=ratio(len(rows), seconds), measured_processing_seconds=seconds)


def compute(rows, native):
    def total(part, label):
        return sum(r['executed_macs'][label] for r in part)
    visual, llm = total(rows, 'visual'), total(rows, 'llm')
    base_visual, base_llm = total(native, 'visual'), total(native, 'llm')
    requests = [q for r in rows for q in r['requests']]
    proposed = sum(q.get('draft_candidate_tokens', 0) for q in requests)
    accepted = sum(q.get('draft_accepted_tokens', 0) for q in requests)
    diag = [r['visual_diagnostics'] for r in rows if r['visual_diagnostics'] is not None]
    total_windows = sum(d['total_windows'] for d in diag)
    reused = sum(d['reused_windows'] for d in diag)
    lengths = [v for r in rows for v in r['llm_forward_lengths']]
    return dict(frames=len(rows), visual_macs=visual, llm_macs=llm,
        native_visual_macs=base_visual, native_llm_macs=base_llm,
        visual_compute_reduction_percent=100 * (1 - visual / base_visual) if base_visual else None,
        llm_compute_reduction_percent=100 * (1 - llm / base_llm) if base_llm else None,
        direct_answers=sum(bool(q.get('llm_skipped')) for q in requests), queries=len(requests),
        direct_answer_rate_percent=ratio(sum(bool(q.get('llm_skipped')) for q in requests), len(requests), 100),
        candidate_tokens=proposed, accepted_candidate_tokens=accepted,
        candidate_token_acceptance_rate_percent=ratio(accepted, proposed, 100),
        prefill_macs=total(rows, 'prefill'), decode_macs=total(rows, 'decode'),
        prefill_input_tokens=sum(v['query_tokens'] for v in lengths if v['phase'] == 'prefill'),
        decode_forward_input_tokens=sum(v['query_tokens'] for v in lengths if v['phase'] == 'decode'),
        total_prefix_windows=total_windows if diag else None,
        refreshed_prefix_windows=sum(d['executed_windows'] for d in diag) if diag else None,
        reused_prefix_windows=reused if diag else None,
        prefix_window_reuse_percent=ratio(reused, total_windows, 100))


def cluster_intervals(rows, statistic, keys, samples=2000, seed=20261030):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row['scene_id']].append(row)
    ids = sorted(clusters)
    if len(ids) < 2 or samples == 0:
        return {k + suffix: None for k in keys for suffix in ('_ci95_low', '_ci95_high')}
    rng = np.random.default_rng(seed)
    observations = {k: [] for k in keys}
    for _ in range(samples):
        sample = [r for i in rng.integers(0, len(ids), len(ids)) for r in clusters[ids[i]]]
        stats = statistic(sample)
        for k in keys:
            if stats[k] is not None:
                observations[k].append(stats[k])
    result = {}
    for k, values in observations.items():
        result[k + '_ci95_low'], result[k + '_ci95_high'] = (
            np.percentile(values, [2.5, 97.5]).tolist() if values else (None, None))
    return result


def response_delays(events, horizon):
    """Optional sampled-video GT delay, explicitly distinct from live wall delay."""
    series = defaultdict(list)
    for event in events:
        series[(event['method'], event['repeat'], event['scene_id'], event['query_id'])].append(event)
    delays = []
    for values in series.values():
        for i, event in enumerate(values):
            if event['fact_status'] != 'Changed' or not event['scorable']:
                continue
            delay, superseded = None, False
            for later in values[i:]:
                if later['timestamp'] - event['timestamp'] > horizon:
                    break
                if later['ground_truth']['known'] and later['ground_truth']['state'] != event['ground_truth']['state']:
                    superseded = True
                    break
                if later['scorable'] and later['correct'] and later['ground_truth']['state'] == event['ground_truth']['state']:
                    delay = (later['timestamp'] - event['timestamp']) * 1000
                    break
            censored = delay is None and not superseded and values[-1]['timestamp'] < event['timestamp'] + horizon
            delays.append(dict(method=event['method'], repeat=event['repeat'], scene_id=event['scene_id'],
                stability_level=event['stability_level'], query_id=event['query_id'], frame_id=event['frame_id'],
                sampled_video_response_delay_ms=delay, responded=delay is not None,
                censored=censored, superseded=superseded))
    return delays


def report(out):
    from .evaluator import verify_manifest
    from tools.worldstream_m123.r0_task_evaluation import sha
    out = Path(out)
    # Refuse to overwrite any previous report; use a new run directory.
    if (out / 'completion.json').exists() or (out / 'raw_qa_results.csv').exists():
        raise FileExistsError('Report already exists; existing outputs will not be overwritten')
    manifest = verify_manifest(out)
    rows, runtimes, segments = {}, {}, {}
    for method in METHODS:
        directory = out / method
        status = read(directory / 'inference_complete.json')
        if status['status'] != 'COMPLETED' or status['receipts_sha256'] != sha(directory / 'receipts.jsonl'):
            raise ValueError('Incomplete/modified inference: ' + method)
        if status['segments_sha256'] != sha(directory / 'processing_segments.json'):
            raise ValueError('Modified throughput measurement')
        rows[method] = list(jsonl(directory / 'receipts.jsonl'))
        if len(rows[method]) != status['frames']:
            raise ValueError('Wrong completed frame count')
        segments[method] = read(directory / 'processing_segments.json')
        runtimes[method] = read(directory / 'effective_runtime.json')
    windows = list(jsonl(out / 'test_windows.jsonl'))
    validate_pair(manifest, windows, rows, runtimes)
    gt = list(jsonl(out / 'query_ground_truth.jsonl'))
    objects = {r['frame_id']: r['objects'] for r in jsonl(out / 'evaluation_objects.jsonl')}
    events = [e for method in METHODS for e in score_rows(rows[method], gt, objects)]
    dump_jsonl(out / 'scored_queries.jsonl', events)
    frame_fields = ('method', 'repeat', 'scene_id', 'frame_id', 'frame_index', 'timestamp',
                    'window_id', 'stability_level', 'e2e_ms', 'execution_status')
    csv_write(out / 'raw_frame_results.csv', [{k: r[k] for k in frame_fields} for method in METHODS for r in rows[method]])
    csv_write(out / 'raw_qa_results.csv', [{k: v for k, v in e.items() if k != 'quality'} for e in events])
    config = manifest['config']
    latency_rows, accuracy_rows, compute_rows, stale_types = [], [], [], []
    ci_args = dict(samples=config['bootstrap_samples'], seed=config['seed'])
    for level in (*LEVELS, 'Unclassified', 'ALL'):
        select = lambda r: level == 'ALL' or r['stability_level'] == level
        native = [r for r in rows['native'] if select(r)]
        for method in METHODS:
            part = [r for r in rows[method] if select(r)]
            qs = [r for r in events if r['method'] == method and select(r)]
            segs = [s for s in segments[method] if select(s)]
            label = dict(method=method, stability_level=level, independent_sequences=len({r['scene_id'] for r in part}))
            lat = dict(**label, **latency(part, segs))
            # E2E CI clusters complete sequences; repeats never become new clusters.
            lat.update(cluster_intervals(part, lambda s: dict(average_e2e_ms=float(np.mean([r['e2e_ms'] for r in s])),
                p95_e2e_ms=float(np.percentile([r['e2e_ms'] for r in s], 95))),
                ['average_e2e_ms', 'p95_e2e_ms'], **ci_args))
            lat.update(cluster_intervals(segs, lambda s: dict(sustained_fps=ratio(sum(r['frames'] for r in s),
                sum(r['wall_seconds'] for r in s))), ['sustained_fps'], **ci_args))
            latency_rows.append(lat)
            acc = dict(**label, **accuracy(qs))
            acc.update(cluster_intervals(qs, accuracy, ['overall_accuracy_percent', 'stable_fact_accuracy_percent',
                'changed_fact_accuracy_percent', 'stale_answer_rate_percent'], **ci_args))
            accuracy_rows.append(acc)
            compute_rows.append(dict(**label, **compute(part, native)))
            for kind in sorted({q['change_type'] for q in qs if q['change_type']}):
                changes = [q for q in qs if q['change_type'] == kind and q['stale_eligible']]
                stale_types.append(dict(**label, change_type=kind, queries_after_changes=len(changes),
                    stale_answers=sum(bool(q['stale_answer']) for q in changes),
                    stale_answer_rate_percent=ratio(sum(bool(q['stale_answer']) for q in changes), len(changes), 100)))
    paired = []
    for w in windows:
        a = [r for r in rows['native'] if r['window_id'] == w['window_id']]
        b = [r for r in rows['worldstream'] if r['window_id'] == w['window_id']]
        if not a:
            continue
        am, bm = np.mean([r['e2e_ms'] for r in a]), np.mean([r['e2e_ms'] for r in b])
        paired.append(dict(window_id=w['window_id'], scene_id=w['scene_id'],
            stability_level=w['stability_level'], change_rate=w['change_rate'], measured_frames=len(a),
            complete_measured_window=len(a) == w['frames'] * config['repeats'],
            native_mean_e2e_ms=float(am), worldstream_mean_e2e_ms=float(bm), speedup=float(am / bm)))
    csv_write(out / 'paired_window_summary.csv', paired)
    csv_write(out / 'latency_summary.csv', latency_rows)
    csv_write(out / 'accuracy_summary.csv', accuracy_rows)
    csv_write(out / 'accuracy_by_query_group.csv', [dict(method=method, group=group,
        **accuracy([q for q in events if q['method'] == method and q['group'] == group]))
        for method in METHODS for group in sorted({q['group'] for q in events})])
    csv_write(out / 'compute_summary.csv', compute_rows)
    csv_write(out / 'stale_by_change_type.csv', stale_types,
              fields=['method', 'stability_level', 'independent_sequences', 'change_type',
                      'queries_after_changes', 'stale_answers', 'stale_answer_rate_percent'])
    delays = response_delays(events, config['response_horizon_seconds'])
    dump_jsonl(out / 'sampled_response_events.jsonl', delays)
    delay_summary = []
    for method in METHODS:
        for level in (*LEVELS, 'Unclassified', 'ALL'):
            part = [d for d in delays if d['method'] == method and (level == 'ALL' or d['stability_level'] == level)]
            successful = [d['sampled_video_response_delay_ms'] for d in part if d['responded']]
            delay_summary.append(dict(method=method, stability_level=level, changed_events=len(part),
                responded_events=len(successful), unresponded_events=sum(not d['responded'] for d in part),
                unresponded_fraction=ratio(sum(not d['responded'] for d in part), len(part)),
                censored_events=sum(d['censored'] for d in part), superseded_events=sum(d['superseded'] for d in part),
                sampled_video_response_delay_ms=float(np.mean(successful)) if successful else None,
                live_wall_response_delay_ms=None, live_wall_response_status='NOT_EVALUABLE_OFFLINE_VIDEO',
                horizon_seconds=config['response_horizon_seconds']))
    csv_write(out / 'response_delay_summary.csv', delay_summary)
    from .plotting import plot_results
    figures = plot_results(out, latency_rows, accuracy_rows, paired, smoke=manifest['smoke'])
    summary = dict(status='COMPLETED', smoke=manifest['smoke'], manifest=manifest,
        data_coverage=read(out / 'data_coverage.json'), thresholds=manifest['thresholds'],
        latency=latency_rows, accuracy=accuracy_rows, compute=compute_rows, response_delay=delay_summary,
        figures=figures, fair_inputs_verified=True, metrics_available=True,
        accuracy_definition='100 * mean existing main QA fact score over known GT; original QA may be fractional. Fully-correct accuracy is also given; UNKNOWN excluded and counted.',
        stable_changed_definition='Same exact query, same sequence, previous valid comparable GT; target identity alone does not change answer state.',
        stale_definition='Changed GT + current score < 1 + previous GT score = 1; identical evaluator for both methods.',
        CI_definition='Percentile 95% bootstrap over sequence clusters, 2000 samples by default; fewer than two sequences gives N/A.',
        response_delay_limitation='Only dataset-time observed-change response delay is identifiable. Offline decoding is not synchronized to live GT event wall time; live wall delay is unavailable.',
        conclusion='Smoke verifies execution only; no generalization claim.' if manifest['smoke'] else 'Descriptive results on previously exposed validation data; no independent TEST claim.')
    write(out / 'experiment_summary.json', summary)
    write(out / 'completion.json', dict(status='COMPLETED', smoke=manifest['smoke'],
          frames_per_method=len(rows['native']), queries_per_method=len(events) // 2,
          fair_inputs_verified=True, figures=figures))
    print('REPORT_COMPLETED', out, flush=True)
