"""Auditable, CPU-only change response supplement for completed paired runs.

The time domain is the sampled video's timestamps, never offline inference wall
time. Historical errors are regraded against the GT BEFORE the focal change,
including when subsequent frames themselves have fact_status='Stable'.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .common import (LEVELS, METHODS, ROOT, csv_write, dump_jsonl, jsonl, read,
                     require_affinity, write)
from .metrics import cluster_intervals, ratio, score_rows, validate_pair


def event_key(row):
    return row['repeat'], row['scene_id'], row['query_id'], row['frame_id']


def _target_instances(gt, frame, objects):
    reference = (gt or {}).get('reference')
    if not isinstance(reference, dict):
        return None
    selected = reference.get('selected_annotation')
    annotations = [selected] if selected else reference.get('annotation_ids', [])
    ids = {o['instance_id'] for o in objects.get(frame, [])
           if o['annotation_id'] in annotations}
    return ids if len(ids) == 1 else None


def change_events(scored, frames, objects, horizon=5., max_gap=1., grade_fn=None):
    """Follow a fixed post-change state until response or an observation boundary.

    Every frozen Changed event is retained. A stricter cohort additionally
    requires the old GT to have been observed on the immediately preceding frame.
    Unknown GT, a missing query, and time gaps censor rather than bridge evidence.
    """
    if not math.isfinite(horizon) or not math.isfinite(max_gap) or min(horizon, max_gap) <= 0:
        raise ValueError('horizon and max_gap must be finite and positive')
    if grade_fn is None:
        from tools.worldstream_m123.continuous_stream_report import grade as grade_fn
    lookup = {}
    for q in scored:
        key = (q['method'], q['repeat'], q['scene_id'], q['frame_id'], q['query_id'])
        if key in lookup:
            raise ValueError('Duplicate scored query')
        lookup[key] = q
    frame_series = defaultdict(list)
    for method in METHODS:
        for row in frames[method]:
            frame_series[(method, row['repeat'], row['scene_id'])].append(row)
    indices = {}
    for key, series in frame_series.items():
        series.sort(key=lambda r: r['timestamp'])
        if any(b['timestamp'] <= a['timestamp'] for a, b in zip(series, series[1:])):
            raise ValueError('Frame timestamps must increase within a scene/repeat')
        for i, row in enumerate(series):
            indices[key + (row['frame_id'],)] = i
    result = []
    for event in scored:
        if event['fact_status'] != 'Changed' or not event['scorable']:
            continue
        old = event['previous_ground_truth']
        if not old or not old['known'] or not event['ground_truth']['known']:
            raise ValueError('Changed event requires known old and new GT')
        if old['state'] == event['ground_truth']['state']:
            raise ValueError('Changed event has identical answer state')
        key = (event['method'], event['repeat'], event['scene_id'])
        series = frame_series[key]
        index = indices[key + (event['frame_id'],)]
        previous = series[index - 1] if index else None
        previous_q = (lookup.get(key + (previous['frame_id'], event['query_id'])) if previous else None)
        onset_gap = event['timestamp'] - event['previous_timestamp']
        contiguous = bool(previous_q and previous_q['scorable'] and
                          previous['frame_id'] == event['previous_frame_id'] and
                          0 < onset_gap <= max_gap)
        old_instance = _target_instances(old, event['previous_frame_id'], objects)
        new_instance = _target_instances(event['ground_truth'], event['frame_id'], objects)
        same_target = (old_instance == new_instance if old_instance and new_instance else None)
        observations, runs, active = [], [], []
        first_correct, reason, boundary = None, 'SCENE_END', series[-1]['timestamp']

        def close_run(next_timestamp=None):
            if active:
                runs.append(dict(frames=len(active), start_timestamp=active[0],
                    last_stale_timestamp=active[-1],
                    observed_span_ms=(active[-1] - active[0]) * 1000,
                    # If the next sampled output is non-stale, it brackets the
                    # end; it does not locate a continuous-time end exactly.
                    next_nonstale_timestamp=next_timestamp,
                    end_bracket_ms=((next_timestamp - active[0]) * 1000
                                    if next_timestamp is not None else None),
                    right_censored=next_timestamp is None))
                active.clear()

        last_timestamp = None
        for frame in series[index:]:
            timestamp = frame['timestamp']
            if timestamp - event['timestamp'] > horizon:
                reason, boundary = 'HORIZON', event['timestamp'] + horizon
                break
            if last_timestamp is not None and timestamp - last_timestamp > max_gap:
                reason, boundary = 'OBSERVATION_GAP', last_timestamp
                break
            later = lookup.get(key + (frame['frame_id'], event['query_id']))
            if later is None:
                reason, boundary = 'QUERY_NOT_ASKED', timestamp
                break
            if not later['scorable'] or not later['ground_truth']['known']:
                reason, boundary = 'UNKNOWN_GT', timestamp
                break
            if later['ground_truth']['state'] != event['ground_truth']['state']:
                reason, boundary = 'SUPERSEDED', timestamp
                break
            historical = grade_fn(event['question'], later['prediction'],
                                  event['previous_frame_id'], objects)
            historical_score = historical.get('score') if historical.get('known') else None
            if historical_score is not None and (not isinstance(historical_score, (int, float)) or
                    not math.isfinite(historical_score) or not 0 <= historical_score <= 1):
                raise ValueError('Invalid historical factual score')
            stale = historical_score == 1 and later['score'] < 1
            request = frame['requests'][later['query_index']]
            observations.append(dict(frame_id=frame['frame_id'], timestamp=timestamp,
                current_score=later['score'], frozen_prechange_GT_score=historical_score,
                historical_fact_scorable=historical_score is not None,
                stale=stale, llm_skipped=bool(request.get('llm_skipped')),
                prediction=later['prediction']))
            if stale:
                active.append(timestamp)
            else:
                close_run(timestamp if historical_score is not None else None)
            last_timestamp = timestamp
            if later['correct']:
                first_correct, reason, boundary = timestamp, 'RESPONDED', timestamp
                break
        close_run()
        if reason == 'SCENE_END' and last_timestamp - event['timestamp'] >= horizon:
            reason, boundary = 'HORIZON', event['timestamp'] + horizon
        censored = reason in ('SCENE_END', 'OBSERVATION_GAP', 'QUERY_NOT_ASKED', 'UNKNOWN_GT')
        initial_run = runs[0] if runs and runs[0]['start_timestamp'] == event['timestamp'] else None
        eligible = [o for o in observations if o['historical_fact_scorable']]
        record = {k: event[k] for k in ('method', 'repeat', 'scene_id', 'query_id',
                    'frame_id', 'question', 'group', 'stability_level', 'change_type')}
        record.update(t_change=event['timestamp'], t_previous_GT=event['previous_timestamp'],
            change_onset_bracket_ms=onset_gap * 1000, contiguous_onset=contiguous,
            same_persistent_target=same_target, previous_GT=old['state'], new_GT=event['ground_truth']['state'],
            t_first_correct=first_correct, response_delay_ms=(
                (first_correct - event['timestamp']) * 1000 if first_correct is not None else None),
            responded=first_correct is not None, stop_reason=reason, censored=censored,
            superseded=reason == 'SUPERSEDED', observation_boundary_timestamp=boundary,
            last_evaluable_timestamp=last_timestamp,
            followup_observed_ms=(last_timestamp - event['timestamp']) * 1000,
            wrong_outputs_before_response=sum(o['current_score'] < 1 for o in observations),
            initial_consecutive_stale_outputs=initial_run['frames'] if initial_run else 0,
            max_consecutive_stale_outputs=max((r['frames'] for r in runs), default=0),
            max_stale_observed_span_ms=max((r['observed_span_ms'] for r in runs), default=0.),
            stale_outputs=sum(o['stale'] for o in observations),
            historical_scorable_outputs=len(eligible),
            historical_unscorable_outputs=len(observations) - len(eligible),
            stale_direct_outputs=sum(o['stale'] and o['llm_skipped'] for o in observations),
            observed_outputs=len(observations), stale_runs=runs, observations=observations)
        result.append(record)
    paired = defaultdict(dict)
    for row in result:
        paired[event_key(row)][row['method']] = row
    for rows in paired.values():
        if set(rows) != set(METHODS):
            raise ValueError('Changed events are not paired')
        for field in ('t_change', 't_previous_GT', 'previous_GT', 'new_GT', 'contiguous_onset'):
            if rows['native'][field] != rows['worldstream'][field]:
                raise ValueError('Different paired GT: ' + field)
    return result


def statistics(events):
    delays = [e['response_delay_ms'] for e in events if e['responded']]
    return dict(changed_events=len(events), responded_events=len(delays),
        observed_response_rate_percent=ratio(len(delays), len(events), 100),
        response_delay_mean_ms=float(np.mean(delays)) if delays else None,
        response_delay_median_ms=float(np.median(delays)) if delays else None,
        response_delay_p95_ms=float(np.percentile(delays, 95)) if delays else None,
        immediate_response_events=sum(e['response_delay_ms'] == 0 for e in events),
        censored_events=sum(e['censored'] for e in events),
        superseded_events=sum(e['superseded'] for e in events),
        horizon_exhausted_events=sum(e['stop_reason'] == 'HORIZON' for e in events),
        mean_wrong_outputs_before_response=float(np.mean([e['wrong_outputs_before_response'] for e in events])) if events else None,
        mean_max_consecutive_stale_outputs=float(np.mean([e['max_consecutive_stale_outputs'] for e in events])) if events else None,
        max_consecutive_stale_outputs=max((e['max_consecutive_stale_outputs'] for e in events), default=None),
        mean_max_stale_observed_span_ms=float(np.mean([e['max_stale_observed_span_ms'] for e in events])) if events else None,
        events_with_stale_outputs=sum(e['stale_outputs'] > 0 for e in events),
        stale_outputs=sum(e['stale_outputs'] for e in events),
        stale_direct_outputs=sum(e['stale_direct_outputs'] for e in events),
        historical_unscorable_outputs=sum(e['historical_unscorable_outputs'] for e in events),
        stop_reasons=dict(Counter(e['stop_reason'] for e in events)))


COHORTS = {
    'ALL_OBSERVED_GT_CHANGES': lambda e: True,
    'CONTIGUOUS_ONSET': lambda e: e['contiguous_onset'],
    'PERSISTENT_TARGET_CONTIGUOUS_ONSET': lambda e: e['contiguous_onset'] and e['same_persistent_target'] is True,
}


def paired_statistics(events):
    pairs = defaultdict(dict)
    for event in events:
        pairs[event_key(event)][event['method']] = event
    if any(set(v) != set(METHODS) for v in pairs.values()):
        raise ValueError('Missing paired event')
    common = [v for v in pairs.values() if all(e['responded'] for e in v.values())]
    return dict(paired_events=len(pairs), both_responded=len(common),
        native_only_responded=sum(v['native']['responded'] and not v['worldstream']['responded'] for v in pairs.values()),
        worldstream_only_responded=sum(v['worldstream']['responded'] and not v['native']['responded'] for v in pairs.values()),
        neither_responded=sum(not any(e['responded'] for e in v.values()) for v in pairs.values()),
        common_native_mean_ms=float(np.mean([v['native']['response_delay_ms'] for v in common])) if common else None,
        common_worldstream_mean_ms=float(np.mean([v['worldstream']['response_delay_ms'] for v in common])) if common else None,
        common_worldstream_faster=sum(v['worldstream']['response_delay_ms'] < v['native']['response_delay_ms'] for v in common),
        common_native_faster=sum(v['worldstream']['response_delay_ms'] > v['native']['response_delay_ms'] for v in common),
        common_tied=sum(v['worldstream']['response_delay_ms'] == v['native']['response_delay_ms'] for v in common))


def plot(events, destination, horizon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size': 10, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for method, color in zip(METHODS, ('#777777', '#1b9e77')):
            part = [e for e in events if e['method'] == method]
            if not part:
                continue
            times = np.linspace(0, horizon, 201)
            fraction = [sum(e['responded'] and e['response_delay_ms'] <= t * 1000
                            for e in part) / len(part) for t in times]
            axes[0].step(times, fraction, where='post', label=f'{method} (n={len(part)})', color=color)
            counts = Counter(e['max_consecutive_stale_outputs'] for e in part)
            xx = np.arange(max(counts) + 1)
            offset = -.2 if method == 'native' else .2
            axes[1].bar(xx + offset, [counts[x] for x in xx], width=.4, label=method, color=color)
        axes[0].set(xlabel='Sampled video time since observed change (s)',
                    ylabel='Observed responses / all change events', ylim=(0, 1.02))
        axes[1].set(xlabel='Longest consecutive historical-error run (outputs)', ylabel='Change events')
        for ax in axes:
            ax.legend(); ax.grid(axis='y', alpha=.2)
        fig.suptitle('All observed GT changes; unknown/gaps censor follow-up; offline video time')
        fig.tight_layout()
        names = []
        for extension in ('pdf', 'png'):
            name = 'change_response.' + extension
            fig.savefig(destination / name, dpi=180, bbox_inches='tight')
            names.append(name)
        plt.close(fig)
        return names


def supplement(source, destination, horizon=None, max_gap=None, bootstrap_samples=2000):
    """Recompute from verified frozen inputs; never overwrite the original report."""
    from .evaluator import verify_manifest
    from tools.worldstream_m123.r0_task_evaluation import sha
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Supplement destination must be new: ' + str(destination))
    completed = read(source / 'completion.json')
    if completed['status'] != 'COMPLETED' or not completed['fair_inputs_verified']:
        raise ValueError('A completed, paired source experiment is required')
    print('VERIFY_SOURCE_MANIFEST_AND_RECEIPTS', source, flush=True)
    manifest = verify_manifest(source)
    horizon = manifest['config']['response_horizon_seconds'] if horizon is None else horizon
    max_gap = manifest['config']['max_transition_gap_seconds'] if max_gap is None else max_gap
    if bootstrap_samples < 0:
        raise ValueError('bootstrap_samples must be nonnegative')
    frames, runtimes = {}, {}
    consumed = [source / n for n in ('manifest.json', 'completion.json', 'query_ground_truth.jsonl',
                                     'evaluation_objects.jsonl', 'test_windows.jsonl')]
    for method in METHODS:
        folder = source / method
        status = read(folder / 'inference_complete.json')
        if status['status'] != 'COMPLETED' or any(sha(folder / n) != status[k] for n, k in (
                ('receipts.jsonl', 'receipts_sha256'), ('processing_segments.json', 'segments_sha256'))):
            raise ValueError('Modified or incomplete inference: ' + method)
        frames[method] = list(jsonl(folder / 'receipts.jsonl'))
        if len(frames[method]) != status['frames']:
            raise ValueError('Wrong completed frame count')
        runtimes[method] = read(folder / 'effective_runtime.json')
        consumed.extend(folder / n for n in ('receipts.jsonl', 'processing_segments.json',
                                             'inference_complete.json', 'effective_runtime.json'))
    validate_pair(manifest, list(jsonl(source / 'test_windows.jsonl')), frames, runtimes)
    gt = list(jsonl(source / 'query_ground_truth.jsonl'))
    objects = {r['frame_id']: r['objects'] for r in jsonl(source / 'evaluation_objects.jsonl')}
    print('REGRADE_CURRENT_AND_FROZEN_PRECHANGE_FACTS', flush=True)
    scored = [e for method in METHODS for e in score_rows(frames[method], gt, objects)]
    events = change_events(scored, frames, objects, horizon, max_gap)
    summary, paired = [], {}
    for cohort, select in COHORTS.items():
        chosen = [e for e in events if select(e)]
        paired[cohort] = paired_statistics(chosen)
        groups = [('stability_level', v) for v in (*LEVELS, 'Unclassified', 'ALL')]
        groups += [('change_type', v) for v in sorted({e['change_type'] for e in chosen})]
        for grouping, group in groups:
            for method in METHODS:
                part = [e for e in chosen if e['method'] == method and
                        (group == 'ALL' or e[grouping] == group)]
                values = statistics(part)
                if grouping == 'stability_level' and group == 'ALL':
                    values.update(cluster_intervals(part, statistics, [
                        'observed_response_rate_percent', 'response_delay_mean_ms',
                        'mean_max_consecutive_stale_outputs'],
                        samples=bootstrap_samples, seed=manifest['config']['seed']))
                summary.append(dict(cohort=cohort, grouping=grouping, group=group,
                                    method=method, **values))
    definitions = dict(
        formula='T_response = t_first_correct - t_change',
        time_domain='Sampled dataset/video timestamps in seconds; delays exported in ms. Not live wall time.',
        t_change='First observed known changed answer-state, bracketed by previous valid GT time and current time.',
        t_first_correct='First fully correct output for the focal new state before next change, unknown GT, missing query, >max_gap interval, scene end, or horizon.',
        stale='Current fact score <1 AND identical prediction scores 1 against GT frozen BEFORE this event. Not just any wrong output.',
        duration='Each run exports observed span (last stale timestamp - first stale timestamp), a lower bound. One stale output has span 0. A next nonstale sample, if available, provides an end bracket, not an exact continuous duration.',
        cohorts='All frozen GT changes retained; CONTIGUOUS_ONSET requires known comparable GT on immediately preceding measured frame within max_gap; persistent-target subset also requires a unique same instance ID.',
        response_rate='Observed responses / all cohort events, including unresolved events in denominator. Not a survival estimate; censoring/superseding counts reported separately.',
        conditional_mean='Response delay mean/median/P95 only among observed responses; method-specific responders differ. Common-responder paired statistics also supplied.',
        wrong_run='Number of consecutively observed incorrect outputs from change to first correct or observation boundary; censored counts are lower bounds.',
        diagnostic='llm_skipped diagnoses direct-answer routing only; no causal attribution to OGSM/QCSE without matched module ablations.',
        confidence_interval='2000 sequence-cluster bootstrap resamples by default; repeated passes are not independent clusters.',
        generalization='Existing exposed validation scenes, not independent TEST.')
    destination.mkdir(parents=True, exist_ok=False)
    dump_jsonl(destination / 'change_response_events.jsonl', events)
    csv_write(destination / 'change_response_events.csv', events)
    csv_write(destination / 'change_response_summary.csv', summary)
    paired_rows = [dict(cohort=key, **value) for key, value in paired.items()]
    csv_write(destination / 'paired_response_summary.csv', paired_rows)
    figures = plot(events, destination, horizon)
    provenance = {str(p): sha(p) for p in consumed}
    implementation = [Path(__file__), Path(__file__).with_name('metrics.py'),
                      Path(__file__).with_name('common.py')]
    report = dict(status='COMPLETED', source=str(source), source_smoke=completed['smoke'],
        generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        horizon_seconds=horizon, max_gap_seconds=max_gap, bootstrap_samples=bootstrap_samples,
        definitions=definitions, source_hashes=provenance,
        implementation_hashes={str(p.resolve()): sha(p) for p in implementation},
        source_manifest_and_pairing_verified=True, summary=summary, paired=paired, figures=figures)
    write(destination / 'change_response_report.json', report)
    _markdown(destination, report)
    write(destination / 'completion.json', dict(status='COMPLETED', source=str(source),
        events_per_method=len(events) // 2, generated_at=report['generated_at'],
        output_hashes={p.name: sha(p) for p in destination.iterdir() if p.is_file()}))
    print('CHANGE_RESPONSE_COMPLETED', destination, flush=True)
    return report


def _markdown(destination, report):
    lines = ['# Change Response Delay 补充评估', '',
        '来源：`' + report['source'] + '`。原实验结果未覆盖，无新增推理。', '',
        '`T_response = t_first_correct − t_change`，使用采样视频时间；不等于在线墙钟时延。', '',
        f'观察上限 {report["horizon_seconds"]:g} 秒，最大连续帧间隔 {report["max_gap_seconds"]:g} 秒。', '',
        '历史错误：当前 GT 得分 <1，且同一回答对本次变化前冻结 GT 的得分 =1；后续 Stable 帧也检测。', '',
        '| 事件集合 | 方法 | 变化数 | 已响应 | 响应率 | 成功样本平均时延(ms) | 最长连续历史错误次数（均值/最大） | 历史错误输出数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in report['summary']:
        if r['grouping'] != 'stability_level' or r['group'] != 'ALL':
            continue
        def fmt(value):
            return 'N/A' if value is None else f'{value:.2f}'
        lines.append(f'| {r["cohort"]} | {r["method"]} | {r["changed_events"]} | '
                     f'{r["responded_events"]} | {fmt(r["observed_response_rate_percent"])}% | '
                     f'{fmt(r["response_delay_mean_ms"])} | {fmt(r["mean_max_consecutive_stale_outputs"])}/'
                     f'{r["max_consecutive_stale_outputs"]} | {r["stale_outputs"]} |')
    lines += ['', '## 配对响应（两方法都响应的相同事件）', '']
    for cohort, p in report['paired'].items():
        lines.append(f'- {cohort}：共 {p["paired_events"]} 个事件，双方响应 {p["both_responded"]}；'
                     f'仅 Native 响应 {p["native_only_responded"]}，仅 WorldStream 响应 '
                     f'{p["worldstream_only_responded"]}，双方未观察到响应 {p["neither_responded"]}。'
                     f'共同响应样本平均时延 Native={p["common_native_mean_ms"]} ms，'
                     f'WorldStream={p["common_worldstream_mean_ms"]} ms。')
    lines += ['', '## 解释边界', '',
        '- ALL 保留原先全部变化事件，但变化前 GT 可能隔了多个未知帧；更严格的连续起点和同一持久目标子集单独列出。',
        '- 再次变化（SUPERSEDED）与缺失 GT/查询、观测间隔、场景末尾分别记录；未响应不填 0，也不能当成已经正确。',
        '- 时延均值只包含成功响应事件，必须结合响应率、共同响应事件统计与删失数阅读。',
        '- 连续错误次数仅为采样输出次数；最长历史错误持续时间使用已观察首末错误的时间跨度下界，单次错误的跨度是 0，并不证明实际持续时间为 0。',
        '- 该对比评价 Native 与完整 WorldStream，不能单凭两者对比证明 OGSM 与 QCSE 的因果协同收益。需要同协议的模块消融。',
        '- 原始事件时间、回答、当前/变化前 GT 得分、连续段和路由诊断均在 events 文件；分组与序列 bootstrap 区间见 summary CSV/JSON。',
        '- 既有验证集结果，不是独立 TEST；真实在线事件发生时间及答案完成的同步墙钟尚不可评估。', '']
    (destination / '结果.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--horizon-seconds', type=float)
    parser.add_argument('--max-gap-seconds', type=float)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    require_affinity()
    destination = args.output or args.source / (
        'change_response_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    supplement(args.source, destination, args.horizon_seconds, args.max_gap_seconds,
               args.bootstrap_samples)


if __name__ == '__main__':
    main()
