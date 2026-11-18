"""Tiny analytic fixtures for scene dependence; never experiment results."""
from copy import deepcopy
import numpy as np
import pytest

from .statistics import sample_counts, paired_intervals, repeat_statistics
from .data import sample_coverage
from .protocol import STAGES, stage_counts, validate_rows, PROTOCOL_VERSION


def fixture():
    frames, events = [], []
    for i, (scene, score, latency) in enumerate([('a', 0., 10.), ('b', 1., 100.)]):
        frames.append(dict(scene_id=scene, frame_id=scene + '0', repeat=0,
            e2e_ms=latency, processing_wall_ms=latency, instrumentation_ms=0.))
        events.append(dict(scene_id=scene, frame_id=scene + '0', repeat=0, query_index=0,
            scorable=True, score=score, correct=score == 1, fact_status='Changed',
            stale_eligible=True, stale_answer=not bool(score)))
    return frames, events


def test_repeats_do_not_create_independent_scenes_or_transitions():
    frames, events = fixture()
    repeated_frames = [dict(f, repeat=r) for r in range(3) for f in frames]
    repeated_events = [dict(e, repeat=r) for r in range(3) for e in events]
    counts = sample_counts(repeated_frames, repeated_events)
    assert counts['independent_scenes'] == 2
    assert counts['unique_stale_eligible_query_transitions'] == 2
    assert counts['query_observations'] == 6
    assert counts['completed_repeats'] == 3
    a, _ = paired_intervals({'native': frames, 'worldstream': frames},
        {'native': events, 'worldstream': events}, samples=500, seed=10)
    b, delta = paired_intervals({'native': repeated_frames, 'worldstream': repeated_frames},
        {'native': repeated_events, 'worldstream': repeated_events}, samples=500, seed=10)
    key = 'factual_qa_accuracy_percent'
    assert a['native'][key + '_ci95_low'] == b['native'][key + '_ci95_low'] == 0
    assert a['native'][key + '_ci95_high'] == b['native'][key + '_ci95_high'] == 100
    assert delta[key + '_worldstream_minus_native']['ci95_low'] == 0
    assert delta[key + '_worldstream_minus_native']['ci95_high'] == 0
    assert delta['e2e_latency_reduction_percent']['ci95_low'] == 0


def test_pairing_preserves_latency_reduction_under_scene_variation():
    frames, events = fixture()
    faster = [dict(r, e2e_ms=r['e2e_ms'] * .5) for r in frames]
    _, deltas = paired_intervals({'native': frames, 'worldstream': faster},
        {'native': events, 'worldstream': events}, samples=500, seed=42)
    assert deltas['e2e_latency_reduction_percent']['ci95_low'] == 50
    assert deltas['e2e_latency_reduction_percent']['ci95_high'] == 50


def test_one_scene_or_missing_change_denominator_has_no_invented_interval():
    frames, events = fixture()
    one = paired_intervals({'native': frames[:1]}, {'native': events[:1]}, samples=100)[0]['native']
    assert one['average_e2e_ms_ci95_low'] is None
    assert one['average_e2e_ms_ci_status'] == 'insufficient_scene_clusters'
    no_changes = [dict(e, stale_eligible=False, stale_answer=None) for e in events]
    stats = paired_intervals({'native': frames}, {'native': no_changes}, samples=100)[0]['native']
    assert stats['stale_answer_rate_percent_ci_valid_draws'] == 0
    assert stats['stale_answer_rate_percent_ci95_low'] is None


def test_zero_errors_does_not_claim_zero_width_reliable_interval():
    frames, events = fixture()
    zero_errors = [dict(e, stale_answer=False) for e in events]
    stats = paired_intervals({'native': frames}, {'native': zero_errors}, samples=100)[0]['native']
    assert stats['stale_answer_rate_percent_ci95_low'] is None
    assert stats['stale_answer_rate_percent_ci_status'].startswith('boundary_rate')
    assert stats['stale_answer_rate_percent_ci_empirically_degenerate']


def test_repeat_p95_is_separate_from_pooled_and_sd_uses_repeats():
    frames, events = fixture()
    frames = [dict(f, repeat=r, e2e_ms=f['e2e_ms'] + r * 10) for r in range(3) for f in frames]
    events = [dict(e, repeat=r) for r in range(3) for e in events]
    stats = repeat_statistics(frames, events)
    assert stats['average_e2e_ms_repeat_mean'] == 65
    assert stats['average_e2e_ms_repeat_sd'] == 10
    assert stats['p95_e2e_ms_repeat_count'] == 3
    assert stats['factual_qa_accuracy_percent_repeat_sd'] == 0


def test_gt_coverage_assigns_phase_by_each_log_and_never_links_scenes():
    scenes = [dict(scene_id=s, tokens=[s + str(i) for i in range(9)]) for s in ('a', 'b')]
    queries = [dict(scene_id=s, frame_id=s + str(i), group='fact',
        ground_truth=dict(known=True), fact_status='Changed' if i in (2, 8) else 'Stable')
        for s in ('a', 'b') for i in range(9)]
    rows = sample_coverage(scenes, queries)
    early = next(r for r in rows if r['scene_id'] == 'ALL' and r['stage'] == 'Early')
    middle = next(r for r in rows if r['scene_id'] == 'ALL' and r['stage'] == 'Middle')
    assert early['unique_frames'] == 6
    assert early['unique_changed_query_transitions'] == 2
    assert early['scenes_with_changed_query_transitions'] == 2
    assert middle['unique_changed_query_transitions'] == 0
    assert next(r for r in rows if r['scene_id'] == 'ALL' and r['stage'] == 'ALL')['unique_queries'] == 18


def test_multi_scene_boundary_allows_cold_start_without_phase_reset():
    from .test_protocol import stream_rows
    first = stream_rows()
    for r in first:
        r['scene_id'] = 'scene-a'
    second = deepcopy(stream_rows())
    for r in second:
        r['scene_id'] = 'scene-b'
        r['frame_id'] = 'b' + r['frame_id']
        for key in ('state_before', 'state_after'):
            state = r[key]
            if state:
                state['epoch'] = 8
                for field in ('frame_id', 'cache_frame', 'candidate_frame'):
                    state[field] = 'b' + state[field]
    scenes = [dict(scene_id='scene-a', tokens=[str(i) for i in range(8)], timestamps=list(map(float, range(8)))),
              dict(scene_id='scene-b', tokens=['b' + str(i) for i in range(8)], timestamps=list(map(float, range(8))))]
    manifest = dict(protocol_version=PROTOCOL_VERSION, stages=STAGES, scenes=scenes)
    assert validate_rows(manifest, first + second, 'worldstream', 0)
    # A truncation in scene two must not become a completed 18-scene run.
    assert not validate_rows(manifest, first + second[:2], 'worldstream', 0, allow_partial=True)


def test_complete_paired_multi_scene_report_keeps_scene_counts(tmp_path, monkeypatch):
    import sys
    import json
    from . import report as report_module
    from .test_protocol import stream_rows
    from .protocol import write, dump_jsonl
    from tools.worldstream_m123.r0_task_evaluation import sha
    scenes = []
    world = []
    for prefix in ('a', 'b'):
        rows = deepcopy(stream_rows())
        for r in rows:
            r['scene_id'] = prefix
            r['frame_id'] = prefix + r['frame_id']
            for name in ('state_before', 'state_after'):
                if r[name]:
                    for key in ('frame_id', 'cache_frame', 'candidate_frame'):
                        r[name][key] = prefix + r[name][key]
            r.update(fairness={}, processing_wall_ms=12., instrumentation_ms=1.,
                aging=dict(ogsm=dict(state_size_mb=.5), davp=dict(cache_size_mb=1.), qcse=dict(cache_size_mb=.2)),
                memory=dict(gpu_peak_allocated_mb=12., gpu_peak_reserved_mb=20.))
        scenes.append(dict(scene_id=prefix, name=prefix, tokens=[r['frame_id'] for r in rows],
            timestamps=list(map(float, range(8))), measured_duration_seconds=7., stage_counts=stage_counts(8)))
        world.extend(rows)
    manifest = dict(protocol_version=PROTOCOL_VERSION, stages=STAGES, effective_repeats=1,
        scenes=scenes, smoke=True, git_commit=None, config=dict(bootstrap_samples=100),
        response_delay_reason='No exact timestamp', horizon_limitation='Test fixture')
    write(tmp_path / 'manifest.json', manifest)
    dump_jsonl(tmp_path / 'query_ground_truth.jsonl', [])
    dump_jsonl(tmp_path / 'evaluation_objects.jsonl', [])
    for method in ('native', 'worldstream'):
        directory = tmp_path / 'runs/repeat_000' / method / 'attempt_000'
        directory.mkdir(parents=True)
        rows = deepcopy(world)
        for row in rows:
            row['method'] = method
            if method == 'native':
                row.update(state_before=None, state_after=None, aging=None)
        dump_jsonl(directory / 'receipts.jsonl', rows)
        write(directory / 'complete.json', dict(frames=16, receipts_sha256=sha(directory / 'receipts.jsonl')))
        write(directory / 'runtime_identity.json', dict(checkpoint_sha256='fixture', device='cpu',
            precision_by_dtype=[], generation={}, gpu_uuid=None))
    def score(rows, *args):
        return [dict(method=r['method'], repeat=r['repeat'], scene_id=r['scene_id'], frame_id=r['frame_id'],
            query_index=i, scorable=True, score=.5, correct=False, fact_status='Stable',
            stale_eligible=False, stale_answer=None) for r in rows for i in range(5)]
    monkeypatch.setattr('tools.environment_stability_eval.metrics.score_rows', score)
    empty_aging = report_module.aging_aggregate([])
    monkeypatch.setattr(report_module, 'aging_aggregate', lambda rows: empty_aging)
    monkeypatch.setattr(report_module, 'plots', lambda *args: None)
    report_module.report(tmp_path)
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['status'] == 'COMPLETED'
    assert len(summary['stage_summary']) == 6
    assert all(r['independent_scenes'] == 2 for r in summary['stage_summary'])
    assert all(r['stale_answer_rate_percent'] is None for r in summary['stage_summary'])
    assert all(r['average_e2e_ms_repeat_sd'] is None for r in summary['stage_summary'])
    assert len(summary['paired_comparison']) == 20
    assert (tmp_path / '表1.md').exists()
    assert (tmp_path / '结果.md').read_text() == (tmp_path / 'report.md').read_text()
    for name in ('sample_coverage', 'scene_summary', 'repeat_summary', 'paired_comparison'):
        if name != 'sample_coverage':
            assert (tmp_path / (name + '.csv')).exists()
