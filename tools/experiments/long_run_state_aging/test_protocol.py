"""CPU regression tests. Fixtures are never published as experiment results."""
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .protocol import (STAGES, append_record, read_journal, stage_counts, stage_index,
                       check_continuity, validate_rows)
from .instrumentation import payload_bytes, distribution, collect
from .report import aggregate, aging_aggregate, memory_aggregate


@pytest.mark.parametrize('length,counts', [(9, [3, 3, 3]), (41, [14, 14, 13]), (8, [3, 3, 2]), (3, [1, 1, 1])])
def test_thirds_preserve_every_frame(length, counts):
    assert list(stage_counts(length).values()) == counts
    assert [stage_index(i, length) for i in range(length)] == sorted(stage_index(i, length) for i in range(length))


def test_torn_journal_keeps_only_committed_frames(tmp_path):
    path = tmp_path / 'journal.jsonl'
    with path.open('w') as handle:
        append_record(handle, dict(frame=0))
        append_record(handle, dict(frame=1))
        handle.write('{"frame":')
    assert read_journal(path) == [dict(frame=0), dict(frame=1)]
    path.write_text('{broken}\n')
    with pytest.raises(json.JSONDecodeError):
        read_journal(path)


def state(i):
    return dict(epoch=4, commits=i + 1, frame_id=str(i), cache_version=i + 1,
                cache_frame=str(i), candidate_frame_index=i, candidate_frame=str(i), candidate_entries=3)


def stream_rows():
    from tools.worldstream_m123.selective_answer_reuse import QUESTIONS
    from tools.worldstream_m123.task_metrics import FACT_GROUPS
    questions = list(QUESTIONS) + ['original query one', 'original query two']
    groups = list(FACT_GROUPS[:3]) + ['other', 'other']
    rows = []
    for i in range(8):
        rows.append(dict(method='worldstream', repeat=0, scene_id='scene', frame_id=str(i),
            frame_index=i, timestamp=float(i), stage=STAGES[stage_index(i, 8)],
            e2e_ms=10., questions=questions, groups=groups, answers=['a'] * 5, requests=[{}] * 5,
            state_before=state(i - 1) if i else None, state_after=state(i)))
    return rows


def test_stage_boundaries_do_not_reset_state_or_relabel_partial_stream():
    manifest = dict(protocol_version=3, stages=STAGES, scenes=[dict(scene_id='scene', tokens=[str(i) for i in range(8)],
                                timestamps=[float(i) for i in range(8)])])
    rows = stream_rows()
    assert validate_rows(manifest, rows, 'worldstream', 0)
    assert not validate_rows(manifest, rows[:3], 'worldstream', 0, allow_partial=True)
    assert [r['stage'] for r in rows[:3]] == ['Early', 'Early', 'Early']
    rows[2]['state_before'] = dict(state(1), commits=0)
    with pytest.raises(RuntimeError, match='between measured frames'):
        validate_rows(manifest, rows, 'worldstream', 0)


@pytest.mark.parametrize('field,value', [('epoch', 5), ('commits', 0), ('cache_version', 1), ('candidate_frame_index', 0)])
def test_detects_internal_reset(field, value):
    with pytest.raises(RuntimeError):
        check_continuity(state(1), state(1), dict(state(2), **{field: value}), '2')


def test_memory_accounts_for_torch_113_storage_and_deduplicates_views():
    tensor = torch.zeros(16384)
    one = payload_bytes(tensor)
    shared = payload_bytes([tensor, tensor[:], tensor])
    distinct = payload_bytes([tensor, tensor.clone()])
    assert one >= tensor.numel() * tensor.element_size()
    assert one < shared < distinct
    arr = np.zeros(16384, dtype=np.float32)
    assert arr.nbytes <= payload_bytes([arr, arr[:]]) < arr.nbytes + 2000


def test_real_ogsm_precheck_commit_crosses_three_statistics_stages():
    from worldstream.m1.state_manager import WorldStateManager
    from worldstream.m1.config import M1Config
    from worldstream.m1.state_types import Frame, CameraFrame, FullOutputs, PerceptionAnchor
    from worldstream.m1.geometry import box_geometry
    manager = WorldStateManager(M1Config(logging=False, visualization=False))
    intrinsic = np.array([[30., 0., 20.], [0., 30., 20.], [0., 0., 1.]])
    image = np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3)
    observed, epochs, versions = [], [], []
    for i in range(9):
        timestamp = 1. + .1 * i
        camera = CameraFrame('front', image, intrinsic, np.eye(4), timestamp,
                             'preprocessed_image', np.eye(3), raw_intrinsic=intrinsic)
        frame = Frame('scene', str(i), timestamp, np.eye(4), {'front': camera}, 'test')
        precheck = manager.precheck(frame)
        prediction = PerceptionAnchor('object', 'car', box_geometry([0., 0., 10.], [2., 2., 2.]), source_score=.9)
        manager.commit(frame, FullOutputs([prediction]), precheck)
        observed.append(STAGES[stage_index(i, 9)])
        epochs.append(manager.store.generation)
        versions.append(manager.store.commit_count)
    assert observed == [s for s in STAGES for _ in range(3)]
    assert len(set(epochs)) == 1
    assert versions == list(range(1, 10))


def test_partial_report_does_not_contribute_to_paired_results(tmp_path, monkeypatch):
    from . import report as report_module
    from .protocol import write, dump_jsonl
    manifest = dict(protocol_version=3, stages=STAGES, effective_repeats=1, smoke=True, git_commit=None,
        scenes=[dict(scene_id='scene', name='fixture', measured_duration_seconds=7.,
                     stage_counts=stage_counts(8), tokens=[str(i) for i in range(8)],
                     timestamps=list(map(float, range(8))))],
        response_delay_reason='No exact change timestamp', horizon_limitation='Test fixture')
    write(tmp_path / 'manifest.json', manifest)
    dump_jsonl(tmp_path / 'query_ground_truth.jsonl', [])
    dump_jsonl(tmp_path / 'evaluation_objects.jsonl', [])
    attempt = tmp_path / 'runs/repeat_000/native/attempt_000'
    attempt.mkdir(parents=True)
    rows = []
    for row in stream_rows()[:3]:
        row.update(method='native', state_before=None, state_after=None, aging=None,
                   processing_wall_ms=12., instrumentation_ms=0.)
        rows.append(row)
    dump_jsonl(attempt / 'receipts.jsonl', rows)
    # Isolate journal/stage aggregation from factual-grading tests.
    monkeypatch.setattr('tools.environment_stability_eval.metrics.score_rows', lambda *a: [])
    report_module.report(tmp_path)
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['status'] == 'PARTIAL'
    assert summary['completed_paired_repeats'] == []
    assert all(r['frames'] == 0 for r in summary['stage_summary'])
    assert not (tmp_path / 'figures').exists()
    import csv
    stats = list(csv.DictReader((tmp_path / 'stage_summary.csv').open()))
    partial = [r for r in stats if r['scope'] == 'attempt']
    assert [int(r['frames']) for r in partial] == [3, 0, 0]
    # Reporting again is idempotent and does not rewrite the input journal.
    original = (attempt / 'receipts.jsonl').read_bytes()
    report_module.report(tmp_path)
    assert (attempt / 'receipts.jsonl').read_bytes() == original


def test_real_module_structures_readonly_instrumentation():
    """Actual StateStore/StateCache/AnswerMemory, tiny CPU allocations only."""
    from worldstream.m1.state_store import StateStore
    from worldstream.m1.config import M1Config
    from worldstream.m1.state_types import WorldAnchor, M1Result
    from worldstream.m1.geometry import box_geometry
    from worldstream.m2.state_cache import StateCache
    from worldstream.m2.global_residual import GlobalAnchor
    from worldstream.m3.answer_memory import AnswerMemory
    store = StateStore(M1Config())
    store.anchors['a'] = WorldAnchor('a', 'scene', 'object', 'car',
        box_geometry([0., 0., 10.], [2., 2., 2.]), 2., track_birth_timestamp=1.,
        last_actual_observation_timestamp=2., last_verified_timestamp=1.5)
    cache = StateCache(1024 ** 2)
    context = dict(scene_id='scene', frame_id='f', timestamp=2., state_epoch=1, camera_ids=['front'])
    x = torch.ones(1, 2, 2, 2)
    anchor = GlobalAnchor('scene', 'f', 2., 1, ('s',), ('front',), 1, tuple(x.shape),
                          tuple(x.stride()), x.clone(), x.clone(), 0, 1, 1)
    cache.commit(x, {0: torch.ones(2, 2)}, context, ('s',), block0_input=x,
                 sources={0: [dict(computation_timestamp=2.), dict(computation_timestamp=1.5)]},
                 window_ages={0: torch.tensor([0, 1])}, global_anchor=anchor)
    memory = AnswerMemory()
    ctx = dict(context, frame_index=0)
    memory.remember('What is ahead?', [1, 2], [3, 4, 5], ctx, ('s',), extract_facts=False)
    pre = M1Result('scene', 'f', 2., valid_anchor_ids=['a'])
    manager = SimpleNamespace(store=store, last_result=pre,
        last_commit_diagnostics=dict(association_pairs=[['a', 0]], selected_prediction_indices=[0],
                                     committed_anchor_ids=['a']))
    visual = SimpleNamespace(cache=cache, config=SimpleNamespace(max_reuse_frames=8, global_anchor_interval=2))
    language = SimpleNamespace(answer_memory=memory, _frame_index=1)
    model = SimpleNamespace(m1_manager=manager, m2_runtime=visual, m3_runtime=language,
        m2_diagnostics=dict(per_block=[dict(block=0, executed_window_ids=[0])],
            total_windows=2, reused_windows=1, approx_global_residual_reuse=False, mode='incremental'),
        m3_diagnostics=dict(requests=[dict(draft_candidate_tokens=4, draft_accepted_tokens=3,
                                          draft_source_frame='f', llm_skipped=False)]))
    revision = (cache.version, cache.storage_revision, list(memory.records), tensor_digest(cache))
    result = collect(model, 2., {'f': 1.5})
    assert result['ogsm']['record_storage_age_s'] == [1.]
    assert result['ogsm']['source_observation_age_s'] == [0.]
    assert result['davp']['reused_window_age_frames'] == [1]
    assert result['qcse']['candidate_token_count'] == 3
    assert result['qcse']['token_acceptance_percent'] == 75
    assert result['davp']['cache_size_mb'] >= result['davp']['tensor_cache_size_mb']
    assert revision == (cache.version, cache.storage_revision, list(memory.records), tensor_digest(cache))
    frame = dict(aging=result, video_interval_s=.5)
    summary = aging_aggregate([frame])
    assert summary['token_acceptance_percent'] == 75
    assert summary['average_window_reuse_age_frames'] == 1
    assert summary['state_update_frequency_hz'] == 2


def tensor_digest(cache):
    return tuple((tuple(v.reshape(-1).tolist()), v._version) for v in
                 [cache.images, cache.block0_input, *cache.outputs.values(), *cache.window_ages.values()])


def test_empty_metrics_are_na_not_zero():
    assert distribution([])['mean'] is None
    summary = aggregate([], [])
    for key in ('factual_qa_accuracy_percent', 'stale_answer_rate_percent', 'sustained_fps', 'p95_e2e_ms'):
        assert summary[key] is None


def test_five_question_contract_and_missing_original_qa_rejected():
    from .protocol import validate_workload
    from tools.worldstream_m123.selective_answer_reuse import QUESTIONS
    from tools.worldstream_m123.task_metrics import FACT_GROUPS
    questions = list(QUESTIONS) + ['original one', 'original two']
    groups = list(FACT_GROUPS[:3]) + ['other', 'other']
    validate_workload(questions, groups)
    for bad_questions in (questions[:4], questions + ['extra'], questions[1:] + questions[:1],
                          questions[:4] + [questions[3]]):
        with pytest.raises(ValueError):
            validate_workload(bad_questions, groups)


def test_memory_peaks_are_stage_maxima_and_missing_is_not_zero(monkeypatch):
    from .memory import FrameMemory, MIB
    import sys
    memory_module = sys.modules[FrameMemory.__module__]
    calls = []
    cuda = SimpleNamespace(reset_peak_memory_stats=lambda device: calls.append(device),
        max_memory_allocated=lambda device: 12 * MIB,
        max_memory_reserved=lambda device: 20 * MIB,
        memory_allocated=lambda device: 8 * MIB)
    rss = iter([30., 40.])
    monkeypatch.setattr(memory_module, 'host_rss_mb', lambda: next(rss))
    probe = FrameMemory(cuda)
    probe.start()
    row = probe.stop()
    assert calls == [0]
    assert row['gpu_peak_allocated_mb'] == 12
    assert row['gpu_peak_reserved_mb'] == 20
    result = memory_aggregate([dict(memory=row), dict(memory=dict(gpu_peak_allocated_mb=10.,
        gpu_peak_reserved_mb=25., host_rss_after_mb=35.))])
    assert result['peak_gpu_allocated_mb'] == 12
    assert result['peak_gpu_reserved_mb'] == 25
    assert result['max_observed_host_rss_mb'] == 40
    assert memory_aggregate([])['peak_gpu_allocated_mb'] is None


def test_config_rejects_any_load_except_three_plus_two(tmp_path):
    import yaml
    from .run import load_config
    config = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
    for count in (0, 1, 3):
        config['other_questions'] = count
        path = tmp_path / 'config.yaml'
        path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match='exactly two'):
            load_config(path)


def test_reordered_and_incomplete_frames_are_rejected():
    manifest = dict(protocol_version=3, stages=STAGES, scenes=[dict(scene_id='scene', tokens=[str(i) for i in range(8)],
                                timestamps=[float(i) for i in range(8)])])
    rows = stream_rows()
    with pytest.raises(ValueError, match='Incomplete stream'):
        validate_rows(manifest, rows[:3], 'worldstream', 0)
    rows[1], rows[2] = rows[2], rows[1]
    with pytest.raises(ValueError, match='reordered'):
        validate_rows(manifest, rows, 'worldstream', 0)
