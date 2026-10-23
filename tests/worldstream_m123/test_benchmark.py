import json
from pathlib import Path

import pytest

from tools.worldstream_m123.benchmark.metrics import (
    answer_quality, compare, distribution, paired_scene_ci, quality_summary,
    ratio, validate_sequence,
)
from tools.worldstream_m123.benchmark.report import generate


def test_missing_denominators_and_gt_are_not_accuracy_zero():
    assert ratio(0, 0) is None
    assert distribution([])['mean'] is None
    assert answer_quality('success', None) is None
    result = quality_summary([{'quality': None}])
    assert result['valid_gt'] == 0 and result['answer_exact_match'] is None
    assert result['missing_gt'] == 1
    with pytest.raises(ValueError):
        distribution([float('nan')])


def test_numeric_scoring_requires_real_gt_numbers_and_equal_counts_for_mae():
    no_number = answer_quality('There are 2 cars.', 'A car is ahead.')
    assert no_number['numeric_count_match'] is None
    mismatch = answer_quality('2 cars and 4 people', '3 cars')
    assert mismatch['numeric_count_match'] is False
    assert mismatch['numeric_mae'] is None
    assert answer_quality('2 cars', '3 cars')['numeric_mae'] == 1.


def test_latency_sign_and_scene_pairing():
    assert compare(100, 120)['latency_change_percent'] == -20.
    assert compare(None, 120)['speedup'] is None
    rows = [dict(condition=c, workload='real', repeat=0, scene_id=str(i),
                 frame_id=str(i), request_ms=v)
            for i in range(3) for c, v in [('B0', 100), ('B3', 120)]]
    result = paired_scene_ci(rows)
    assert result['paired_scenes'] == 3
    assert result['ci95_ms'] == [20., 20.]
    for row in rows[1::2]:
        row['workload'] = 'different'
    assert paired_scene_ci(rows)['ci95_ms'] is None


def test_scene_sequence_rejects_duplicate_interleaved_and_backwards_frames():
    def f(s, t, n):
        return dict(scene_token=s, timestamp=t, token=n)
    with pytest.raises(ValueError):
        validate_sequence([f('a', 2, '1'), f('a', 1, '2')])
    with pytest.raises(ValueError):
        validate_sequence([f('a', 1, '1'), f('b', 2, '2'), f('a', 3, '3')])
    with pytest.raises(ValueError):
        validate_sequence([f('a', 1, '1'), f('b', 2, '1')])
    assert validate_sequence([f('a', 1, '1'), f('b', 1, '2')])['status'] == 'PASS'


def test_empty_report_preserves_unknown_and_failed_status(tmp_path):
    (tmp_path/'experiments'/'failed').mkdir(parents=True)
    (tmp_path/'experiments'/'failed'/'status.json').write_text(json.dumps(dict(status='FAILED')))
    status = generate(tmp_path)
    assert status['complete'] is False
    assert status['experiments']['failed']['status'] == 'FAILED'
    modules = json.loads((tmp_path/'module_metrics.json').read_text())
    assert modules['B3']['m3'] is None
    assert 'NOT_RUN' in (tmp_path/'report.md').read_text()
    assert (tmp_path/'per_frame_metrics.jsonl').read_text() == ''


def test_observer_keeps_commit_and_camera_delegates_separate(monkeypatch):
    from types import SimpleNamespace as NS
    import torch
    from tools.worldstream_m123.benchmark.observe import Observer
    from worldstream.m3 import visual_state
    class Event:
        def __init__(self, **kw):
            pass
        def record(self):
            pass
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(visual_state, 'base_model', lambda model: model)
    called = []
    def native_commit(frame, outputs, result):
        called.append('commit')
        return 42
    forward = lambda *a, **kw: None
    model = NS(img_backbone=NS(forward=forward), pts_bbox_head=NS(forward=forward),
        map_head=NS(forward=forward), lm_head=NS(generate=forward, forward=forward),
        m3_runtime=None, m1_manager=NS(precheck=forward, precheck_with_observation=forward, commit=native_commit),
        m2_runtime=None)
    observer = Observer(model)
    assert model.m1_manager.commit('frame', NS(anchors=[]), 'precheck') == 42
    assert called == ['commit']
    assert observer.commits == [('frame', [], 'precheck')]
    model.m1_manager.precheck_with_observation('frame',validate_observation=True)
    assert sum(label=='m1_precheck' for label, *_ in observer.events)==1


def test_zero_continuity_events_are_retained_with_null_association_rates():
    from tools.worldstream_m123.benchmark.observe import IdentityScorer
    result = IdentityScorer({}).summary()
    assert result['continuity']['continuous_identity_switches'] == 0
    assert result['continuity']['predicted_track_fragments'] == 0
    assert result['continuity']['bridged_visible_gaps'] == 0
    assert result['association_precision'] is None


def test_live_jsonl_reader_ignores_only_incomplete_last_record(tmp_path):
    from tools.worldstream_m123.benchmark.report import read_rows
    p = tmp_path/'live.jsonl'
    p.write_text('{"frame": 1}\n{"frame":')
    assert read_rows(p) == [{'frame': 1}]
    p.write_text('{"frame": 1}\ninvalid\n')
    with pytest.raises(json.JSONDecodeError):
        read_rows(p)


def test_official_undefined_metrics_become_null_without_changing_valid_scores():
    import numpy as np
    from tools.worldstream_m123.benchmark.metrics import finite_metrics
    raw = dict(mAP=.377397, NDS=.471342, traffic_cone=dict(orient_err=float('nan')),
               array=np.array([1., float('inf')]))
    clean, paths = finite_metrics(raw)
    assert clean['mAP'] == raw['mAP'] and clean['NDS'] == raw['NDS']
    assert clean['traffic_cone']['orient_err'] is None
    assert clean['array'] == [1., None]
    assert set(paths) == {'/traffic_cone/orient_err', '/array/1'}
    json.dumps(clean, allow_nan=False)


def test_map_evaluator_metadata_passes_official_validator(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from tools.worldstream_m123.benchmark.worker import map_evaluate
    import openlanev2.centerline.evaluation as evaluation
    from openlanev2.centerline.preprocessing.check import check_results
    def checked_evaluate(ground_truth, predictions):
        check_results(predictions)
        return {'OpenLane-V2 Score': {'DET_l': .25}}
    monkeypatch.setattr(evaluation, 'evaluate', checked_evaluate)
    result = map_evaluate(SimpleNamespace(data_infos=[], lane_anno_file='unused'), [], tmp_path)
    assert result['OpenLane-V2 Score']['DET_l'] == .25


def test_window_report_counts_actual_admission_and_attention_batches():
    from tools.worldstream_m123.benchmark.report import module_summary
    row=dict(condition='B3',native_eva_cameras=0,
        observed_window_attention_batches={'0':1,'1':1},m2=dict(cache_boundary='window',
            actual_skipped_windows=34,verified_exact_windows=34,cache_bytes=100,M2_validation_ms=0.,
            stages=[dict(total_windows=18,executed_windows=1,reused_windows=17)]*2))
    m2=module_summary([row],[],{})['B3']['m2']
    assert m2['boundary']=='window' and m2['admitted_frames']==1
    assert m2['actual_reused_windows']==34 and m2['executed_prefix_windows']==2
    assert m2['observed_attention_window_batches']==2 and m2['window_skip_audit']=='PASS'
    assert 'actual_reused_cameras' not in m2
    row['observed_window_attention_batches']['1']=18
    assert module_summary([row],[],{})['B3']['m2']['window_skip_audit']=='FAIL'
