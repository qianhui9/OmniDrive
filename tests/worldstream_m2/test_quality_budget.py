import pytest

from tools.worldstream_m2.quality_budget import assess_quality, quality_budgets, METRICS


def metrics(loss):
    return {key: {'loss': loss} for key in METRICS}


def test_relaxed_budget_accepts_previously_rejected_detection_loss():
    assert assess_quality(metrics(.01635492645), quality_budgets())['within_quality_budgets']
    assert not assess_quality(metrics(.01635492645), quality_budgets(.005, .02))['within_quality_budgets']
    vlm_only = metrics(0.)
    vlm_only['Token_F1']['loss'] = .04
    assert assess_quality(vlm_only, quality_budgets())['within_quality_budgets']
    assert not assess_quality(vlm_only, quality_budgets(.005, .02))['within_quality_budgets']


def test_absolute_inclusive_boundary_and_each_task():
    assert assess_quality(metrics(.4-.35), quality_budgets())['within_quality_budgets']
    assert assess_quality(metrics(-.2), quality_budgets())['within_quality_budgets']
    for key in METRICS:
        values = metrics(0.)
        values[key]['loss'] = .050001
        result = assess_quality(values, quality_budgets())
        assert not result['within_quality_budgets']
        assert not result['metric_budget_pass'][key]


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_measurements_fail(value):
    assert not assess_quality(metrics(value), quality_budgets())['within_quality_budgets']


@pytest.mark.parametrize('value', [-.1, 1.01, float('nan'), float('inf')])
def test_invalid_budgets_rejected(value):
    with pytest.raises(ValueError):
        quality_budgets(value)


def test_selection_accepts_relaxed_loss_but_rejects_faster_over_budget_candidate(tmp_path, monkeypatch):
    import json
    from tools.worldstream_m2 import global_residual_analysis as analysis
    (tmp_path/'calibration_parameters.json').write_text(json.dumps({'threshold': .97}))
    monkeypatch.setattr(analysis, 'quality_latency_plot', lambda *args: None)
    monkeypatch.setattr(analysis, 'report_rejected_calibration', lambda *args: None)
    losses = {'cal_age2': .01, 'cal_age3': .04, 'cal_age4': .051}
    durations = {'cal_native': 100., 'cal_age2': 95., 'cal_age3': 80., 'cal_age4': 60.}
    monkeypatch.setattr(analysis, 'compare_quality', lambda root, native, candidate: dict(
        **assess_quality(metrics(losses[candidate]), quality_budgets()),
        new_critical_proxy_regressions_vs_wsgrr=0))
    monkeypatch.setattr(analysis, 'service_summary', lambda path, formal: dict(
        request_ms={'mean': durations[path.name]}, visual_cuda_span_ms={'mean': 10.},
        global_anchor_interval=int(path.name[-1]) if formal else None, global_drift_threshold=.97))
    analysis.calibration(tmp_path)
    selected = json.loads((tmp_path/'selection.json').read_text())
    assert selected['interval'] == 3
    assert not selected['validation_used']
    losses.update({name: .051 for name in losses})
    analysis.calibration(tmp_path)
    assert not (tmp_path/'selection.json').exists()
    assert (tmp_path/'calibration_rejection.json').exists()


@pytest.mark.parametrize('failure', ['modified_manifest', 'excluded_scene'])
def test_heldout_identity_rejected_before_gpu_execution(tmp_path, monkeypatch, failure):
    import argparse
    import json
    from tools.worldstream_m2 import global_residual_benchmark as benchmark
    (tmp_path/'selection.json').write_text(json.dumps({'interval': 2, 'threshold': .9}))
    (tmp_path/'development_sequence40.json').write_text(json.dumps({'scene_splits': {'dev': []}}))
    (tmp_path/'validation_sequence24.json').write_text(json.dumps({
        'tokens': ['frame'], 'scene_splits': {'old': ['frame']},
        'excluded_scenes': ['old'] if failure == 'excluded_scene' else []}))
    if failure == 'modified_manifest':
        (tmp_path/'experiment_plan.json').write_text(json.dumps({
            'validation': {'manifest_sha256': 'predeclared-different-content'}}))
    called = []
    monkeypatch.setattr(benchmark.os, 'sched_getaffinity', lambda _: {0})
    monkeypatch.setattr(benchmark, 'service', lambda *args, **kwargs: called.append(True))
    with pytest.raises(ValueError, match='manifest was modified|excluded previous scene'):
        benchmark.run(argparse.Namespace(root=str(tmp_path), stage='validation',
            quality_budget=.05, vlm_quality_budget=None, validation_sequence=None))
    assert not called


@pytest.mark.parametrize('operation', ['calibration', 'service', 'evaluate'])
def test_existing_experiment_outputs_are_preserved(tmp_path, monkeypatch, operation):
    import argparse
    from tools.worldstream_m2 import global_residual_benchmark as benchmark
    name = 'cal_age2_official' if operation == 'evaluate' else 'cal_age2'
    folder = tmp_path/name
    folder.mkdir()
    marker = folder/'original.txt'
    marker.write_text('original evidence')
    called = []
    monkeypatch.setattr(benchmark.os, 'sched_getaffinity', lambda _: {0})
    monkeypatch.setattr(benchmark, 'wait_gpu', lambda: called.append(True))
    with pytest.raises(ValueError, match='already exists'):
        if operation == 'calibration':
            benchmark.run(argparse.Namespace(root=str(tmp_path), stage='calibration',
                quality_budget=.05, vlm_quality_budget=None))
        elif operation == 'service':
            benchmark.service(tmp_path, 'cal_age2', tmp_path/'absent.json')
        else:
            benchmark.evaluate(tmp_path, 'cal_age2')
    assert marker.read_text() == 'original evidence'
    assert not called
