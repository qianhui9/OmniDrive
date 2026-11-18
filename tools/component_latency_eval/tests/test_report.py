import pytest
from tools.component_latency_eval.report import aggregate_samples, summarize_trial, summarize_trials


def sample(component, value, frame='f', trial='r', clock='cpu_host_ms'):
    return dict(trial_id=trial, sequence_id='s', frame_id=frame, query_id='', component_name=component,
                execution_path='', sample_unit='frame', execution_status='COMPLETED', invocation_count=1,
                **{clock:value})


def test_between_trial_std_not_pooled():
    a = summarize_trial([sample('OGSM.Precheck', 0), sample('OGSM.Precheck', 2, 'g')], 'r1', 2, 0)
    b = summarize_trial([sample('OGSM.Precheck', 10)], 'r2', 1, 0)
    row = next(r for r in summarize_trials(a+b) if r['component_name']=='OGSM.Precheck')
    assert row['mean_ms'] == 5.5  # Not pooled 4.0.
    assert row['mean_std_ms'] == pytest.approx(6.3639610307)
    assert row['p95_ms'] == 5.95
    assert row['samples'] == 3


def test_empty_conditional_paths_no_invented_zero_latency():
    rows = summarize_trial([sample('Frame E2E', 10, clock='component_wall_ms')], 'r', 1, 5)
    row = next(r for r in rows if r['component_name']=='QCSE.Direct Answer')
    assert row['sample_count'] == 0 and row['mean_ms'] is None and row['p95_ms'] is None
    assert row['invocation_rate'] == 0 and row['global_average_ms'] == 0
    assert summarize_trials([row])[0]['mean_std_ms'] is None


def test_same_component_frame_sum_and_no_double_counting():
    rows = [sample('DAVP.Visual Planning', 2), sample('DAVP.Visual Planning', 3),
            sample('OGSM.Precheck', 4), sample('OGSM.Commit', 6),
            sample('Detection Head', 20, clock='gpu_device_ms'), sample('Mapping Head', 30, clock='gpu_device_ms')]
    result = aggregate_samples(rows)
    assert next(r for r in result if r['component_name']=='DAVP.Visual Planning')['cpu_host_ms'] == 5
    assert next(r for r in result if r['component_name']=='OGSM.Total')['cpu_host_ms'] == 10
    perception = next(r for r in result if r['component_name']=='Original Perception')
    assert perception['gpu_device_ms'] == 50
    assert perception['cpu_host_ms'] is None


def test_invocation_vs_frame_counts():
    samples = aggregate_samples([sample('DAVP.Visual Planning', 1), sample('DAVP.Visual Planning', 2)])
    row = next(r for r in summarize_trial(samples, 'r', 1, 5) if r['component_name']=='DAVP.Visual Planning')
    assert row['sample_count'] == 1 and row['invocation_count'] == 2
    assert row['invocation_rate'] == 1 and row['mean_ms'] == 3
