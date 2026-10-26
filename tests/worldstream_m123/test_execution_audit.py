"""Audit rejects inconsistent success/reuse flags and preserves real failure counts."""
import copy
from types import SimpleNamespace

import pytest

from tools.worldstream_m123.benchmark.audit import execution_summary, validate_execution_row
from tools.worldstream_m123.benchmark.common import write
from tools.worldstream_m123.benchmark.worker import append


def frame():
    m1 = dict(precheck_entered=True, precheck_completed=True, commit_entered=True,
              commit_completed=True, transaction_status='committed', state_epoch=1,
              valid_states=2, expired_states=1, full_refresh_reasons=['periodic'])
    m2 = dict(runtime_entered=True, execution_completed=True, executor='WindowExecutor',
              actual_skipped_windows=0, actual_window_reuse=False, full_recompute=True,
              stages=[dict(total_windows=8, executed_windows=8, reused_windows=0)],
              cache_saved=False, full_recompute_reason='no_valid_cache')
    q = dict(executor='PrefillExecutor', prefill_entered=True, prefill_completed=True,
             decode_completed=True, full_prefill=True, reused_tokens=0, matched_prefix=3,
             full_prefill_reason='prefix_evidence_unavailable', historical_text_consumed=False,
             cache_stored=False, cache_bytes=0, cache_budget_bytes=512,
             prefill_including_cache_gpu_ms=4., decode_loop_gpu_span_ms=6.)
    m3 = dict(runtime_entered=True, execution_completed=True, requests=[q])
    identity = dict(config='formal.py', checkpoint='formal.pth', config_sha256='a'*64,
                    checkpoint_sha256='b'*64, source_manifest_sha256='c'*64)
    return dict(scene_id='s', frame_id='f', timestamp=1., identity=identity, m2=m2, m3=m3,
        module_execution=dict(scene_id='s', frame_id='f', timestamp=1., completed=True,
                              identity=identity,
                              m1=m1, m2=copy.deepcopy(m2), m3=copy.deepcopy(m3)),
        request_ms=20., peak_gpu_allocated_bytes=1000, peak_gpu_reserved_bytes=2000,
        observed_window_attention_batches={'0':8}, stages={},
        m1_cpu_timings_ms={'precheck_total':1., 'observation_validation':.5},
        m1_commit_timings_ms={'commit_total_ms':2., 'm1_total_including_adapters_ms':3.})


def test_consistent_full_compute_and_matched_but_unapproved_prefix():
    assert validate_execution_row(frame()) == []


@pytest.mark.parametrize('case', ['identity', 'm1', 'm2', 'm3', 'reuse', 'attention', 'native', 'history', 'prefix', 'run_identity', 'visual_incomplete', 'language_incomplete'])
def test_audit_rejects_fabricated_completion_or_reuse(case):
    row = frame()
    if case == 'identity': row['module_execution']['frame_id'] = 'stale'
    elif case == 'm1': row['module_execution']['m1']['commit_completed'] = False
    elif case == 'm2': row['m2']['executor'] = 'native'
    elif case == 'm3': row['m3']['requests'][0]['decode_completed'] = False
    elif case == 'reuse': row['m2']['actual_window_reuse'] = True
    elif case == 'attention': row['observed_window_attention_batches']['0'] = 7
    elif case == 'native': row['stages']['native_generation'] = dict(calls=1)
    elif case == 'history': row['m3']['requests'][0]['historical_text_consumed'] = True
    elif case == 'run_identity': row['identity'] = {}
    elif case == 'visual_incomplete': row['m2']['execution_completed'] = False
    elif case == 'language_incomplete': row['m3']['execution_completed'] = False
    else: row['m3']['requests'][0].update(reused_tokens=4, full_prefill=False)
    assert validate_execution_row(row)


def test_summary_keeps_completion_history_prefix_and_cache_write_distinct(tmp_path):
    row = frame()
    append(tmp_path/'per_frame_metrics.jsonl', row)
    write(tmp_path/'status.json', dict(status='COMPLETED'))
    summary = execution_summary(tmp_path)
    assert summary['status'] == 'PASS'
    assert summary['m1']['precheck']['rate'] == summary['m1']['commit']['rate'] == 1
    assert summary['m1']['expired_states_total'] == 1
    assert summary['m2']['actual_window_reuse']['count'] == 0
    assert summary['m2']['full_recompute']['count'] == 1
    assert summary['m2']['per_frame'][0]['total_windows'] == 8
    assert summary['m3']['prefill_completed'] == summary['m3']['decode_completed'] == 1
    assert summary['m3']['matched_prefix_requests'] == 1
    assert summary['m3']['kv_reuse_requests'] == summary['m3']['historical_text_consumed']['count'] == 0
    assert summary['m3']['cache_write_failures'] == 1
    assert summary['official_detection'] is None
    assert summary['latency_ms']['mean'] == 20.
    assert summary['throughput_frames_s'] == 50.
    assert summary['m3']['timings']['prefill_including_cache_gpu_ms']['mean'] == 4.
    append(tmp_path/'request_failures.jsonl', dict(module_execution=dict(error=dict(stage='m2_vision', message='OOM'))))
    summary = execution_summary(tmp_path)
    assert summary['status'] == 'FAIL'
    assert summary['failures']['rate'] == .5
    assert summary['failure_stages'] == {'m2_vision':1}


def test_empty_run_has_unknown_metrics_not_measured_zero(tmp_path):
    summary = execution_summary(tmp_path)
    assert summary['status'] == 'FAIL'
    assert summary['latency_ms']['mean'] is None
    assert summary['m1']['commit']['rate'] is None
    assert summary['throughput_frames_s'] is None


def test_observer_keeps_verified_topology_and_covers_window_executor(eva, monkeypatch):
    import torch
    from tools.worldstream_m123.benchmark.observe import Observer
    from worldstream.m2 import IncrementalVisionRuntime
    runtime = IncrementalVisionRuntime(eva)
    model = SimpleNamespace(img_backbone=eva, m2_runtime=runtime)
    observer = Observer.__new__(Observer)
    observer.model, observer.handles, observer.events = model, [], []
    class Event:
        def record(self): pass
        def elapsed_time(self, end): return 1.
    monkeypatch.setattr(torch.cuda, 'Event', lambda **kwargs: Event())
    observer.observe_module(eva.patch_embed, 'm2_patch_embedding')
    for i, block in enumerate(eva.blocks):
        if i not in runtime.prefix:
            observer.observe_module(block, f'm2_suffix_block_{i}')
    observer.wrap(runtime.executor, 'execute', 'm2_window_block', window=True)
    assert runtime.planner.cacheable_prefix() == runtime.prefix
    assert runtime.planner.rejection_reason is None
    with torch.no_grad():
        x = eva.patch_embed(torch.randn(2, 3, 40, 40))
        runtime._execute(x, dict(scene_id='s',state_epoch=1,frame_id='f', timestamp=1., camera_ids=['a','b']), None, 'no_cache')
    labels = [event[0] for event in observer.events]
    assert labels[0] == 'm2_patch_embedding'
    assert labels.count('m2_window_block_0') == labels.count('m2_window_block_1') == 1
    assert 'm2_suffix_block_2' in labels


@pytest.fixture
def eva():
    import importlib.util
    spec = importlib.util.spec_from_file_location('audit_eva', 'tests/worldstream_m2/conftest.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.eva.__wrapped__()
