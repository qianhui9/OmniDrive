"""Recovery regressions: transient driver errors, existing outputs and final reports."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from . import recover
from .run import read, sha, write


def test_driver_exit18_retries_then_runs_only_when_idle(tmp_path, monkeypatch):
    replies = iter([
        subprocess.CalledProcessError(18, ['nvidia-smi'], stderr='Unknown Error'),
        SimpleNamespace(stdout='12345\n'), SimpleNamespace(stdout='10000, 99\n'),
        SimpleNamespace(stdout=''), SimpleNamespace(stdout='100, 0\n')])

    def call(*args, **kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    sleeps, stages = [], []
    monkeypatch.setattr(recover.subprocess, 'run', call)
    monkeypatch.setattr(recover.time, 'sleep', sleeps.append)
    monkeypatch.setattr(recover.run, 'stage', lambda out, name, **kw: stages.append(name))
    recover.wait_gpu(tmp_path, interval=30)
    assert stages == ['WAIT_GPU_DRIVER', 'WAIT_GPU_IDLE']
    assert sleeps == [30, 30]


@pytest.mark.parametrize('error', [FileNotFoundError('nvidia-smi'),
                                  subprocess.TimeoutExpired(['nvidia-smi'], 15)])
def test_probe_failure_never_grants_gpu_access(tmp_path, monkeypatch, error):
    def call(*args, **kwargs):
        raise error
    monkeypatch.setattr(recover.subprocess, 'run', call)
    with pytest.raises(RuntimeError, match='no inference started'):
        recover.wait_gpu(tmp_path, once=True)


def test_failed_attempt_is_preserved_and_complete_variant_cannot_be_archived(tmp_path):
    out = tmp_path / 'run'
    failed = out / 'failed'
    failed.mkdir(parents=True)
    (failed / 'receipts.jsonl').write_text('partial evidence\n')
    write(failed / 'failure.json', {'returncode': 1})
    history = out / 'recovery_history' / 'attempt'
    new = recover.prepare_attempt(out, 'failed', history)
    assert not list(new.iterdir())
    assert (history / 'failed' / 'receipts.jsonl').read_text() == 'partial evidence\n'
    write(new / 'inference_complete.json', {'status': 'COMPLETED'})
    with pytest.raises(RuntimeError, match='completed inference'):
        recover.prepare_attempt(out, 'failed', out / 'other_history')
    assert (new / 'inference_complete.json').exists()


def test_unmarked_partial_worker_directory_is_not_moved(tmp_path):
    directory = tmp_path / 'running'
    directory.mkdir()
    (directory / 'receipts.jsonl').write_text('active worker evidence')
    with pytest.raises(RuntimeError, match='no failure marker'):
        recover.prepare_attempt(tmp_path, 'running', tmp_path / 'history')
    assert (directory / 'receipts.jsonl').exists()


def test_only_missing_variants_are_selected_and_bad_completed_data_is_rejected(tmp_path):
    manifest = dict(variants=['done', 'failed', 'not_started'], smoke=False,
                    scenes=[dict(tokens=['a', 'b'])])
    d = tmp_path / 'done'
    d.mkdir()
    (d / 'receipts.jsonl').write_text('{"frame_id":"a"}\n{"frame_id":"b"}\n')
    write(d / 'inference_complete.json', dict(status='COMPLETED', smoke=False, frames=2,
          receipts_sha256=sha(d / 'receipts.jsonl')))
    assert recover.plan(tmp_path, manifest)['missing'] == ['failed', 'not_started']
    (d / 'receipts.jsonl').write_text('{"frame_id":"a"}\n')
    with pytest.raises(RuntimeError, match='refusing replacement'):
        recover.plan(tmp_path, manifest)


def test_two_recovery_launchers_cannot_mutate_same_output(tmp_path):
    with recover.recovery_lock(tmp_path):
        with pytest.raises(RuntimeError, match='already running'):
            with recover.recovery_lock(tmp_path):
                pytest.fail('Acquired second recovery lock')


def test_complete_report_does_not_keep_obsolete_missing_status(tmp_path):
    # Explicit rendering fixture only; these copied metric values are not new inference.
    import copy
    from .summarize_results import render
    source = Path(__file__).resolve().parents[2] / 'outputs/worldstream_ablation_full_run01'
    manifest = read(source / 'manifest.json')
    metrics = copy.deepcopy(read(source / 'metrics.json')['metrics'])
    for name in manifest['variants']:
        if name not in metrics:
            metrics[name] = copy.deepcopy(metrics['qcse_exact_cache'])
    audit = dict(generated_at='fixture only', status='COMPLETE', planned_variants=14,
                 completed_variants=14, variants={n: dict(status='COMPLETED') for n in metrics})
    render(tmp_path, manifest, metrics, audit, read(source / 'scenario_tags.json'))
    text = (tmp_path / '结果.md').read_text()
    assert '整套任务已完成：14/14' in text
    assert 'NOT_STARTED' not in text and '两组缺失' not in text and '未启动' not in text
    assert 'QCSE 的所有替代配置已齐全' in text
    assert '当前核验完成 14/14' in (tmp_path / 'ABLATION_REPORT.md').read_text()
