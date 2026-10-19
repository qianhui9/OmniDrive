"""CPU tests of CUDA job admission and result accounting; no GPU evidence."""
import json
from types import SimpleNamespace
from tools.worldstream_m2 import state_interface_cuda_batch as batch
from tools.worldstream_m1.state_interface_gpu_report import summarize


def test_busy_gpu_is_not_selected_when_utilization_is_zero(tmp_path,monkeypatch):
    responses=iter([
        SimpleNamespace(returncode=0,stderr="",stdout=
            "0, GPU-busy, NVIDIA RTX 3090, 0, 24576, 0\n"
            "1, GPU-free, NVIDIA RTX 3090, 0, 24576, 0\n"),
        SimpleNamespace(returncode=0,stderr="",stdout="GPU-busy, 1234, python, 0\n")])
    monkeypatch.setattr(batch.subprocess,"run",lambda *a,**kw:next(responses))
    selected=batch.wait_idle(tmp_path)
    assert selected["index"]==1
    recorded=json.loads((tmp_path/"gpu_availability.jsonl").read_text())
    assert recorded["processes"][0][1]=="1234"
    assert len(recorded["available"])==1


def test_missing_driver_access_is_not_reported_as_a_free_gpu(monkeypatch):
    import pytest
    monkeypatch.setattr(batch.subprocess,"run",lambda *a,**kw:
        SimpleNamespace(returncode=9,stderr="NVML unavailable",stdout=""))
    with pytest.raises(RuntimeError,match="GPU access unavailable"):
        batch.query_gpus()


def test_missing_results_remain_not_run(tmp_path):
    result=summarize(tmp_path)
    assert result["status"]=="Not Run"
    assert not result["arms"] and not result["all_quality_pass"]


def test_warmup_latency_is_excluded_from_gpu_summary(tmp_path):
    (tmp_path/"status.json").write_text(json.dumps(dict(status="completed")))
    frame=dict(frame="f",E2E_wall_ms=20.,E2E_GPU_ms=15.,M1_time_ms=3.,
        runtime=dict(actual_reused_cameras=1,source_age=.5))
    warm=dict(frame, E2E_wall_ms=1000.)
    runs=[dict(mode="original",warmup=True,frames=[warm],peak_gpu_allocated=100,
        peak_gpu_reserved=200,cpu_rss_high_water_kib=300,scene="s",repeat=0),
        dict(mode="original",warmup=False,frames=[frame],peak_gpu_allocated=100,
        peak_gpu_reserved=200,cpu_rss_high_water_kib=300,scene="s",repeat=1)]
    (tmp_path/"runs.json").write_text(json.dumps(runs))
    summary=summarize(tmp_path)
    assert summary["arms"]["original"]["requests"]==1
    assert summary["arms"]["original"]["E2E_wall_ms"]["p50"]==20.
    assert not summary["all_quality_pass"]


def test_fresh_zero_cost_profile_does_not_grant_task_qualification(tmp_path, monkeypatch, eva):
    import torch
    from tools.worldstream_m2.state_interface_integration import profile_zero_camera_cost
    from worldstream.m2 import profiler
    from worldstream.m2.camera_executor import camera_cost_key
    monkeypatch.setattr(profiler, 'measure', lambda fn, **kw: dict(wall_ms=1., gpu_ms=1.))
    images = torch.randn(2, 3, 40, 40)
    result = profile_zero_camera_cost(eva, images, tmp_path/'costs.json')
    assert len(result['records']) == 1
    record = result['records'][0]
    assert record['key'] == camera_cost_key(eva, images)
    assert record['changed_cameras'] == 0
    assert record['quality_certified'] is False
    assert record['feature_error']['max_abs'] == 0.
