import copy
from dataclasses import replace
from threading import Event,get_ident
import time
import pytest
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.runtime import UnifiedRuntime
def helpers():
    import importlib.util
    spec=importlib.util.spec_from_file_location('overlap_test_helpers','tests/worldstream_m1/test_integration.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def runtime(**kwargs):
    return UnifiedRuntime(dict(model_version='checkpoint-sha',lora_version='lora-sha',**kwargs))


def model_at(path,overlap='serial',**config):
    h=helpers();model=h.make_model(path,WorldStateManager(M1Config(enabled=True)))
    model.training=False;model.stream_runtime=runtime(cost_aware=False,m1_overlap=overlap,**config)
    return model


def test_worker_owns_store_only_while_full_eva_and_identical_m1_results(tmp_path):
    h=helpers();serial=model_at(tmp_path/'serial');overlap=model_at(tmp_path/'overlap','experimental',reuse_m1_signature=True)
    entered=Event();main_progress=Event();original=overlap.m1_manager.precheck
    def check(frame):
        entered.set();assert main_progress.wait(2)
        assert get_ident()!=main_thread
        return original(frame)
    overlap.m1_manager.precheck=check
    extract=overlap.extract_img_feat
    def eva(img):
        assert entered.wait(2);main_progress.set()
        return extract(img)
    overlap.extract_img_feat=eva;main_thread=get_ident()
    import torch
    for timestamp in (100.,100.5):
        torch.manual_seed(4);serial.simple_test(*h.inputs(timestamp)[:1],**h.inputs(timestamp)[1])
        torch.manual_seed(4);overlap.simple_test(*h.inputs(timestamp)[:1],**h.inputs(timestamp)[1])
        assert serial.m1_manager.last_result.valid_anchor_ids==overlap.m1_manager.last_result.valid_anchor_ids
        assert serial.m1_manager.store.feature_versions()==overlap.m1_manager.store.feature_versions()
        assert serial.m1_manager.store.execution_signature==overlap.m1_manager.store.execution_signature
        assert torch.equal(serial.lm_inputs[-1],overlap.lm_inputs[-1])
        assert overlap.stream_runtime.last_record.m1['overlap_active']
    overlap.stream_runtime.close()


def test_worker_error_is_propagated_and_drained_before_reset(tmp_path):
    h=helpers();model=model_at(tmp_path,'experimental')
    def fail(frame):raise RuntimeError('controlled worker failure')
    model.m1_manager.precheck=fail
    with pytest.raises(RuntimeError,match='controlled worker'):model.simple_test(*h.inputs()[:1],**h.inputs()[1])
    assert model.stream_runtime._m1_worker.future is None
    assert model.m1_manager._pending is None and not model.m1_manager.store.anchors
    assert model.pts_bbox_head.calls==0 and model.stream_runtime.previous is None
    model.stream_runtime.close()


def test_timeout_drains_worker_before_lifecycle_reset(tmp_path):
    h=helpers();model=model_at(tmp_path,'experimental',m1_timeout_s=.001)
    original=model.m1_manager.precheck;finished=Event()
    def slow(frame):
        time.sleep(.04);result=original(frame);finished.set();return result
    model.m1_manager.precheck=slow
    with pytest.raises(TimeoutError):model.simple_test(*h.inputs()[:1],**h.inputs()[1])
    assert finished.is_set() and model.m1_manager._pending is None
    assert model.m1_manager.last_result is None and model.stream_runtime.previous is None
    model.stream_runtime.close()


def test_auto_without_certificate_stays_serial(tmp_path):
    h=helpers();model=model_at(tmp_path,'auto')
    model.stream_runtime=runtime(m1_overlap='auto')
    model.simple_test(*h.inputs()[:1],**h.inputs()[1])
    assert not model.stream_runtime.last_record.m1['overlap_active'] and model.stream_runtime._m1_worker is None


def test_incremental_vision_dependency_forces_serial(tmp_path):
    from types import SimpleNamespace
    h=helpers();model=model_at(tmp_path,'experimental')
    cache=SimpleNamespace(bytes=0)
    model.m2_runtime=SimpleNamespace(config=SimpleNamespace(m1_service='maintain',reuse_boundary='camera'),
        reset=lambda:None,pre_m1_admission=lambda *a:dict(reject=False,reason='needs_precheck'),
        forward=lambda image,*a:(model.img_backbone(image),{}),cache=cache,
        camera_executor=SimpleNamespace(cache=cache))
    model.simple_test(*h.inputs()[:1],**h.inputs()[1])
    assert not model.stream_runtime.last_record.m1['overlap_active']
    assert model.stream_runtime._m1_worker is None


def test_eva_failure_drains_precheck_before_store_invalidation(tmp_path):
    h=helpers();model=model_at(tmp_path,'experimental');entered=Event();finished=Event()
    original=model.m1_manager.precheck
    def slow(frame):
        entered.set();time.sleep(.04);result=original(frame);finished.set();return result
    def fail(image):
        assert entered.wait(2);raise RuntimeError('controlled full EVA failure')
    model.m1_manager.precheck=slow;model.extract_img_feat=fail
    with pytest.raises(RuntimeError,match='EVA failure'):model.simple_test(*h.inputs()[:1],**h.inputs()[1])
    assert finished.is_set() and model.m1_manager.last_result is None
    assert model.stream_runtime._m1_worker.future is None and model.stream_runtime.previous is None
    model.stream_runtime.close()
