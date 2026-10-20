"""Cross-module service/lifecycle regressions, independent of pretrained weights."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from worldstream.execution import FrameIdentity, FrameExecutionRecord, StageRecorder
from worldstream.runtime import UnifiedRuntime
from worldstream.m1 import M1Config, WorldStateManager


def helpers():
    spec = importlib.util.spec_from_file_location('stream_test_helpers',
        Path('tests/worldstream_m1/test_integration.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime(service='persistent_state', **kwargs):
    return UnifiedRuntime(dict(service=service, model_version='checkpoint-sha',
        lora_version='lora-sha', **kwargs))


@pytest.mark.parametrize('change,reason', [
    ({'scene_id': 'other'}, 'scene_change'),
    ({'camera_ids': ('b', 'a')}, 'camera_order_change'),
    ({'timestamp': 1.}, 'timestamp_reversal_or_repeat'),
    ({'timestamp': 8.}, 'stale_stream'),
    ({'execution_signature': 'new'}, 'model_or_execution_change'),
    ({'lora_version': 'new'}, 'model_or_execution_change'),
])
def test_lifecycle(change, reason):
    from dataclasses import replace
    r = runtime()
    r.previous = FrameIdentity('s', 'f', 1., ('a', 'b'), 'm', 'l', 'sig')
    current = replace(r.previous, frame_id='next', timestamp=1.5, **change) if 'timestamp' not in change else replace(r.previous, **change)
    assert r.lifecycle_reason(current) == reason


def test_current_output_cannot_export_history_and_still_generates(tmp_path):
    h = helpers(); manager = WorldStateManager(M1Config(enabled=True))
    model = h.make_model(tmp_path, manager); model.training = False
    model.stream_runtime = runtime('current_output')
    for ts in (100., 100.5):
        meta, data = h.inputs(ts)
        model.simple_test(meta, **data)
        assert not manager.store.anchors and manager._pending is None
        assert model.m3_verified_state is None
        assert model.stream_runtime.last_record.m1['history_preserved'] is False
    assert len(model.lm_inputs) == 2
    assert model.map_head.calls == model.pts_bbox_head.calls == 2


def test_persistent_no_gpu_hit_retains_real_commit_sources(tmp_path):
    h = helpers(); manager = WorldStateManager(M1Config(enabled=True))
    model = h.make_model(tmp_path, manager); model.training = False
    model.stream_runtime = runtime(trace_path=str(tmp_path/'trace.jsonl'))
    for ts in (100., 100.5):
        meta, data = h.inputs(ts)
        model.simple_test(meta, **data)
        assert manager.store.last_timestamp == ts
        assert all(a.source_frame_id == str(ts) for a in manager.store.anchors.values())
        assert all(a.evidence_origin == 'actual_perception' for a in manager.store.anchors.values())
        assert all(f.source_frame_id == str(ts) for f in manager.store.features.values())
    assert len((tmp_path/'trace.jsonl').read_text().splitlines()) == 2


def test_exception_after_head_mutation_clears_all_caches_and_heads(tmp_path):
    h = helpers(); model = h.make_model(tmp_path, WorldStateManager(M1Config(enabled=True)))
    model.training = False
    model.stream_runtime = runtime(cost_aware=False)
    reset = []
    for name in ('pts_bbox_head', 'map_head'):
        getattr(model, name).reset_memory = lambda name=name: reset.append(name)
    model.m3_runtime = SimpleNamespace(reset=lambda: reset.append('m3'), config=SimpleNamespace(state_context='none'))
    def failed(*args, **kw):
        raise RuntimeError('failure after temporal mutation')
    model.map_head.get_bboxes = failed
    meta, data = h.inputs()
    with pytest.raises(RuntimeError, match='temporal mutation'):
        model.simple_test(meta, **data)
    assert model.stream_runtime.previous is None
    assert model.stream_runtime.last_record.status == 'failed'
    assert not model.m1_manager.store.anchors
    assert all(reset.count(name) == 2 for name in ('pts_bbox_head', 'map_head', 'm3'))


def test_persistent_rejects_output_only_admission(tmp_path):
    h = helpers(); model = h.make_model(tmp_path, WorldStateManager(M1Config(enabled=True)))
    model.training = False; model.stream_runtime = runtime()
    model.m2_runtime = SimpleNamespace(reset=lambda: None, config=SimpleNamespace(m1_service='output_only'))
    meta, data = h.inputs()
    with pytest.raises(ValueError, match='Persistent service'):
        model.simple_test(meta, **data)


def test_memory_budget_includes_physical_history_and_explicitly_invalidates(tmp_path):
    h=helpers();manager=WorldStateManager(M1Config(enabled=True))
    model=h.make_model(tmp_path,manager);model.training=False
    model.stream_runtime=runtime(memory_budget_mb=0)
    meta,data=h.inputs()
    model.simple_test(meta,**data)
    record=model.stream_runtime.last_record
    assert record.memory['m1_payload_bytes']>0
    assert 'persistent_state_explicitly_invalidated_for_budget' in record.fallback_reasons
    assert manager.last_result is None and not manager.store.anchors


def test_nonfinite_timestamp_invalidates_previous_state(tmp_path):
    h = helpers(); model = h.make_model(tmp_path, WorldStateManager(M1Config(enabled=True)))
    model.training = False; model.stream_runtime = runtime()
    for head in (model.pts_bbox_head, model.map_head):
        head.reset_memory = lambda head=head: setattr(head, 'memory_timestamp', None)
    meta, data = h.inputs(float('nan'))
    with pytest.raises(ValueError, match='lifecycle'):
        model.simple_test(meta, **data)
    assert model.pts_bbox_head.memory_timestamp is None


def test_camera_metadata_must_cover_actual_batch(tmp_path):
    h=helpers();model=h.make_model(tmp_path,WorldStateManager(M1Config(enabled=True)))
    model.training=False;model.stream_runtime=runtime()
    meta,data=h.inputs();meta[0]['cam_infos']={'front':{}}
    with pytest.raises(ValueError,match='lifecycle'):
        model.simple_test(meta,**data)


def test_nested_timers_do_not_double_count():
    identity = FrameIdentity('s', 'f', 1., ('a',), 'm', 'l', 'sig')
    record = FrameExecutionRecord(identity, 1, 'persistent_state')
    timer = StageRecorder(record)
    with timer.stage('outer'):
        with timer.stage('inner', dict(frame='historical', computation_version=2)):
            sum(range(1000))
    timer.finish()
    assert sum(s['exclusive_wall_ms'] for s in record.stages) == pytest.approx(record.stages[0]['wall_ms'])
    assert record.stages[1]['computed_source']['frame'] == 'historical'
    assert record.gpu_kernel_ms is None


def test_instrumentation_preserves_hf_generation_signature(llm,carriers):
    import inspect
    identity=FrameIdentity('s','f',1.,('a',),'m','l','sig')
    record=FrameExecutionRecord(identity,1,'current_output')
    model=SimpleNamespace(lm_head=llm)
    original=inspect.signature(llm.forward)
    ids=torch.tensor([[1,3,-200,4]])
    expected=llm.generate(inputs=ids,images=carriers,max_new_tokens=4,do_sample=False)
    with runtime().instrument(model,StageRecorder(record)):
        assert inspect.signature(llm.forward)==original
        actual=llm.generate(inputs=ids,images=carriers,max_new_tokens=4,do_sample=False)
    assert inspect.signature(llm.forward)==original
    assert torch.equal(expected,actual)


def test_cached_suffix_embedding_forward_is_prefill(llm,carriers):
    from worldstream.m3 import expand_input
    identity=FrameIdentity('s','f',1.,('a',),'m','l','sig')
    record=FrameExecutionRecord(identity,1,'current_output')
    model=SimpleNamespace(lm_head=llm)
    expanded=expand_input(llm,torch.tensor([1,3,-200,4,5]),carriers)
    with runtime().instrument(model,StageRecorder(record)):
        prefix=llm(inputs_embeds=expanded.embeddings[:,:-1],use_cache=True)
        llm(inputs_embeds=expanded.embeddings[:,-1:],past_key_values=prefix.past_key_values,use_cache=True)
    calls=[s for s in record.stages if s['name'].startswith('llm_')]
    assert [s['name'] for s in calls]==['llm_prefill','llm_prefill']
    assert calls[-1]['forward_tokens']==1 and len(calls[-1]['input_shape'])==3


def test_cost_admission_needs_identical_workload_independent_quality_and_net_cost():
    key = dict(model_version='m', lora_version='l', execution_key='e', generation_key='g',
        input_shape=[6, 3, 640, 640], question_lengths=[20, 30], service='persistent_state',
        strategy='batch', cache_state='cold', memory_budget_bytes=100, refresh_interval=10, frame_gap_s=.5)
    costs = dict(key=key, includes_all_incremental_costs=True, full_wall_ms=10., optimized_wall_ms=8., peak_cache_bytes=90)
    quality = dict(key=key, exact_targets_pass=True, heldout_samples=10, calibration_scenes=['a'],
        heldout_scenes=['b'], evidence_sha256='sha', protocol_sha256='protocol')
    assert UnifiedRuntime.admit(key, costs, quality)[0]
    assert not UnifiedRuntime.admit(key, costs, quality, overhead_ms=3)[0]
    assert not UnifiedRuntime.admit(key, costs, dict(quality, heldout_scenes=['a']))[0]
    assert not UnifiedRuntime.admit(key, costs, dict(quality, exact_targets_pass=False))[0]
    assert not UnifiedRuntime.admit(dict(key, question_lengths=[20]), costs, quality)[0]
    assert not UnifiedRuntime.admit(key, dict(costs, optimized_wall_ms=float('nan')), quality)[0]
    assert not UnifiedRuntime.admit(key, dict(costs, peak_cache_bytes=101), quality)[0]


def test_offline_strategy_cannot_inherit_full_quality_certificate():
    r=runtime()
    model=SimpleNamespace(m3_runtime=SimpleNamespace(config=SimpleNamespace(offline_experiment=True)))
    allowed,reason=r.optimization_admission(model,{})
    assert not allowed and reason=='offline_strategy_cannot_inherit_production_certificate'


def test_inplace_weight_change_cannot_reuse_old_content_certificate(tmp_path):
    h=helpers();model=h.make_model(tmp_path,WorldStateManager(M1Config(enabled=True)))
    model.training=False;model.stream_runtime=runtime()
    weight=torch.nn.Parameter(torch.ones(1))
    model.named_parameters=lambda:[('actual_weight_version_witness',weight)]
    meta,data=h.inputs();model.simple_test(meta,**data)
    with torch.no_grad():weight.add_(1)
    meta,data=h.inputs(100.5);model.simple_test(meta,**data)
    assert model.stream_runtime._certificate_invalidated
    assert 'model_changed_without_new_content_version' in model.stream_runtime.last_record.fallback_reasons
