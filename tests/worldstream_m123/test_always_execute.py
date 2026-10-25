"""Exercise the real inference methods and runtimes without deployed weights."""
import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from mmcv import Config

from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m3 import M3Runtime


def load_helpers(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = load_helpers('always_execute_native', 'tests/worldstream_m1/test_integration.py')
llm = load_helpers('always_execute_llm', 'tests/worldstream_m3/conftest.py').llm
eva = load_helpers('always_execute_eva', 'tests/worldstream_m2/conftest.py').eva


def formal_model(tmp_path, eva, llm, monkeypatch):
    cfg = Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    settings = replace(M1Config.load(cfg.model.m1_config), logging=True,
                       log_path=str(tmp_path / 'state.jsonl'))
    model = native.make_model(tmp_path, WorldStateManager(settings))
    model.training = False
    model.worldstream_require_m123 = cfg.model.worldstream_require_m123
    model.img_backbone = eva
    model.m2_runtime = IncrementalVisionRuntime(eva, dict(cfg.model.m2_config))
    model.m3_runtime = M3Runtime(dict(cfg.model.m3_config))

    class Head(native.Head):
        num_extra = 1

        def __call__(self, metas, pos, **data):
            self.calls += 1
            self.memory_timestamp -= data['timestamp']
            return {}, data['img_feats'].mean((-1, -2))

    model.pts_bbox_head = Head()
    model.map_head = Head(True)
    model.lm_head = llm
    # Bound only the test's decode work. Production kwargs are observed intact.
    original = model.m3_runtime.generate
    llm.generation_config.eos_token_id = None

    def short_generation(**kwargs):
        assert kwargs['max_new_tokens'] == 320 and kwargs['do_sample']
        kwargs['max_new_tokens'] = 3
        return original(**kwargs)

    monkeypatch.setattr(model.m3_runtime, 'generate', short_generation)
    return model


def request(timestamp):
    meta, data = native.inputs(timestamp)
    data['img'] = data['img'][..., :40, :40].contiguous() / 255
    meta[0]['img_shape'] = [(40, 40, 3)] * 2
    data['input_ids'] = [[torch.tensor([1, 3, -200, 4])]]
    return meta, data


@torch.no_grad()
@pytest.mark.parametrize('evidence', ['formal_unapproved', 'missing'])
def test_formal_request_always_runs_all_modules(tmp_path, eva, llm, monkeypatch, evidence):
    model = formal_model(tmp_path, eva, llm, monkeypatch)
    if evidence == 'missing':
        # Replace artifact readers, never the protected production artifacts.
        model.m2_runtime.costs.records = []
        model.m3_runtime.scheduler.costs = {}
        model.m3_runtime.scheduler.quality = {}
        model.m3_runtime.scheduler.state_quality = {}
    assert not model.m3_runtime.scheduler.quality.get('approved')
    forward_lengths = []
    original = llm.forward

    def observe_forward(*args, **kwargs):
        x = kwargs.get('inputs_embeds', kwargs.get('input_ids'))
        forward_lengths.append(x.shape[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(llm, 'forward', observe_forward)
    # Enabled modules must execute even if a native answer file already exists.
    (tmp_path / '100.0').write_text('previous answer')
    traces = []
    for timestamp in (100., 100.1):
        meta, data = request(timestamp)
        result = model.simple_test(meta, **data)
        trace = copy.deepcopy(model.worldstream_execution)
        traces.append(trace)
        assert trace['completed']
        assert all(trace['m1'][key] for key in
                   ('precheck_entered', 'precheck_completed', 'commit_entered', 'commit_completed'))
        assert trace['m1']['transaction_status'] == 'committed'
        assert trace['m2']['runtime_entered'] and trace['m2']['execution_completed']
        assert trace['m2']['full_recompute'] and not trace['m2']['actual_window_reuse']
        assert trace['m2']['executor'] == 'WindowExecutor'
        assert trace['m2']['full_recompute_reason']
        assert trace['m3']['runtime_entered'] and trace['m3']['execution_completed']
        row = trace['m3']['requests'][0]
        assert row['reused_tokens'] == 0 and row['full_prefill']
        assert row['prefill_completed'] and row['decode_completed']
        assert not row['historical_text_consumed']
        assert row['history_reason'] and row['full_prefill_reason']
        assert model.m3_diagnostics['requests'][0]['visual_tokens'] == 4
        assert result[0]['pts_bbox'] and result[0]['lane_results'] and result[0]['text_out']
        assert model.m1_manager._pending is None
    assert model.pts_bbox_head.calls == model.map_head.calls == 2
    assert forward_lengths == [7, 1, 1, 7, 1, 1]  # Full prefill, then native decode.
    logs = [json.loads(line) for line in (tmp_path / 'state.jsonl').read_text().splitlines()]
    assert [r['experiment']['module_execution'] for r in logs] == traces


@torch.no_grad()
@pytest.mark.parametrize('missing', ['m1_manager', 'm2_runtime', 'm3_runtime'])
def test_formal_route_rejects_missing_runtime(tmp_path, eva, llm, monkeypatch, missing):
    model = formal_model(tmp_path, eva, llm, monkeypatch)
    setattr(model, missing, None)
    meta, data = request(100.)
    with pytest.raises(RuntimeError, match='requires M1, M2, M3'):
        model.simple_test(meta, **data)
    assert not model.worldstream_execution['completed']
    assert not model.lm_inputs


@torch.no_grad()
def test_formal_route_detects_visual_bypass(tmp_path, eva, llm, monkeypatch):
    model = formal_model(tmp_path, eva, llm, monkeypatch)
    # Simulate a future accidental native-backbone shortcut, retaining task heads.
    def bypass(image):
        n = image.shape[1]
        return eva(image.reshape(-1, 3, 40, 40))[0].reshape(1, n, 32, 10, 10)
    monkeypatch.setattr(model, 'extract_img_feat', bypass)
    meta, data = request(100.)
    with pytest.raises(RuntimeError, match='bypassed or did not complete'):
        model.simple_test(meta, **data)
    assert not model.worldstream_execution['m2']['runtime_entered']
    assert not model.worldstream_execution['completed']


@torch.no_grad()
def test_formal_empty_prompt_cannot_claim_prefill_completion(tmp_path, eva, llm, monkeypatch):
    model = formal_model(tmp_path, eva, llm, monkeypatch)
    meta, data = request(100.)
    data['input_ids'] = [[]]
    with pytest.raises(ValueError, match='requires at least one language prompt'):
        model.simple_test(meta, **data)
    assert not model.worldstream_execution['completed']
    assert not model.worldstream_execution['m3']['execution_completed']


@torch.no_grad()
def test_request_trace_does_not_inherit_previous_success(tmp_path, eva, llm, monkeypatch):
    model = formal_model(tmp_path, eva, llm, monkeypatch)
    meta, data = request(100.)
    model.simple_test(meta, **data)
    assert model.worldstream_execution['completed']
    def fail(*a, **kw):raise RuntimeError('unexpected source failure before M2 forward')
    monkeypatch.setattr(model.m1_manager, 'export_m2_inputs', fail)
    meta, data = request(100.1)
    with pytest.raises(RuntimeError, match='unexpected source failure'):
        model.simple_test(meta, **data)
    trace = model.worldstream_execution
    assert trace['m1']['transaction_status'] == 'aborted'
    assert trace['m2']['runtime_entered'] and not trace['m2']['execution_completed']
    assert not trace['m3']['runtime_entered'] and not trace['completed']


@torch.no_grad()
@pytest.mark.parametrize('failure', ['vision', 'commit', 'generation', 'invalid_answers', 'publish'])
def test_failure_diagnostics_and_transaction_lifecycle(tmp_path, eva, llm, monkeypatch, failure):
    model = formal_model(tmp_path, eva, llm, monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError('injected execution failure')

    if failure == 'vision':
        monkeypatch.setattr(model.m2_runtime.executor, 'execute', fail)
    elif failure == 'commit':
        monkeypatch.setattr(model.m1_manager, '_commit', fail)
    elif failure == 'generation':
        monkeypatch.setattr(model.m3_runtime.executor, 'generate', fail)
    elif failure == 'invalid_answers':
        monkeypatch.setattr(model.m3_runtime, 'generate', lambda **kwargs: (None, {}))
    else:
        import worldstream.m3.world_state as state
        monkeypatch.setattr(state, 'freeze_verified_state', fail)
    meta, data = request(100.)
    with pytest.raises(RuntimeError):
        model.simple_test(meta, **data)
    trace = model.worldstream_execution
    assert not trace['completed'] and trace['error']
    assert trace['m1']['precheck_completed']
    assert model.m1_manager._pending is None
    committed = failure in ('generation', 'invalid_answers')
    assert trace['m1']['commit_completed'] == committed
    assert trace['m1']['transaction_status'] == ('committed' if committed else 'aborted')
    assert trace['m1']['commit_entered'] == (failure not in ('vision', 'publish'))
    assert not trace['m3']['execution_completed']
    # A failure must never silently invoke the independent native generator.
    assert not model.lm_inputs
    log = json.loads((tmp_path / 'state.jsonl').read_text())
    assert log['experiment']['module_execution'] == trace


@torch.no_grad()
def test_prefill_rejection_does_not_reenter_native_generate(llm, monkeypatch):
    runtime = M3Runtime(dict(min_prefix_tokens=2))
    calls = []
    original = runtime.executor.generate

    def observe(model, expanded, **kwargs):
        calls.append((expanded, kwargs['reuse_length']))
        return original(model, expanded, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('native multimodal generate bypass')

    monkeypatch.setattr(runtime.executor, 'generate', observe)
    monkeypatch.setattr(llm, 'generate', forbidden)
    ids = torch.tensor([1, 3, -200, 4])
    vision = torch.randn(1, 4, 32)
    ctx = dict(scene_id='s', frame_id='f', camera_ids=['front'], timestamp=1.)
    _, diag = runtime.generate(llm, [ids, ids], vision, context=ctx, max_new_tokens=3)
    assert diag['runtime_entered'] and diag['execution_completed']
    assert [length for _, length in calls] == [0, 0]
    assert diag['requests'][1]['matched_prefix'] == 7
    assert diag['requests'][1]['prefix_admission'] == 'prefix_evidence_unavailable'
    for (expanded, _), row in zip(calls, diag['requests']):
        assert row['full_prefill'] and row['forward_prefill_lengths'] == [7]
        assert row['decode_completed'] and not row['historical_text_consumed']
        assert expanded.token_ids[0, [0, 1, -1]].tolist() == [1, 3, 4]
        assert torch.equal(expanded.embeddings[:, 2:6], vision)
