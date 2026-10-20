"""Run actual Petr3D methods, preserving heads and answer-file behavior."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch


def test_entry_m3_reexecutes_existing_answer_and_keeps_heads(tmp_path):
    spec = importlib.util.spec_from_file_location('m3_integration_helpers', Path('tests/worldstream_m1/test_integration.py'))
    helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
    model = helpers.make_model(tmp_path, None)
    model.pts_bbox_head.num_extra = 1
    model.training = False
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return [torch.tensor([[1, 7, 2]])], dict(actual=True)
    model.m3_runtime = SimpleNamespace(generate=generate)
    # A previous answer must not skip an enabled M3 request.
    (tmp_path/'100.0').write_text('old_answer')
    meta, data = helpers.inputs(100.)
    result = model.simple_test(meta, **data)
    assert len(calls) == 1
    assert calls[0]['layout'].object_tokens == 1
    assert calls[0]['layout'].ego_tokens == 1
    assert calls[0]['context']['timestamp'] == 100.
    assert calls[0]['do_sample'] is True and calls[0]['max_new_tokens'] == 320
    assert result[0]['text_out'] == [dict(Q='Q', A=[[1, 7, 2]])]
    assert model.pts_bbox_head.calls == model.map_head.calls == 1
    assert model.pts_bbox_head.memory_timestamp.item() == -100.
    assert model.m3_diagnostics == dict(actual=True)


def test_entry_disabled_keeps_original_existing_file_behavior(tmp_path):
    spec = importlib.util.spec_from_file_location('m3_disabled_helpers', Path('tests/worldstream_m1/test_integration.py'))
    helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
    model = helpers.make_model(tmp_path, None)
    (tmp_path/'100.0').write_text('old_answer')
    meta, data = helpers.inputs(100.)
    assert model.simple_test(meta, **data)[0]['text_out'] == []
    assert not model.lm_inputs
