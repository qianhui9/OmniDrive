"""Verify pure-forward boundaries and deferred export without a GPU model load."""
from contextlib import contextmanager
import pytest
from tools.component_latency_eval.forward import DeferredExports, forward_inference
from tools.component_latency_eval.timing import Recorder


def test_export_outside_synchronized_forward_and_returned_list_filled(tmp_path):
    active = []
    order = ['input_loaded', 'preprocessed', 'h2d_completed', 'prompt_encoded']
    exports = DeferredExports()
    result = []
    class Meter:
        @contextmanager
        def span(self, name, sync):
            assert name == 'Frame Forward' and sync
            active.append(name)
            order.append('forward_start_fence')
            try:
                yield
            finally:
                order.append('forward_end_fence_and_clock')
                active.pop()
    class Model:
        def __call__(self, **data):
            assert active == ['Frame Forward']
            assert data == dict(return_loss=False, rescale=True, image='prepared')
            order.append('actual_forward')
            def export():
                assert not active
                result.append('decoded answer')
                (tmp_path/'answer.json').write_text('answer')
                order.append('text_decode_and_file_write')
            exports.defer(self, object(), export)
            return result
    outputs = forward_inference(Model(), dict(image='prepared'), Meter(), exports)
    assert outputs is result and outputs == ['decoded answer']
    assert order[-4:] == ['forward_start_fence', 'actual_forward', 'forward_end_fence_and_clock', 'text_decode_and_file_write']
    exports.require_empty()


def test_do_not_defer_native_lm_generation_or_drop_pending_export():
    exports = DeferredExports()
    with pytest.raises(RuntimeError, match='native generation'):
        exports.defer(object(), None, lambda: None)
    exports.defer(object(), object(), lambda: None)
    with pytest.raises(RuntimeError, match='preceding frame'):
        exports.require_empty()
    exports.flush()
    exports.require_empty()


def test_forward_error_propagates_without_export():
    exports = DeferredExports()
    recorder = Recorder('test')
    recorder.frame('seq', 'frame')
    class Model:
        def __call__(self, **data):
            raise ValueError('model failed')
    with pytest.raises(ValueError, match='model failed'):
        forward_inference(Model(), {}, recorder, exports)
    assert recorder.rows[0]['component_name'] == 'Frame Forward'
    assert recorder.rows[0]['execution_status'] == 'FAILED'


def test_shares_use_forward_denominator_and_keep_legacy_scope():
    from tools.component_latency_eval.report import summarize_trial, summarize_trials
    rows = [dict(component_name='Frame Forward', component_wall_ms=10, invocation_count=1),
            dict(component_name='DAVP.Total', cpu_host_ms=2, gpu_device_ms=3, invocation_count=1)]
    stats = summarize_trial(rows, 'r', 1, 5, frame_component='Frame Forward')
    assert not any(r['component_name']=='Frame E2E' for r in stats)
    davp = next(r for r in stats if r['component_name']=='DAVP.Total' and r['metric']=='gpu_device_ms')
    assert davp['e2e_share_percent'] == 30
    assert davp['share_denominator'] == 'Frame Forward'
    final = summarize_trials([davp])[0]
    assert final['share_denominator'] == 'Frame Forward'
