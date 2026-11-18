"""Synchronized model-call timing, independent of data preparation and export.

No torch import at module scope: the clock's synchronization function is injected
so its interval accounting can be tested without a GPU.
"""
from collections import defaultdict
from contextlib import contextmanager
import math
import time


class ForwardClock:
    def __init__(self, synchronize, clock=time.perf_counter):
        self.synchronize = synchronize
        self.clock = clock
        self.active = False
        self.depth = 0
        self.last = None

    def start(self):
        if self.active:
            raise RuntimeError('Nested root forward calls are not supported')
        # Finish transfers and preprocessing kernels BEFORE the start timestamp.
        self.synchronize()
        self.excluded = defaultdict(float)
        self.exclusion_calls = defaultdict(int)
        self.depth = 0
        self.last = None
        self.tick = self.clock()
        self.active = True

    @contextmanager
    def exclude(self, label):
        if not self.active:
            yield
            return
        outermost = self.depth == 0
        if outermost:
            # Charge pending model kernels to forward, not to output handling.
            self.synchronize()
            tick = self.clock()
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            if outermost:
                self.synchronize()
                self.excluded[label] += (self.clock() - tick) * 1000
                self.exclusion_calls[label] += 1

    def stop(self):
        if not self.active or self.depth:
            raise RuntimeError('No complete root forward interval')
        self.synchronize()
        raw_ms = (self.clock() - self.tick) * 1000
        self.active = False
        excluded_ms = sum(self.excluded.values())
        forward_ms = raw_ms - excluded_ms
        if not all(math.isfinite(v) for v in (raw_ms, excluded_ms, forward_ms)) or forward_ms <= 0:
            raise ValueError('Invalid measured forward time')
        self.last = dict(forward_ms=forward_ms, forward_raw_ms=raw_ms,
            forward_excluded_ms=excluded_ms,
            forward_excluded_by_label_ms=dict(self.excluded),
            forward_exclusion_calls=dict(self.exclusion_calls))
        return self.last

    def abort(self):
        self.active = False
        self.depth = 0
        self.last = None


@contextmanager
def observe_forward(model, timer):
    """Observe one root model call and only its final answer export operations.

    The Petr3D module's local `open` binding is scoped to this worker and restored
    afterwards. Actual answers are still decoded and written without alteration.
    Token decoding inside QCSE algorithms is retained in forward latency; only
    direct calls by Petr3D.simple_test_pts for final answer export are excluded.
    """
    import builtins
    from functools import wraps
    import inspect
    import sys

    module = sys.modules[model.__class__.__module__]
    had_open = 'open' in vars(module)
    original_open = getattr(module, 'open', builtins.open)
    original_decode = model.tokenizer.batch_decode
    had_decode = 'batch_decode' in vars(model.tokenizer)
    saved_decode = vars(model.tokenizer).get('batch_decode')

    @contextmanager
    def answer_file(*args, **kwargs):
        with timer.exclude('answer_file_write'):
            with original_open(*args, **kwargs) as handle:
                yield handle

    @wraps(original_open)
    def local_open(*args, **kwargs):
        path = args[0] if args else kwargs.get('file')
        mode = args[1] if len(args) > 1 else kwargs.get('mode', 'r')
        if timer.active and mode == 'w' and str(path).startswith(str(model.save_path)):
            return answer_file(*args, **kwargs)
        return original_open(*args, **kwargs)

    @wraps(original_decode)
    def batch_decode(*args, **kwargs):
        caller = inspect.currentframe().f_back
        try:
            export = (caller.f_code.co_name == 'simple_test_pts' and
                      caller.f_globals is vars(module))
        finally:
            del caller
        if export:
            with timer.exclude('answer_text_decode'):
                return original_decode(*args, **kwargs)
        return original_decode(*args, **kwargs)

    def start(module, args):
        timer.start()

    def stop(module, args, output):
        timer.stop()

    handles = [model.register_forward_pre_hook(start), model.register_forward_hook(stop)]
    module.open = local_open
    model.tokenizer.batch_decode = batch_decode
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        timer.abort()
        if had_open:
            module.open = original_open
        else:
            del module.open
        if had_decode:
            model.tokenizer.batch_decode = saved_decode
        else:
            del model.tokenizer.batch_decode
