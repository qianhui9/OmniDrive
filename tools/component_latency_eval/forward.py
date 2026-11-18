"""One synchronized inference on prepared device inputs, with export afterward."""
import torch


class DeferredExports:
    """Delay only the existing post-M3 text export; never delay generation."""
    def __init__(self):
        self.callbacks = []

    def defer(self, model, runtime, callback):
        if runtime is None:
            raise RuntimeError('Cannot defer native generation as output export; full QCSE is required')
        self.callbacks.append(callback)

    def require_empty(self):
        if self.callbacks:
            raise RuntimeError('Unflushed output export from a preceding frame')

    def flush(self):
        callbacks, self.callbacks = self.callbacks, []
        for callback in callbacks:
            callback()


def forward_inference(model, data, recorder, exports):
    """Data loading/preprocessing/H2D/prompt encoding happen before this call.

    The real forward still produces every perception/semantic result and updates
    the same caches. The captured export callback fills the returned text list
    and writes the same answer file after the end fence and wall-clock reading.
    """
    exports.require_empty()
    with torch.no_grad():
        with recorder.span('Frame Forward', sync=True):
            outputs = model(return_loss=False, rescale=True, **data)
    exports.flush()
    return outputs
