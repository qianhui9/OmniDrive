"""Opt-in request-local sampling for native and target-aware generation.

Transformers 4.31 sample() and our block verifier draw from the default Torch
generators. fork_rng preserves their semantics while isolating request draws.
No target forward, sampling parameter change, or condition-dependent seed.
"""
from contextlib import contextmanager
import hashlib
import json
import torch
from time import perf_counter


def request_seed(context, question_index):
    key=[str(context['scene_id']),str(context['frame_id']),int(question_index),
         int(context.get('repeat',0)),int(context.get('experiment_seed',0))]
    return int.from_bytes(hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).digest()[:8],
                          'big') % (2**63-1)


def rng_digest(device=None):
    value=hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if device is not None and torch.device(device).type=='cuda':
        value.update(torch.cuda.get_rng_state(device).cpu().numpy().tobytes())
    return value.hexdigest()


@contextmanager
def question_rng(context, question_index, device, enabled=False):
    if not enabled:
        yield dict(enabled=False)
        return
    setup_tick=perf_counter()
    device=torch.device(device)
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    audit=bool(context.get('rng_audit',False))
    diagnostic=dict(enabled=True,seed=request_seed(context,question_index),
                    key_fields=['scene_id','frame_id','question_index','repeat','experiment_seed'])
    if audit:diagnostic['outer_before']=rng_digest(device)
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(diagnostic['seed'])
        if devices:
            with torch.cuda.device(devices[0]):torch.cuda.manual_seed(diagnostic['seed'])
        if audit:diagnostic['sampling_before']=rng_digest(device)
        diagnostic['setup_wall_ms']=(perf_counter()-setup_tick)*1000
        try:
            yield diagnostic
        finally:
            restore_tick=perf_counter()
            if audit:diagnostic['sampling_after']=rng_digest(device)
    if audit:diagnostic['outer_after']=rng_digest(device)
    diagnostic['restore_wall_ms']=(perf_counter()-restore_tick)*1000
    diagnostic['management_wall_ms']=diagnostic['setup_wall_ms']+diagnostic['restore_wall_ms']
