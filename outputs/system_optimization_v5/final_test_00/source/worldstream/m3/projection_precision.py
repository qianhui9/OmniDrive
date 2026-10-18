"""Offline FP32 BASE Q/K/V prefill GEMMs; original weights/decode stay intact."""
from contextlib import contextmanager
from functools import wraps
import torch
import torch.nn.functional as F


@contextmanager
def fp32_qkv_prefill(model):
    """A genuinely different arithmetic candidate, not the already-FP32 LoRA.

    Temporary FP32 operands, FP16 outputs and original LoRA add/scaling order.
    This is a serial experiment context, not an approved production strategy.
    """
    if torch.is_autocast_enabled():raise ValueError('FP32 projection probe requires autocast disabled')
    changed=[];counts=dict(prefill_gemms=0,native_decode_calls=0,modules=0)
    try:
        for name,module in model.named_modules():
            if not isinstance(module,torch.nn.Linear):continue
            if not name.startswith('model.layers.'):continue
            if not any(name.endswith('self_attn.'+q+'.base_layer') or name.endswith('self_attn.'+q) for q in ('q_proj','k_proj','v_proj')):continue
            original=module.forward
            @wraps(original)
            def forward(x,*a,module=module,original=original,**kw):
                if x.ndim<3 or x.shape[-2]==1:
                    counts['native_decode_calls']+=1
                    return original(x,*a,**kw)
                if a or kw:raise ValueError('Unexpected Linear arguments')
                counts['prefill_gemms']+=1
                return F.linear(x.float(),module.weight.float(),
                    module.bias.float() if module.bias is not None else None).to(x.dtype)
            changed.append((module,original));module.forward=forward
        if not changed:raise ValueError('No supported native Q/K/V projections found')
        counts['modules']=len(changed)
        yield counts
    finally:
        for module,original in reversed(changed):module.forward=original
