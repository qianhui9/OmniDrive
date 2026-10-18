"""Reversible, offline-only precision probes. No parameter is edited or merged."""
from contextlib import contextmanager
from functools import wraps
import torch
import torch.nn.functional as F


@contextmanager
def precision_probe(model, policy='native'):
    """Compare against the unchanged native baseline; none is an equality proof.

    This scopes module forward overrides to a serial research invocation. It is
    deliberately not an Auto strategy or safe to use concurrently on one model.
    """
    if policy not in ('native','fp32_lora','no_fp16_reduction','deterministic'):
        raise ValueError('Unknown numerical probe')
    changed=[]
    reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    deterministic=torch.are_deterministic_algorithms_enabled()
    warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        if policy=='no_fp16_reduction':
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        if policy=='deterministic':
            torch.use_deterministic_algorithms(True)
        if policy=='fp32_lora':
            for name,module in model.named_modules():
                if '.lora_A.' not in name and '.lora_B.' not in name:continue
                if not isinstance(module,torch.nn.Linear):raise ValueError('Unsupported LoRA projection')
                original=module.forward
                @wraps(original)
                def forward(x, module=module):
                    # Each LoRA projection retains its original FP16 output and
                    # scaling/add order; only its GEMM accumulates in FP32.
                    return F.linear(x.float(),module.weight.float(),
                        module.bias.float() if module.bias is not None else None).to(x.dtype)
                changed.append((module,original));module.forward=forward
        yield
    finally:
        for module,original in reversed(changed):module.forward=original
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=reduction
        torch.use_deterministic_algorithms(deterministic,warn_only=warn_only)
