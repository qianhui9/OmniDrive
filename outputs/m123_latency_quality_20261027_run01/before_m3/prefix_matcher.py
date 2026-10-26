"""Exact causal LCP after native image expansion, never semantic similarity."""
import torch
from dataclasses import dataclass


def input_revision(x):
    return (tuple((getattr(x, name).data_ptr(), getattr(x, name)._version,
                   tuple(getattr(x, name).shape), getattr(x, name).dtype, getattr(x, name).device)
                  for name in ('embeddings', 'token_ids', 'position_ids', 'attention_mask', 'visual_mask')),
            x.scope, x.signature, x.source, x.layout_signature, x.visual_group_spans)


@dataclass(frozen=True)
class PrefixProof:
    current_revision: tuple
    cached_revision: tuple
    contiguous_limit: int
    policy_id: str
    risk: float
    risk_limit: float
    rms: float
    max_rms: float
    max_token: float
    max_token_change: float
    max_age_frames: int
    max_approx_depth: int


@torch.no_grad()
def prefix_statistics(previous, current, block_size=32):
    """One metadata transfer; blocks are LLM spans, never EVA windows/IDs."""
    if (previous.scope != current.scope or previous.signature != current.signature
            or previous.layout_signature != current.layout_signature
            or previous.visual_group_spans != current.visual_group_spans
            or previous.embeddings.dtype != current.embeddings.dtype
            or previous.embeddings.device != current.embeddings.device
            or previous.embeddings.shape[2:] != current.embeddings.shape[2:]):
        return 0, 0, []
    n = min(previous.length, current.length)
    if not n:
        return 0, 0, []
    a, b = previous.embeddings[:, :n], current.embeddings[:, :n]
    embed_equal = (a == b).all(-1)[0]
    metadata = torch.ones(n, dtype=torch.bool, device=a.device)
    for name in ('token_ids', 'position_ids', 'attention_mask', 'visual_mask'):
        metadata &= (getattr(previous, name)[:, :n] == getattr(current, name)[:, :n])[0]
    metadata &= previous.attention_mask[0, :n].bool() & current.attention_mask[0, :n].bool()
    finite = torch.isfinite(a).all(-1)[0] & torch.isfinite(b).all(-1)[0]
    legal = metadata & finite & (current.visual_mask[0, :n] | embed_equal)
    indices = torch.arange(n, device=a.device)
    values = [torch.where(embed_equal & metadata & finite, n, indices).min().float(),
              torch.where(legal, n, indices).min().float()]
    spans = []
    for group, start, stop in current.visual_group_spans:
        for lo in range(start, min(stop, n), block_size):
            hi = min(lo + block_size, stop, n)
            old, new = a[0, lo:hi].float(), b[0, lo:hi].float()
            delta = (new - old).square().sum(-1)
            scale = torch.maximum(old.square().sum(-1), new.square().sum(-1)).clamp_min(1e-12)
            values += [(delta.sum() / scale.sum()).sqrt(), (delta / scale).sqrt().max()]
            spans.append((group, lo, hi))
    stats = torch.stack(values).cpu().tolist()
    return int(stats[0]), int(stats[1]), [dict(group=g, start=a, end=b,
        rms=stats[2+2*i], max_token=stats[3+2*i]) for i, (g, a, b) in enumerate(spans)]


def exact_lcp(previous, current):
    if (previous.scope != current.scope or previous.signature != current.signature
            or previous.layout_signature != current.layout_signature):
        return 0
    a, b = previous.embeddings, current.embeddings
    if a.dtype != b.dtype or a.device != b.device or a.shape[2:] != b.shape[2:]:
        return 0
    n = min(previous.length, current.length)
    same = (a[:, :n] == b[:, :n]).all(-1)[0]
    same &= torch.isfinite(a[:, :n]).all(-1)[0] & torch.isfinite(b[:, :n]).all(-1)[0]
    for name in ('token_ids', 'position_ids', 'attention_mask', 'visual_mask'):
        same &= (getattr(previous, name)[:, :n] == getattr(current, name)[:, :n])[0]
    # Do not reuse padding. Arbitrary 4D/noncausal masks never enter ExpandedInput.
    same &= previous.attention_mask[0, :n].bool() & current.attention_mask[0, :n].bool()
    # One scalar transfer, with no dynamic nonzero allocation/metadata sync.
    return int(torch.where(same, n, torch.arange(n,device=a.device)).min()) if n else 0
