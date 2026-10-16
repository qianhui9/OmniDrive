"""Exact causal LCP after native image expansion, never semantic similarity."""
import torch


def exact_lcp(previous, current):
    if previous.scope != current.scope or previous.signature != current.signature:
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
    mismatches = (~same).nonzero(as_tuple=False)
    return int(mismatches[0, 0]) if len(mismatches) else n
