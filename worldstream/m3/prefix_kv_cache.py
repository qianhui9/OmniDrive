"""One budgeted immutable legacy KV entry; query branches receive owned clones."""
from dataclasses import dataclass
import torch
from .prefix_matcher import exact_lcp


def kv_bytes(kv):
    return sum(t.numel() * t.element_size() for layer in kv for t in layer)


def minimum_native_prefix_bytes(model,expanded,length):
    """Rejection-only lower bound for inspected floating legacy LLaMA KV.

    Native FP16/BF16/FP32 keys/values use at least two bytes per element.
    Last logits are omitted, so this never approves a cache allocation.
    """
    cfg=model.config
    heads=getattr(cfg,'num_key_value_heads',cfg.num_attention_heads)
    return expanded.prefix(length).nbytes+length*cfg.num_hidden_layers*heads*(cfg.hidden_size//cfg.num_attention_heads)*2*2


def clone_kv(kv, length):
    if not isinstance(kv, (tuple, list)) or any(len(layer) != 2 for layer in kv):
        raise ValueError('Only Transformers 4.31 legacy Llama K/V tuples are supported')
    if any(t.ndim != 4 or t.shape[-2] < length for layer in kv for t in layer):
        raise ValueError('Invalid legacy cache shape')
    return tuple(tuple(t[..., :length, :].detach().clone() for t in layer) for layer in kv)


@dataclass
class CacheEntry:
    expanded: object
    kv: tuple
    last_logits: torch.Tensor
    computed_source: tuple
    nbytes: int
    prefix_lineage: tuple = ()


class PrefixKVCache:
    def __init__(self, budget_bytes):
        self.budget_bytes = int(budget_bytes)
        self.entry = None

    def reset(self):
        self.entry = None

    @property
    def nbytes(self):
        return self.entry.nbytes if self.entry else 0

    def match(self, expanded):
        return exact_lcp(self.entry.expanded, expanded) if self.entry else 0

    def put(self, expanded, kv, logits, prefix_lineage=()):
        size = expanded.nbytes + kv_bytes(kv) + logits[:, -1:].numel() * logits.element_size()
        # Budget includes the owned stored entry. Clones used by a live decoder
        # are separately reported as request workspace, not hidden cache memory.
        if size > self.budget_bytes:
            self.reset()
            return False
        self.reset()
        self.entry = CacheEntry(expanded.prefix(expanded.length, clone=True), clone_kv(kv, expanded.length),
                                logits[:, -1:].detach().clone(), expanded.source, size, tuple(prefix_lineage))
        return True

    def branch(self, length, read_only=False):
        if self.entry is None or not 0 < length <= self.entry.expanded.length:
            raise ValueError('Invalid cache branch')
        if read_only:
            # Internal loan, only after the executor certifies the installed
            # out-of-place torch.cat attention implementation. Never public COW.
            return tuple(tuple(t[..., :length, :].detach() for t in layer) for layer in self.entry.kv)
        return clone_kv(self.entry.kv, length)
