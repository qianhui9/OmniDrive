"""One budgeted immutable legacy KV entry; query branches receive owned clones."""
from dataclasses import dataclass
import torch
from .prefix_matcher import exact_lcp


def kv_bytes(kv):
    return sum(t.numel() * t.element_size() for layer in kv for t in layer)


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
        self.revision = None

    def reset(self):
        self.entry = None
        self.revision = None

    @staticmethod
    def _revision(entry):
        tensors=[getattr(entry.expanded,n) for n in
            ('embeddings','token_ids','position_ids','attention_mask','visual_mask')]
        tensors += [t for layer in entry.kv for t in layer]+[entry.last_logits]
        return (tuple((t.data_ptr(),t._version,tuple(t.shape),tuple(t.stride()),t.dtype,t.device) for t in tensors),
                repr((entry.expanded.scope,entry.expanded.signature,entry.expanded.source,entry.prefix_lineage,
                      entry.expanded.layout_signature,entry.expanded.visual_group_spans)))

    @property
    def nbytes(self):
        return self.entry.nbytes if self.entry else 0

    def match(self, expanded):
        if self.entry is not None and self.revision!=self._revision(self.entry):self.reset()
        return exact_lcp(self.entry.expanded, expanded) if self.entry else 0

    def put(self, expanded, kv, logits, prefix_lineage=()):
        size = expanded.nbytes + kv_bytes(kv) + logits[:, -1:].numel() * logits.element_size()
        # Budget includes the owned stored entry. Clones used by a live decoder
        # are separately reported as request workspace, not hidden cache memory.
        if size > self.budget_bytes:
            self.reset()
            return False
        try:
            entry = CacheEntry(expanded.prefix(expanded.length, clone=True), clone_kv(kv, expanded.length),
                              logits[:, -1:].detach().clone(), expanded.source, size, tuple(prefix_lineage))
            revision=self._revision(entry)
        except torch.cuda.OutOfMemoryError:
            self.reset()
            return False
        self.entry,self.revision=entry,revision
        return True

    def branch(self, length):
        if self.entry is None or not 0 < length <= self.entry.expanded.length:
            raise ValueError('Invalid cache branch')
        return clone_kv(self.entry.kv, length)
