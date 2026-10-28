"""Trainable fixed-budget world prefix and current-vision resampling adapter.

This changes the representation and MUST be trained/validated without KV first.
No native visual token is frozen. Every object/map token participates in the
current suffix; ego is retained explicitly. Compression is not a quality proof.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from .stable_world import digest

LAYOUT_VERSION = 'wrcps_stable_dynamic_v1'


@dataclass(frozen=True)
class WorldPrefixLayout:
    stable_tokens: int = 128
    total_tokens: int = 513
    order_contract: str = LAYOUT_VERSION

    def __post_init__(self):
        if (type(self.stable_tokens) is not int or not 0 < self.stable_tokens < 511
                or self.total_tokens != 513 or self.order_contract != LAYOUT_VERSION):
            raise ValueError('Invalid WRCPS visual budget/layout')

    @property
    def groups(self):
        return (('stable_world', self.stable_tokens), ('dynamic_current', self.total-self.stable_tokens))

    @property
    def total(self):
        return self.total_tokens


class VisualResampler(nn.Module):
    def __init__(self, hidden_size, count, rank):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(count, rank)/math.sqrt(rank))
        self.positions = nn.Parameter(torch.randn(256, rank)/math.sqrt(rank))
        self.key = nn.Linear(hidden_size, rank, bias=False)
        self.value = nn.Linear(hidden_size, rank, bias=False)
        self.output = nn.Linear(rank, hidden_size)

    def forward(self, current):
        keys = self.key(current)+self.positions
        attention = torch.softmax(self.queries @ keys.transpose(-1, -2)/math.sqrt(keys.shape[-1]), -1)
        return self.output(attention @ self.value(current))


class WorldPrefixComposer(nn.Module):
    def __init__(self, hidden_size, stable_tokens=128, rank=128, max_state_bytes=8192):
        super().__init__()
        self.layout = WorldPrefixLayout(stable_tokens)
        if type(hidden_size) is not int or hidden_size < 1 or type(rank) is not int or rank < 1 or max_state_bytes < 1:
            raise ValueError('Invalid composer dimensions')
        self.spec = dict(hidden_size=hidden_size, stable_tokens=stable_tokens, rank=rank, max_state_bytes=max_state_bytes)
        self.state_bytes = nn.Embedding(257, rank)
        self.state_positions = nn.Embedding(max_state_bytes, rank)
        self.stable_queries = nn.Parameter(torch.randn(stable_tokens, rank)/math.sqrt(rank))
        self.stable_output = nn.Linear(rank, hidden_size)
        count = 512-stable_tokens
        self.objects = VisualResampler(hidden_size, count//2, rank)
        self.maps = VisualResampler(hidden_size, count-count//2, rank)
        self.adapter_sha256 = ''
        self._cached_key = self._cached_stable = self._cached_revision = None

    def reset(self):
        self._cached_key = self._cached_stable = self._cached_revision = None

    def revision(self):
        return tuple((n, p.data_ptr(), p._version, p.dtype, p.device) for n, p in self.named_parameters())

    def implementation_signature(self):
        from worldstream.code_signature import stable_function_hash
        return tuple(stable_function_hash(f) for f in
            (self.forward,self.encode_stable,self.objects.forward,self.maps.forward))

    def encode_stable(self, anchor):
        data = json.dumps(anchor.facts, separators=(',', ':'), ensure_ascii=False).encode('utf8')
        if len(data) > self.spec['max_state_bytes']:
            raise ValueError('Stable state exceeds declared adapter capacity; no silent truncation')
        device = self.stable_queries.device
        ids = torch.tensor(list(data) or [256], dtype=torch.long, device=device)
        state = self.state_bytes(ids)+self.state_positions(torch.arange(len(ids), device=device))
        weights = torch.softmax(self.stable_queries @ state.T/math.sqrt(state.shape[-1]), -1)
        return self.stable_output(weights @ state).unsqueeze(0)

    def forward(self, current, anchor, *, use_anchor_cache=False):
        if tuple(current.shape) != (1, 513, self.spec['hidden_size']):
            raise ValueError('Composer requires the complete native 256/1/256 current carrier')
        if digest(anchor.facts) != anchor.stable_state_digest:
            raise ValueError('Stable anchor content mismatch')
        revision = self.revision()
        key = (anchor.anchor_id, anchor.stable_state_digest, revision)
        if use_anchor_cache:
            if self.training or torch.is_grad_enabled():
                raise ValueError('Anchor embedding cache is inference-only')
            if self._cached_stable is not None and self._cached_stable._version != self._cached_revision:
                raise ValueError('Cached stable embedding was modified')
            if key != self._cached_key:
                self._cached_stable = self.encode_stable(anchor).detach()
                self._cached_revision = self._cached_stable._version
                self._cached_key = key
            stable = self._cached_stable
        else:
            stable = self.encode_stable(anchor)
        # Explicit ego lane; both learned resamplers see ALL current tokens.
        dynamic = torch.cat((self.objects(current[:, :256]), current[:, 256:257],
                             self.maps(current[:, 257:])), dim=1)
        return torch.cat((stable, dynamic), dim=1)

    @classmethod
    def load_adapter(cls, path, hidden_size, device, dtype):
        raw = Path(path).read_bytes()
        artifact = torch.load(path, map_location='cpu')
        if (artifact.get('layout_version') != LAYOUT_VERSION or artifact.get('trained_steps', 0) < 1
                or not artifact.get('training_scene_ids') or not artifact.get('base_weights_sha256')
                or not artifact.get('data_manifest_sha256')
                or artifact['spec']['hidden_size'] != hidden_size):
            raise ValueError('A trained, provenance-bound WRCPS adapter is required')
        adapter = cls(**artifact['spec'])
        adapter.load_state_dict(artifact['state_dict'], strict=True)
        if not all(torch.isfinite(p).all() for p in adapter.parameters()):
            raise ValueError('Nonfinite adapter weights')
        adapter.to(device=device, dtype=dtype).eval().requires_grad_(False)
        adapter.adapter_sha256 = hashlib.sha256(raw).hexdigest()
        adapter.base_weights_sha256 = artifact['base_weights_sha256']
        adapter.loaded_revision = adapter.revision()
        return adapter
