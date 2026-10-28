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

LAYOUT_VERSION = 'wrcps_semantic_core_v2'
SF_LAYOUT_VERSION = 'sf_wrcps_slot_preserving_v1'


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
        # Preserve native embedding dimensions through attention pooling;
        # learn a low-rank residual instead of forcing all current evidence
        # through a rank-128 value bottleneck before the frozen LLM.
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self, current):
        keys = self.key(current)+self.positions
        attention = torch.softmax(self.queries @ keys.transpose(-1, -2)/math.sqrt(keys.shape[-1]), -1)
        return attention @ current + self.output(attention @ self.value(current))


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
        # Opaque tracking IDs control membership/lifetime, never adapter input.
        # Sorting semantic fields makes arbitrary ID renaming nonsemantic.
        semantic_content=sorted((kind,label) for _,kind,label in anchor.facts)
        data = json.dumps(semantic_content, separators=(',', ':'), ensure_ascii=False).encode('utf8')
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
        if (artifact.get('layout_version') not in (LAYOUT_VERSION,SF_LAYOUT_VERSION) or artifact.get('trained_steps', 0) < 1
                or not artifact.get('training_scene_ids') or not artifact.get('base_weights_sha256')
                or not artifact.get('data_manifest_sha256')
                or artifact['spec']['hidden_size'] != hidden_size):
            raise ValueError('A trained, provenance-bound WRCPS adapter is required')
        constructor=SpatiallyFaithfulComposer if artifact['layout_version']==SF_LAYOUT_VERSION else cls
        adapter = constructor(**artifact['spec'])
        adapter.load_state_dict(artifact['state_dict'], strict=True)
        if not all(torch.isfinite(p).all() for p in adapter.parameters()):
            raise ValueError('Nonfinite adapter weights')
        adapter.to(device=device, dtype=dtype).eval().requires_grad_(False)
        adapter.adapter_sha256 = hashlib.sha256(raw).hexdigest()
        adapter.base_weights_sha256 = artifact['base_weights_sha256']
        adapter.semantic_qualification_sha256=artifact.get('semantic_qualification_sha256','')
        adapter.loaded_revision = adapter.revision()
        return adapter


@dataclass(frozen=True)
class SpatialPrefixLayout:
    stable_tokens: int = 64
    order_contract: str = SF_LAYOUT_VERSION

    def __post_init__(self):
        if self.stable_tokens not in (32,64) or self.order_contract!=SF_LAYOUT_VERSION:
            raise ValueError('SF-WRCPS permits only the declared K=64/32 layouts')

    @property
    def groups(self):
        count=256-self.stable_tokens//2
        return (('stable_world',self.stable_tokens),('object',count),('ego',1),('map',count))

    @property
    def total(self):return 513


class SlotPreservingCompression(nn.Module):
    """Merge only a bounded subset using current feature dependence.

    Extra VLM queries are learned latent slots, NOT a physical raster. Even/odd
    is only a bipartite matching partition: edges use cosine similarity, never
    index distance as a spatial proxy. Unmerged native slots remain bit-exact,
    kept slots retain native order, and every removed slot contributes to its
    selected destination. This is a constrained approximation, not a claim of
    lossless spatial reconstruction.
    """
    def __init__(self,hidden_size,count,rank):
        super().__init__()
        self.removed=256-count
        self.merge_logit=nn.Parameter(torch.zeros(()))
        self.value=nn.Linear(hidden_size,rank,bias=False)
        self.output=nn.Linear(rank,hidden_size,bias=False)
        nn.init.zeros_(self.output.weight)

    def plan(self,current):
        with torch.no_grad():
            feature=torch.nn.functional.normalize(current[0].float(),dim=-1)
            scores=feature[::2]@feature[1::2].T
            similarity,destination=scores.max(-1)
            selected=torch.argsort(similarity,descending=True,stable=True)[:self.removed]
            source=2*selected;destination=2*destination[selected]+1
            keep=torch.ones(256,device=current.device,dtype=torch.bool);keep[source]=False
            retained=torch.arange(256,device=current.device)[keep]
            return source,destination,retained

    def forward(self,current):
        source,destination,retained=self.plan(current)
        contributions=torch.zeros_like(current).index_add(1,destination,current[:,source])
        counts=torch.zeros(256,device=current.device,dtype=current.dtype)
        counts.index_add_(0,destination,torch.ones_like(destination,dtype=current.dtype))
        mean=contributions/counts.clamp_min(1)[None,:,None]
        changed=(counts>0)[None,:,None]
        residual=self.output(self.value(mean-current))
        merged=current+changed*(torch.sigmoid(self.merge_logit)*(mean-current)+residual)
        return merged[:,retained]


class SpatiallyFaithfulComposer(WorldPrefixComposer):
    """Adaptive input layout within the SAME M3 runtime/executor."""
    def __init__(self,hidden_size,stable_tokens=64,rank=128,max_state_bytes=8192):
        super().__init__(hidden_size,stable_tokens,rank,max_state_bytes)
        self.layout=SpatialPrefixLayout(stable_tokens)
        count=256-stable_tokens//2
        self.objects=SlotPreservingCompression(hidden_size,count,rank)
        self.maps=SlotPreservingCompression(hidden_size,count,rank)

    def forward(self,current,anchor,*,use_anchor_cache=False):
        if tuple(current.shape)!=(1,513,self.spec['hidden_size']):
            raise ValueError('Composer requires the complete native current carrier')
        if digest(anchor.facts)!=anchor.stable_state_digest:
            raise ValueError('Stable anchor content mismatch')
        if not anchor.stable_state_valid:
            if anchor.facts:raise ValueError('Invalid core must not publish facts')
            return current  # identity, including dtype, layout and storage
        if not anchor.facts:raise ValueError('A valid semantic core requires facts')
        return super().forward(current,anchor,use_anchor_cache=use_anchor_cache)

    def implementation_signature(self):
        from worldstream.code_signature import stable_function_hash
        return super().implementation_signature()+(stable_function_hash(self.objects.plan),
            stable_function_hash(WorldPrefixComposer.forward))
