"""One historical visual-region memory, for gated WRVMP research only.

The retained feature is a historical observation, including its old contextual
position/appearance. It is never a claim about unchanged current visual state.
"""
from dataclasses import dataclass
import hashlib
import math
import torch


@dataclass
class VisualMemory:
    anchor_id: str
    semantic_class: str
    camera_id: str
    generating_frame: str
    source_state_version: str
    latest_frame: str
    age: int
    revision: int
    content_sha256: str
    features: torch.Tensor
    grid_indices: tuple
    scope: tuple
    tensor_version: int
    association_qualification: str
    invalidation_reason: str = ''


class VisualMemoryManager:
    """Create/keep/replace/revoke one memory after EVERY current observation.

    No GT enters this interface. Actual qualified IDs come from the existing
    independent semantic policy; region validity is checked separately. Fewer
    than K real grid cells cannot be expanded into K purported observations.
    """
    def __init__(self, tokens=64, max_age_frames=8):
        if tokens not in (64, 128, 256) or max_age_frames < 1:
            raise ValueError('Unsupported finite memory budget')
        self.tokens, self.max_age_frames = tokens, max_age_frames
        self.memory = None
        self.revision = 0
        self.previous = None

    def update(self, context, regions, qualified_ids, qualification_sha256):
        required = ('scene_id', 'frame_id', 'timestamp', 'state_epoch', 'state_version',
                    'model_sha256', 'adapter_sha256', 'memory_version')
        if any(k not in context for k in required) or not qualification_sha256:
            raise ValueError('Memory requires fully versioned current context')
        if (not math.isfinite(context['timestamp']) or type(context['state_version']) is not int
                or context['state_version'] < 0):
            raise ValueError('Invalid current memory time/state')
        scope = tuple(context[k] for k in ('scene_id', 'state_epoch', 'model_sha256',
                      'adapter_sha256', 'memory_version')) + (qualification_sha256, self.tokens)
        available = {r.anchor_id:r for r in regions if r.anchor_id in qualified_ids
            and r.frame_id == context['frame_id'] and len(r.grid_indices) >= self.tokens}
        old = self.memory
        if old is not None and old.features._version != old.tensor_version:
            self.memory = None
            raise ValueError('Historical visual memory tensor mutated')
        reason = 'cold'
        if self.previous is not None:
            if (context['scene_id'], context['frame_id']) == (self.previous['scene_id'], self.previous['frame_id']):
                if context != self.previous:
                    raise ValueError('Same-frame memory context changed')
                if old is not None and old.anchor_id in available and old.scope == scope:
                    return old, 'same_frame'
                self.memory = None
                return None, 'revoked_same_frame'
            if context['scene_id'] != self.previous['scene_id']:
                reason = 'scene_changed'
            elif context['state_version'] < self.previous['state_version']:
                reason = 'state_rollback'
            elif not 0 < context['timestamp']-self.previous['timestamp'] <= 1.:
                reason = 'time_discontinuity'
            elif old is not None and old.scope != scope:
                reason = 'execution_or_memory_version_changed'
            elif old is not None and old.anchor_id not in available:
                reason = 'current_identity_or_region_unavailable'
            elif old is not None and available[old.anchor_id].semantic_class != old.semantic_class:
                reason = 'semantic_class_changed'
            elif old is not None and old.age+1 >= self.max_age_frames:
                reason = 'memory_horizon_expired'
            elif old is not None:
                old.age += 1
                old.latest_frame = context['frame_id']
                self.previous = dict(context)
                return old, 'keep'
        self.previous = dict(context)
        self.memory = None
        if old is not None:
            old.invalidation_reason = reason
        if not available:
            return None, 'revoke:'+reason if old else 'no_memory'
        self.revision += 1
        # Identity orders lifecycle only; it is never passed to a learned module.
        region = available[sorted(available)[0]]
        indices = torch.linspace(0, len(region.grid_indices)-1, self.tokens,
            device=region.features.device).long()
        values = region.features.index_select(0, indices).detach().clone()
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite regional memory')
        content = hashlib.sha256(values.cpu().contiguous().numpy().tobytes()).hexdigest()
        selected = tuple(region.grid_indices[i] for i in indices.cpu().tolist())
        self.memory = VisualMemory(region.anchor_id, region.semantic_class, region.camera_id,
            region.frame_id, region.source_state_version, region.frame_id, 0, self.revision,
            content, values, selected, scope, values._version, qualification_sha256)
        return self.memory, 'create:'+reason


class VisualMemoryProjection(torch.nn.Module):
    """Cost-only projection until all gates license a trained artifact.

    No identity, pose, velocity, signal or class-string embedding is accepted.
    Repeated calls on a retained memory return the SAME projected tensor.
    """
    def __init__(self, input_dim=1024, output_dim=4096):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, output_dim)
        self._key = self._embedding = None

    def forward(self, memory):
        signature = tuple((p.data_ptr(), p._version) for p in self.parameters())
        key = (memory.scope, memory.revision, memory.content_sha256, memory.features.data_ptr(),
               memory.features._version, signature)
        if memory.features._version != memory.tensor_version:
            raise ValueError('Historical memory was mutated')
        if self._key != key:
            self._embedding = self.projection(memory.features).unsqueeze(0)
            self._key = key
        return self._embedding

    def compose(self, current, memory):
        if memory is None:
            return current
        return torch.cat((self(memory).to(current), current), 1)
