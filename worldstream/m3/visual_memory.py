"""One historical visual-region memory, for gated WRVMP research only.

The retained feature is a historical observation, including its old contextual
position/appearance. It is never a claim about unchanged current visual state.
"""
from dataclasses import dataclass
import hashlib
import math
from time import perf_counter
import numpy as np
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
    storage_ptr: int = 0
    provenance_binding: tuple = ()

    def binding(self):
        return (self.anchor_id,self.semantic_class,self.camera_id,self.generating_frame,
                self.source_state_version,self.revision,self.content_sha256,self.grid_indices,
                self.scope,self.association_qualification)

    def validate(self):
        if (self.features._version != self.tensor_version
                or self.features.data_ptr() != self.storage_ptr
                or self.binding() != self.provenance_binding):
            raise ValueError('Historical visual memory tensor or provenance mutated')


def select_observations(region, count, cpu_values):
    """Deterministic farthest coverage in normalized position/feature space.

    One bounded D2H on creation, no per-selection GPU scalar synchronizations.
    Each selected row is a distinct observed grid cell, never interpolation.
    """
    positions = np.asarray([(i % region.grid_width, i // region.grid_width)
                            for i in region.grid_indices], dtype=np.float32)
    positions = (positions-positions.min(0))/np.maximum(np.ptp(positions, axis=0), 1)
    features = cpu_values.float().numpy()
    features = features/np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    joint = np.concatenate((positions, features*.5), axis=1)
    # The first cell is closest to the spatial centre; ties follow grid order.
    first = int(np.square(positions-.5).sum(1).argmin())
    selected, distance = [], np.full(len(joint), np.inf)
    for _ in range(count):
        selected.append(first)
        distance = np.minimum(distance, np.square(joint-joint[first]).sum(1))
        distance[selected] = -1
        first = int(distance.argmax())
    return sorted(selected)


class VisualMemoryManager:
    """Create/keep/replace/revoke one memory after EVERY current observation.

    No GT enters this interface. Actual qualified IDs come from the existing
    independent semantic policy; region validity is checked separately. Fewer
    than K real grid cells cannot be expanded into K purported observations.
    """
    def __init__(self, tokens=64, max_age_frames=8):
        if tokens not in (8, 16, 64, 128, 256) or max_age_frames < 1:
            raise ValueError('Unsupported finite memory budget')
        self.tokens, self.max_age_frames = tokens, max_age_frames
        self.memory = None
        self.revision = 0
        self.previous = None
        self.last_timings = {}

    def reset(self):
        self.memory = self.previous = None
        # Revision never rolls back: old identical visual KV cannot resurrect.
        self.revision += 1

    def update(self, context, regions, qualified_ids, qualification_sha256):
        tick = perf_counter()
        self.last_timings = {}
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
            and r.frame_id == context['frame_id'] and len(r.grid_indices) >= self.tokens
            and len(set(r.grid_indices)) == len(r.grid_indices)}
        old = self.memory
        version_tick = perf_counter()
        if old is not None:
            try:
                old.validate()
            except ValueError:
                self.memory = None
                raise
        self.last_timings['version_check_ms'] = (perf_counter()-version_tick)*1000
        reason = 'cold'
        if self.previous is not None:
            if (context['scene_id'], context['frame_id']) == (self.previous['scene_id'], self.previous['frame_id']):
                if context != self.previous:
                    raise ValueError('Same-frame memory context changed')
                if old is not None and old.anchor_id in available and old.scope == scope:
                    if available[old.anchor_id].semantic_class != old.semantic_class:
                        self.memory = None
                        return None, 'revoked_same_frame_class_changed'
                    self.last_timings['memory_keep_ms'] = (perf_counter()-tick)*1000
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
                self.last_timings['memory_keep_ms'] = (perf_counter()-tick)*1000
                return old, 'keep'
        self.previous = dict(context)
        self.memory = None
        if old is not None:
            old.invalidation_reason = reason
            self.last_timings['memory_invalidation_ms'] = (perf_counter()-tick)*1000
        if not available:
            return None, 'revoke:'+reason if old else 'no_memory'
        self.revision += 1
        # Identity orders lifecycle only; it is never passed to a learned module.
        region = available[sorted(available)[0]]
        gather_tick = perf_counter()
        observed = region.read_features()
        if observed.ndim != 2 or len(observed) != len(region.grid_indices):
            raise ValueError('Real feature/grid provenance mismatch')
        self.last_timings['eva_gather_host_ms'] = (perf_counter()-gather_tick)*1000
        sync_tick = perf_counter()
        cpu_values = observed.detach().cpu()
        self.last_timings['gpu_sync_d2h_ms'] = (perf_counter()-sync_tick)*1000
        if not torch.isfinite(cpu_values).all():
            raise ValueError('Nonfinite regional memory')
        selection_tick = perf_counter()
        chosen = select_observations(region, self.tokens, cpu_values)
        indices = torch.tensor(chosen, device=observed.device)
        values = observed.index_select(0, indices).detach()
        self.last_timings['observation_selection_ms'] = (perf_counter()-selection_tick)*1000
        hash_tick = perf_counter()
        content = hashlib.sha256(cpu_values[chosen].contiguous().numpy().tobytes()).hexdigest()
        selected = tuple(region.grid_indices[i] for i in chosen)
        self.last_timings['hash_version_ms'] = (perf_counter()-hash_tick)*1000
        self.memory = VisualMemory(region.anchor_id, region.semantic_class, region.camera_id,
            region.frame_id, region.source_state_version, region.frame_id, 0, self.revision,
            content, values, selected, scope, values._version, qualification_sha256)
        self.memory.storage_ptr = values.data_ptr()
        self.memory.provenance_binding = self.memory.binding()
        self.last_timings['memory_create_ms'] = (perf_counter()-tick)*1000
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
        self._embedding_version = None

    def reset(self):
        self._key = self._embedding = self._embedding_version = None

    def forward(self, memory):
        memory.validate()
        if self.training and torch.is_grad_enabled():
            return self.projection(memory.features).unsqueeze(0)
        signature = tuple((p.data_ptr(), p._version) for p in self.parameters())
        key = (memory.scope, memory.revision, memory.content_sha256, memory.features.data_ptr(),
               memory.features._version, signature)
        if self._embedding is not None and self._embedding._version != self._embedding_version:
            raise ValueError('Historical projected embedding was mutated')
        if self._key != key:
            self._embedding = self.projection(memory.features).unsqueeze(0)
            self._key = key
            self._embedding_version = self._embedding._version
        return self._embedding

    def compose(self, current, memory):
        if memory is None:
            return current
        return torch.cat((self(memory).to(current), current), 1)
