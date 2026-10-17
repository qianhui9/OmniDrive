"""Minimal M1/M2/M3 lifecycle and service coordination; no new research module."""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
import math
from time import perf_counter
from .execution import FrameExecutionRecord, FrameIdentity, StageRecorder


@dataclass(frozen=True)
class RuntimeConfig:
    service: str = 'persistent_state'  # or current_output
    model_version: str = ''
    lora_version: str = ''
    max_frame_gap_s: float = 2.
    memory_budget_mb: float = 1024.
    trace_path: str = ''
    profile_stages: bool = False
    cost_aware: bool = True
    cost_table: str = ''
    quality_certificate: str = ''

    def __post_init__(self):
        if self.service not in ('persistent_state', 'current_output'):
            raise ValueError('Unknown service contract')
        if not self.model_version or not self.lora_version:
            raise ValueError('Explicit checkpoint and LoRA content versions required')
        if not math.isfinite(self.max_frame_gap_s) or self.max_frame_gap_s <= 0:
            raise ValueError('Invalid freshness limit')
        if not math.isfinite(self.memory_budget_mb) or self.memory_budget_mb < 0:
            raise ValueError('Invalid memory budget')


class UnifiedRuntime:
    def __init__(self, config):
        self.config = config if isinstance(config, RuntimeConfig) else RuntimeConfig(**config)
        self.previous = None
        self.active = False
        self.last_record = None
        self.input_preparation_wall_ms = None
        from pathlib import Path
        self.costs = json.loads(Path(self.config.cost_table).read_text()) if self.config.cost_table else {}
        self.quality = json.loads(Path(self.config.quality_certificate).read_text()) if self.config.quality_certificate else {}

    def optimization_admission(self, model, data):
        """Cold negative decision avoids all module signatures/cache captures."""
        if not self.config.cost_aware:
            return True, 'explicit_independent_module_experiment'
        if not self.costs.get('records') or not self.quality.get('records'):
            return False, 'no_system_workload_certificate'
        from .m3.visual_state import deployment_key
        key = dict(model_version=self.config.model_version, lora_version=self.config.lora_version,
            execution_key=deployment_key(model.lm_head),
            # This is the fixed native Petr3D generation call. Alternate callers
            # must obtain their own coordinator/certificate, never reuse this key.
            generation_key='native_sampling_temp0.1_topp0.75_beams1_max320_cache',
            input_shape=list(data['img'].shape), question_lengths=[x.numel() for x in data['input_ids'][0]],
            service=self.config.service, strategy='m1_m2_m3',
            cache_state='cold' if self.previous is None else 'warm',
            memory_budget_bytes=int(self.config.memory_budget_mb*1024**2),
            refresh_interval=getattr(getattr(getattr(model, 'm1_manager', None), 'config', None), 'full_refresh_interval', 0),
            frame_gap_s=0. if self.previous is None else float(data['timestamp'].reshape(-1)[0])-self.previous.timestamp)
        for costs in self.costs['records']:
            for quality in self.quality['records']:
                admitted, reason = self.admit(key, costs, quality)
                if admitted:
                    return admitted, reason
        return False, 'no_matching_quality_and_cost_evidence'

    @staticmethod
    def admit(key, costs, quality, overhead_ms=0.):
        """One exact shape/contract key; complete measured cost, no weighted score.

        The modules still independently verify their own cache dependencies.
        A system certificate never proves M1 validity or M2/M3 cache equality.
        """
        if not key or not isinstance(costs, dict) or not isinstance(quality, dict):
            return False, 'missing_workload_evidence'
        required = ('model_version', 'lora_version', 'execution_key', 'generation_key',
                    'input_shape', 'question_lengths', 'service', 'strategy', 'cache_state',
                    'memory_budget_bytes', 'refresh_interval', 'frame_gap_s')
        if any(k not in key or key[k] is None for k in required):
            return False, 'incomplete_workload_key'
        if costs.get('key') != key or quality.get('key') != key:
            return False, 'workload_evidence_mismatch'
        if (quality.get('exact_targets_pass') is not True or not quality.get('heldout_samples')
                or not quality.get('calibration_scenes') or not quality.get('heldout_scenes')
                or set(quality['calibration_scenes']) & set(quality['heldout_scenes'])
                or not quality.get('evidence_sha256') or not quality.get('protocol_sha256')):
            return False, 'uncertified_quality'
        if costs.get('includes_all_incremental_costs') is not True:
            return False, 'incomplete_cost_scope'
        try:
            full, optimized = float(costs['full_wall_ms']), float(costs['optimized_wall_ms'])
            overhead = float(overhead_ms)
            budget = int(key['memory_budget_bytes'])
            memory = int(costs['peak_cache_bytes'])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False, 'invalid_measured_cost'
        if (not all(math.isfinite(x) and x >= 0 for x in (full, optimized, overhead))
                or memory < 0 or budget < 0 or memory > budget):
            return False, 'invalid_cost_or_memory_budget'
        return (True, 'measured_quality_and_net_cost') if optimized+overhead < full else (False, 'no_net_benefit')

    def identity(self, model, metas, data):
        import torch
        meta = metas[0]
        cameras = tuple(meta.get('cam_infos', {}))
        if not cameras:
            from pathlib import Path
            cameras = tuple(Path(f).parent.name for f in meta['filename'])
        timestamp = float(data['timestamp'].reshape(-1)[0])
        if (not math.isfinite(timestamp) or not cameras or len(set(cameras)) != len(cameras)
                or not meta.get('scene_token') or not meta.get('sample_idx')):
            raise ValueError('Invalid stream lifecycle metadata')
        parameters = [(n, p.data_ptr(), p._version, tuple(p.shape), str(p.dtype), str(p.device))
                      for n, p in model.named_parameters()]
        buffers = [(n, b.data_ptr(), b._version) for n, b in model.named_buffers()] if hasattr(model, 'named_buffers') else []
        flags = (model.training, torch.is_autocast_enabled(), torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32)
        # This version witness is runtime-local, never a portable quality key.
        signature = hashlib.sha256(repr((parameters, buffers, flags)).encode()).hexdigest()
        return FrameIdentity(str(meta['scene_token']), str(meta['sample_idx']), timestamp,
            cameras, self.config.model_version, self.config.lora_version, signature)

    @staticmethod
    def invalidate(model, reason):
        for attr in ('m1_manager', 'm2_runtime', 'm3_runtime'):
            obj = getattr(model, attr, None)
            if obj is not None:
                obj.reset()
        model.m3_verified_state = None
        for attr in ('m1_admission_diagnostics', 'm2_diagnostics', 'm3_diagnostics'):
            setattr(model, attr, {})
        for attr in ('pts_bbox_head', 'map_head'):
            head = getattr(model, attr, None)
            if hasattr(head, 'reset_memory'):
                head.reset_memory()
        model.test_flag = False
        model.stream_invalidation_reason = reason

    def lifecycle_reason(self, current):
        previous = self.previous
        if previous is None:
            return 'cold_stream'
        if current.scene_id != previous.scene_id:
            return 'scene_change'
        if current.camera_ids != previous.camera_ids:
            return 'camera_order_change'
        if (current.model_version, current.lora_version, current.execution_signature) != (
                previous.model_version, previous.lora_version, previous.execution_signature):
            return 'model_or_execution_change'
        if current.timestamp <= previous.timestamp:
            return 'timestamp_reversal_or_repeat'
        if current.timestamp-previous.timestamp > self.config.max_frame_gap_s:
            return 'stale_stream'
        return None

    @contextmanager
    def instrument(self, model, recorder):
        """Request-scoped wrappers always restored, including on exceptions."""
        with ExitStack() as stack:
            def wrap(obj, attr, name, dynamic=False, source=None):
                if obj is None or not hasattr(obj, attr):
                    return
                owned = attr in vars(obj)
                original = getattr(obj, attr)
                @wraps(original)
                def call(*args, **kwargs):
                    label = name
                    if dynamic:
                        label = 'llm_prefill' if kwargs.get('past_key_values') is None else 'llm_decode'
                    with recorder.stage(label, source() if source else None):
                        return original(*args, **kwargs)
                setattr(obj, attr, call)
                stack.callback(setattr, obj, attr, original) if owned else stack.callback(delattr, obj, attr)
            wrap(getattr(model, 'm1_manager', None), 'precheck', 'm1_precheck')
            wrap(getattr(model, 'm1_manager', None), 'commit', 'm1_commit')
            if getattr(model, 'm1_manager', None) is not None:
                from .m1 import omnidrive_adapter
                wrap(omnidrive_adapter, 'frame_from_model', 'm1_input_adapter_and_d2h')
                wrap(omnidrive_adapter, 'outputs_from_model', 'm1_output_adapter_and_d2h')
            m2 = getattr(model, 'm2_runtime', None)
            wrap(m2, 'pre_m1_admission', 'm2_cheap_admission')
            wrap(m2, 'forward', 'm2_cache_and_execution')
            m3 = getattr(model, 'm3_runtime', None)
            wrap(m3, 'generate', 'm3_context_cache_and_generation')
            if m3 is not None:
                wrap(m3.cache, 'put', 'm3_kv_capture')
                wrap(m3.cache, 'match', 'm3_prefix_verification')
                wrap(m3.cache, 'branch', 'm3_kv_branch', source=lambda:
                    dict(prefix_lineage=m3.cache.entry.prefix_lineage) if m3.cache.entry else dict(source='missing_cache'))
            wrap(model, 'extract_img_feat', 'vision_service')
            for attr, name in (('img_backbone', 'eva'), ('pts_bbox_head', 'detection_head'), ('map_head', 'map_head')):
                wrap(getattr(model, attr, None), 'forward', name)
            if getattr(model, 'lm_head', None) is not None:
                from .m3.visual_state import base_model
                base = base_model(model.lm_head)
                wrap(base, 'forward', 'llm', dynamic=True)
                wrap(base, 'prepare_inputs_labels_for_multimodal', 'm3_input_preparation')
            yield

    def cache_memory(self, model):
        m1, m2, m3 = (getattr(model, name, None) for name in ('m1_manager', 'm2_runtime', 'm3_runtime'))
        gate = getattr(getattr(m2, 'cheap_admission', None), 'previous', None)
        gate_bytes = gate.numel()*gate.element_size() if gate is not None else 0
        for name in ('rows', 'cols'):
            index = getattr(getattr(m2, 'cheap_admission', None), name, None)
            if index is not None:
                gate_bytes += index.numel()*index.element_size()
        return dict(m1_tensor_bytes=m1.store.tensor_bytes if m1 else 0,
            m2_cache_bytes=(m2.cache.bytes+m2.camera_executor.cache.bytes if m2 else 0),
            m2_gate_bytes=gate_bytes, m3_cache_bytes=m3.cache.nbytes if m3 else 0)

    def execute(self, model, metas, data, forward):
        import torch
        tick = perf_counter()
        preparation = self.input_preparation_wall_ms
        self.input_preparation_wall_ms = None
        record = None
        self.last_record = None
        cuda = data['img'].is_cuda
        try:
            identity = self.identity(model, metas, data)
            ids = data.get('input_ids', [[]])[0]
            record = FrameExecutionRecord(identity, len(ids), self.config.service,
                input_preparation_wall_ms=preparation)
            self.last_record = record
            reason = self.lifecycle_reason(identity)
            if reason:
                self.invalidate(model, reason)
                # Already reset both heads at this actual inference boundary.
                # forward_test must not discard this frame's commit next time.
                model.test_flag = True
                self.previous = None
                record.lifecycle_resets.append(reason)
            if self.config.service == 'persistent_state':
                if getattr(model, 'm1_manager', None) is None:
                    raise ValueError('Persistent-state service requires an M1 manager')
                m2 = getattr(model, 'm2_runtime', None)
                if m2 and m2.config.m1_service != 'maintain':
                    raise ValueError('Persistent service cannot use output-only M1 admission')
            else:
                # This service never exports persistent history, even if some
                # visual/LLM cache happens to hit. Full heads and generation run.
                manager = getattr(model, 'm1_manager', None)
                if manager is not None:
                    manager.reset(identity.scene_id)
            recorder = StageRecorder(record, cuda and self.config.profile_stages)
            self.active = True
            with ExitStack() as stack:
                admitted, reason = self.optimization_admission(model, data)
                if not admitted:
                    record.fallback_reasons.append(reason)
                # Temporary bypass preserves each module's bounded cache but
                # cannot use it as a certificate; lifecycle reset remains shared.
                if not admitted:
                    for attr in ('m2_runtime', 'm3_runtime'):
                        saved = getattr(model, attr, None)
                        if saved is not None:
                            saved.reset()
                        setattr(model, attr, None)
                        stack.callback(setattr, model, attr, saved)
                for attr in ('m2_diagnostics', 'm3_diagnostics'):
                    setattr(model, attr, {})
                if self.config.service == 'current_output':
                    saved = getattr(model, 'm1_manager', None)
                    model.m1_manager = None
                    stack.callback(setattr, model, 'm1_manager', saved)
                if self.config.profile_stages:
                    stack.enter_context(self.instrument(model, recorder))
                with recorder.stage('inference'):
                    output = forward()
            if cuda:
                with recorder.stage('completion_synchronization'):
                    torch.cuda.synchronize(data['img'].device)
            recorder.finish()
            manager = getattr(model, 'm1_manager', None)
            record.m1 = dict(getattr(manager, 'last_commit_diagnostics', {}) or {})
            if self.config.service == 'current_output':
                model.m3_verified_state = None
                record.m1 = dict(service='current_output', history_preserved=False)
            elif manager:
                result = manager.last_result
                record.m1.update(state_epoch=manager.store.generation,
                    valid_anchor_count=len(result.valid_anchor_ids) if result else 0,
                    source_versions=manager.store.feature_versions())
            record.m2 = dict(getattr(model, 'm2_diagnostics', {}) or {})
            record.m3 = dict(getattr(model, 'm3_diagnostics', {}) or {})
            # Module cache/source fields are kept separately. No cross-module
            # validity inference is made from either module's hit.
            for diag in (record.m2, record.m3):
                why = diag.get('fallback_reason') or diag.get('reason')
                if why:
                    record.fallback_reasons.append(why)
            record.memory = self.cache_memory(model)
            record.memory['budget_bytes'] = int(self.config.memory_budget_mb*1024**2)
            record.memory['tracked_cache_bytes'] = sum(v for k, v in record.memory.items() if k != 'budget_bytes')
            if record.memory['tracked_cache_bytes'] > record.memory['budget_bytes']:
                # Evict computation caches first; persistent physical history
                # stays governed by its own bounded M1 store.
                for attr in ('m3_runtime', 'm2_runtime'):
                    obj = getattr(model, attr, None)
                    if obj:
                        obj.reset()
                record.fallback_reasons.append('aggregate_cache_budget_eviction')
                record.memory['after_eviction'] = self.cache_memory(model)
                if sum(record.memory['after_eviction'].values()) > record.memory['budget_bytes']:
                    if manager:
                        manager.reset(identity.scene_id)
                    model.m3_verified_state = None
                    record.fallback_reasons.append('persistent_state_explicitly_invalidated_for_budget')
            record.status = 'completed'
            self.previous = identity
            return output
        except Exception as error:
            # A head may already have mutated StreamPETR memory. Never retry
            # against that partially advanced state or publish partial output.
            self.invalidate(model, 'execution_error')
            self.previous = None
            if record:
                record.status = 'failed'
                record.fallback_reasons.append(type(error).__name__+': '+str(error))
            raise
        finally:
            self.active = False
            if record:
                record.inference_wall_ms = (perf_counter()-tick)*1000
                record.e2e_wall_ms = record.inference_wall_ms+(preparation or 0.)
                if self.config.trace_path:
                    record.append(self.config.trace_path)
