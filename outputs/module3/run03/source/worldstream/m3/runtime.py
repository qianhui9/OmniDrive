"""Request-local generation with scene/camera/model isolation and full fallback."""
from dataclasses import asdict
import math
from time import perf_counter
import torch
import transformers
from .config import M3Config
from .visual_state import base_model, expand_input, execution_signature, deployment_key
from .carrier_selector import select_carriers
from .prefix_kv_cache import PrefixKVCache
from .prefill_executor import PrefillExecutor
from .scheduler import Scheduler


class M3Runtime:
    def __init__(self, config=None):
        self.config = config if isinstance(config, M3Config) else M3Config(**(config or {}))
        self.cache = PrefixKVCache(self.config.memory_budget_mb * 1024**2)
        self.scheduler = Scheduler(self.config)
        self._scope = None
        self._signature = None
        self._timestamp = None

    def reset(self):
        self.cache.reset()
        self._scope = self._signature = self._timestamp = None

    @torch.no_grad()
    def generate(self, lm_head, input_ids_list, vision_carriers, world_state=None,
                 visual_execution=None, context=None, questions=None, layout=None, **generation):
        context = dict(context or {})
        ids_list = list(input_ids_list)
        questions = list(questions) if questions is not None else [''] * len(ids_list)
        if len(questions) != len(ids_list):
            raise ValueError('Question metadata order must match input IDs')
        base = base_model(lm_head)
        supported = (transformers.__version__ == '4.31.0' and not base.training
                     and not getattr(base.config, 'rope_scaling', None)
                     and generation.get('num_beams', 1) == 1
                     and generation.get('num_return_sequences', 1) == 1
                     and generation.get('use_cache', True)
                     and not generation.get('output_attentions') and not generation.get('output_hidden_states')
                     and not generation.get('constraints') and not generation.get('force_words_ids'))
        # Batched masks and alternate generation configurations have distinct
        # contracts. The native generator owns those until explicitly adapted.
        if any(k in generation for k in ('attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds', 'generation_config')):
            supported = False
        if self.config.mode == 'full' or not supported:
            self.reset()
            return [lm_head.generate(inputs=x.unsqueeze(0) if x.ndim == 1 else x,
                    images=vision_carriers, **generation) for x in ids_list], dict(
                        mode='full', reason='explicit_full' if supported else 'unsupported_execution_contract', requests=[])
        if self.config.mode == 'auto' and not self.scheduler.costs.get('approved') and not self.scheduler.quality.get('approved'):
            self.reset()
            return [lm_head.generate(inputs=x.unsqueeze(0) if x.ndim == 1 else x,
                    images=vision_carriers, **generation) for x in ids_list], dict(
                        mode='full', reason='no_calibrated_policy', requests=[])
        cameras = context.get('camera_ids', ())
        timestamp_valid = (context.get('timestamp') is None or
                           math.isfinite(float(context['timestamp'])))
        if (not context.get('scene_id') or not context.get('frame_id') or not cameras
                or len(set(cameras)) != len(cameras) or not timestamp_valid
                or (self.config.cross_frame and context.get('timestamp') is None)):
            self.reset()
            return [lm_head.generate(inputs=x.unsqueeze(0) if x.ndim == 1 else x,
                    images=vision_carriers, **generation) for x in ids_list], dict(mode='full', reason='missing_lifecycle', requests=[])
        tick = perf_counter()
        signature = execution_signature(lm_head, context.get('model_version', ''))
        if self.config.cost_table or self.config.quality_certificate:
            context['execution_key'] = deployment_key(lm_head)
        scope = (str(context['scene_id']), tuple(context['camera_ids']),
                 None if self.config.cross_frame else str(context['frame_id']))
        timestamp = context.get('timestamp')
        if (scope != self._scope or signature != self._signature or
                (timestamp is not None and self._timestamp is not None and timestamp <= self._timestamp)):
            self.cache.reset()
        self._scope, self._signature, self._timestamp = scope, signature, timestamp
        signature_ms = (perf_counter() - tick) * 1000
        answers, rows = [], []
        for index, (ids, question) in enumerate(zip(ids_list, questions)):
            tick = perf_counter()
            carriers = vision_carriers
            selection = dict(approximate=False, reason='full_carrier')
            if layout is not None and vision_carriers is not None and self.config.mode in ('compact', 'combined', 'auto'):
                selected = select_carriers(carriers, layout, question, self.config.selector,
                    self.config.keep_ratio, self.config.random_seed, world_state, context)
                allowed, reason = self.scheduler.compact_allowed(selected.diagnostics, context)
                selection = dict(selected.diagnostics, admitted=allowed, admission_reason=reason)
                if allowed:
                    carriers = carriers.index_select(1, selected.indices)
            selection_ms = (perf_counter() - tick) * 1000
            tick = perf_counter()
            try:
                expanded = expand_input(lm_head, ids, carriers, scope=scope, signature=signature,
                    source=(context['scene_id'], context['frame_id'], timestamp, index))
            except (ValueError, IndexError) as error:
                self.cache.reset()
                answers.append(lm_head.generate(inputs=ids.unsqueeze(0) if ids.ndim == 1 else ids,
                                                images=vision_carriers, **generation))
                rows.append(dict(mode='full', reason=str(error), reused_tokens=0))
                continue
            preparation_ms = (perf_counter() - tick) * 1000
            tick = perf_counter()
            matched_prefix = self.cache.match(expanded) if self.config.mode != 'compact' else 0
            prefix = matched_prefix
            match_ms = (perf_counter() - tick) * 1000
            allowed, reason = self.scheduler.prefix_allowed(expanded, prefix, context, match_ms + signature_ms / max(1, len(ids_list)))
            prefix = prefix if allowed else 0
            # Uncertified auto calls the original generate. Preparing embeddings
            # above is admission overhead and remains inside the request timing.
            capture_allowed = self.scheduler.capture_allowed(expanded, context, len(ids_list))
            if self.config.mode == 'auto' and not prefix and not selection.get('admitted') and not capture_allowed:
                answer = lm_head.generate(inputs=ids.unsqueeze(0) if ids.ndim == 1 else ids,
                                          images=vision_carriers, **generation)
                diag = dict(input_tokens=expanded.length, reused_tokens=0,
                            effective_prefill_tokens=expanded.length, mode='full')
            else:
                answer, diag = PrefillExecutor(self.cache).generate(base, expanded, prefix,
                    capture=self.config.mode != 'compact' and (self.config.mode != 'auto' or capture_allowed),
                    profile=context.get('profile', False), **generation)
                diag['mode'] = ('compact+' if selection.get('admitted') else '') + ('exact' if prefix else 'independent')
            # Cache a true prefill in exact experiment mode on a miss. Auto
            # capture is intentionally disabled without calibrated cost evidence.
            diag.update(selection=selection, selection_cpu_ms=selection_ms,
                preparation_cpu_ms=preparation_ms, match_cpu_ms=match_ms,
                matched_prefix=matched_prefix, prefix_admission=reason,
                visual_tokens=int(expanded.visual_mask.sum()), signature_cpu_ms=signature_ms / max(1, len(ids_list)))
            answers.append(answer)
            rows.append(diag)
        return answers, dict(mode=self.config.mode, requests=rows, cache_bytes=self.cache.nbytes,
                             config=asdict(self.config), m2_required=False)
