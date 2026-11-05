"""Frozen R0: current committed facts, full-current prefill and exact historical K8 proposals."""
from dataclasses import asdict, replace
import hashlib, json, math
from time import perf_counter
import torch, transformers
from .config import M3Config
from .visual_state import base_model, expand_input, execution_signature
from .prefill_executor import PrefillExecutor
from .answer_memory import AnswerMemory

def language_request_completed(runtime, row, current_world=None):
    """Explicit selective completion; default deployment still requires LLM."""
    if row.get('executor') == 'PrefillExecutor':
        return bool(row.get('prefill_completed') and row.get('decode_completed'))
    p = row.get('fast_answer') or {}
    return bool(runtime is not None and runtime.config.selective_answer_mode == 'online' and runtime.fast_answer_policy and (current_world is not None) and (row.get('executor') == 'QuestionReferencedFactReuse') and row.get('llm_skipped') and (row.get('llm_forward_calls') == 0) and (row.get('layout_preparation_ms') == 0.0) and (not row.get('prefill_completed')) and (not row.get('decode_completed')) and p.get('supported') and p.get('admitted') and p.get('facts') and (tuple(p.get('current_commit', ())) == (current_world.epoch, current_world.version, current_world.frame_id)) and all((f.get('evidence_origin') == 'actual_perception' and f.get('source_frame') == current_world.frame_id and (f.get('source_timestamp') == current_world.timestamp) and (f.get('state_version') == current_world.version) for f in p['facts'])))

class M3Runtime:

    def __init__(self, config=None):
        from pathlib import Path
        self.config = config if isinstance(config, M3Config) else M3Config(**config or {})
        self.executor = PrefillExecutor()
        self.answer_memory = AnswerMemory()
        self.fast_answer_policy = json.loads(Path(self.config.selective_answer_policy).read_text())
        from worldstream.r0 import validate_fact_policy
        validate_fact_policy(self.fast_answer_policy)
        self._scope = self._signature = self._timestamp = None
        self._frame_id = None
        self._frame_index = 0
        self.last_diagnostics = {}
        self.semantic_qualifier = None
        self.last_composed_carriers = None
        self._decode_model_revision = self._decode_model_key = None
        self._decode_tokenizer_revision = self._decode_tokenizer_key = None
        self._decode_previous_world = None

    def reset(self):
        self.answer_memory.reset()
        self._scope = self._signature = self._timestamp = None
        self._frame_id = None
        self._frame_index = 0
        self._decode_previous_world = None

    @torch.no_grad()
    def generate(self, lm_head, input_ids_list, vision_carriers, context=None, questions=None, layout=None, verified_state=None, tokenizer=None, m2_diagnostics=None, stable_world_state=None, visual_memory_inputs=None, committed_world_state=None, **generation):
        runtime_tick = perf_counter()
        rows = []
        self.last_diagnostics = dict(runtime_entered=True, execution_completed=False, requests=rows)
        context = dict(context or {})
        context.setdefault('model_version', self.config.model_version)
        if not context['model_version']:
            context['model_version'] = context.get('weights_sha256', '')
        ids_list = list(input_ids_list)
        questions = list(questions) if questions is not None else [''] * len(ids_list)
        if len(questions) != len(ids_list):
            raise ValueError('Question order must match input IDs')
        base = base_model(lm_head)
        self.validate_contract(base, generation)
        if layout is not None and vision_carriers is not None and (vision_carriers.shape[1] != layout.total):
            raise ValueError('M3 requires the complete declared current visual layout')
        cameras = context.get('camera_ids', ())
        timestamp = context.get('timestamp')
        if not context.get('scene_id') or not context.get('frame_id') or (not cameras) or (len(set(cameras)) != len(cameras)) or (type(timestamp) not in (int, float)) or (not math.isfinite(timestamp)):
            raise ValueError('M3 requires a bound scene/frame/time/camera lifecycle')
        return self._decode_current_frame(lm_head, base, ids_list, questions, vision_carriers, layout, context, tokenizer, committed_world_state, generation, runtime_tick)

    @staticmethod
    def validate_contract(base, generation):
        from transformers.models.llama.modeling_llama import LlamaAttention
        effective = lambda key, default=None: generation.get(key, getattr(base.generation_config, key, default))
        layers = getattr(getattr(base, 'model', None), 'layers', ())
        supported = transformers.__version__ == '4.31.0' and len(layers) and all((type(layer.self_attn) is LlamaAttention for layer in layers)) and (not any((m.training for m in base.modules()))) and (not getattr(base.config, 'rope_scaling', None)) and (effective('num_beams', 1) == 1) and (effective('num_return_sequences', 1) == 1) and (effective('use_cache', True) is True) and (not effective('output_attentions')) and (not effective('output_hidden_states')) and (not effective('constraints')) and (not effective('force_words_ids')) and (effective('penalty_alpha') is None)
        forbidden = {'attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds', 'generation_config'}
        if not supported or forbidden & generation.keys():
            raise ValueError('Unsupported M3 model or generation contract; PrefillExecutor is required')

    def _decode_binding(self, lm_head, base, context, tokenizer, generation):
        """Cache expensive metadata, while checking real mutable revisions."""
        revision = (context['model_version'], repr(base.config.to_dict()), tuple(((id(p), p._version, p.data_ptr(), p.dtype, p.device) for p in base.parameters())), tuple(((id(b), b._version, b.data_ptr()) for (n, b) in base.named_buffers() if not n.endswith(('cos_cached', 'sin_cached')))), tuple(((id(m), repr(getattr(m, 'active_adapters', None)), repr(getattr(m, 'disable_adapters', None)), repr(getattr(m, 'merged_adapters', None))) for m in base.modules() if hasattr(m, 'lora_A'))), tuple((id(getattr(m.forward, '__func__', m.forward)) for m in base.modules())), torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        model_hit = revision == self._decode_model_revision
        if not model_hit:
            self._decode_model_key = hashlib.sha256(repr(execution_signature(lm_head, context['model_version'])).encode()).hexdigest()
            self._decode_model_revision = revision
        if tokenizer is None:
            vocab = None
        elif hasattr(tokenizer, 'sp_model'):
            vocab = hashlib.sha256(tokenizer.sp_model.serialized_model_proto()).hexdigest()
            vocab = (vocab, repr(tokenizer.get_added_vocab()))
        elif hasattr(tokenizer, 'backend_tokenizer'):
            vocab = tokenizer.backend_tokenizer.to_str()
        else:
            vocab = sorted(tokenizer.get_vocab().items())
        tokenizer_revision = (id(tokenizer), repr(getattr(tokenizer, 'special_tokens_map', None)), repr(vocab))
        tokenizer_hit = tokenizer_revision == self._decode_tokenizer_revision
        if not tokenizer_hit:
            self._decode_tokenizer_key = hashlib.sha256(repr(tokenizer_revision).encode()).hexdigest()
            self._decode_tokenizer_revision = tokenizer_revision
        genkey = hashlib.sha256(json.dumps(dict(defaults=base.generation_config.to_dict(), overrides=generation), sort_keys=True, default=repr).encode()).hexdigest()
        return ((self._decode_model_key, self._decode_tokenizer_key, genkey), model_hit, tokenizer_hit)

    def _decode_current_frame(self, lm_head, base, ids_list, questions, carriers, layout, context, tokenizer, current_world, generation, runtime_tick):
        """History and world-guided proposals share the same current-context verifier.

        No prefix planner, inherited visual prefix, or persistent prefill K/V.
        This branch begins before any legacy prefix-layout/signature work.
        """
        guided = self.config.decode_strategy == 'world_guided'
        if current_world is not None:
            current_world.validate(context)
        tick = perf_counter()
        (binding, model_hit, tokenizer_hit) = self._decode_binding(lm_head, base, context, tokenizer, generation)
        metadata_ms = (perf_counter() - tick) * 1000
        scope = (context['scene_id'], tuple(context['camera_ids']), context.get('state_epoch'))
        same_frame = context['frame_id'] == self._frame_id and context['timestamp'] == self._timestamp
        invalid = scope != self._scope or binding[0] != self._signature or (self._timestamp is not None and context['timestamp'] <= self._timestamp and (not same_frame))
        if invalid:
            self.answer_memory.reset()
            self._frame_index = 0
            self._decode_previous_world = None
        elif not same_frame:
            self._frame_index += 1
        (self._scope, self._signature) = (scope, binding[0])
        (self._frame_id, self._timestamp) = (context['frame_id'], context['timestamp'])
        context['frame_index'] = self._frame_index
        context['current_state_version'] = current_world.version if current_world else context.get('current_state_version')
        answers = []
        rows = self.last_diagnostics['requests']
        for (index, (ids, question)) in enumerate(zip(ids_list, questions)):
            question_tick = perf_counter()
            tick = perf_counter()
            prompt_ids = tuple(ids.reshape(-1).tolist())
            self.last_diagnostics['active_request_index'] = index
            proposal = None
            if self.config.selective_answer_mode != 'off':
                from .answer_validity import propose_fact_answer
                proposal = propose_fact_answer(question, current_world, context, self.answer_memory, binding, self.fast_answer_policy, self._decode_previous_world, target_binding_enabled=self.config.target_binding_enabled, temporal_risk_enabled=self.config.temporal_risk_enabled, target_completeness_enabled=self.config.target_completeness_enabled)
                if proposal['admitted'] and self.config.selective_answer_mode == 'online' and (tokenizer is not None) and (not generation.get('output_scores', False)):
                    construct_tick = perf_counter()
                    token_ids = list(tokenizer.encode(proposal['answer_text'], add_special_tokens=False))
                    eos = generation.get('eos_token_id', base.generation_config.eos_token_id)
                    if isinstance(eos, (list, tuple)):
                        eos = eos[0] if eos else None
                    if eos is not None:
                        token_ids.append(eos)
                    bos = base.generation_config.bos_token_id
                    answer = ids.new_tensor([[bos if bos is not None else 1] + token_ids])
                    if generation.get('return_dict_in_generate', False):
                        from transformers.generation.utils import SampleDecoderOnlyOutput, GreedySearchDecoderOnlyOutput
                        answer = (SampleDecoderOnlyOutput if generation.get('do_sample', False) else GreedySearchDecoderOnlyOutput)(sequences=answer)
                    proposal['fast_answer_construction_wall_ms'] += (perf_counter() - construct_tick) * 1000
                    diag = dict(request_index=index, executor='QuestionReferencedFactReuse', mode='fast_answer', decode_strategy=self.config.decode_strategy, llm_skipped=True, llm_forward_calls=0, fast_answer=proposal, answer_source=proposal['answer_source'], output_token_ids=token_ids, input_tokens=0, visual_tokens=0, current_visual_complete=True, layout_preparation_ms=0.0, prefill_completed=False, decode_completed=False, reused_tokens=0, full_prefill=False, prefill_including_cache_gpu_ms=0.0, prefill_including_cache_cpu_enqueue_ms=0.0, prefill_including_cache_wall_ms=0.0, decode_loop_gpu_span_ms=0.0, decode_forward_cuda_ms=0.0, decode_forward_host_enqueue_ms=0.0, decode_forward_calls=0, draft_status='not_run_fast_answer', prefix_planner_executed=False, persistent_prefill_kv=False, historical_text_consumed=False, metadata_wall_ms=metadata_ms / max(1, len(ids_list)), end_to_end_ms=(perf_counter() - question_tick) * 1000)
                    answers.append(answer)
                    rows.append(diag)
                    continue
            selection = self.answer_memory.retrieve(question, prompt_ids, context, binding, current_world, guided and (not self.config.target_aware_exact_only))
            retrieval_ms = (perf_counter() - tick) * 1000
            retrieval_only_ms = retrieval_ms - selection.world_diff_ms
            selection = self.answer_memory.construct(selection, question, prompt_ids, context, binding, None if self.config.target_aware_exact_only else current_world, tokenizer)
            draft = selection.record.tokens if selection.record is not None else None
            draft = None
            k = min(self.config.draft_length, selection.length) if draft else self.config.draft_length
            if draft and selection.length < 8:
                draft = draft[:selection.length]
            tick = perf_counter()
            expanded = expand_input(lm_head, ids, carriers, scope=scope, signature=(binding[0],), source=(context['scene_id'], context['frame_id'], context['timestamp'], index), carrier_layout=layout)
            preparation_ms = (perf_counter() - tick) * 1000
            self.last_diagnostics['active_request_index'] = index
            from .request_rng import question_rng
            with question_rng(context, index, ids.device, self.config.question_rng_enabled) as rng:
                (answer, diag) = self.executor.generate(base, expanded, capture=False, profile=context.get('profile', False), draft_tokens=draft, draft_candidates=selection.candidates if guided else None, draft_length=k, experimental_sampling=self.config.experimental_speculative_sampling, **generation)
            diag['question_rng'] = rng
            tick = perf_counter()
            tokens = answer.sequences if hasattr(answer, 'sequences') else answer
            token_ids = tokens[0, 1:].tolist()
            answer_start = tokenizer.decode(token_ids if guided else token_ids[:8], skip_special_tokens=True) if tokenizer is not None else ''
            candidate_diag = [c.metadata() for c in selection.candidates]
            selected = diag.get('selected_candidate_index')
            c = selection.candidates[selected] if selected is not None else None
            selection = replace(selection, record=c.record if c else None, length=c.block_length if c else 0, match_type=('exact' if c.kind == 'exact' else 'semantic') if c else 'none', delta=c.delta if c else selection.delta)
            diag['output_token_ids'] = token_ids
            diag['draft_selection_wall_ms'] = retrieval_ms + selection.construction_ms + selection.tokenizer_ms
            self.answer_memory.observe_acceptance(selection, diag, self._frame_index)
            self.answer_memory.remember(question, prompt_ids, token_ids, context, binding, current_world, answer_start, extract_facts=guided)
            management_ms = (perf_counter() - tick) * 1000
            diag.update(request_index=index, executor='PrefillExecutor', mode='full_prefill', decode_strategy=self.config.decode_strategy, fast_answer=proposal, llm_skipped=False, llm_forward_calls=1 + diag.get('decode_forward_calls', 0), answer_source='current_target_model_generation', current_visual_complete=True, visual_tokens=int(expanded.visual_mask.sum()), historical_text_consumed=False, prefix_planner_executed=False, persistent_prefill_kv=False, draft_match_type=selection.match_type, draft_exact_available=selection.exact_available, draft_semantic_available=selection.semantic_available, draft_source_frame=selection.record.frame_id if selection.record else None, draft_source_question=selection.record.question if selection.record else None, proposed_block_length=selection.length if selection.record is not None else 0, draft_decision_reason=selection.reason, draft_candidates=candidate_diag, draft_construction_wall_ms=selection.construction_ms, draft_tokenizer_wall_ms=selection.tokenizer_ms, world_delta=selection.delta.to_dict() if selection.delta else None, output_token_ids=token_ids, draft_selection_wall_ms=max(0.0, retrieval_only_ms + selection.construction_ms + selection.tokenizer_ms), draft_management_wall_ms=management_ms, metadata_wall_ms=metadata_ms / max(1, len(ids_list)), model_metadata_cache_hit=model_hit, tokenizer_metadata_cache_hit=tokenizer_hit, layout_preparation_ms=preparation_ms, prefill_ms=diag.get('prefill_including_cache_gpu_ms'), decode_ms=diag.get('decode_loop_gpu_span_ms'), world_diff_wall_ms=selection.world_diff_ms, end_to_end_ms=(perf_counter() - question_tick) * 1000)
            answers.append(answer)
            rows.append(diag)
        self._decode_previous_world = current_world
        elapsed = (perf_counter() - runtime_tick) * 1000
        self.last_diagnostics.update(mode=self.config.decode_strategy, execution_completed=True, runtime_host_elapsed_ms=elapsed, complete_m3_wall_ms=elapsed, cache_bytes=0, config=asdict(self.config), numerical_contract='approximate_block_fp16', prefix_planner_executed=False)
        if any((r.get('llm_skipped') for r in rows)):
            self.last_diagnostics['numerical_contract'] = 'task_limited_approximate_fact_response; target_aware_fp16_on_misses'
        return (answers, self.last_diagnostics)
