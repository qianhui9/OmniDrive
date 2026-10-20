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
        self._decode_graphs = None
        from .scheduler import read_artifact
        self.direct_task_certificate = read_artifact(self.config.direct_task_certificate)

    def reset(self):
        self.cache.reset()
        self._scope = self._signature = self._timestamp = None

    @torch.no_grad()
    def generate(self, lm_head, input_ids_list, vision_carriers, world_state=None,
                 visual_execution=None, context=None, questions=None, layout=None,
                 verified_state=None, tokenizer=None, task_evidence=None, **generation):
        if self.config.experimental_rng_isolated_microbatch:
            from .sampling_microbatch import SamplingMicrobatchExecutor
            ids=list(input_ids_list);qs=list(questions) if questions is not None else ['']*len(ids)
            executor=SamplingMicrobatchExecutor(self.config.sampling_microbatch_size,self.config.sampling_microbatch_seed)
            answer,diag=executor.try_generate(lm_head,ids,vision_carriers,qs,dict(context or {}),tokenizer,**generation)
            if answer is not None:return answer,dict(mode='experimental_rng_isolated_microbatch',sampling_microbatch=diag)
            # Attempt work is charged; failed temporary KV never enters PrefixCache.
            self.reset()
            native=[lm_head.generate(inputs=x[None] if x.ndim==1 else x,images=vision_carriers,**generation) for x in ids]
            return native,dict(mode='full',reason='sampling_microbatch_native_fallback',sampling_microbatch=diag)
        args=dict(world_state=world_state,visual_execution=visual_execution,context=context,
            questions=questions,layout=layout,verified_state=verified_state,tokenizer=tokenizer,
            task_evidence=task_evidence,**generation)
        if not self.config.decode_cuda_graph:
            return self._generate_impl(lm_head,input_ids_list,vision_carriers,**args)
        from .native_decode import StaticDecodeGraphs
        if self._decode_graphs is None:self._decode_graphs=StaticDecodeGraphs()
        try:
            with self._decode_graphs.activate(lm_head) as stats:
                out,diagnostics=self._generate_impl(lm_head,input_ids_list,vision_carriers,**args)
            diagnostics['decode_execution']=stats
            return out,diagnostics
        except Exception:
            self.reset()
            self._decode_graphs.clear()
            raise

    @torch.no_grad()
    def _generate_impl(self, lm_head, input_ids_list, vision_carriers, world_state=None,
                 visual_execution=None, context=None, questions=None, layout=None,
                 verified_state=None, tokenizer=None, task_evidence=None, **generation):
        ids = list(input_ids_list)
        qs = list(questions) if questions is not None else [''] * len(ids)
        if len(ids) != len(qs):
            raise ValueError('Question metadata order must match input IDs')
        arguments = dict(world_state=world_state, visual_execution=visual_execution,
            context=context, layout=layout, verified_state=verified_state, tokenizer=tokenizer)
        if not self.config.task_evidence_enabled:
            return self._generate_visual(lm_head, ids, vision_carriers, questions=qs,
                **arguments, **generation)
        from .task_evidence import evaluate_task_evidence,native_prompt_matches
        tick = perf_counter()
        records = task_evidence if task_evidence is not None else getattr(verified_state, 'records', ())
        decisions = [evaluate_task_evidence(q, context or {}, records, self.direct_task_certificate) for q in qs]
        admission_ms = (perf_counter() - tick) * 1000
        format_supported = not any(generation.get(k) for k in ('return_dict_in_generate', 'output_scores', 'output_attentions', 'output_hidden_states'))
        format_supported = format_supported and not any(k in generation for k in ('attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds', 'stopping_criteria', 'constraints', 'force_words_ids'))
        bindings = [native_prompt_matches(tokenizer, x, q) if d.quality_qualified else False for x, q, d in zip(ids, qs, decisions)]
        direct = [bool(bound and format_supported and self.config.experimental_direct_query and not self.config.shadow_evaluation
            and d.evidence_complete and d.quality_qualified and tokenizer is not None) for d, bound in zip(decisions, bindings)]
        execution_rejections = [None] * len(ids)
        answers = [None] * len(ids)
        for i, allowed in enumerate(direct):
            if allowed:
                # Encode only the final schema-compatible answer; no model forward,
                # multimodal expansion, prefill or decode is invoked for this query.
                encoded = tokenizer.encode(decisions[i].candidate, add_special_tokens=False)
                limit = generation.get('max_new_tokens')
                if not encoded or (limit is not None and len(encoded) > limit):
                    direct[i] = False
                    execution_rejections[i] = 'direct_answer_exceeds_generation_budget'
                else:
                    answers[i] = ids[i].new_tensor(encoded).reshape(1, -1)
        fallback_indices = [i for i, allowed in enumerate(direct) if not allowed]
        fallback_diag = dict(mode='no_visual_requests', requests=[])
        if fallback_indices:
            if self.config.direct_query_prefix_fallback and not self.config.shadow_evaluation:
                fallback, fallback_diag = self._generate_visual(lm_head,
                    [ids[i] for i in fallback_indices], vision_carriers,
                    questions=[qs[i] for i in fallback_indices], **arguments, **generation)
            else:
                self.reset()
                fallback = [lm_head.generate(inputs=ids[i].unsqueeze(0) if ids[i].ndim == 1 else ids[i],
                    images=vision_carriers, **generation) for i in fallback_indices]
                fallback_diag = dict(mode='full', reason='evidence_full_visual_fallback', requests=[])
            for i, answer in zip(fallback_indices, fallback):answers[i] = answer
        elif direct:
            # A direct-only transaction never leaves an unrelated prefix live.
            self.reset()
        rows = []
        for i, d in enumerate(decisions):
            rows.append(dict(d.to_dict(), route='direct' if direct[i] else 'visual_fallback',
                shadow_evaluation=self.config.shadow_evaluation,native_prompt_bound=bindings[i],
                actual_avoided_prefill_requests=int(direct[i]), actual_avoided_decode_requests=int(direct[i]),
                hypothetical_savings_credited=False,
                fallback_reason=None if direct[i] else execution_rejections[i] or ('shadow_requires_original_vlm' if self.config.shadow_evaluation
                    and d.quality_qualified else d.reason if not d.quality_qualified else 'native_prompt_or_output_contract_mismatch' if not bindings[i] or not format_supported else 'experimental_execution_disabled')))
        return answers, dict(mode='task_evidence', requests=rows, visual_fallback=fallback_diag,
            admission_cpu_ms=admission_ms, cache_bytes=self.cache.nbytes,
            actual_direct_requests=sum(direct), actual_visual_requests=len(fallback_indices))

    def _generate_visual(self, *args, **kwargs):
        try:
            return self._generate_visual_impl(*args, **kwargs)
        except Exception:
            # No prefix from an interrupted/OOM generation may survive into a
            # later request. The caller owns retry and downstream memory policy.
            self.reset()
            raise

    @torch.no_grad()
    def _generate_visual_impl(self, lm_head, input_ids_list, vision_carriers, world_state=None,
                 visual_execution=None, context=None, questions=None, layout=None,
                 verified_state=None, tokenizer=None, **generation):
        context = dict(context or {})
        context.setdefault('model_version', self.config.model_version)
        ids_list = list(input_ids_list)
        questions = list(questions) if questions is not None else [''] * len(ids_list)
        if len(questions) != len(ids_list):
            raise ValueError('Question metadata order must match input IDs')
        base = base_model(lm_head)
        def effective(name, default=None):
            return generation.get(name, getattr(base.generation_config, name, default))
        supported = (transformers.__version__ == '4.31.0' and not any(m.training for m in base.modules())
                     and not getattr(base.config, 'rope_scaling', None)
                     and effective('num_beams', 1) == 1
                     and effective('num_return_sequences', 1) == 1
                     and effective('use_cache', True)
                     and not effective('output_attentions') and not effective('output_hidden_states')
                     and not effective('constraints') and not effective('force_words_ids')
                     and effective('penalty_alpha') is None)
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
        if (self.config.single_question_fastpath and len(ids_list)==1
                and (self.config.mode=='exact' or (self.config.mode=='auto' and self.config.common_prefix_seed)) and not self.config.cross_frame
                and self.config.state_context=='none'):
            self.reset()
            return [lm_head.generate(inputs=x.unsqueeze(0) if x.ndim==1 else x,
                images=vision_carriers,**generation) for x in ids_list],dict(
                mode='full',reason='single_question_no_shared_prefix',requests=[])
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
        prepared=None;shared_seed=None
        if self.config.common_prefix_seed and len(ids_list)>1:
            from .prefix_matcher import exact_lcp
            tick=perf_counter()
            prepared=[expand_input(lm_head,ids,vision_carriers,scope=scope,signature=signature,
                source=(context['scene_id'],context['frame_id'],timestamp,i)) for i,ids in enumerate(ids_list)]
            prefix=min(min(exact_lcp(prepared[0],x) for x in prepared[1:]),min(x.length for x in prepared)-1)
            import hashlib,json
            try:
                key=hashlib.sha256(json.dumps(dict(defaults=base.generation_config.to_dict(),overrides=generation),
                    sort_keys=True,allow_nan=False).encode()).hexdigest()
            except (TypeError,ValueError):key=None
            allowed,reason=self.scheduler.common_seed_allowed(prepared,prefix,context,key)
            if self.config.mode=='auto' and not allowed:
                self.reset()
                return [lm_head.generate(inputs=x.unsqueeze(0) if x.ndim==1 else x,
                    images=vision_carriers,**generation) for x in ids_list],dict(
                    mode='full',reason=reason,requests=[])
            if prefix>=self.config.min_prefix_tokens:
                x=prepared[0].prefix(prefix)
                with torch.autograd.profiler.record_function('m3_independent_shared_prefix'):
                    seed=base(inputs_embeds=x.embeddings,position_ids=x.position_ids,
                        attention_mask=x.attention_mask,use_cache=True,return_dict=True)
                stored=self.cache.put(x,seed.past_key_values,seed.logits)
                del seed
                shared_seed=dict(prefix_tokens=prefix,cache_stored=stored,
                    actual_saved_prefix_tokens=prefix*(len(ids_list)-1) if stored else 0,
                    preparation_and_seed_cpu_enqueue_ms=(perf_counter()-tick)*1000,
                    note='Independent true common prefix, not a slice of the first full-question prefill')
        batch_diagnostics=None
        if self.config.batched_decode:
            from .batched_executor import BatchedExecutor, AdaptiveBatchExecutor
            expanded_batch=[expand_input(lm_head,ids,vision_carriers,scope=scope,signature=signature,
                source=(context['scene_id'],context['frame_id'],timestamp,i)) for i,ids in enumerate(ids_list)]
            workspace=int(self.config.batch_workspace_mb*1024**2)
            executor=BatchedExecutor(self.cache,workspace)
            if self.config.adaptive_batch:
                import hashlib,json
                from .scheduler import read_artifact
                try:
                    contract=hashlib.sha256(json.dumps(dict(defaults=base.generation_config.to_dict(),overrides=generation),
                        sort_keys=True,allow_nan=False).encode()).hexdigest()
                except (TypeError,ValueError):contract=None
                cost_context=dict(model_version=context.get('model_version'),execution_key=deployment_key(lm_head),
                    generation_key=contract,workspace_bytes=workspace,memory_budget_bytes=self.cache.budget_bytes)
                executor=AdaptiveBatchExecutor(self.cache,workspace,self.config.max_batch_size,
                    read_artifact(self.config.batch_cost_table),cost_context)
            answers,batch_diagnostics=executor.try_generate(base,expanded_batch,**generation)
            if answers is not None:
                return answers,dict(mode='batched',requests=batch_diagnostics['requests'],batch=batch_diagnostics,
                    cache_bytes=self.cache.nbytes,signature_cpu_ms=signature_ms,m2_required=False)
            self.reset()
            return [lm_head.generate(inputs=x.unsqueeze(0) if x.ndim==1 else x,
                images=vision_carriers,**generation) for x in ids_list],dict(
                mode='full',reason='batch_contract_full_fallback',requests=[],batch_fallback=batch_diagnostics)
        for index, (ids, question) in enumerate(zip(ids_list, questions)):
            tick = perf_counter()
            carriers = vision_carriers
            selection = dict(approximate=False, reason='full_carrier')
            state_retrieval=None
            if self.config.state_context!='none':
                from .world_state import query_world_state,insert_state_context
                state_retrieval=query_world_state(verified_state,question,context)
                if state_retrieval['admitted'] and tokenizer is not None:
                    try:
                        enriched=insert_state_context(ids,state_retrieval['text'],tokenizer)
                        length=enriched.numel()-1+(0 if self.config.state_context=='only' else carriers.shape[1])
                        if length>base.config.max_position_embeddings:raise ValueError('State context exceeds model position limit')
                    except ValueError as error:
                        state_retrieval=dict(state_retrieval,admitted=False,reason=str(error))
                    else:
                        ids=enriched
                        if self.config.state_context=='only':carriers=carriers[:,:0]
                        selection=dict(approximate=True,admitted=True,reason='offline_state_context',
                            original_tokens=vision_carriers.shape[1],selected_tokens=carriers.shape[1])
                elif state_retrieval['admitted']:
                    state_retrieval=dict(state_retrieval,admitted=False,reason='missing_tokenizer')
            if (self.config.state_context=='none' and layout is not None and vision_carriers is not None
                    and self.config.mode in ('compact', 'combined', 'auto')):
                selected = select_carriers(carriers, layout, question, self.config.selector,
                    self.config.keep_ratio, self.config.random_seed, world_state, context)
                allowed, reason = self.scheduler.compact_allowed(selected.diagnostics, context)
                if allowed and self.config.mode == 'auto' and not self.scheduler.compact_cost_allowed(selected.diagnostics, context):
                    allowed, reason = False, 'no_profitable_compact_cost_measurement'
                selection = dict(selected.diagnostics, admitted=allowed, admission_reason=reason)
                if allowed:
                    carriers = carriers.index_select(1, selected.indices)
            selection_ms = (perf_counter() - tick) * 1000
            if state_retrieval is not None and not state_retrieval['admitted']:
                self.cache.reset()
                answers.append(lm_head.generate(inputs=ids.unsqueeze(0) if ids.ndim==1 else ids,
                    images=vision_carriers,**generation))
                visual=vision_carriers.shape[1] if vision_carriers is not None else 0
                n=ids.numel()+int((ids==-200).sum())*(visual-1)
                rows.append(dict(mode='full',reason='no_usable_verified_state',state_retrieval=state_retrieval,
                    input_tokens=n,effective_prefill_tokens=n,reused_tokens=0,visual_tokens=visual,
                    selection=selection,selection_cpu_ms=selection_ms))
                continue
            tick = perf_counter()
            try:
                expanded = prepared[index] if prepared is not None else expand_input(lm_head, ids, carriers, scope=scope, signature=signature,
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
            if shared_seed and shared_seed['cache_stored'] and prefix>=shared_seed['prefix_tokens']:
                allowed,reason=True,'independently_admitted_common_prefix_plan'
            prefix = prefix if allowed else 0
            if self.config.identical_prefill_only and prefix:
                # An exact input LCP does not certify FP16 KV invariance across
                # different full/suffix GEMM shapes. Reuse saved final logits
                # only for the entire stored prefill, otherwise compute full.
                if prefix != expanded.length or self.cache.entry.expanded.length != expanded.length:
                    prefix=0
                    reason='partial_prefix_has_no_numerical_parity_certificate'
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
                    profile=context.get('profile', False), borrow_prefix=self.config.borrow_prefix, **generation)
                diag['mode'] = ('compact+' if selection.get('admitted') else '') + ('exact' if prefix else 'independent')
            # Cache a true prefill in exact experiment mode on a miss. Auto
            # capture is intentionally disabled without calibrated cost evidence.
            diag.update(selection=selection, selection_cpu_ms=selection_ms,
                preparation_cpu_ms=preparation_ms, match_cpu_ms=match_ms,
                matched_prefix=matched_prefix, prefix_admission=reason,
                visual_tokens=int(expanded.visual_mask.sum()), signature_cpu_ms=signature_ms / max(1, len(ids_list)))
            if state_retrieval is not None:diag['state_retrieval']=state_retrieval
            answers.append(answer)
            rows.append(diag)
        return answers, dict(mode=self.config.mode, requests=rows, cache_bytes=self.cache.nbytes,
                             config=asdict(self.config), m2_required=False,batch_fallback=batch_diagnostics,
                             shared_seed=shared_seed)
