"""Suffix-only forward plus the installed HF generation loop, without monkeypatching.

Llava.generate refuses inputs_embeds. A request-local GenerationMixin facade
uses native expanded embeddings for its first forward and delegates every decode
to the original model. HF retains its BOS convention, processors, RNG, EOS and
stopping criteria. No images are passed back into the model.

History blocks are an opt-in research path. Their ideal-arithmetic selection
rule is target-equivalent, but real FP16 block/serial numerical identity must
be qualified separately before deployment (the 3090 study found differences).
"""
import copy
import math
from time import perf_counter
import torch
from torch import nn
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
from .prefix_matcher import input_revision


class GenerationFacade(nn.Module, GenerationMixin):
    main_input_name = 'input_ids'

    def __init__(self, model, expanded, first_forward):
        super().__init__()
        self.model = model
        self.config = model.config
        self.generation_config = copy.deepcopy(model.generation_config)
        self.expanded = expanded
        self.first_forward = first_forward
        self.first = True
        self.draft_tokens = ()
        self.draft_length = 8
        self.diagnostics = dict(decode_forward_calls=0,decode_forward_input_lengths=[],
                                decode_forward_host_enqueue_ms=0.)
        self.decode_events = []
        self.profile = False
        # HF appends exactly one unmasked token per step for batch-one decoding.
        # Keep the prompt count on-device instead of rescanning its complete
        # attention mask with cumsum/masked_fill for every generated token.
        self.prompt_unmasked = expanded.attention_mask.long().sum(-1, keepdim=True)

    @property
    def device(self):
        return self.expanded.embeddings.device

    @staticmethod
    def can_generate():
        return True

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, attention_mask=None, **kwargs):
        if self.first:
            return dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                        use_cache=True)
        added = attention_mask.shape[-1] - self.expanded.length
        position_ids = self.prompt_unmasked + (added - 1)
        return dict(input_ids=input_ids[:, -1:], position_ids=position_ids,
            past_key_values=past_key_values, attention_mask=attention_mask, use_cache=True)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=True,
                return_dict=True, output_attentions=None, output_hidden_states=None):
        if self.first:
            self.first = False
            return self.first_forward()
        tick = perf_counter()
        if self.profile and self.device.type == 'cuda':
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2)); a.record()
        result = self.model(input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=True, return_dict=True, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        if self.profile and self.device.type == 'cuda':
            b.record(); self.decode_events.append((a, b, input_ids.shape[-1]))
        self.diagnostics['decode_forward_calls'] = self.diagnostics.get('decode_forward_calls', 0) + 1
        self.diagnostics['decode_forward_input_lengths'].append(input_ids.shape[-1])
        self.diagnostics['decode_forward_host_enqueue_ms'] += (perf_counter()-tick)*1000
        return result

    def greedy_search(self, input_ids, logits_processor=None, stopping_criteria=None, **kwargs):
        if not self.draft_tokens:
            return super().greedy_search(input_ids, logits_processor=logits_processor,
                stopping_criteria=stopping_criteria, **kwargs)
        return self._history_decode(input_ids, logits_processor, stopping_criteria, **kwargs)

    def sample(self, input_ids, logits_processor=None, stopping_criteria=None, logits_warper=None, **kwargs):
        if not self.draft_tokens:
            return super().sample(input_ids, logits_processor=logits_processor,
                stopping_criteria=stopping_criteria, logits_warper=logits_warper, **kwargs)
        return self._history_decode(input_ids, logits_processor, stopping_criteria,
            logits_warper=logits_warper, sampling=True, **kwargs)

    def _history_decode(self, ids, processors, stopping, logits_warper=None, sampling=False,
                        eos_token_id=None, pad_token_id=None, output_scores=False,
                        return_dict_in_generate=False, synced_gpus=False, streamer=None, **kwargs):
        """One request-local target verifier; K/V always derives from this prefill.

        A correction/bonus is pending until the next forward. Feed it together
        with the next block so a fully accepted block costs one forward. A
        rejected suffix is sliced off before any continuation. History is
        abandoned after the first rejection to bound wasted verification work.
        HF generate has already prepared the actual processors/warpers/stoppers.
        """
        if synced_gpus or streamer is not None or ids.shape != (1, 1):
            raise ValueError('History verification supports batch-one local decoding')
        output = self.forward(inputs_embeds=self.expanded.embeddings,
                              attention_mask=self.expanded.attention_mask)
        kv, logits = output.past_key_values, output.logits[:, -1]
        del output
        eos = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or ())
        pending = None
        active = True
        scores = () if return_dict_in_generate and output_scores else None
        diag = self.diagnostics
        diag.update(verification_forward_calls=0, draft_candidate_tokens=0,
                    draft_accepted_tokens=0, verification_log=[], kv_crop_host_ms=0.)

        def transform(raw):
            value = processors(ids, raw.clone()) if processors is not None else raw
            return logits_warper(ids, value) if sampling and logits_warper is not None else value

        def emit(token, value):
            nonlocal ids, scores
            ids = torch.cat((ids, token.reshape(1, 1)), -1)
            if scores is not None:
                scores += (value,)
            return int(token) in eos or bool(stopping(ids, scores))

        while True:
            generated = ids.shape[-1]-1
            room = stopping.max_length-ids.shape[-1]
            candidates = self.draft_tokens[generated:generated+min(self.draft_length, room)] if active else ()
            if not candidates:
                if pending is not None:
                    length = kv[0][0].shape[-2]
                    output = self._target_forward(pending.reshape(1,1), kv, length)
                    kv, logits = output.past_key_values, output.logits[:, -1]
                    del output
                value = transform(logits)
                token = torch.multinomial(torch.softmax(value, -1), 1)[0,0] if sampling else value.argmax(-1)[0]
                pending = token
                if emit(token, value):
                    break
                continue
            proposed = ids.new_tensor(candidates).reshape(1,-1)
            block = torch.cat((pending.reshape(1,1), proposed), -1) if pending is not None else proposed
            length = kv[0][0].shape[-2]
            output = self._target_forward(block, kv, length)
            target = output.logits
            kv = output.past_key_values
            # logits before d_0 come from prefill, or from the pending token.
            distributions = target[:, :-1] if pending is not None else torch.cat((logits[:,None],target[:,:-1]),1)
            base_length = length + int(pending is not None)
            log = dict(output_offset=generated, proposed=list(candidates), accepted=0,
                       rejection_position=None, forward_input_tokens=block.shape[-1])
            diag['verification_forward_calls'] += 1
            diag['draft_candidate_tokens'] += len(candidates)
            diag['verification_log'].append(log)
            rejected = False
            finished = False
            for j, candidate in enumerate(candidates):
                value = transform(distributions[:,j])
                if sampling:
                    token, accepted = deterministic_draft_sample(torch.softmax(value,-1), candidate)
                else:
                    token = value.argmax(-1)[0]
                    accepted = int(token) == candidate
                if accepted:
                    log['accepted'] += 1
                    diag['draft_accepted_tokens'] += 1
                else:
                    rejected = True
                    active = False
                    log['rejection_position'] = j
                finished = emit(token, value)
                if rejected or finished:
                    break
            crop_tick = perf_counter()
            kv = tuple(tuple(t[..., :base_length+log['accepted'], :] for t in layer) for layer in kv)
            diag['kv_crop_host_ms'] += (perf_counter()-crop_tick)*1000
            if finished:
                break
            if rejected:
                pending = token
            else:
                value = transform(target[:,-1])
                pending = torch.multinomial(torch.softmax(value,-1),1)[0,0] if sampling else value.argmax(-1)[0]
                if emit(pending, value):
                    break
            del output, target, distributions
        if return_dict_in_generate:
            cls = SampleDecoderOnlyOutput if sampling else GreedySearchDecoderOnlyOutput
            return cls(sequences=ids, scores=scores)
        return ids

    def _target_forward(self, tokens, kv, length):
        count = tokens.shape[-1]
        added = length-self.expanded.length
        attention = torch.cat((self.expanded.attention_mask,
            self.expanded.attention_mask.new_ones((1, added+count))), -1)
        positions = self.prompt_unmasked + added + torch.arange(count,device=self.device)[None]
        return self.forward(input_ids=tokens, attention_mask=attention,
            position_ids=positions, past_key_values=kv)


def deterministic_draft_sample(probabilities, candidate, generator=None):
    """q=delta(candidate): accept p(d); reject using p with d removed.

    Mixing accepted proposals and residual corrections returns a token with law p.
    Kept separate for exact transition-law and Monte Carlo boundary tests.
    """
    p = probabilities.reshape(-1)
    if torch.rand((), device=p.device, generator=generator) < p[candidate]:
        return p.new_tensor(candidate, dtype=torch.long), True
    residual = p.clone()
    residual[candidate] = 0
    return torch.multinomial(residual, 1, generator=generator)[0], False


class PrefillExecutor:
    def __init__(self, cache):
        self.cache = cache

    def effective_reuse_length(self,expanded,length):
        if length==expanded.length and self.cache.entry is not None and self.cache.entry.expanded.length!=expanded.length:
            return length-1
        return length

    @torch.no_grad()
    def generate(self, model, expanded, reuse_length=0, capture=True, profile=False, decision=None,
                 draft_tokens=None, draft_length=8, experimental_sampling=False,
                 **generation):
        if any(m.training for m in model.modules()):
            raise ValueError('M3 execution is inference only')
        config = generation.get('generation_config') or model.generation_config
        effective = lambda name, default=None: generation.get(name, getattr(config, name, default))
        if effective('num_beams', 1) != 1 or effective('num_return_sequences', 1) != 1 or effective('penalty_alpha') is not None:
            raise ValueError('M3 supports one greedy or sampling sequence')
        if generation.get('use_cache', True) is not True:
            raise ValueError('M3 requires use_cache=True')
        if effective('output_attentions') or effective('output_hidden_states'):
            raise ValueError('Cached prefix does not retain attentions/hidden states')
        if generation.get('max_new_tokens', 1) < 1:
            raise ValueError('max_new_tokens must be positive')
        n = expanded.length
        self.cache.validate_storage()
        if decision is not None:
            if reuse_length:
                raise ValueError('Specify ReuseDecision rather than a second reuse length')
            reuse_length = decision.reuse_length
            if decision.reuse_type not in ('none', 'exact', 'bounded_approx'):
                raise ValueError('Invalid reuse decision type')
            if (decision.reuse_type == 'none') != (reuse_length == 0):
                raise ValueError('Reuse decision type and length disagree')
        if type(reuse_length) is not int or not 0 <= reuse_length <= n:
            raise ValueError('Invalid continuous prefix length')
        if reuse_length:
            if self.cache.entry is None:
                raise ValueError('Missing or modified source KV')
            if decision is not None and decision.reuse_type == 'bounded_approx':
                proof = decision.proof
                source=self.cache.entry
                expected_depth=source.approx_reuse_depth+int(expanded.source[:2]!=source.expanded.source[:2])
                if (proof is None or not proof.policy_id or proof.current_revision != input_revision(expanded)
                        or proof.cached_revision != self.cache.revision or reuse_length > proof.contiguous_limit
                        or any(not math.isfinite(v) for v in (proof.risk,proof.risk_limit,proof.rms,proof.max_rms,proof.max_token,proof.max_token_change))
                        or proof.risk > proof.risk_limit or proof.rms > proof.max_rms
                        or proof.max_token > proof.max_token_change
                        or not 0 < decision.approx_reuse_depth == expected_depth
                        or decision.root_frame_index!=source.root_frame_index
                        or decision.cache_age_frames!=decision.frame_index-source.root_frame_index
                        or decision.cache_age_frames<0
                        or decision.cache_age_frames > proof.max_age_frames
                        or decision.approx_reuse_depth > proof.max_approx_depth
                        or decision.estimated_quality_risk != proof.risk):
                    raise ValueError('Bounded prefix requires current contiguous policy evidence')
                source=None  # Do not retain the replaced owned KV during capture.
            elif self.cache.entry.approx_reuse_depth or self.cache.match(expanded) < reuse_length:
                raise ValueError('Prefix equality must be certified before exact execution')
        # Only the end of the stored prefill has saved last-token logits.
        reuse_length=self.effective_reuse_length(expanded,reuse_length)
        branch_workspace_bytes = (sum(t[..., :reuse_length, :].numel() * t.element_size()
            for layer in self.cache.entry.kv for t in layer) if reuse_length else 0)
        diagnostics = dict(input_tokens=n, reused_tokens=reuse_length,
            prefill_entered=False,prefill_completed=False,decode_completed=False,full_prefill=False,
            effective_prefill_tokens=n - reuse_length, forward_prefill_lengths=[],
            computed_source=list(self.cache.entry.computed_source) if reuse_length else list(expanded.source),
            suffix_computed_source=list(expanded.source))
        diagnostics.update(decode_forward_calls=0, decode_forward_input_lengths=[],
            decode_forward_host_enqueue_ms=0., draft_status='none')
        diagnostics.update(reuse_type=decision.reuse_type if decision else ('exact' if reuse_length else 'none'),
            cache_age_frames=decision.cache_age_frames if decision else 0,
            approx_reuse_depth=decision.approx_reuse_depth if decision else 0)
        self.last_diagnostics = diagnostics
        diagnostics['prefix_storage']='owned_clone'
        events = []
        branch_events=[]
        diagnostics['cache_branch_cpu_enqueue_ms']=0.
        profile_cuda = profile and expanded.embeddings.is_cuda
        if profile_cuda:
            generation_start = torch.cuda.Event(enable_timing=True)
            generation_end = torch.cuda.Event(enable_timing=True)
            generation_start.record()

        def first_forward():
            diagnostics['prefill_entered'] = True
            tick = perf_counter()
            if profile and expanded.embeddings.is_cuda:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            def branch(length):
                if not length:return None
                t=perf_counter()
                if profile_cuda:
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record()
                value=self.cache.branch(length)
                if profile_cuda:b.record();branch_events.append((a,b))
                diagnostics['cache_branch_cpu_enqueue_ms']+=(perf_counter()-t)*1000
                return value
            if reuse_length == n:
                output = CausalLMOutputWithPast(logits=self.cache.entry.last_logits.clone(),
                                               past_key_values=branch(n))
            else:
                with torch.autograd.profiler.record_function('m3_prefix_branch'):
                    kv = branch(reuse_length)
                with torch.autograd.profiler.record_function('m3_suffix_prefill' if reuse_length else 'm3_full_prefill'):
                    output = model(inputs_embeds=expanded.embeddings[:, reuse_length:],
                        position_ids=expanded.position_ids[:, reuse_length:],
                        attention_mask=expanded.attention_mask, past_key_values=kv,
                        use_cache=True, return_dict=True)
                diagnostics['forward_prefill_lengths'].append(n - reuse_length)
                # Refresh on a new frame after legal text-prefix reuse, so its
                # later questions can share this frame's freshly computed vision.
                # Preserve the true computation source of the inherited prefix.
                refresh = (reuse_length and expanded.source[:2] != self.cache.entry.expanded.source[:2])
                if capture and (not reuse_length or refresh):
                    lineage = []
                    if reuse_length:
                        cursor = 0
                        for span_start, span_end, source in self.cache.entry.prefix_lineage:
                            if span_start >= reuse_length:
                                break
                            stop = min(span_end, reuse_length)
                            lineage.append((span_start, stop, source)); cursor = stop
                        if cursor < reuse_length:
                            lineage.append((cursor, reuse_length, self.cache.entry.computed_source))
                    update_tick = perf_counter()
                    diagnostics['cache_stored'] = self.cache.put(expanded, output.past_key_values,
                        output.logits, lineage,
                        frame_index=decision.frame_index if decision else 0,
                        root_frame_index=decision.root_frame_index if decision else 0,
                        approx_reuse_depth=decision.approx_reuse_depth if decision else 0,
                        quality_risk=decision.estimated_quality_risk if decision else 0.,
                        risk_profile=decision.risk_profile if decision else 'full_visual')
                    diagnostics['cache_update_cpu_enqueue_ms'] = (perf_counter()-update_tick)*1000
            if profile and expanded.embeddings.is_cuda:
                end.record()
                events.append((start, end))
            diagnostics['prefill_including_cache_cpu_enqueue_ms'] = (perf_counter() - tick) * 1000
            diagnostics['prefill_completed'] = True
            diagnostics['full_prefill'] = reuse_length == 0
            return output

        facade = GenerationFacade(model, expanded, first_forward)
        facade.diagnostics = diagnostics
        facade.profile = profile
        if type(draft_length) is not int or not 1 <= draft_length <= 32:
            raise ValueError('draft_length must be in [1,32]')
        draft = list(draft_tokens) if draft_tokens is not None else []
        valid = all(type(t) is int and 0 <= t < model.config.vocab_size for t in draft)
        if draft and valid:
            if effective('do_sample', False) and not experimental_sampling:
                raise ValueError('Sampling verifier requires explicit experimental_sampling validation opt-in')
            facade.draft_tokens = tuple(draft)
            facade.draft_length = draft_length
            diagnostics['draft_status'] = 'target_verification'
        elif draft:
            diagnostics['draft_status'] = 'invalid_tokens_native'
        tick = perf_counter()
        answer = facade.generate(inputs_embeds=expanded.embeddings,
            attention_mask=expanded.attention_mask, **generation)
        if not diagnostics['prefill_completed']:
            raise RuntimeError('M3 generation completed without PrefillExecutor prefill')
        diagnostics['decode_completed'] = True
        if profile_cuda:
            generation_end.record()
            torch.cuda.synchronize(expanded.embeddings.device)
            diagnostics['prefill_including_cache_gpu_ms'] = sum(a.elapsed_time(b) for a, b in events)
            diagnostics['decode_loop_gpu_span_ms'] = events[-1][1].elapsed_time(generation_end)
            diagnostics['generation_gpu_span_ms'] = generation_start.elapsed_time(generation_end)
            diagnostics['cache_branch_gpu_span_ms']=sum(a.elapsed_time(b) for a,b in branch_events)
            diagnostics['decode_forward_cuda_ms'] = sum(a.elapsed_time(b) for a,b,_ in facade.decode_events)
            diagnostics['verification_forward_cuda_ms'] = sum(a.elapsed_time(b) for a,b,k in facade.decode_events if k>1)
            diagnostics['decode_forward_cuda_events'] = [dict(input_tokens=k,cuda_ms=a.elapsed_time(b)) for a,b,k in facade.decode_events]
            diagnostics['timing_method'] = 'CUDA event stream spans; includes host gaps; decode includes HF sampling loop'
        diagnostics['llm_wall_ms' if profile else 'llm_cpu_enqueue_ms'] = (perf_counter() - tick) * 1000
        diagnostics['cache_bytes'] = self.cache.nbytes
        diagnostics['cache_budget_bytes'] = self.cache.budget_bytes
        diagnostics['prefix_lineage'] = self.cache.entry.prefix_lineage if self.cache.entry else ()
        diagnostics['branch_workspace_bytes'] = branch_workspace_bytes
        return answer, diagnostics
