"""Suffix-only forward plus the installed HF generation loop, without monkeypatching.

Llava.generate refuses inputs_embeds. A request-local GenerationMixin facade
uses native expanded embeddings for its first forward and delegates every decode
to the original model. HF retains its BOS convention, processors, RNG, EOS and
stopping criteria. No images are passed back into the model.
"""
import copy
import math
from time import perf_counter
import torch
from torch import nn
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
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
        return self.model.prepare_inputs_for_generation(input_ids,
            past_key_values=past_key_values, attention_mask=attention_mask,
            use_cache=True)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=True,
                return_dict=True, output_attentions=None, output_hidden_states=None):
        if self.first:
            self.first = False
            return self.first_forward()
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=True, return_dict=True, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)


class PrefillExecutor:
    def __init__(self, cache):
        self.cache = cache

    def effective_reuse_length(self,expanded,length):
        if length==expanded.length and self.cache.entry is not None and self.cache.entry.expanded.length!=expanded.length:
            return length-1
        return length

    @torch.no_grad()
    def generate(self, model, expanded, reuse_length=0, capture=True, profile=False, decision=None,
                 **generation):
        if any(m.training for m in model.modules()):
            raise ValueError('M3 execution is inference only')
        effective = lambda name, default=None: generation.get(name, getattr(model.generation_config, name, default))
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
            diagnostics['timing_method'] = 'CUDA event stream spans; includes host gaps; decode includes HF sampling loop'
        diagnostics['llm_wall_ms' if profile else 'llm_cpu_enqueue_ms'] = (perf_counter() - tick) * 1000
        diagnostics['cache_bytes'] = self.cache.nbytes
        diagnostics['cache_budget_bytes'] = self.cache.budget_bytes
        diagnostics['prefix_lineage'] = self.cache.entry.prefix_lineage if self.cache.entry else ()
        diagnostics['branch_workspace_bytes'] = branch_workspace_bytes
        return answer, diagnostics
