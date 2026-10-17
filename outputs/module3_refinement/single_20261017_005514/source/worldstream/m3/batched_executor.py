"""Experimental shared-prefix prefill and native HF greedy batched decode.

Suffixes are LEFT padded after a read-only prefix, with per-row positions and
full masks. The installed HF loop owns EOS/max length and original output order.
Sampling and caller stopping hooks fall back to sequential generation: neither
batch RNG draw order nor global stopping can equal independent requests.
"""
from time import perf_counter
import torch
from .prefill_executor import GenerationFacade,readonly_cache_safe
from .prefix_matcher import exact_lcp
from .prefix_kv_cache import kv_bytes


class BatchedExecutor:
    def __init__(self,cache,workspace_bytes=4096*1024**2):
        self.cache=cache;self.workspace_bytes=workspace_bytes

    @torch.no_grad()
    def try_generate(self,model,inputs,**generation):
        def fallback(reason):return None,dict(admitted=False,reason=reason)
        if len(inputs)<2:return fallback('one_question')
        effective=lambda k,d=None:generation.get(k,getattr(model.generation_config,k,d))
        if any(m.training for m in model.modules()) or not readonly_cache_safe(model):return fallback('unsupported_cache_update_semantics')
        if effective('do_sample',False):return fallback('sampling_requires_sequential_rng_isolation')
        if (effective('num_beams',1)!=1 or effective('num_return_sequences',1)!=1 or effective('penalty_alpha') is not None
                or not effective('use_cache',True)):return fallback('unsupported_generation_contract')
        if any(generation.get(k) is not None for k in ('stopping_criteria','logits_processor','prefix_allowed_tokens_fn','constraints','force_words_ids','generation_config')):
            return fallback('request_local_generation_hook_requires_sequential')
        if effective('return_dict_in_generate',False) or effective('output_scores',False) or effective('output_hidden_states',False) or effective('output_attentions',False):
            return fallback('structured_outputs_require_sequential')
        first=inputs[0]
        if any(x.attention_mask.shape[0]!=1 or not bool(x.attention_mask.all()) for x in inputs):return fallback('unsupported_input_padding')
        prefix=min(exact_lcp(first,x) for x in inputs[1:])
        # Every row needs a real last suffix token for its next-token logits.
        prefix=min(prefix,min(x.length for x in inputs)-1)
        if prefix<1:return fallback('no_shared_prefix')
        batch=len(inputs);suffix_lengths=[x.length-prefix for x in inputs];width=max(suffix_lengths)
        n=prefix+width;new_tokens=effective('max_new_tokens',None)
        if not isinstance(new_tokens,int) or new_tokens<1:return fallback('explicit_positive_token_limit_required')
        cfg=model.config;dtype=first.embeddings.dtype;device=first.embeddings.device
        per_token=cfg.num_hidden_layers*getattr(cfg,'num_key_value_heads',cfg.num_attention_heads)*(cfg.hidden_size//cfg.num_attention_heads)*2*first.embeddings.element_size()
        # Live KV plus transient concatenation input/output and prefix staging.
        estimate=2*batch*(n+new_tokens)*per_token+batch*width*cfg.vocab_size*4
        if estimate>self.workspace_bytes:return fallback('batch_workspace_budget')
        if device.type=='cuda':
            free,_=torch.cuda.mem_get_info(device)
            reusable=torch.cuda.memory_reserved(device)-torch.cuda.memory_allocated(device)
            if estimate+128*1024**2>free+reusable:return fallback('insufficient_free_gpu_workspace')
        tick=perf_counter()
        with torch.autograd.profiler.record_function('m3_batch_shared_prefix'):
            seed=model(inputs_embeds=first.embeddings[:,:prefix],attention_mask=first.attention_mask[:,:prefix],
                position_ids=first.position_ids[:,:prefix],use_cache=True,return_dict=True)
        if not self.cache.put(first.prefix(prefix),seed.past_key_values,seed.logits):
            del seed
            return fallback('persistent_prefix_budget')
        del seed
        embeds=first.embeddings.new_zeros((batch,width,cfg.hidden_size))
        mask=first.attention_mask.new_ones((batch,n));positions=first.position_ids.new_zeros((batch,width))
        for i,(x,length) in enumerate(zip(inputs,suffix_lengths)):
            pad=width-length
            embeds[i,pad:]=x.embeddings[0,prefix:]
            mask[i,prefix:prefix+pad]=0
            positions[i,pad:]=x.position_ids[0,prefix:]
        borrowed=self.cache.branch(prefix,read_only=True)
        shared=tuple(tuple(t.expand(batch,-1,-1,-1) for t in layer) for layer in borrowed)
        def first_forward():
            with torch.autograd.profiler.record_function('m3_batch_suffix_prefill'):
                return model(inputs_embeds=embeds,attention_mask=mask,position_ids=positions,
                    past_key_values=shared,use_cache=True,return_dict=True)
        facade=GenerationFacade(model,first,first_forward)
        output=facade.generate(inputs_embeds=embeds,attention_mask=mask,**generation)
        eos=effective('eos_token_id',None);eos=[eos] if isinstance(eos,int) else eos
        answers=[]
        for row in output:
            end=len(row)
            if eos:
                stopped=(row[1:,None]==row.new_tensor(eos)[None,:]).any(-1).nonzero().flatten()
                if len(stopped):end=int(stopped[0])+2
            answers.append(row[:end].unsqueeze(0))
        diagnostics=dict(admitted=True,reason='offline_shared_prefix_greedy_batch',batch_size=batch,
            shared_prefix_tokens=prefix,actual_reused_tokens=prefix*(batch-1),
            suffix_lengths=suffix_lengths,suffix_padding_tokens=batch*width-sum(suffix_lengths),
            decode_padding_tokens=batch*(output.shape[1]-1)-sum(x.shape[1]-1 for x in answers),
            effective_prefill_tokens=prefix+sum(suffix_lengths),physical_prefill_tokens=prefix+batch*width,
            estimated_workspace_bytes=estimate,persistent_bytes=self.cache.nbytes,
            shared_prefix_is_readonly=True,full_sequence_equality_guaranteed=False,
            llm_cpu_enqueue_ms=(perf_counter()-tick)*1000,
            requests=[dict(input_tokens=x.length,reused_tokens=prefix if i else 0,
                effective_prefill_tokens=x.length-prefix if i else x.length,
                shared_prefix_tokens=prefix,visual_tokens=int(x.visual_mask.sum()),mode='batched') for i,x in enumerate(inputs)])
        return answers,diagnostics
