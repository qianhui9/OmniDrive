"""Opt-in research sampling batches. No exact-FP16 or production certificate.

Original single-request prefill is retained; independent legacy K/V rows are
left padded once, with explicit positions. Each request owns a private RNG and
history; completed rows leave the next forward. No persistent cache is touched.
The transaction returns answers in source order only after every group succeeds.
"""
import copy
from dataclasses import dataclass,field
import hashlib
import json
from time import perf_counter
import torch
from transformers import LogitsProcessorList
from .visual_state import base_model,expand_input
from .question_dependencies import analyze_questions,ready_groups
from .microbatch_kv import pack_independent_kv,select_active_kv
from .prefill_executor import readonly_cache_safe


def request_seed(seed,context,index):
    key=(int(seed),str(context.get('scene_id','')),str(context.get('frame_id','')),int(index))
    return int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:8],'little')%(2**63-1)


def sampling_tools(base,generation):
    import transformers
    if transformers.__version__!='4.31.0':raise ValueError('Uninspected generation backend')
    config=copy.deepcopy(base.generation_config)
    unused=config.update(**generation)
    if unused:raise ValueError('Custom generation hooks/kwargs require Native')
    if (not config.do_sample or config.num_beams!=1 or config.num_return_sequences!=1
            or not config.use_cache or not isinstance(config.max_new_tokens,int) or config.max_new_tokens<1
            or config.return_dict_in_generate or config.output_scores or config.output_hidden_states or config.output_attentions
            or config.constraints or config.force_words_ids or config.penalty_alpha is not None
            or config.guidance_scale not in (None,1) or config.encoder_repetition_penalty!=1.
            or config.num_beam_groups!=1 or config.exponential_decay_length_penalty is not None
            or config.max_time is not None):
        raise ValueError('Unsupported isolated sampling contract')
    if config.bos_token_id is None:raise ValueError('Native embedding-input BOS convention required')
    config.max_length=config.max_new_tokens+1
    processor=base._get_logits_processor(config,1,None,None,LogitsProcessorList())
    warper=base._get_logits_warper(config)
    def probabilities(ids,logits):
        return torch.softmax(warper(ids,processor(ids,logits)),dim=-1)
    return config,probabilities


@dataclass
class RequestState:
    request_id: int
    source: tuple
    seed: int
    generator: object
    token_ids: list=field(default_factory=list)
    stop_reason: str=''
    complete_ms: float=0.


class SamplingMicrobatchExecutor:
    def __init__(self,batch_size=2,seed=0):
        if batch_size not in (2,4):raise ValueError('Research executor supports fixed 2/4 groups')
        self.batch_size,self.seed=batch_size,seed
        self.last_diagnostics={}

    @torch.no_grad()
    def try_generate(self,lm_head,ids,carriers,questions,context,tokenizer,**generation):
        tick=perf_counter();self.last_diagnostics=dict(admitted=False,batch_size_limit=self.batch_size,
            production_qualified=False,exact_sampling_claim=False)
        def reject(reason):
            self.last_diagnostics.update(fallback_reason=reason,failed_attempt_wall_ms=(perf_counter()-tick)*1000)
            return None,self.last_diagnostics
        base=base_model(lm_head)
        if (any(m.training for m in base.modules()) or not readonly_cache_safe(base)
                or getattr(base,'_worldstream_sampling_microbatch_owner',None) is not None):
            return reject('unsupported_training_cache_backend_or_concurrent_owner')
        if (not context.get('scene_id') or not context.get('frame_id') or
                len(set(context.get('camera_ids',())))!=6 or tokenizer is None):return reject('missing_six_camera_source_contract')
        if carriers is None or carriers.shape[0]!=1:return reject('shared_complete_carriers_required')
        if len(ids)!=len(questions) or not ids:return reject('question_metadata_mismatch')
        try:config,probs=sampling_tools(base,generation)
        except (ValueError,TypeError):return reject('unsupported_generation_contract')
        rows=analyze_questions(tokenizer,ids,questions,inputs_ready=True,shared_carriers=True,
            state_writes=context.get('question_state_writes',False),ordered_execution=context.get('ordered_execution',False))
        groups=ready_groups(rows,self.batch_size)
        if any(not d.independent for d in rows):return reject('dependency_or_prompt_contract_requires_native')
        if len(ids)<2:return reject('one_question_no_batch_opportunity')
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(carriers.device) if carriers.is_cuda else None
        base._worldstream_sampling_microbatch_owner=self
        failed=None;result=None
        try:
            result=self._generate(base,lm_head,ids,carriers,context,groups,config,probs,tick)
        except (RuntimeError,ValueError,IndexError) as error:
            failed=type(error).__name__+': '+str(error)[:240]
        finally:
            delattr(base,'_worldstream_sampling_microbatch_owner')
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,carriers.device)
        if failed:
            # Every generated answer/KV was request-local and not published.
            # No native fallback runs until the failed frame's stack is gone.
            if carriers.is_cuda:torch.cuda.empty_cache()
            return reject(failed)
        answers,diag=result
        diag.update(admitted=True,dependencies=[vars(r) for r in rows],groups=groups,
            transaction_committed_in_original_order=True,private_rng_does_not_advance_global=True,
            persistent_prefix_cache_accessed=False,production_qualified=False,exact_sampling_claim=False)
        self.last_diagnostics=diag
        return answers,diag

    def _generate(self,base,lm_head,ids,carriers,context,groups,config,probs,ready_tick):
        device=carriers.device;answers=[None]*len(ids);requests=[];steps=[];prefills=[]
        eos=config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
        packing_ms=0.;compaction_ms=0.;sampling_ms=0.;actual_batched=set();max_kv=0;max_padding=0
        events=[]
        for group in groups:
            states=[];caches=[];first_logits=[]
            for index in group:
                expanded=expand_input(lm_head,ids[index],carriers,source=(context['scene_id'],context['frame_id'],index))
                start=perf_counter();a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2)) if carriers.is_cuda else (None,None)
                if a:a.record()
                out=base(inputs_embeds=expanded.embeddings,attention_mask=expanded.attention_mask,
                    position_ids=expanded.position_ids,use_cache=True,return_dict=True)
                if b:b.record();events.append(('prefill',a,b))
                seed=request_seed(self.seed,context,index)
                state=RequestState(index,(context['scene_id'],context['frame_id'],index),seed,
                    torch.Generator(device=device).manual_seed(seed),[config.bos_token_id])
                states.append(state);caches.append(out.past_key_values);first_logits.append(out.logits[:,-1].clone())
                prefills.append(dict(request_id=index,input_tokens=expanded.length,cpu_submission_ms=(perf_counter()-start)*1000))
                del out,expanded
            def sample(state,logits):
                nonlocal sampling_ms
                start=perf_counter()
                history=torch.tensor([state.token_ids],device=device,dtype=torch.long)
                p=probs(history,logits)
                token=int(torch.multinomial(p,1,generator=state.generator))
                state.token_ids.append(token)
                if token in eos:state.stop_reason='eos'
                elif len(state.token_ids)-1>=config.max_new_tokens:state.stop_reason='max_new_tokens'
                if state.stop_reason:state.complete_ms=(perf_counter()-ready_tick)*1000
                sampling_ms+=(perf_counter()-start)*1000
            for state,logits in zip(states,first_logits):sample(state,logits)
            first_logits.clear()
            active=[i for i,s in enumerate(states) if not s.stop_reason]
            if active:
                start=perf_counter()
                kv,mask,positions=pack_independent_kv([caches[i] for i in active],device=device)
                packing_ms+=(perf_counter()-start)*1000
                caches.clear()
                live=[states[i] for i in active]
                max_padding=max(max_padding,int((mask==0).sum()))
                while live:
                    max_kv=max(max_kv,sum(t.numel()*t.element_size() for layer in kv for t in layer))
                    input_ids=torch.tensor([[s.token_ids[-1]] for s in live],device=device)
                    mask=torch.cat((mask,mask.new_ones((len(live),1))),1)
                    a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2)) if carriers.is_cuda else (None,None)
                    if a:a.record()
                    start=perf_counter()
                    out=base(input_ids=input_ids,past_key_values=kv,attention_mask=mask,
                        position_ids=positions,use_cache=True,return_dict=True)
                    if b:b.record();events.append(('decode',a,b))
                    steps.append(dict(batch=len(live),request_ids=[s.request_id for s in live],
                        cpu_submission_ms=(perf_counter()-start)*1000,physical_kv_tokens=mask.shape[1],
                        positions=positions.detach().cpu().flatten().tolist()))
                    if len(live)>1:actual_batched.update(s.request_id for s in live)
                    for i,state in enumerate(live):sample(state,out.logits[i:i+1,-1])
                    survivors=[i for i,s in enumerate(live) if not s.stop_reason]
                    kv=out.past_key_values;positions=positions+1
                    del out
                    if not survivors:
                        del kv,mask,positions;break
                    if len(survivors)!=len(live):
                        start=perf_counter();indices=torch.tensor(survivors,device=device)
                        kv=select_active_kv(kv,indices);mask=mask.index_select(0,indices);positions=positions.index_select(0,indices)
                        compaction_ms+=(perf_counter()-start)*1000
                        live=[live[i] for i in survivors]
            caches.clear()
            for state in states:
                answers[state.request_id]=torch.tensor([state.token_ids],device=device,dtype=torch.long)
                requests.append(dict(request_id=state.request_id,source=state.source,seed=state.seed,
                    generated_tokens=len(state.token_ids)-1,stop_reason=state.stop_reason,
                    compute_complete_ms=state.complete_ms,
                    final_rng_sha256=hashlib.sha256(state.generator.get_state().cpu().numpy().tobytes()).hexdigest()))
        if carriers.is_cuda:torch.cuda.synchronize(device)
        completed_ms=(perf_counter()-ready_tick)*1000
        for r in requests:r['ordered_transaction_commit_ms']=completed_ms
        return answers,dict(requests=sorted(requests,key=lambda r:r['request_id']),steps=steps,prefills=prefills,
            ready_task_count=len(ids),actual_batched_requests=len(actual_batched),
            generated_tokens=sum(r['generated_tokens'] for r in requests),
            full_generation_wall_ms=completed_ms,aggregate_tokens_s=sum(r['generated_tokens'] for r in requests)*1000/completed_ms,
            target_forward_calls=len(prefills)+len(steps),max_independent_kv_bytes=max_kv,max_padding_tokens=max_padding,
            packing_cpu_ms=packing_ms,compaction_cpu_ms=compaction_ms,independent_sampling_wall_ms=sampling_ms,
            prefill_gpu_ms=sum(a.elapsed_time(b) for stage,a,b in events if stage=='prefill'),
            decode_gpu_ms=sum(a.elapsed_time(b) for stage,a,b in events if stage=='decode'),
            warning='Different batched FP16 logits may change the target distribution; private RNG is not an exact sampling certificate.')
