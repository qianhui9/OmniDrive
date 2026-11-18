"""Experimental answer caches and a causal, text-prefix-only KV cache.

Image-conditioned KV is never reused across changing visual inputs. Every KV
hit still computes the current visual suffix and runs target-model Decode.
"""
from collections import OrderedDict
from dataclasses import asdict
from time import perf_counter
from types import SimpleNamespace
import hashlib
import torch
from worldstream.m3.visual_state import base_model, expand_input
from worldstream.m3.prefill_executor import GenerationFacade, PrefillExecutor
from worldstream.m3.request_rng import question_rng


def answer_tensor(ids, tokenizer, base, text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    eos = base.generation_config.eos_token_id
    if isinstance(eos,(tuple,list)):
        eos = eos[0] if eos else None
    if eos is not None:
        tokens.append(eos)
    return ids.new_tensor([[base.generation_config.bos_token_id or 1]+tokens])


class PromptKV:
    def __init__(self, budget_bytes=512*1024**2):
        self.budget_bytes = budget_bytes
        self.reset()

    def reset(self):
        self.cache = OrderedDict()
        self.bytes = 0

    @torch.no_grad()
    def generate(self, model, expanded, key, **generation):
        visual = expanded.visual_mask[0].nonzero(as_tuple=False)
        prefix = int(visual[0]) if len(visual) else expanded.length-1
        prefix = min(prefix,expanded.length-1)
        entry = self.cache.get(key)
        reuse = 0
        if entry is not None and entry['length'] == prefix and prefix > 0:
            if all(torch.equal(entry[n],getattr(expanded,n)[:,:prefix])
                   for n in ('embeddings','attention_mask','position_ids')):
                reuse = prefix
                self.cache.move_to_end(key)
        diag = dict(executor='PromptKV', reused_tokens=reuse, input_tokens=expanded.length,
            effective_prefill_tokens=expanded.length-reuse, persistent_prefill_kv=True,
            visual_KV_reused=False, decode_forward_calls=0, decode_forward_input_lengths=[],
            decode_forward_host_enqueue_ms=0., prefill_completed=False, decode_completed=False)

        def first():
            output = model(inputs_embeds=expanded.embeddings[:,reuse:],
                position_ids=expanded.position_ids[:,reuse:], attention_mask=expanded.attention_mask,
                past_key_values=entry['kv'] if reuse else None, use_cache=True, return_dict=True)
            diag['prefill_completed'] = True
            if prefix > 0:
                # Own only the text prefix. No generated-token or visual KV history.
                kv = tuple(tuple(t[:,:,:prefix].detach().clone() for t in layer) for layer in output.past_key_values)
                value = dict(length=prefix, kv=kv,
                    **{n:getattr(expanded,n)[:,:prefix].detach().clone()
                       for n in ('embeddings','attention_mask','position_ids')})
                size = sum(t.numel()*t.element_size() for layer in kv for t in layer)
                size += sum(value[n].numel()*value[n].element_size() for n in ('embeddings','attention_mask','position_ids'))
                value['bytes'] = size
                old = self.cache.pop(key,None)
                if old:
                    self.bytes -= old['bytes']
                if size <= self.budget_bytes:
                    while self.cache and self.bytes+size > self.budget_bytes:
                        self.bytes -= self.cache.popitem(last=False)[1]['bytes']
                    self.cache[key] = value
                    self.bytes += size
            return output
        facade = GenerationFacade(model,expanded,first)
        facade.diagnostics = diag
        answer = facade.generate(inputs_embeds=expanded.embeddings,
            attention_mask=expanded.attention_mask,**generation)
        diag.update(decode_completed=True, cache_bytes=self.bytes)
        return answer,diag


def alternative_fact(question, state, policy):
    """Same fact grammar/confidence/risk admission, with baseline-valid witnesses.

    Historical source timestamps/origins stay intact in receipts. This adapter
    deliberately trusts the *chosen baseline's* validity decision, so it can
    produce stale answers; it never fabricates current-perception provenance.
    """
    from worldstream.m3.task_conditioner import question_facts
    from worldstream.m3.answer_validity import fact_region, fact_directions, render_facts
    from worldstream.m3.world_delta import VEHICLES
    from worldstream.m3.target_binding import target_binding_risk
    import math
    p = question_facts(question)
    if p.intent == 'unsupported' or state is None:
        return None
    classes = set(p.classes)
    if 'vehicle' in classes:
        classes = (classes-{'vehicle'})|VEHICLES
    if 'object' in classes:
        classes = set()
    witnesses = [a for a in state.anchors if a.observed and a.cameras and a.score is not None and
        a.score >= .35 and (not classes or a.semantic_class in classes) and fact_region(a.center_ego,p)]
    if not witnesses:
        return None
    witnesses.sort(key=lambda a:(math.hypot(*a.center_ego[:2]),-a.score,a.anchor_id))
    witness = max(witnesses,key=lambda a:a.score) if p.selector == 'any' else witnesses[0]
    if p.selector == 'nearest' and any(a.semantic_class != witness.semantic_class and
        abs(math.hypot(*a.center_ego[:2])-math.hypot(*witness.center_ego[:2]))<2 for a in witnesses[1:]):
        return None
    if p.intent == 'position' and any(abs(abs(v)-1)<.5 for v in witness.center_ego[:2]):
        return None
    if p.intent == 'motion':
        return None  # These simple alternatives make no velocity certification.
    risk = target_binding_risk(p,witness,witnesses,state,None,False)
    if witness.score < policy['min_confidence'] or risk['total_risk'] > policy['max_binding_risk']:
        return None
    return dict(text=render_facts(p,witness.semantic_class,fact_directions(witness.center_ego)),
        witness=asdict(witness), target_binding=risk, source_frame=witness.source_frame,
        history_reused=witness.source_frame != state.frame_id)


class LanguagePolicy:
    """Petr3D protocol adapter; production M3Config is never weakened or edited."""
    def __init__(self, original, policy, state_manager=None, semantic_threshold=.95):
        self.original, self.policy, self.state_manager = original,policy,state_manager
        self.semantic_threshold = semantic_threshold
        # Prevent detector's production-only snapshot builder for alternative stores.
        cfg = asdict(original.config)
        cfg['decode_strategy'] = 'ablation_'+policy
        self.config = SimpleNamespace(**cfg)
        self.semantic_qualifier = None
        self.executor = PrefillExecutor()
        self.kv = PromptKV()
        self.reset()

    def reset(self):
        self.original.reset()
        self.kv.reset()
        self.cache = OrderedDict()
        self.scope = None
        self.frame_index = -1
        self.frame_id = None
        self.last_diagnostics = {}

    @torch.no_grad()
    def generate(self, lm_head, input_ids_list, vision_carriers, context=None, questions=None,
                 layout=None, committed_world_state=None, tokenizer=None, **kwargs):
        context = dict(context or {})
        context.setdefault('model_version',context.get('weights_sha256') or '')
        if self.policy == 'candidate_only':
            kwargs['committed_world_state'] = None
            result,diag = self.original.generate(lm_head=lm_head,input_ids_list=input_ids_list,
                vision_carriers=vision_carriers,context=context,questions=questions,layout=layout,
                tokenizer=tokenizer,**kwargs)
            self.last_diagnostics = diag
            return result,diag
        if self.policy == 'alternative_state':
            committed_world_state = self.state_manager.snapshot()
        base = base_model(lm_head)
        generation_keys = ('do_sample','temperature','top_p','num_beams','max_new_tokens','use_cache')
        generation = {k:kwargs[k] for k in generation_keys if k in kwargs}
        scope = (context['scene_id'],context.get('weights_sha256'),tuple(context['camera_ids']),
                 repr(generation),self.policy)
        if scope != self.scope:
            self.cache.clear()
            self.kv.reset()
            self.original.reset()
            self.frame_index = -1
            self.frame_id = None
            self.scope = scope
        if self.frame_id != context['frame_id']:
            self.frame_index += 1
            self.frame_id = context['frame_id']
        context['frame_index'] = self.frame_index
        answers, rows = [],[]
        ids_list = list(input_ids_list)
        for index,(ids,question) in enumerate(zip(ids_list,questions)):
            tick = perf_counter()
            prompt = tuple(ids.reshape(-1).tolist())
            exact = (question,prompt)
            hit,vector,similarity = self.cache.get(exact),None,None
            proposal = None
            if self.policy == 'alternative_state':
                proposal = alternative_fact(question,committed_world_state,self.original.fast_answer_policy)
            if self.policy == 'semantic_cache':
                qids = tokenizer.encode(question,add_special_tokens=False)
                embedding = base.get_input_embeddings()
                vector = embedding(ids.new_tensor(qids)).float().mean(0)
                vector = torch.nn.functional.normalize(vector,dim=0)
                if hit is None and self.cache:
                    entries = list(self.cache.values())
                    similarities = torch.stack([entry['vector'] for entry in entries])@vector
                    best = int(similarities.argmax())
                    similarity = float(similarities[best])
                    if similarity >= self.semantic_threshold:
                        hit = entries[best]
            if proposal is not None:
                answer = answer_tensor(ids,tokenizer,base,proposal['text'])
                diag = dict(executor='AlternativeStateFactAnswer',llm_skipped=True,llm_forward_calls=0,
                    direct_answer=True,history_reused=proposal['history_reused'],
                    source_frame=proposal['source_frame'],alternative_fact=proposal)
            elif self.policy in ('exact_cache','semantic_cache') and hit is not None:
                answer = ids.new_tensor([hit['tokens']])
                diag = dict(executor='AnswerCache',llm_skipped=True,llm_forward_calls=0,direct_answer=False,
                    history_reused=True,source_frame=hit['frame_id'],source_question=hit['question'],
                    semantic_similarity=similarity)
            elif self.policy == 'alternative_state':
                # Full QCSE target verifier on misses, with current-only facts that
                # genuinely came from this frame. Historical direct answers above
                # use explicit method validity, not fabricated perception metadata.
                # Run the full frame together below to preserve question RNG indices.
                answer,diag = None,None
            else:
                expanded = expand_input(lm_head,ids,vision_carriers,carrier_layout=layout)
                with question_rng(context,index,ids.device,True):
                    if self.policy == 'prompt_kv':
                        answer,diag = self.kv.generate(base,expanded,exact,**generation)
                    else:
                        answer,diag = self.executor.generate(base,expanded,**generation)
                diag.update(llm_skipped=False,history_reused=False,direct_answer=False,
                            llm_forward_calls=1+diag.get('decode_forward_calls',0))
                if self.policy in ('exact_cache','semantic_cache'):
                    tokens = answer.sequences if hasattr(answer,'sequences') else answer
                    self.cache[exact] = dict(tokens=tokens[0].tolist(),frame_id=context['frame_id'],
                        question=question,vector=vector)
                    while len(self.cache)>256:
                        self.cache.popitem(last=False)
            answers.append(answer)
            rows.append(dict(request_index=index,**diag,end_to_end_ms=(perf_counter()-tick)*1000) if diag else None)
        if self.policy == 'alternative_state' and any(a is None for a in answers):
            # The same current-context target verifier on misses. Keep original
            # frame/question RNG indices even when other questions answered directly.
            for index,answer in enumerate(answers):
                if answer is not None:
                    continue
                with question_rng(context,index,ids_list[index].device,True):
                    expanded = expand_input(lm_head,ids_list[index],vision_carriers,carrier_layout=layout)
                    binding,_,_ = self.original._decode_binding(lm_head,base,context,tokenizer,generation)
                    selection = self.original.answer_memory.retrieve(questions[index],tuple(ids_list[index].reshape(-1).tolist()),
                        context,binding,committed_world_state,False)
                    selection = self.original.answer_memory.construct(selection,questions[index],tuple(ids_list[index].reshape(-1).tolist()),
                        context,binding,None,tokenizer)
                    answer,diag = self.executor.generate(base,expanded,draft_candidates=selection.candidates,
                        draft_length=8,experimental_sampling=True,**generation)
                    self.original.answer_memory.observe_acceptance(selection,diag,self.frame_index)
                    tokens = answer[0,1:].tolist()
                    self.original.answer_memory.remember(questions[index],tuple(ids_list[index].reshape(-1).tolist()),
                        tokens,context,binding,committed_world_state,tokenizer.decode(tokens,skip_special_tokens=True),extract_facts=True)
                answers[index] = answer
                rows[index] = dict(request_index=index,executor='PrefillExecutor',llm_skipped=False,
                    history_reused=False,direct_answer=False,llm_forward_calls=1+diag.get('decode_forward_calls',0),
                    **{k:v for k,v in diag.items() if k!='executor'})
        self.last_diagnostics = dict(execution_completed=True,requests=rows,experimental_policy=self.policy)
        return answers,self.last_diagnostics
