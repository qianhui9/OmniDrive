"""Frozen history supplements full current vision; every request uses PrefillExecutor."""
from dataclasses import asdict
import hashlib
import json
import math
from time import perf_counter
import torch
import transformers
from .config import M3Config
from .visual_state import base_model,expand_input,execution_signature,deployment_key
from .prefix_kv_cache import PrefixKVCache
from .prefill_executor import PrefillExecutor
from .scheduler import Scheduler
from .world_state import query_world_state,insert_state_context


class M3Runtime:
    def __init__(self,config=None):
        self.config=config if isinstance(config,M3Config) else M3Config(**(config or {}))
        self.cache=PrefixKVCache(self.config.memory_budget_mb*1024**2)
        self.executor=PrefillExecutor(self.cache)
        self.scheduler=Scheduler(self.config)
        self._scope=self._signature=self._timestamp=None

    def reset(self):
        self.cache.reset();self._scope=self._signature=self._timestamp=None

    @staticmethod
    def validate_contract(base,generation):
        from transformers.models.llama.modeling_llama import LlamaAttention
        effective=lambda key,default=None:generation.get(key,getattr(base.generation_config,key,default))
        layers=getattr(getattr(base,'model',None),'layers',())
        supported=(transformers.__version__=='4.31.0' and len(layers)
            and all(type(layer.self_attn) is LlamaAttention for layer in layers)
            and not any(m.training for m in base.modules()) and not getattr(base.config,'rope_scaling',None)
            and effective('num_beams',1)==1 and effective('num_return_sequences',1)==1
            and effective('use_cache',True) is True and not effective('output_attentions')
            and not effective('output_hidden_states') and not effective('constraints')
            and not effective('force_words_ids') and effective('penalty_alpha') is None)
        forbidden={'attention_mask','position_ids','past_key_values','inputs_embeds','generation_config'}
        if not supported or forbidden&generation.keys():
            raise ValueError('Unsupported M3 model or generation contract; PrefillExecutor is required')

    @torch.no_grad()
    def generate(self,lm_head,input_ids_list,vision_carriers,context=None,questions=None,layout=None,verified_state=None,tokenizer=None,**generation):
        context=dict(context or {});context.setdefault('model_version',self.config.model_version)
        ids_list=list(input_ids_list);questions=list(questions) if questions is not None else ['']*len(ids_list)
        if len(questions)!=len(ids_list):raise ValueError('Question order must match input IDs')
        base=base_model(lm_head);self.validate_contract(base,generation)
        if layout is not None and vision_carriers is not None and vision_carriers.shape[1]!=layout.total:
            raise ValueError('M3 requires the complete declared current visual layout')
        cameras=context.get('camera_ids',());timestamp=context.get('timestamp')
        if (not context.get('scene_id') or not context.get('frame_id') or not cameras
                or len(set(cameras))!=len(cameras) or type(timestamp) not in (int,float) or not math.isfinite(timestamp)):
            raise ValueError('M3 requires a bound scene/frame/time/camera lifecycle')
        start=perf_counter();signature=execution_signature(lm_head,context['model_version'])
        scope=(str(context['scene_id']),tuple(cameras),context.get('state_epoch'))
        if (scope!=self._scope or signature!=self._signature
                or self._timestamp is not None and timestamp<=self._timestamp):self.cache.reset()
        self._scope,self._signature,self._timestamp=scope,signature,timestamp
        # Historical text has its own quality gate, independent of exact KV equality.
        if any((self.config.cost_table,self.config.quality_certificate,self.config.state_quality_certificate)):
            context['execution_key']=deployment_key(lm_head)
        context['generation_key']=hashlib.sha256(json.dumps(dict(defaults=base.generation_config.to_dict(),
            overrides=generation),sort_keys=True,default=repr).encode()).hexdigest()
        signature_ms=(perf_counter()-start)*1000
        answers=[];rows=[]
        for index,(ids,question) in enumerate(zip(ids_list,questions)):
            retrieval=query_world_state(verified_state,question,context)
            retrieval=dict(retrieval)
            if retrieval['admitted']:
                allowed,why=self.scheduler.state_allowed(retrieval,context)
                if not allowed:retrieval.update(admitted=False,reason=why)
                elif tokenizer is None:retrieval.update(admitted=False,reason='missing_tokenizer')
                else:
                    try:
                        enriched=insert_state_context(ids,retrieval['text'],tokenizer)
                        visual=vision_carriers.shape[1] if vision_carriers is not None else 0
                        if enriched.numel()-1+visual>base.config.max_position_embeddings:
                            raise ValueError('Historical context exceeds position budget')
                    except ValueError as error:retrieval.update(admitted=False,reason=str(error))
                    else:ids=enriched;retrieval['task_quality_certified']=True
            tick=perf_counter()
            expanded=expand_input(lm_head,ids,vision_carriers,scope=scope,signature=signature,
                source=(context['scene_id'],context['frame_id'],timestamp,index))
            preparation_ms=(perf_counter()-tick)*1000;tick=perf_counter()
            matched=self.cache.match(expanded)
            match_ms=(perf_counter()-tick)*1000
            allowed,reason=self.scheduler.prefix_allowed(expanded,matched,context,match_ms+signature_ms/max(1,len(ids_list)))
            prefix=matched if allowed else 0
            answer,diag=self.executor.generate(base,expanded,reuse_length=prefix,capture=True,
                profile=context.get('profile',False),**generation)
            diag.update(mode='prefix_prefill' if prefix else 'full_prefill',matched_prefix=matched,
                prefix_admission=reason,state_retrieval=retrieval,visual_tokens=int(expanded.visual_mask.sum()),
                current_visual_complete=True,preparation_cpu_ms=preparation_ms,match_cpu_ms=match_ms,
                signature_cpu_ms=signature_ms/max(1,len(ids_list)),executor='PrefillExecutor')
            answers.append(answer);rows.append(diag)
        return answers,dict(mode='exact_prefill',requests=rows,cache_bytes=self.cache.nbytes,config=asdict(self.config),
                           effective_history_records=sum(len(r['state_retrieval']['records']) for r in rows if r['state_retrieval']['admitted']))
