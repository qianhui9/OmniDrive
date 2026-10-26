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
from .reuse_planner import ReusePlanner,WSDR_VERSION,organize_carriers,GROUPS


class M3Runtime:
    def __init__(self,config=None):
        self.config=config if isinstance(config,M3Config) else M3Config(**(config or {}))
        self.cache=PrefixKVCache(self.config.memory_budget_mb*1024**2)
        self.executor=PrefillExecutor(self.cache)
        self.scheduler=Scheduler(self.config)
        self.planner=ReusePlanner()
        self._scope=self._signature=self._timestamp=None
        self._state_version=self._frame_id=None
        self.last_diagnostics={}

    def reset(self):
        self.cache.reset();self._scope=self._signature=self._timestamp=None
        self._state_version=self._frame_id=None

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
    def generate(self,lm_head,input_ids_list,vision_carriers,context=None,questions=None,layout=None,verified_state=None,tokenizer=None,m2_diagnostics=None,**generation):
        runtime_tick=perf_counter()
        rows=[]
        self.last_diagnostics=dict(runtime_entered=True,execution_completed=False,requests=rows)
        context=dict(context or {});context.setdefault('model_version',self.config.model_version)
        wsdr=self.config.reuse_strategy=='wsdr_kv'
        if wsdr and not context['model_version']:context['model_version']=context.get('weights_sha256','')
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
        if wsdr:signature=(signature,WSDR_VERSION)
        scope=(str(context['scene_id']),tuple(cameras),context.get('state_epoch'))
        invalidation=None
        if self._scope is None:invalidation='cold_cache'
        elif scope!=self._scope:invalidation='scene_camera_or_epoch_changed'
        elif signature!=self._signature:invalidation='model_execution_signature_changed'
        elif self._timestamp is not None and timestamp<=self._timestamp:
            same_frame=(wsdr and timestamp==self._timestamp and context['frame_id']==self._frame_id
                and context.get('state_version')==self._state_version)
            if not same_frame:invalidation='nonmonotonic_timestamp'
        version=context.get('state_version')
        if wsdr and type(version) is int and type(self._state_version) is int and version<self._state_version:
            invalidation='state_version_rollback'
        if invalidation:self.cache.reset()
        self._scope,self._signature,self._timestamp=scope,signature,timestamp
        self._state_version,self._frame_id=version,context['frame_id']
        # Historical text has its own quality gate, independent of exact KV equality.
        if wsdr or any((self.config.cost_table,self.config.quality_certificate,self.config.state_quality_certificate)):
            context['execution_key']=deployment_key(lm_head)
        context['generation_key']=hashlib.sha256(json.dumps(dict(defaults=base.generation_config.to_dict(),
            overrides=generation),sort_keys=True,default=repr).encode()).hexdigest()
        signature_ms=(perf_counter()-start)*1000
        answers=[]
        for index,(ids,question) in enumerate(zip(ids_list,questions)):
            question_tick=perf_counter();q_invalidation=invalidation if index==0 else None
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
                source=(context['scene_id'],context['frame_id'],timestamp,index),
                **(dict(carrier_layout=layout) if wsdr and layout is not None else {}))
            plan=None
            if wsdr:
                # Verify storage revisions before planner reads group views.
                had_stored=self.cache.entry is not None
                self.cache.match(expanded)
                if had_stored and self.cache.entry is None:q_invalidation='cache_storage_revision_changed'
                previous=self.cache.entry.expanded if self.cache.entry is not None else None
                plan=self.planner.plan(question,expanded,layout,previous=previous,history=retrieval,
                    context=context,m2_diagnostics=m2_diagnostics)
                if plan.layout_permutation!=GROUPS:
                    organized=organize_carriers(vision_carriers,layout,plan)
                    expanded=expand_input(lm_head,ids,organized,scope=scope,signature=signature,
                        source=expanded.source,carrier_layout=layout,carrier_order=plan.layout_permutation)
                expanded.layout_signature=plan.layout_signature
                if not plan.m1_state_valid:
                    self.cache.reset();q_invalidation=q_invalidation or 'm1_snapshot_unavailable_or_invalid'
                elif previous is not None and previous.layout_signature!=plan.layout_signature:
                    self.cache.reset();q_invalidation='layout_signature_changed'
                previous=None
                context.update(plan.diagnostics())
            preparation_ms=(perf_counter()-tick)*1000;tick=perf_counter()
            had_cache=self.cache.entry is not None
            matched=self.cache.match(expanded)
            if had_cache and self.cache.entry is None:q_invalidation='cache_storage_revision_changed'
            match_ms=(perf_counter()-tick)*1000
            context['effective_reuse_length']=self.executor.effective_reuse_length(expanded,matched)
            allowed,reason=self.scheduler.prefix_allowed(expanded,matched,context,
                match_ms+(preparation_ms if wsdr else 0)+signature_ms/max(1,len(ids_list)))
            prefix=matched if allowed else 0
            self.last_diagnostics['active_request_index'] = index
            self.executor.last_diagnostics = {}
            try:
                answer,diag=self.executor.generate(base,expanded,reuse_length=prefix,capture=True,
                    profile=context.get('profile',False),**generation)
            except Exception as error:
                rows.append(dict(self.executor.last_diagnostics, request_index=index,
                    executor='PrefillExecutor', matched_prefix=matched, prefix_admission=reason,
                    state_retrieval=retrieval, error_type=type(error).__name__, error=str(error)))
                raise
            reused=diag['reused_tokens']
            diag.update(request_index=index,mode='prefix_prefill' if reused else 'full_prefill',matched_prefix=matched,
                prefix_admission=reason,state_retrieval=retrieval,visual_tokens=int(expanded.visual_mask.sum()),
                historical_text_consumed=bool(retrieval['admitted'] and diag['prefill_completed']),
                full_prefill_reason=(reason if not prefix else 'prefix_adjusted_for_last_token_logits') if not reused else None,
                current_visual_complete=True,preparation_cpu_ms=preparation_ms,match_cpu_ms=match_ms,
                signature_cpu_ms=signature_ms/max(1,len(ids_list)),executor='PrefillExecutor')
            if plan is not None:diag.update(plan.diagnostics())
            diag.update(reuse_strategy=self.config.reuse_strategy,
                matched_prefix_tokens=matched,approved_prefix_tokens=prefix,
                prefix_rejection_reason=None if allowed else reason,cache_invalidation_reason=q_invalidation,
                layout_fallback_reason=plan.fallback_reason if plan else None,
                layout_preparation_ms=preparation_ms,prefix_matching_ms=match_ms,
                cache_branch_ms=diag.get('cache_branch_gpu_span_ms'),
                prefill_ms=diag.get('prefill_including_cache_gpu_ms'),decode_ms=diag.get('decode_loop_gpu_span_ms'),
                end_to_end_ms=(perf_counter()-question_tick)*1000,
                diagnostic_timing_scope='M3 question host wall; prefill/decode/branch CUDA spans only in profiling mode')
            answers.append(answer);rows.append(diag)
        self.last_diagnostics.update(mode='exact_prefill',execution_completed=True,
            reuse_strategy=self.config.reuse_strategy,runtime_host_elapsed_ms=(perf_counter()-runtime_tick)*1000,
            cache_bytes=self.cache.nbytes,config=asdict(self.config),
            effective_history_records=sum(len(r['state_retrieval']['records']) for r in rows if r['historical_text_consumed']))
        return answers,self.last_diagnostics
