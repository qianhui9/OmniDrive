"""Frozen history supplements full current vision; every request uses PrefillExecutor."""
from dataclasses import asdict
import hashlib
import json
import math
from time import perf_counter
import torch
import transformers
from .config import M3Config
from .visual_state import base_model,expand_input,execution_signature,deployment_key,carrier_layout_signature,CarrierLayout
from .prefix_kv_cache import PrefixKVCache
from .prefill_executor import PrefillExecutor
from .scheduler import Scheduler
from .world_state import query_world_state,insert_state_context
from .reuse_planner import ReusePlanner,WSDR_VERSION


class M3Runtime:
    def __init__(self,config=None):
        self.config=config if isinstance(config,M3Config) else M3Config(**(config or {}))
        self.cache=PrefixKVCache(self.config.memory_budget_mb*1024**2)
        self.executor=PrefillExecutor(self.cache)
        self.scheduler=Scheduler(self.config)
        self.planner=ReusePlanner(self.scheduler)
        self._scope=self._signature=self._timestamp=None
        self._state_version=self._frame_id=None
        self._frame_index=0
        self._deployment_key=None
        self.last_diagnostics={}
        from .stable_world import StableAnchorManager
        self.stable_anchors=StableAnchorManager(self.config.stable_anchor_max_frames)
        self.composer=None
        self.last_composed_carriers=None  # current-request observer reference, never history lookup
        from .semantic_core import SemanticQualifier
        self.semantic_qualifier=SemanticQualifier(self.config.semantic_qualification) if self.config.semantic_qualification else None

    def reset(self):
        self.cache.reset();self._scope=self._signature=self._timestamp=None
        self._state_version=self._frame_id=None
        self._frame_index=0
        self.stable_anchors.reset()
        if self.composer is not None:self.composer.reset()
        self.last_composed_carriers=None

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
    def generate(self,lm_head,input_ids_list,vision_carriers,context=None,questions=None,layout=None,verified_state=None,tokenizer=None,m2_diagnostics=None,stable_world_state=None,**generation):
        runtime_tick=perf_counter()
        rows=[]
        self.last_diagnostics=dict(runtime_entered=True,execution_completed=False,requests=rows)
        context=dict(context or {});context.setdefault('model_version',self.config.model_version)
        if not context['model_version']:context['model_version']=context.get('weights_sha256','')
        ids_list=list(input_ids_list);questions=list(questions) if questions is not None else ['']*len(ids_list)
        if len(questions)!=len(ids_list):raise ValueError('Question order must match input IDs')
        base=base_model(lm_head);self.validate_contract(base,generation)
        if layout is not None and vision_carriers is not None and vision_carriers.shape[1]!=layout.total:
            raise ValueError('M3 requires the complete declared current visual layout')
        cameras=context.get('camera_ids',());timestamp=context.get('timestamp')
        if (not context.get('scene_id') or not context.get('frame_id') or not cameras
                or len(set(cameras))!=len(cameras) or type(timestamp) not in (int,float) or not math.isfinite(timestamp)):
            raise ValueError('M3 requires a bound scene/frame/time/camera lifecycle')
        composer_ms=anchor_ms=anchor_update_ms=core_extraction_ms=0.
        self.last_composed_carriers=None
        anchor=None;anchor_hit=False
        if self.config.prefix_adapter:
            from .world_prefix_composer import WorldPrefixComposer
            if not isinstance(layout,CarrierLayout) or layout.groups != (('object',256),('ego',1),('map',256)):
                raise ValueError('WRCPS requires the native source carrier contract')
            if stable_world_state is None:
                raise ValueError('WRCPS requires current committed M1 stable state')
            tick=perf_counter()
            anchor,anchor_hit=self.stable_anchors.update(stable_world_state,
                dict(context,state_version=context.get('stable_state_version')))
            anchor_update_ms=(perf_counter()-tick)*1000
            core_extraction_ms=context.get('stable_snapshot_wall_ms',0.)
            anchor_ms=anchor_update_ms+core_extraction_ms
            tick=perf_counter()
            if self.composer is None:
                weight=base.get_input_embeddings().weight
                self.composer=WorldPrefixComposer.load_adapter(self.config.prefix_adapter,
                    base.config.hidden_size,weight.device,torch.float32)
            if self.composer.base_weights_sha256 != context.get('weights_sha256'):
                raise ValueError('WRCPS adapter base checkpoint mismatch')
            if (self.composer.semantic_qualification_sha256
                    and self.composer.semantic_qualification_sha256!=stable_world_state.qualification_id):
                raise ValueError('WRCPS adapter semantic qualification mismatch')
            if self.composer.revision() != self.composer.loaded_revision:
                raise ValueError('Loaded WRCPS adapter weights were modified')
            vision_carriers=self.composer(vision_carriers.to(dtype=self.composer.stable_queries.dtype),
                anchor,use_anchor_cache=True)
            if context.get('profile'):self.last_composed_carriers=vision_carriers
            layout=self.composer.layout
            composer_ms=(perf_counter()-tick)*1000
            context['adapter_sha256']=self.composer.adapter_sha256
            context['semantic_qualification_sha256']=stable_world_state.qualification_id
        start=perf_counter();signature=execution_signature(lm_head,context['model_version'])
        signature=(signature,WSDR_VERSION)
        scope=(str(context['scene_id']),tuple(cameras),context.get('state_epoch'))
        if anchor is not None:
            # Visual anchor metadata is checked at its causal span boundary.
            # A changed/invalid anchor can still reuse identical fixed TEXT;
            # equal null embeddings never legalize cross-frame visual reuse.
            signature+=(self.composer.adapter_sha256,self.composer.revision(),
                self.composer.implementation_signature(),layout.order_contract)
        invalidation=None
        if self._scope is None:invalidation='cold_cache'
        elif scope!=self._scope:invalidation='scene_camera_or_epoch_changed'
        elif signature!=self._signature:invalidation='model_execution_signature_changed'
        elif self._timestamp is not None and timestamp<=self._timestamp:
            same_frame=(timestamp==self._timestamp and context['frame_id']==self._frame_id
                and context.get('state_version')==self._state_version)
            if not same_frame:invalidation='nonmonotonic_timestamp'
        version=context.get('state_version')
        if type(version) is int and type(self._state_version) is int and version<self._state_version:
            invalidation='state_version_rollback'
        new_frame=context['frame_id']!=self._frame_id
        if invalidation:self.cache.reset()
        if signature!=self._signature or self._deployment_key is None:
            key=deployment_key(lm_head)
            self._deployment_key=(hashlib.sha256((key+self.composer.adapter_sha256+layout.order_contract
                +repr(self.composer.implementation_signature())+anchor.qualification_id
                +repr(tuple((n,str(p.dtype)) for n,p in self.composer.named_parameters()))
                +(self.semantic_qualifier.implementation_signature() if self.semantic_qualifier else '')).encode()).hexdigest()
                if anchor is not None else key)
        if invalidation:self._frame_index=0
        elif new_frame:self._frame_index+=1
        context['frame_index']=self._frame_index
        self._scope,self._signature,self._timestamp=scope,signature,timestamp
        self._state_version,self._frame_id=version,context['frame_id']
        context['execution_key']=self._deployment_key
        context['generation_key']=hashlib.sha256(json.dumps(dict(defaults=base.generation_config.to_dict(),
            overrides=generation),sort_keys=True,default=repr).encode()).hexdigest()
        context['layout_signature']=carrier_layout_signature(layout) if layout else ''
        context['dtype']=str(base.get_input_embeddings().weight.dtype)
        signature_ms=(perf_counter()-start)*1000
        answers=[]
        for index,(ids,question) in enumerate(zip(ids_list,questions)):
            question_tick=perf_counter();q_invalidation=invalidation if index==0 else None
            retrieval=query_world_state(verified_state,question,context)
            retrieval=dict(retrieval)
            if retrieval['admitted']:
                if not self.scheduler.history_enabled(context):
                    retrieval.update(admitted=False,reason='history_text_not_calibrated')
                elif tokenizer is None:retrieval.update(admitted=False,reason='missing_tokenizer')
                else:
                    ids=insert_state_context(ids,retrieval['text'],tokenizer)
                    visual=vision_carriers.shape[1] if vision_carriers is not None else 0
                    if ids.numel()-1+visual>base.config.max_position_embeddings:
                        raise ValueError('Historical context exceeds position budget')
            tick=perf_counter()
            expanded=expand_input(lm_head,ids,vision_carriers,scope=scope,signature=signature,
                source=(context['scene_id'],context['frame_id'],timestamp,index),carrier_layout=layout,
                stable_anchor=(anchor.anchor_id,anchor.stable_state_valid) if anchor else ())
            preparation_ms=(perf_counter()-tick)*1000
            had_cache=self.cache.entry is not None
            decision,plan=self.planner.plan(question,expanded,self.cache,retrieval,context,m2_diagnostics,
                overhead_ms=preparation_ms+(signature_ms+composer_ms+anchor_ms)/max(1,len(ids_list)))
            if had_cache and self.cache.entry is None:q_invalidation='cache_storage_revision_changed'
            matched=plan.get('matched_prefix_tokens',0)
            reason=decision.decision_reason
            source_frame=(self.cache.entry.computed_source[1] if decision.reuse_length
                and self.cache.entry is not None and len(self.cache.entry.computed_source)>1 else None)
            self.last_diagnostics['active_request_index'] = index
            self.executor.last_diagnostics = {}
            try:
                answer,diag=self.executor.generate(base,expanded,decision=decision,capture=True,
                    profile=context.get('profile',False),**generation)
            except Exception as error:
                rows.append(dict(self.executor.last_diagnostics, request_index=index,
                    executor='PrefillExecutor', matched_prefix=matched, prefix_admission=reason,
                    state_retrieval=retrieval, error_type=type(error).__name__, error=str(error)))
                raise
            diag.update(plan,**decision.diagnostics())
            reused_visual=int(expanded.visual_mask[:,:diag['reused_tokens']].sum())
            diag.update(request_index=index,mode='prefix_prefill' if diag['reused_tokens'] else 'full_prefill',
                matched_prefix=matched,prefix_admission=reason,state_retrieval=retrieval,
                full_prefill_reason=reason if not diag['reused_tokens'] else None,
                visual_tokens=int(expanded.visual_mask.sum()),
                visual_group_spans=expanded.visual_group_spans,
                historical_text_consumed=bool(retrieval['admitted']),current_visual_complete=True,
                executor='PrefillExecutor',reuse_strategy=self.config.reuse_strategy,
                prefix_rejection_reason=reason if not decision.reuse_length else None,
                cache_invalidation_reason=q_invalidation,layout_preparation_ms=preparation_ms,
                composer_wall_ms=composer_ms/max(1,len(ids_list)),
                stable_anchor_wall_ms=anchor_ms/max(1,len(ids_list)),
                stable_core_extraction_wall_ms=core_extraction_ms/max(1,len(ids_list)),
                anchor_update_wall_ms=anchor_update_ms/max(1,len(ids_list)),
                stable_anchor_id=anchor.anchor_id if anchor else None,
                stable_anchor_hit=anchor_hit,stable_anchor_valid=anchor.stable_state_valid if anchor else False,
                stable_anchor_source_frame=anchor.source_frame if anchor else None,
                stable_anchor_age=anchor.anchor_age if anchor else None,
                semantic_qualification_sha256=anchor.qualification_id if anchor else None,
                stable_anchor_update_reason=anchor.update_reason if anchor and not anchor_hit else None,
                stable_prefix_tokens=layout.stable_tokens if anchor else 0,
                adapter_sha256=context.get('adapter_sha256'),
                planned_actual_reuse_equal=decision.reuse_length==diag['reused_tokens'],
                actual_exact_visual_kv_tokens=reused_visual if decision.reuse_type=='exact' else 0,
                crossframe_exact_visual_kv_tokens=(reused_visual if decision.reuse_type=='exact'
                    and source_frame is not None and source_frame!=context['frame_id'] else 0),
                prefix_matching_ms=plan['planning_ms'],
                cache_branch_ms=diag.get('cache_branch_gpu_span_ms'),
                prefill_ms=diag.get('prefill_including_cache_gpu_ms'),decode_ms=diag.get('decode_loop_gpu_span_ms'),
                end_to_end_ms=(perf_counter()-question_tick)*1000,
                diagnostic_timing_scope='M3 question host wall; CUDA spans only with profile=True')
            answers.append(answer);rows.append(diag)
        runtime_elapsed_ms=(perf_counter()-runtime_tick)*1000
        self.last_diagnostics.update(mode='wrcps_eq_ssc_exact' if anchor is not None else 'quality_bounded_wsdr',execution_completed=True,
            reuse_strategy=self.config.reuse_strategy,runtime_host_elapsed_ms=runtime_elapsed_ms,
            complete_m3_wall_ms=runtime_elapsed_ms+core_extraction_ms,
            cache_bytes=self.cache.nbytes,config=asdict(self.config),
            composer_wall_ms=composer_ms,stable_anchor_wall_ms=anchor_ms,
            stable_core_extraction_wall_ms=core_extraction_ms,anchor_update_wall_ms=anchor_update_ms,
            effective_history_records=sum(len(r['state_retrieval']['records']) for r in rows if r['historical_text_consumed']))
        return answers,self.last_diagnostics
