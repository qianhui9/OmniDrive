"""Frozen history supplements full current vision; every request uses PrefillExecutor."""
from dataclasses import asdict,replace
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
from .answer_memory import AnswerMemory


def language_request_completed(runtime,row,current_world=None):
    """Explicit selective completion; default deployment still requires LLM."""
    if row.get('executor')=='PrefillExecutor':
        return bool(row.get('prefill_completed') and row.get('decode_completed'))
    p=row.get('fast_answer') or {}
    return bool(runtime is not None and runtime.config.selective_answer_mode=='online'
        and runtime.fast_answer_policy and current_world is not None
        and row.get('executor')=='QuestionReferencedFactReuse' and row.get('llm_skipped')
        and row.get('llm_forward_calls')==0 and row.get('layout_preparation_ms')==0.
        and not row.get('prefill_completed') and not row.get('decode_completed')
        and p.get('supported') and p.get('admitted') and p.get('facts')
        and tuple(p.get('current_commit',()))==(current_world.epoch,current_world.version,current_world.frame_id)
        and all(f.get('evidence_origin')=='actual_perception' and f.get('source_frame')==current_world.frame_id
                and f.get('source_timestamp')==current_world.timestamp
                and f.get('state_version')==current_world.version for f in p['facts']))


class M3Runtime:
    def __init__(self,config=None):
        self.config=config if isinstance(config,M3Config) else M3Config(**(config or {}))
        self.cache=PrefixKVCache(self.config.memory_budget_mb*1024**2)
        self.executor=PrefillExecutor(self.cache)
        self.answer_memory=AnswerMemory()
        self.fast_answer_policy=None
        if self.config.selective_answer_policy:
            from pathlib import Path
            self.fast_answer_policy=json.loads(Path(self.config.selective_answer_policy).read_text())
            p=self.fast_answer_policy
            if p.get('main_path','legacy_reproduction') not in ('current_only','legacy_reproduction'):
                raise ValueError('Unknown selective-answer main path')
            if p.get('main_path')=='current_only' and (
                    self.config.temporal_risk_enabled or self.config.target_completeness_enabled
                    or p.get('temporal_risk_enabled') or p.get('target_completeness_enabled')):
                raise ValueError('Current-only bypass cannot enable historical risk modules')
            if p.get('main_path')=='current_only':
                mode=p.get('position_mode','basic')
                floor=p.get('position_min_confidence')
                if mode not in ('basic','restricted','fallback') or (mode=='restricted' and
                        (not isinstance(floor,(int,float)) or not math.isfinite(floor) or not .35<=floor<=1.)):
                    raise ValueError('Invalid current-only Position policy')
            if (p.get('status')!='FROZEN_RESEARCH_POLICY' or not .35<=p.get('min_confidence',0)<=1.
                    or not math.isfinite(p.get('expected_normal_m3_ms',float('nan')))
                    or p['expected_normal_m3_ms']<=0 or not p.get('calibration_sha256')):
                raise ValueError('Invalid selective-answer development policy')
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
        self._adaptive_adapter_hash=''
        if self.config.prefix_layout in ('sf_adaptive','world_compact_memory') and self.config.prefix_adapter:
            from pathlib import Path
            self._adaptive_adapter_hash=hashlib.sha256(Path(self.config.prefix_adapter).read_bytes()).hexdigest()
        self.last_composed_carriers=None  # current-request observer reference, never history lookup
        self._decode_model_revision=self._decode_model_key=None
        self._decode_tokenizer_revision=self._decode_tokenizer_key=None
        self._decode_previous_world=None
        from .semantic_core import SemanticQualifier
        self.semantic_qualifier=SemanticQualifier(self.config.semantic_qualification) if self.config.semantic_qualification else None
        from .visual_memory import VisualMemoryManager
        self.visual_memories=VisualMemoryManager(8, self.config.stable_anchor_max_frames)

    def reset(self):
        self.answer_memory.reset()
        self.cache.reset();self._scope=self._signature=self._timestamp=None
        self._state_version=self._frame_id=None
        self._frame_index=0
        self.stable_anchors.reset()
        self.visual_memories.reset()
        if self.composer is not None:self.composer.reset()
        self.last_composed_carriers=None
        self._decode_previous_world=None

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
    def generate(self,lm_head,input_ids_list,vision_carriers,context=None,questions=None,layout=None,verified_state=None,tokenizer=None,m2_diagnostics=None,stable_world_state=None,visual_memory_inputs=None,committed_world_state=None,**generation):
        runtime_tick=perf_counter()
        rows=[]
        self.last_diagnostics=dict(runtime_entered=True,execution_completed=False,requests=rows)
        context=dict(context or {});context.setdefault('model_version',self.config.model_version)
        if not context['model_version']:context['model_version']=context.get('weights_sha256','')
        ids_list=list(input_ids_list);questions=list(questions) if questions is not None else ['']*len(ids_list)
        if len(questions)!=len(ids_list):raise ValueError('Question order must match input IDs')
        base=base_model(lm_head);self.validate_contract(base,generation)
        history_decode=self.config.decode_strategy in ('history_draft','world_guided')
        if layout is not None and vision_carriers is not None and vision_carriers.shape[1]!=layout.total:
            raise ValueError('M3 requires the complete declared current visual layout')
        cameras=context.get('camera_ids',());timestamp=context.get('timestamp')
        if (not context.get('scene_id') or not context.get('frame_id') or not cameras
                or len(set(cameras))!=len(cameras) or type(timestamp) not in (int,float) or not math.isfinite(timestamp)):
            raise ValueError('M3 requires a bound scene/frame/time/camera lifecycle')
        if history_decode:
            return self._decode_current_frame(lm_head,base,ids_list,questions,vision_carriers,
                layout,context,tokenizer,committed_world_state,generation,runtime_tick)
        composer_ms=anchor_ms=anchor_update_ms=core_extraction_ms=0.
        self.last_composed_carriers=None
        anchor=None;anchor_hit=False
        adaptive=self.config.prefix_layout=='sf_adaptive'
        layout_mode='native_full_visual'
        adapter_binding=()
        compact=self.config.prefix_layout=='world_compact_memory'
        bridge_timings={};memory_event=None
        if compact:
            from types import SimpleNamespace
            from .visual_feature_bridge import update_visual_memory
            from .world_prefix_composer import CompactVisualComposer
            if not isinstance(layout,CarrierLayout) or layout.groups != (('object',256),('ego',1),('map',256)):
                raise ValueError('Compact memory requires native source carrier')
            if stable_world_state is None:
                raise ValueError('Compact memory requires current M1 snapshot')
            stable_world_state.validate(dict(context,state_version=context.get('stable_state_version')))
            if not stable_world_state.qualification_id:
                raise ValueError('Compact memory requires independently qualified M1 identities')
            qualification=stable_world_state.qualification_id
            if self.semantic_qualifier is None or qualification!=self.semantic_qualifier.sha256:
                raise ValueError('Compact qualification does not match configured policy')
            adapter_sha=self._adaptive_adapter_hash
            memory_context=dict(context,state_version=stable_world_state.state_version,
                model_sha256=context.get('weights_sha256',''),adapter_sha256=adapter_sha,
                memory_version='wr_cvms_real_region_v1')
            qualified_ids=[aid for aid,kind,_ in stable_world_state.facts if kind=='object']
            if qualified_ids:
                if visual_memory_inputs is None:
                    raise ValueError('Qualified compact candidates require real EVA source')
                frame=visual_memory_inputs['frame']
                if (frame.scene_id,frame.frame_id,frame.timestamp)!=(context['scene_id'],context['frame_id'],timestamp):
                    raise ValueError('Compact visual source frame mismatch')
                memory,memory_event,_,bridge_timings=update_visual_memory(self.visual_memories,
                    frame,visual_memory_inputs['anchors'],visual_memory_inputs['feature_map'],
                    visual_memory_inputs['config'],memory_context,qualified_ids,qualification)
            else:
                tick=perf_counter()
                memory,memory_event=self.visual_memories.update(memory_context,[],[],qualification)
                bridge_timings=dict(bridge_total_host_ms=(perf_counter()-tick)*1000)
            anchor_ms=bridge_timings['bridge_total_host_ms']
            core_extraction_ms=context.get('stable_snapshot_wall_ms',0.)
            anchor_ms+=core_extraction_ms
            if memory is not None:
                if not self.config.prefix_adapter:
                    raise ValueError('Compact memory cost scaffold is not a deployable trained adapter')
                if self.composer is None:
                    self.composer=CompactVisualComposer.load_adapter(self.config.prefix_adapter,
                        base.config.hidden_size,vision_carriers.device)
                if (self.composer.adapter_sha256!=adapter_sha
                        or self.composer.base_weights_sha256!=context.get('weights_sha256')
                        or self.composer.semantic_qualification_sha256!=qualification
                        or self.composer.revision()!=self.composer.loaded_revision):
                    raise ValueError('Compact adapter binding/revision mismatch')
                tick=perf_counter()
                vision_carriers=self.composer(vision_carriers,memory)
                composer_ms=(perf_counter()-tick)*1000
                layout=self.composer.layout;layout_mode='world_compact_memory'
            anchor=SimpleNamespace(anchor_id=repr((memory.scope,memory.revision)) if memory else 'invalid',
                stable_state_valid=memory is not None,qualification_id=qualification,
                source_frame=memory.generating_frame if memory else None,
                anchor_age=memory.age if memory else 0,update_reason=memory_event)
            anchor_hit=memory_event in ('keep','same_frame')
            adapter_binding=(adapter_sha,'wr_cvms_real_region_v1')
            context.update(adapter_sha256=adapter_sha,semantic_qualification_sha256=qualification)
            if context.get('profile'):self.last_composed_carriers=vision_carriers
        elif self.config.prefix_adapter or adaptive:
            from .world_prefix_composer import WorldPrefixComposer,SpatiallyFaithfulComposer
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
            if not adaptive or anchor.stable_state_valid:
                if not self.config.prefix_adapter:
                    raise ValueError('Valid SF-WRCPS core requires a trained spatial adapter')
                if self.composer is None:
                    weight=base.get_input_embeddings().weight
                    self.composer=WorldPrefixComposer.load_adapter(self.config.prefix_adapter,
                        base.config.hidden_size,weight.device,torch.float32)
                if adaptive != isinstance(self.composer,SpatiallyFaithfulComposer):
                    raise ValueError('Adapter and requested adaptive visual layout mismatch')
                if self.composer.base_weights_sha256 != context.get('weights_sha256'):
                    raise ValueError('WRCPS adapter base checkpoint mismatch')
                if (self.composer.semantic_qualification_sha256
                        and self.composer.semantic_qualification_sha256!=stable_world_state.qualification_id):
                    raise ValueError('WRCPS adapter semantic qualification mismatch')
                if self.composer.revision() != self.composer.loaded_revision:
                    raise ValueError('Loaded WRCPS adapter weights were modified')
                vision_carriers=self.composer(vision_carriers.to(dtype=self.composer.stable_queries.dtype),
                    anchor,use_anchor_cache=True)
                layout=self.composer.layout
                layout_mode='stable_core_visual'
                adapter_binding=(self.composer.adapter_sha256,self.composer.revision(),
                    self.composer.implementation_signature(),layout.order_contract)
            else:
                # No learned computation, casting, padding, permutation or
                # query conditioning: this is the SAME native tensor/layout.
                adapter_binding=(self._adaptive_adapter_hash,'sf_native_identity_v1')
            if context.get('profile'):self.last_composed_carriers=vision_carriers
            composer_ms=(perf_counter()-tick)*1000
            context['adapter_sha256']=self.composer.adapter_sha256 if layout_mode=='stable_core_visual' else self._adaptive_adapter_hash
            context['semantic_qualification_sha256']=stable_world_state.qualification_id
        start=perf_counter();signature=execution_signature(lm_head,context['model_version'])
        signature=(signature,WSDR_VERSION)
        scope=(str(context['scene_id']),tuple(cameras),context.get('state_epoch'))
        if anchor is not None:
            # Visual anchor metadata is checked at its causal span boundary.
            # A changed/invalid anchor can still reuse identical fixed TEXT;
            # equal null embeddings never legalize cross-frame visual reuse.
            signature+=(adapter_binding,layout_mode,anchor.qualification_id,carrier_layout_signature(layout))
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
        if invalidation:
            self.cache.reset()
            self.answer_memory.reset()
        if signature!=self._signature or self._deployment_key is None:
            key=deployment_key(lm_head)
            self._deployment_key=(hashlib.sha256((key+repr(adapter_binding)+layout_mode
                +carrier_layout_signature(layout)+anchor.qualification_id
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
        if history_decode:
            tokenizer_tick=perf_counter()
            tokenizer_key=hashlib.sha256(repr((type(tokenizer).__qualname__,
                getattr(tokenizer,'name_or_path',None),getattr(tokenizer,'special_tokens_map',None),
                sorted(tokenizer.get_vocab().items()) if tokenizer is not None else None)).encode()).hexdigest()
            answer_binding=(signature,tokenizer_key,context['generation_key'])
            tokenizer_binding_ms=(perf_counter()-tokenizer_tick)*1000
        context['layout_signature']=carrier_layout_signature(layout) if layout else ''
        context['dtype']=str(base.get_input_embeddings().weight.dtype)
        signature_ms=(perf_counter()-start)*1000
        answers=[]
        for index,(ids,question) in enumerate(zip(ids_list,questions)):
            question_tick=perf_counter();q_invalidation=invalidation if index==0 else None
            retrieval=query_world_state(verified_state,question,context)
            retrieval=dict(retrieval)
            if history_decode:
                retrieval.update(admitted=False,reason='history_draft_keeps_native_prompt')
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
            draft=None
            selection_ms=0.
            if history_decode:
                from .reuse_planner import ReuseDecision
                decision=ReuseDecision(decision_reason='history_decode_current_full_prefill',frame_index=self._frame_index)
                tick=perf_counter()
                prompt_ids=tuple(ids.reshape(-1).tolist())
                record=self.answer_memory.select(prompt_ids,context,answer_binding)
                draft=record.tokens if record else None
                selection_ms=(perf_counter()-tick)*1000
            if had_cache and self.cache.entry is None:q_invalidation='cache_storage_revision_changed'
            matched=plan.get('matched_prefix_tokens',0)
            reason=decision.decision_reason
            source_frame=(self.cache.entry.computed_source[1] if decision.reuse_length
                and self.cache.entry is not None and len(self.cache.entry.computed_source)>1 else None)
            self.last_diagnostics['active_request_index'] = index
            self.executor.last_diagnostics = {}
            try:
                answer,diag=self.executor.generate(base,expanded,decision=decision,capture=True,
                    profile=context.get('profile',False),draft_tokens=draft,draft_length=self.config.draft_length,
                    experimental_sampling=self.config.experimental_speculative_sampling,**generation)
            except Exception as error:
                rows.append(dict(self.executor.last_diagnostics, request_index=index,
                    executor='PrefillExecutor', matched_prefix=matched, prefix_admission=reason,
                    state_retrieval=retrieval, error_type=type(error).__name__, error=str(error)))
                raise
            diag.update(plan,**decision.diagnostics())
            if history_decode:
                tick=perf_counter()
                tokens=answer.sequences if hasattr(answer,'sequences') else answer
                self.answer_memory.remember(question,prompt_ids,tokens[0,1:].tolist(),context,answer_binding)
                diag.update(decode_strategy='history_draft',draft_selection_wall_ms=selection_ms,
                    draft_management_wall_ms=(perf_counter()-tick)*1000,
                    tokenizer_binding_wall_ms=tokenizer_binding_ms/max(1,len(ids_list)),
                    draft_source_frame=record.frame_id if record else None)
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
                stable_prefix_tokens=getattr(layout,'stable_tokens',0) if anchor else 0,
                visual_layout_mode=layout_mode,
                visual_memory_event=memory_event,visual_bridge_timings=bridge_timings,
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
        self.last_diagnostics.update(mode='sf_wrcps' if adaptive else 'wrcps_eq_ssc_exact' if anchor is not None else 'quality_bounded_wsdr',visual_layout_mode=layout_mode,execution_completed=True,
            reuse_strategy=self.config.reuse_strategy,runtime_host_elapsed_ms=runtime_elapsed_ms,
            complete_m3_wall_ms=runtime_elapsed_ms+core_extraction_ms,
            cache_bytes=self.cache.nbytes,config=asdict(self.config),
            composer_wall_ms=composer_ms,stable_anchor_wall_ms=anchor_ms,
            stable_core_extraction_wall_ms=core_extraction_ms,anchor_update_wall_ms=anchor_update_ms,
            effective_history_records=sum(len(r['state_retrieval']['records']) for r in rows if r['historical_text_consumed']))
        return answers,self.last_diagnostics

    def _decode_binding(self,lm_head,base,context,tokenizer,generation):
        """Cache expensive metadata, while checking real mutable revisions."""
        revision=(context['model_version'],repr(base.config.to_dict()),
            tuple((id(p),p._version,p.data_ptr(),p.dtype,p.device) for p in base.parameters()),
            tuple((id(b),b._version,b.data_ptr()) for n,b in base.named_buffers()
                  if not n.endswith(('cos_cached','sin_cached'))),
            tuple((id(m),repr(getattr(m,'active_adapters',None)),repr(getattr(m,'disable_adapters',None)),
                   repr(getattr(m,'merged_adapters',None))) for m in base.modules() if hasattr(m,'lora_A')),
            tuple(id(getattr(m.forward,'__func__',m.forward)) for m in base.modules()),
            torch.is_autocast_enabled(),torch.get_autocast_gpu_dtype(),
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        model_hit=revision==self._decode_model_revision
        if not model_hit:
            self._decode_model_key=hashlib.sha256(repr(execution_signature(lm_head,context['model_version'])).encode()).hexdigest()
            self._decode_model_revision=revision
        if tokenizer is None:vocab=None
        elif hasattr(tokenizer,'sp_model'):
            vocab=hashlib.sha256(tokenizer.sp_model.serialized_model_proto()).hexdigest()
            vocab=(vocab,repr(tokenizer.get_added_vocab()))
        elif hasattr(tokenizer,'backend_tokenizer'):vocab=tokenizer.backend_tokenizer.to_str()
        else:vocab=sorted(tokenizer.get_vocab().items())
        tokenizer_revision=(id(tokenizer),repr(getattr(tokenizer,'special_tokens_map',None)),repr(vocab))
        tokenizer_hit=tokenizer_revision==self._decode_tokenizer_revision
        if not tokenizer_hit:
            self._decode_tokenizer_key=hashlib.sha256(repr(tokenizer_revision).encode()).hexdigest()
            self._decode_tokenizer_revision=tokenizer_revision
        genkey=hashlib.sha256(json.dumps(dict(defaults=base.generation_config.to_dict(),overrides=generation),
                                       sort_keys=True,default=repr).encode()).hexdigest()
        return (self._decode_model_key,self._decode_tokenizer_key,genkey),model_hit,tokenizer_hit

    def _decode_current_frame(self,lm_head,base,ids_list,questions,carriers,layout,context,
                              tokenizer,current_world,generation,runtime_tick):
        """History and world-guided proposals share the same current-context verifier.

        No prefix planner, inherited visual prefix, or persistent prefill K/V.
        This branch begins before any legacy prefix-layout/signature work.
        """
        guided=self.config.decode_strategy=='world_guided'
        if current_world is not None:current_world.validate(context)
        tick=perf_counter()
        binding,model_hit,tokenizer_hit=self._decode_binding(lm_head,base,context,tokenizer,generation)
        metadata_ms=(perf_counter()-tick)*1000
        scope=(context['scene_id'],tuple(context['camera_ids']),context.get('state_epoch'))
        same_frame=context['frame_id']==self._frame_id and context['timestamp']==self._timestamp
        invalid=(scope!=self._scope or binding[0]!=self._signature or
                 (self._timestamp is not None and context['timestamp']<=self._timestamp and not same_frame))
        if invalid:
            self.answer_memory.reset();self.cache.reset();self._frame_index=0;self._decode_previous_world=None
        elif not same_frame:self._frame_index+=1
        self._scope,self._signature=scope,binding[0]
        self._frame_id,self._timestamp=context['frame_id'],context['timestamp']
        context['frame_index']=self._frame_index
        context['current_state_version']=current_world.version if current_world else context.get('current_state_version')
        answers=[];rows=self.last_diagnostics['requests']
        for index,(ids,question) in enumerate(zip(ids_list,questions)):
            question_tick=perf_counter()
            tick=perf_counter();prompt_ids=tuple(ids.reshape(-1).tolist())
            self.last_diagnostics['active_request_index']=index
            proposal=None
            if self.config.selective_answer_mode!='off':
                from .answer_validity import propose_fact_answer
                proposal=propose_fact_answer(question,current_world,context,self.answer_memory,binding,
                    self.fast_answer_policy,self._decode_previous_world,
                    target_binding_enabled=self.config.target_binding_enabled,
                    temporal_risk_enabled=self.config.temporal_risk_enabled,
                    target_completeness_enabled=self.config.target_completeness_enabled)
                if (proposal['admitted'] and self.config.selective_answer_mode=='online'
                        and tokenizer is not None and not generation.get('output_scores',False)):
                    construct_tick=perf_counter()
                    token_ids=list(tokenizer.encode(proposal['answer_text'],add_special_tokens=False))
                    eos=generation.get('eos_token_id',base.generation_config.eos_token_id)
                    if isinstance(eos,(list,tuple)):eos=eos[0] if eos else None
                    if eos is not None:token_ids.append(eos)
                    bos=base.generation_config.bos_token_id
                    answer=ids.new_tensor([[bos if bos is not None else 1]+token_ids])
                    if generation.get('return_dict_in_generate',False):
                        from transformers.generation.utils import SampleDecoderOnlyOutput,GreedySearchDecoderOnlyOutput
                        answer=(SampleDecoderOnlyOutput if generation.get('do_sample',False) else GreedySearchDecoderOnlyOutput)(sequences=answer)
                    if proposal['answer_source']=='full_answer_reuse':
                        key=(context['scene_id'],tuple(proposal['source_prompt_ids']))
                        source=self.answer_memory.records[key]
                        self.answer_memory.observe_validity(source,context['frame_id'],True,proposal['support_score'])
                    # Fact answers never enter target-generated AnswerMemory.
                    proposal['fast_answer_construction_wall_ms']+=(perf_counter()-construct_tick)*1000
                    diag=dict(request_index=index,executor='QuestionReferencedFactReuse',mode='fast_answer',
                        decode_strategy=self.config.decode_strategy,llm_skipped=True,llm_forward_calls=0,
                        fast_answer=proposal,answer_source=proposal['answer_source'],output_token_ids=token_ids,
                        input_tokens=0,visual_tokens=0,current_visual_complete=True,layout_preparation_ms=0.,
                        prefill_completed=False,decode_completed=False,reused_tokens=0,full_prefill=False,
                        prefill_including_cache_gpu_ms=0.,prefill_including_cache_cpu_enqueue_ms=0.,
                        prefill_including_cache_wall_ms=0.,decode_loop_gpu_span_ms=0.,decode_forward_cuda_ms=0.,
                        decode_forward_host_enqueue_ms=0.,decode_forward_calls=0,draft_status='not_run_fast_answer',
                        prefix_planner_executed=False,persistent_prefill_kv=False,historical_text_consumed=False,
                        metadata_wall_ms=metadata_ms/max(1,len(ids_list)),
                        end_to_end_ms=(perf_counter()-question_tick)*1000)
                    answers.append(answer);rows.append(diag)
                    continue
            selection=self.answer_memory.retrieve(question,prompt_ids,context,binding,current_world,
                guided and not self.config.target_aware_exact_only)
            retrieval_ms=(perf_counter()-tick)*1000
            retrieval_only_ms=retrieval_ms-selection.world_diff_ms
            if guided:
                selection=self.answer_memory.construct(selection,question,prompt_ids,context,binding,
                    None if self.config.target_aware_exact_only else current_world,tokenizer)
            draft=selection.record.tokens if selection.record is not None else None
            if guided:draft=None
            k=min(self.config.draft_length,selection.length) if draft else self.config.draft_length
            # A changed-state proposal is a short prefix, not permission to
            # keep proposing an entire stale answer with four-token blocks.
            if draft and selection.length<8:draft=draft[:selection.length]
            tick=perf_counter()
            expanded=expand_input(lm_head,ids,carriers,scope=scope,signature=(binding[0],),
                source=(context['scene_id'],context['frame_id'],context['timestamp'],index),carrier_layout=layout)
            preparation_ms=(perf_counter()-tick)*1000
            self.last_diagnostics['active_request_index']=index
            from .request_rng import question_rng
            with question_rng(context,index,ids.device,self.config.question_rng_enabled) as rng:
                answer,diag=self.executor.generate(base,expanded,capture=False,profile=context.get('profile',False),
                    draft_tokens=draft,draft_candidates=selection.candidates if guided else None,
                    draft_length=k,experimental_sampling=self.config.experimental_speculative_sampling,**generation)
            diag['question_rng']=rng
            tick=perf_counter()
            tokens=answer.sequences if hasattr(answer,'sequences') else answer
            token_ids=tokens[0,1:].tolist()
            answer_start=tokenizer.decode(token_ids if guided else token_ids[:8],skip_special_tokens=True) if tokenizer is not None else ''
            candidate_diag=[c.metadata() for c in selection.candidates]
            selected=diag.get('selected_candidate_index')
            if guided:
                c=selection.candidates[selected] if selected is not None else None
                selection=replace(selection,record=c.record if c else None,length=c.block_length if c else 0,
                    match_type=('exact' if c.kind=='exact' else 'semantic') if c else 'none',delta=c.delta if c else selection.delta)
            diag['output_token_ids']=token_ids
            diag['draft_selection_wall_ms']=retrieval_ms+selection.construction_ms+selection.tokenizer_ms
            self.answer_memory.observe_acceptance(selection,diag,self._frame_index)
            self.answer_memory.remember(question,prompt_ids,token_ids,context,binding,current_world,answer_start,extract_facts=guided)
            management_ms=(perf_counter()-tick)*1000
            diag.update(request_index=index,executor='PrefillExecutor',mode='full_prefill',decode_strategy=self.config.decode_strategy,
                fast_answer=proposal,llm_skipped=False,llm_forward_calls=1+diag.get('decode_forward_calls',0),
                answer_source='current_target_model_generation',
                current_visual_complete=True,visual_tokens=int(expanded.visual_mask.sum()),historical_text_consumed=False,
                prefix_planner_executed=False,persistent_prefill_kv=False,draft_match_type=selection.match_type,
                draft_exact_available=selection.exact_available,draft_semantic_available=selection.semantic_available,
                draft_source_frame=selection.record.frame_id if selection.record else None,
                draft_source_question=selection.record.question if selection.record else None,
                proposed_block_length=selection.length if selection.record is not None else 0,draft_decision_reason=selection.reason,
                draft_candidates=candidate_diag,draft_construction_wall_ms=selection.construction_ms,
                draft_tokenizer_wall_ms=selection.tokenizer_ms,
                world_delta=selection.delta.to_dict() if selection.delta else None,
                output_token_ids=token_ids,
                draft_selection_wall_ms=max(0.,retrieval_only_ms+selection.construction_ms+selection.tokenizer_ms),draft_management_wall_ms=management_ms,
                metadata_wall_ms=metadata_ms/max(1,len(ids_list)),model_metadata_cache_hit=model_hit,
                tokenizer_metadata_cache_hit=tokenizer_hit,layout_preparation_ms=preparation_ms,
                prefill_ms=diag.get('prefill_including_cache_gpu_ms'),decode_ms=diag.get('decode_loop_gpu_span_ms'),
                world_diff_wall_ms=selection.world_diff_ms,end_to_end_ms=(perf_counter()-question_tick)*1000)
            answers.append(answer);rows.append(diag)
        self._decode_previous_world=current_world
        elapsed=(perf_counter()-runtime_tick)*1000
        self.last_diagnostics.update(mode=self.config.decode_strategy,execution_completed=True,
            runtime_host_elapsed_ms=elapsed,complete_m3_wall_ms=elapsed,cache_bytes=self.cache.nbytes,
            config=asdict(self.config),numerical_contract='approximate_block_fp16',prefix_planner_executed=False)
        if any(r.get('llm_skipped') for r in rows):
            self.last_diagnostics['numerical_contract']='task_limited_approximate_fact_response; target_aware_fp16_on_misses'
        return answers,self.last_diagnostics
