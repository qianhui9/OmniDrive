"""WSGRR is the serving algorithm; every planned window executes and publishes."""
import copy
import math
from time import perf_counter
import torch
from .config import M2Config
from .dependency_planner import DependencyPlanner,block0_input,patch_updates
from .m1_adapter import computation_admission
from .state_cache import StateCache
from .window_executor import WindowExecutor,NUMERICAL_POLICY
from .region_refresh import pooled_descriptors,select_regions


def finite_time(value):
    return type(value) in (int,float) and math.isfinite(value)


class IncrementalVisionRuntime:
    def __init__(self,model,config=None):
        self.model=model
        self.config=config if isinstance(config,M2Config) else M2Config(**(config or {}))
        self.planner=DependencyPlanner(model)
        self.prefix=self.planner.cacheable_prefix()
        if self.planner.rejection_reason:raise ValueError(self.planner.rejection_reason)
        self.executor=WindowExecutor()
        self.cache=StateCache(self.config.memory_budget_mb*1024**2)
        self.stage_observer=None
        self.last_diagnostics={}

    def reset(self):
        self.cache.clear()

    def signature(self,context):
        tensors=tuple((n,t.data_ptr(),t._version,tuple(t.shape),str(t.dtype),str(t.device))
            for n,t in list(self.model.named_parameters())+list(self.model.named_buffers()))
        topology=tuple((b.window_size,b.use_residual_block,b.attn.flash_attn,b.attn.scale,
            b.norm1.eps,b.norm2.eps,getattr(b.attn.inner_attn_ln,'eps',None),getattr(b.mlp.ffn_ln,'eps',None),
            getattr(getattr(b.attn,'inner_attn',None),'softmax_scale',None),
            getattr(getattr(b.attn,'inner_attn',None),'fp16_enabled',None)) for b in self.model.blocks)
        p=self.model.patch_embed.proj
        return (context.get('execution_signature'),NUMERICAL_POLICY,self.config.refresh_signature(),tensors,topology,
            (p.kernel_size,p.stride,p.padding,p.dilation,self.model.pretrain_use_cls_token),
            torch.is_autocast_enabled(),torch.get_autocast_gpu_dtype(),
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.are_deterministic_algorithms_enabled())

    def forward_from_m1(self,images,manager,context):
        if manager is None:
            raise ValueError('missing_m1_manager')
        tick=perf_counter()
        snapshot=manager.export_m2_inputs(expected_context=context,computation_only=True)
        read_ms=(perf_counter()-tick)*1000
        result,diag=self.forward(images,m1_result=snapshot,execution_context=context)
        diag.update(m1_snapshot_read=True,M1_M2_state_read_ms=read_ms,
                    world_identity_authorized_pixel_reuse=False)
        return result,diag

    def evaluate_offline_from_m1(self,images,manager,context,*,variant='D',window_targets=None):
        # Historical tools share the serving executor and cache namespace.
        from tools.worldstream_m2.offline_variants import evaluate_from_m1
        return evaluate_from_m1(self,images,manager,context,variant,window_targets)

    def forward(self,images,m1_result=None,m1_state=None,execution_context=None,computation_state=None):
        return self._forward(images,m1_result,execution_context,computation_state)

    def evaluate_offline(self,images,*,variant='D',m1_result=None,execution_context=None,computation_state=None,window_targets=None):
        from tools.worldstream_m2.offline_variants import evaluate
        return evaluate(self,images,variant,m1_result,execution_context,computation_state,window_targets)

    def _plan_refresh(self,x0,bundle,context,uncertainty,mandatory):
        for i in self.prefix:
            expired=self.cache.window_ages[i]+1 >= self.config.max_reuse_frames
            mandatory |= self.planner.windows_to_tokens(expired,x0.shape[:3],self.model.blocks[i].window_size)
        selected,descriptors,regions,refresh=select_regions(x0,self.cache,bundle,context,
            self.model.patch_embed.proj,self.model.blocks[0].window_size,self.config,
            uncertainty,mandatory)
        return self.planner.plan_windows(selected,x0.shape[:3],stop_after=len(self.prefix)),descriptors,regions,refresh

    def _forward(self,images,m1_result,execution_context,computation_state):
        self.last_diagnostics=dict(runtime_entered=True,execution_completed=False)
        start=perf_counter();context=dict(execution_context or {})
        if any(m.training for m in self.model.modules()) or torch.is_grad_enabled():
            raise ValueError('M2 window execution is inference-only')
        if images.ndim!=4 or not images.is_floating_point():raise ValueError('Expected floating NCHW model input')
        self.prefix=self.planner.cacheable_prefix()
        if self.planner.rejection_reason:raise ValueError(self.planner.rejection_reason)
        cameras=context.get('camera_ids',())
        if len(cameras)!=len(images) or len(set(cameras))!=len(images):
            raise ValueError('One ordered camera identity is required per model input')
        bundle=computation_state if computation_state is not None else getattr(m1_result,'computation_state',None)
        footprints=self.planner.input_footprints(*images.shape[-2:],verified_prefix=self.prefix)
        signature=self.signature(context)
        previous=copy.deepcopy(self.cache.context) if isinstance(self.cache.context,dict) else None
        source_times=[]
        invalid=None;reason=None;details={}
        valid=bool(context.get('scene_id') and context.get('frame_id') and finite_time(context.get('timestamp'))
            and context.get('execution_signature') and type(context.get('state_epoch')) is int
            and context.get('coordinate_space')=='preprocessed_image')
        if not valid:raise ValueError('invalid_execution_context')
        if not self.config.enabled:raise ValueError('M2 runtime is disabled')
        if not self.prefix:raise ValueError('no_cacheable_prefix')
        if not torch.isfinite(images).all().item():raise ValueError('nonfinite_model_input')
        x0=block0_input(self.model,images)
        if not torch.isfinite(x0).all().item():raise ValueError('nonfinite_block0_input')
        mismatch=self.cache.mismatch(images,context,signature,x0)
        # Scene boundaries are lifecycle resets, not a second full executor.
        # Validate stored integrity before discarding history from a prior scene.
        if mismatch in ('cache_metadata_modified','cache_storage_modified'):
            raise ValueError(mismatch)
        if self.cache.context is not None and self.cache.context['scene_id']!=context['scene_id']:
            self.cache.clear()
            mismatch='scene_change'
        elif mismatch not in (None,'no_valid_cache'):
            raise ValueError(mismatch)
        if mismatch is None:
            source_times=[s.get('computation_timestamp') for units in self.cache.sources.values() for s in units]
        invalid,reason=computation_admission(bundle,context,images,self.cache,
            getattr(m1_result,'full_refresh_hint',False),details=details,window_footprints=footprints)
        if reason not in (None,'no_valid_cache','m1_full_refresh'):
            raise ValueError(reason)
        if reason is None and mismatch:reason=mismatch
        if mismatch=='scene_change':reason='scene_change'
        uncertainty=details.pop('_uncertainty_pixels',None)
        scoring_start=perf_counter();refresh={};descriptors=None;region_states=[]
        if reason is None:
            mandatory=patch_updates(torch.as_tensor(invalid,device=images.device,dtype=torch.bool),self.model.patch_embed.proj)
            plans,descriptors,region_states,refresh=self._plan_refresh(x0,bundle,context,uncertainty,mandatory)
        else:plans=None
        scoring_ms=(perf_counter()-scoring_start)*1000
        validation_ms=(perf_counter()-start)*1000
        execution_start=perf_counter()
        features,stages,outputs,sources,executed_ids=self._execute(x0,context,plans,reason)
        execution_ms=(perf_counter()-execution_start)*1000
        if self.prefix:
            if plans is None:
                refresh.update(mandatory_windows=stages[0]['total_windows'],ordinary_windows=0,
                    mandatory_window_ids=stages[0]['executed_window_ids'],ordinary_window_ids=[])
            missed=set(refresh.get('mandatory_window_ids',[]))-set(stages[0]['executed_window_ids'])
            refresh['event_missed_windows']=len(missed)
            mandatory=set(refresh.get('mandatory_window_ids',[]))
            ordinary=set(refresh.get('ordinary_window_ids',[]))
            selected=mandatory|ordinary
            actual=set(stages[0]['executed_window_ids'])
            refresh['selection_consistency']=dict(
                mandatory_ordinary_disjoint=not bool(mandatory & ordinary),
                selected_in_execution=selected.issubset(actual),
                mandatory_in_execution=mandatory.issubset(actual))
            per_camera=stages[0]['total_windows']//len(cameras)
            refresh['per_camera']=[dict(camera_id=cid,
                mandatory_windows=sum(j//per_camera==c for j in mandatory),
                ordinary_windows=sum(j//per_camera==c for j in ordinary),
                executed_windows=sum(j//per_camera==c for j in actual),
                reused_windows=per_camera-sum(j//per_camera==c for j in actual)) for c,cid in enumerate(cameras)]
            # CPU metadata only: map initial selection through the already
            # verified token dependency footprints, including different sizes.
            first_size=self.model.blocks[0].window_size
            cols=(x0.shape[2]+first_size-1)//first_size
            def closure(ids,block):
                result=set();units=footprints[block]
                for c in range(len(cameras)):
                    selected=[j%per_camera for j in ids if j//per_camera==c]
                    for j,unit in enumerate(units):
                        a,b,d,e=unit['token_region']
                        if any((k%cols)*first_size<d and (k%cols+1)*first_size>a
                            and (k//cols)*first_size<e and (k//cols+1)*first_size>b for k in selected):
                            result.add(c*len(units)+j)
                return result
            for stage in stages:
                if stage['block'] not in self.prefix:continue
                required=closure(mandatory,stage['block'])
                optional=closure(ordinary,stage['block'])-required
                stage['mandatory_window_ids']=sorted(required)
                stage['ordinary_window_ids']=sorted(optional)
                for c,counts in enumerate(stage['per_camera']):
                    n=counts['total_windows']
                    counts['mandatory_windows']=sum(j//n==c for j in required)
                    counts['ordinary_windows']=sum(j//n==c for j in optional)
                stage['selection_consistency'].update(mandatory_in_execution=required.issubset(set(stage['executed_window_ids'])),
                    mandatory_ordinary_disjoint=not bool(required&optional))
        skipped=sum(s.get('reused_windows',0) for s in stages)
        publication_start=perf_counter()
        ages={}
        for stage in stages:
            i=stage['block']
            if i not in self.prefix:continue
            updated=executed_ids[i]
            age=self.cache.window_ages[i]+1 if plans is not None else torch.zeros(stage['total_windows'],device=x0.device,dtype=torch.long)
            age[updated]=0;ages[i]=age
        current=descriptors if descriptors is not None else pooled_descriptors(x0,self.model.blocks[0].window_size)
        if plans is not None:
            descriptors=self.cache.descriptors.clone()
            ids=executed_ids[0]
            descriptors.index_copy_(0,ids,current.index_select(0,ids))
        else:
            descriptors=current
            region_states=copy.deepcopy(bundle.region_states)
        if not torch.isfinite(features).all().item():raise ValueError('nonfinite_visual_features')
        self.cache.commit(images,outputs,context,signature,features.stride(),sources,block0_input=x0,
            region_states=region_states,window_ages=ages,descriptors=descriptors)
        publication_ms=(perf_counter()-publication_start)*1000
        now=context.get('timestamp');old=(previous or {}).get('timestamp')
        diag=dict(mode='incremental' if skipped else 'full_update',full_update_reason=reason,
            cache_boundary='window',state_interface='source_linked',cacheable_blocks=list(self.prefix),
            global_barrier_block=len(self.prefix),patch_embedding='full_once',change_detection=self.config.refresh_signature()[0],numerical_policy=NUMERICAL_POLICY,
            refresh_policy=self.config.refresh_signature(),region_refresh=refresh,
            pointwise_execution='selected_window_qkv_projection_and_valid_token_mlp',
            window_count_scope='cached_prefix_output_windows; pointwise work reported separately',
            stages=stages,actual_skipped_windows=skipped,cache_hits=skipped,
            actual_skipped_block_computations=sum(1 for s in stages if s.get('executed_windows')==0),
            skipped_pointwise_window_work=sum(s.get('total_windows',0) for s in stages if s.get('executed_windows')==0),
            recomputed_windows=sum(s.get('executed_windows',0) for s in stages),
            cache_bytes=self.cache.bytes,cache_version=self.cache.version,cache_saved=True,
            output_window_provenance=copy.deepcopy(sources),
            cache_context=copy.deepcopy(context),cache_mismatch_reason=mismatch,
            task_quality_certified=False,window_reuse_quality_certified=False,m1_transformer_certificate_used=False,
            cache_verification_age=now-old if finite_time(now) and finite_time(old) else None,
            source_age=now-min(t for t in source_times if finite_time(t)) if finite_time(now) and any(finite_time(t) for t in source_times) else None,
            M2_validation_ms=validation_ms,M2_region_scoring_ms=scoring_ms,
            M2_execution_ms=execution_ms,M2_window_and_suffix_ms=execution_ms,M2_cache_publication_ms=publication_ms,M2_wall_ms=(perf_counter()-start)*1000,
            prefix_window_geometry={i:[self.model.patch_embed.proj.kernel_size[j]*self.model.blocks[i].window_size for j in (0,1)] for i in self.prefix},
            prefix_dependency_footprints=footprints)
        full_reason = (reason or next((s.get('full_update_reason') for s in stages
            if s.get('full_update_reason')), None) or 'all_windows_required') if not skipped else None
        cached_stages=[s for s in stages if s['block'] in self.prefix]
        diag.update(total_windows=sum(s['total_windows'] for s in cached_stages),
            planned_windows=sum(s['planned_windows'] for s in cached_stages),
            executed_windows=sum(s['executed_windows'] for s in cached_stages),
            reused_windows=skipped,full_refresh_reason=full_reason,
            mandatory_windows=sum(len(s['mandatory_window_ids']) for s in cached_stages),
            ordinary_windows=sum(len(s['ordinary_window_ids']) for s in cached_stages),
            per_block=cached_stages,per_camera=[dict(camera_id=cid,**{
                k:sum(s['per_camera'][c][k] for s in cached_stages)
                for k in ('total_windows','planned_windows','executed_windows','reused_windows','mandatory_windows','ordinary_windows')})
                for c,cid in enumerate(cameras)],M2_total_ms=(perf_counter()-start)*1000)
        diag.update(runtime_entered=True,execution_completed=True,
            actual_window_reuse=bool(skipped),full_recompute=not bool(skipped),
            full_recompute_reason=full_reason,executor='WindowExecutor')
        diag.update(details);self.last_diagnostics.update(diag)
        return [features],self.last_diagnostics

    def _execute(self,x,context,plans,reason):
        stages=[];outputs={};sources={};executed_ids={}
        for i,block in enumerate(self.model.blocks):
            if i not in self.prefix:
                x=block(x);stages.append(dict(block=i,mode='full',reason='global_dependency'))
            else:
                size=block.window_size;total=len(x)*((x.shape[1]+size-1)//size)*((x.shape[2]+size-1)//size)
                ids=plans[i].updated.nonzero(as_tuple=False).flatten() if plans is not None else torch.arange(total,device=x.device)
                why=reason
                planned_ids=list(range(total)) if len(ids)==total else ids.tolist()
                self.executing_block=i
                executed_ids[i]=ids
                cached=self.cache.outputs.get(i) if len(ids)<total else None
                x,packed=self.executor.execute(block,x,cached,ids)
                outputs[i]=packed;id_values=list(range(total)) if len(ids)==total else planned_ids
                computed=set(id_values);per_camera=total//len(x)
                cols=(x.shape[2]+size-1)//size
                valid_tokens=sum(min(size,x.shape[1]-(j%per_camera//cols)*size)*
                    min(size,x.shape[2]-(j%cols)*size) for j in id_values)
                sources[i]=[dict(computation_frame_id=context.get('frame_id'),computation_timestamp=context.get('timestamp'),
                    computation_version=self.cache.version+1,camera_id=context['camera_ids'][j//per_camera],
                    producer_layer=i,dependency_scope='closed_window_prefix',task_quality_certified=False)
                    if j in computed else copy.deepcopy(self.cache.sources[i][j]) for j in range(total)]
                stages.append(dict(block=i,total_windows=total,updated_windows=len(ids),executed_windows=len(ids),
                    planned_windows=len(ids),planned_window_ids=planned_ids,
                    executed_window_ids=id_values,
                    reused_windows=total-len(ids),mode='windows',full_update_reason=why,attention_windows=len(ids),
                    pointwise_tokens=valid_tokens,qkv_tokens=len(ids)*size*size,
                    pointwise_window_work=len(ids)))
                stage=stages[-1]
                stage['selection_consistency']=dict(plan_equals_execution=set(planned_ids)==computed,
                    counts_match=len(computed)==len(ids) and len(ids)+stage['reused_windows']==total)
                stage['per_camera']=[dict(camera_id=cid,total_windows=per_camera,
                    planned_windows=sum(j//per_camera==c for j in planned_ids),
                    executed_windows=sum(j//per_camera==c for j in id_values),
                    reused_windows=per_camera-sum(j//per_camera==c for j in id_values)) for c,cid in enumerate(context['camera_ids'])]
            if self.stage_observer is not None:self.stage_observer(i,x)
        return x.permute(0,3,1,2),stages,outputs,sources,executed_ids
