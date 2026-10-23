"""One window execution graph: cache misses expand to all-window updates."""
import copy
import math
from time import perf_counter
import torch
import numpy as np
from .config import M2Config
from .cost_model import CostModel,cost_key
from .dependency_planner import DependencyPlanner,block0_input,exact_token_changes,patch_updates
from .m1_adapter import computation_admission
from .state_cache import StateCache,NonFiniteCacheError,CachePublicationError
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
        self.costs=CostModel(self.config.cost_table)
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
        return self._from_m1(images,manager,context)

    def evaluate_offline_from_m1(self,images,manager,context,*,variant='D',window_targets=None):
        return self._from_m1(images,manager,context,variant,window_targets)

    def _from_m1(self,images,manager,context,offline=None,window_targets=None):
        tick=perf_counter()
        if manager is None:snapshot=None
        else:
            try:snapshot=manager.export_m2_inputs(expected_context=context,computation_only=True)
            except RuntimeError as error:
                if str(error) not in ('Call precheck first','M1 snapshot does not match the pending M2 request'):
                    raise
                snapshot=None
        read_ms=(perf_counter()-tick)*1000
        if offline:
            result,diag=self.evaluate_offline(images,variant=offline,m1_result=snapshot,execution_context=context,window_targets=window_targets)
        else:result,diag=self.forward(images,m1_result=snapshot,execution_context=context)
        diag.update(m1_snapshot_read=snapshot is not None,M1_M2_state_read_ms=read_ms,
                    world_identity_authorized_pixel_reuse=False)
        return result,diag

    def forward(self,images,m1_result=None,m1_state=None,execution_context=None,computation_state=None):
        return self._forward(images,m1_result,execution_context,computation_state)

    def evaluate_offline(self,images,*,variant='D',m1_result=None,execution_context=None,computation_state=None,window_targets=None):
        """Explicit research entry; no cost table mutation or production approval.

        A/B/C are ablations only. Production forward always uses complete WSGRR.
        A runtime/cache cannot cross the offline/production signature boundary.
        """
        if variant not in ('A','B','C','D','NoM1Region','NoM1RegionMatched'):raise ValueError('Unknown offline ablation')
        if variant=='NoM1RegionMatched':
            if (not isinstance(window_targets,list) or len(window_targets)!=len(images)
                    or any(type(v) is not int or v<0 for v in window_targets)):
                raise ValueError('Explicit per-camera offline total targets required')
        elif window_targets is not None:raise ValueError('Total targets belong only to the offline matched-compute control')
        return self._forward(images,m1_result,execution_context,computation_state,variant,window_targets)

    def _forward(self,images,m1_result,execution_context,computation_state,offline=None,window_targets=None):
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
        signature=(self.signature(context), 'offline_'+offline if offline else 'production')
        previous=copy.deepcopy(self.cache.context) if isinstance(self.cache.context,dict) else None
        source_times=[]
        invalid=None;reason=None;details={}
        valid=bool(context.get('scene_id') and context.get('frame_id') and finite_time(context.get('timestamp'))
            and context.get('execution_signature') and type(context.get('state_epoch')) is int
            and context.get('coordinate_space')=='preprocessed_image')
        x0=block0_input(self.model,images)
        # Inspect storage/metadata before the adapter can dereference corrupted
        # lineage. A cache miss still validates M1's current publication context.
        mismatch=self.cache.mismatch(images,context,signature,x0) if valid else None
        if valid and mismatch is None:
            source_times=[s.get('computation_timestamp') for units in self.cache.sources.values() for s in units]
        if not valid:reason='invalid_execution_context'
        elif not self.config.enabled:reason='disabled'
        elif mismatch not in (None,'no_valid_cache'):reason=mismatch
        else:
            invalid,reason=computation_admission(bundle,context,images,self.cache,
                getattr(m1_result,'full_refresh_hint',False),details=details,
                window_footprints=self.planner.input_footprints(*images.shape[-2:]))
        if reason is None and mismatch:reason=mismatch
        uncertainty=details.pop('_uncertainty_pixels',None)
        scoring_start=perf_counter();refresh={};descriptors=None;region_states=[]
        if not self.prefix and reason is None:reason='no_cacheable_prefix'
        if reason is None:
            mandatory=patch_updates(torch.as_tensor(invalid,device=images.device,dtype=torch.bool),self.model.patch_embed.proj)
            mandatory |= ~torch.isfinite(x0).all(-1)
            if offline != 'A':
                for i in self.prefix:
                    expired=self.cache.window_ages[i]+1 >= self.config.max_reuse_frames
                    mandatory |= self.planner.windows_to_tokens(expired,x0.shape[:3],self.model.blocks[i].window_size)
            if offline=='A':
                from worldstream.m1.mask_codec import decode_mask
                dirty=torch.as_tensor(np.stack([decode_mask(bundle.refresh_contract[c]['dirty_hint_mask']) for c in cameras]),device=x0.device)
                mandatory |= patch_updates(dirty,self.model.patch_embed.proj)
                changed=exact_token_changes(x0,self.cache.block0_input)|mandatory
                plans=self.planner.plan(changed,stop_after=len(self.prefix))
                refresh=dict(exact_token_update_ratio=float(changed.float().mean()))
                region_states=copy.deepcopy(bundle.region_states)
                descriptors=pooled_descriptors(x0,self.model.blocks[0].window_size)
            else:
                selected,descriptors,region_states,refresh=select_regions(x0,self.cache,bundle,context,
                    self.model.patch_embed.proj,self.model.blocks[0].window_size,self.config,
                    uncertainty,mandatory,events=offline not in ('B','NoM1Region','NoM1RegionMatched'),
                    budget=offline not in ('B','C'),window_targets=window_targets)
                plans=self.planner.plan_windows(selected,x0.shape[:3],stop_after=len(self.prefix))
        else:plans=None
        scoring_ms=(perf_counter()-scoring_start)*1000
        validation_ms=(perf_counter()-start)*1000
        execution_start=perf_counter()
        features,stages,outputs,sources,capture_eligible,executed_ids=self._execute(x0,context,plans,reason,offline)
        execution_ms=(perf_counter()-execution_start)*1000
        if self.prefix:
            if plans is None:
                refresh.update(mandatory_windows=stages[0]['total_windows'],ordinary_windows=0,
                    mandatory_window_ids=stages[0]['executed_window_ids'],ordinary_window_ids=[])
            missed=set(refresh.get('mandatory_window_ids',[]))-set(stages[0]['executed_window_ids'])
            refresh['event_missed_windows']=len(missed)
        skipped=sum(s.get('reused_windows',0) for s in stages)
        capture=valid and self.config.enabled and self.config.cache_capture!='never'
        capture &= bool(offline) or self.config.cache_capture=='always' or capture_eligible
        capture &= reason not in ('missing_computation_state','m1_context_mismatch','m1_aborted_transaction',
                                 'm1_state_epoch_mismatch','invalid_refresh_contract','invalid_region_state')
        publication_error=None
        publication_start=perf_counter()
        if capture:
            try:
                ages={}
                for s in stages:
                    if s['block'] not in self.prefix:continue
                    i=s['block'];updated=executed_ids[i]
                    age=(self.cache.window_ages[i]+1).clone() if plans is not None else torch.zeros(s['total_windows'],device=x0.device,dtype=torch.long)
                    age[updated]=0;ages[i]=age
                if self.prefix:
                    size=self.model.blocks[0].window_size
                    current=descriptors if descriptors is not None else pooled_descriptors(x0,size)
                    if plans is not None:
                        descriptors=self.cache.descriptors.clone()
                        ids=stages[0]['executed_window_ids'];descriptors[ids]=current[ids]
                    else:
                        descriptors=current
                        region_states=copy.deepcopy(getattr(bundle,'region_states',[]))
                self.cache.commit(images,outputs,context,signature,features.stride(),sources,block0_input=x0,
                    region_states=region_states,window_ages=ages,descriptors=descriptors)
            except (NonFiniteCacheError,CachePublicationError,torch.cuda.OutOfMemoryError) as error:
                publication_error='nonfinite_cache_features' if isinstance(error,NonFiniteCacheError) else 'cache_publication_failed'
                capture=False;self.cache.clear()
        else:self.cache.clear()
        publication_ms=(perf_counter()-publication_start)*1000
        now=context.get('timestamp');old=(previous or {}).get('timestamp')
        diag=dict(mode='incremental' if skipped else 'full_update',full_update_reason=reason,fallback_reason=reason,
            cache_boundary='window',state_interface='source_linked',cacheable_blocks=list(self.prefix),
            global_barrier_block=len(self.prefix),patch_embedding='full_once',change_detection='exact_block0_input' if offline=='A' else 'wsgrr_v1',numerical_policy=NUMERICAL_POLICY,
            offline_experiment=bool(offline),offline_variant=offline,refresh_policy=self.config.refresh_signature(),region_refresh=refresh,
            pointwise_execution='selected_window_qkv_projection_and_valid_token_mlp',
            window_count_scope='cached_prefix_output_windows; pointwise work reported separately',
            stages=stages,actual_skipped_windows=skipped,cache_hits=skipped,
            actual_skipped_block_computations=sum(1 for s in stages if s.get('executed_windows')==0),
            skipped_pointwise_window_work=sum(s.get('total_windows',0) for s in stages if s.get('executed_windows')==0),
            recomputed_windows=sum(s.get('executed_windows',0) for s in stages),
            cache_bytes=self.cache.bytes,cache_version=self.cache.version,cache_saved=capture,
            cache_publication_error=publication_error,output_window_provenance=copy.deepcopy(sources),
            cache_context=copy.deepcopy(context),cost_table_status=self.costs.status,
            task_quality_certified=False,window_reuse_quality_certified=bool(skipped and not offline),m1_transformer_certificate_used=False,
            cache_verification_age=now-old if finite_time(now) and finite_time(old) else None,
            source_age=now-min(t for t in source_times if finite_time(t)) if finite_time(now) and any(finite_time(t) for t in source_times) else None,
            M2_validation_ms=validation_ms,M2_region_scoring_ms=scoring_ms,
            M2_window_and_suffix_ms=execution_ms,M2_cache_publication_ms=publication_ms,M2_wall_ms=(perf_counter()-start)*1000,
            prefix_window_geometry={i:[self.model.patch_embed.proj.kernel_size[j]*self.model.blocks[i].window_size for j in (0,1)] for i in self.prefix},
            prefix_dependency_footprints=self.planner.input_footprints(*images.shape[-2:]))
        diag.update(details);self.last_diagnostics=diag
        return [features],diag

    def _execute(self,x,context,plans,reason,offline=None):
        stages=[];outputs={};sources={};capture_eligible=False;executed_ids={}
        for i,block in enumerate(self.model.blocks):
            if i not in self.prefix:
                x=block(x);stages.append(dict(block=i,mode='full',reason='global_dependency'))
            else:
                size=block.window_size;total=len(x)*((x.shape[1]+size-1)//size)*((x.shape[2]+size-1)//size)
                ids=plans[i].updated.nonzero(as_tuple=False).flatten() if plans is not None else torch.arange(total,device=x.device)
                why=reason
                key=cost_key(block,x,i,model=self.model,refresh_policy=self.config.refresh_signature()) if self.costs.records else None
                capture_eligible |= key is not None and self.costs.has_beneficial(key)
                if len(ids)<total and not offline:
                    allowed,why=self.costs.beneficial(key,len(ids))
                    if not allowed:
                        ids=torch.arange(total,device=x.device)
                        # Actual upstream recomputation must close through later
                        # prefix blocks even when cost admission expands a plan.
                        expanded=self.planner.plan(torch.ones(x.shape[:3],device=x.device,dtype=torch.bool),stop_after=len(self.prefix))
                        for later in range(i+1,len(self.prefix)):plans[later]=expanded[later]
                self.executing_block=i
                executed_ids[i]=ids
                cached=self.cache.outputs.get(i) if len(ids)<total else None
                x,packed=self.executor.execute(block,x,cached,ids)
                outputs[i]=packed;id_values=ids.tolist();computed=set(id_values);per_camera=total//len(x)
                rows=(x.shape[1]+size-1)//size;cols=(x.shape[2]+size-1)//size
                valid_tokens=sum(min(size,x.shape[1]-(j%per_camera//cols)*size)*
                    min(size,x.shape[2]-(j%cols)*size) for j in id_values)
                sources[i]=[dict(computation_frame_id=context.get('frame_id'),computation_timestamp=context.get('timestamp'),
                    computation_version=self.cache.version+1,camera_id=context['camera_ids'][j//per_camera],
                    producer_layer=i,dependency_scope='closed_window_prefix',task_quality_certified=False)
                    if j in computed else copy.deepcopy(self.cache.sources[i][j]) for j in range(total)]
                stages.append(dict(block=i,total_windows=total,updated_windows=len(ids),executed_windows=len(ids),
                    planned_windows=int(plans[i].updated.sum()) if plans is not None else total,
                    executed_window_ids=id_values,
                    reused_windows=total-len(ids),mode='windows',full_update_reason=why,attention_windows=len(ids),
                    pointwise_tokens=valid_tokens,qkv_tokens=len(ids)*size*size,
                    pointwise_window_work=len(ids)))
            if self.stage_observer is not None:self.stage_observer(i,x)
        return x.permute(0,3,1,2),stages,outputs,sources,capture_eligible,executed_ids
