"""One window execution graph: cache misses expand to all-window updates."""
import copy
import math
from time import perf_counter
import torch
from .config import M2Config
from .cost_model import CostModel,cost_key
from .dependency_planner import DependencyPlanner,exact_pixel_changes,patch_updates
from .m1_adapter import computation_admission
from .state_cache import StateCache,NonFiniteCacheError
from .window_executor import WindowExecutor,NUMERICAL_POLICY


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
        return (context.get('execution_signature'),NUMERICAL_POLICY,tensors,topology,
            (p.kernel_size,p.stride,p.padding,p.dilation,self.model.pretrain_use_cls_token),
            torch.is_autocast_enabled(),torch.get_autocast_gpu_dtype(),
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.are_deterministic_algorithms_enabled())

    def forward_from_m1(self,images,manager,context):
        tick=perf_counter()
        try:snapshot=manager.export_m2_inputs(expected_context=context)
        except (AttributeError,RuntimeError,ValueError,TypeError,KeyError):snapshot=None
        read_ms=(perf_counter()-tick)*1000
        result,diag=self.forward(images,m1_result=snapshot,execution_context=context)
        diag.update(m1_snapshot_read=snapshot is not None,M1_M2_state_read_ms=read_ms,
                    world_identity_authorized_pixel_reuse=False)
        return result,diag

    def forward(self,images,m1_result=None,m1_state=None,execution_context=None,computation_state=None):
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
        signature=self.signature(context);previous=copy.deepcopy(self.cache.context)
        source_times=[s.get('computation_timestamp') for units in self.cache.sources.values() for s in units]
        invalid=None;reason=None;details={}
        valid=bool(context.get('scene_id') and context.get('frame_id') and finite_time(context.get('timestamp'))
            and context.get('execution_signature') and type(context.get('state_epoch')) is int
            and context.get('coordinate_space')=='preprocessed_image')
        if not valid:reason='invalid_execution_context'
        elif not self.config.enabled:reason='disabled'
        else:
            invalid,reason=computation_admission(bundle,context,images,self.cache,
                getattr(m1_result,'full_refresh_hint',False),details=details,
                window_footprints=self.planner.input_footprints(*images.shape[-2:]))
        mismatch=self.cache.mismatch(images,context,signature) if valid else None
        if reason is None and mismatch:reason=mismatch
        if reason is None:
            changed=exact_pixel_changes(images,self.cache.images)
            if invalid is not None:changed|=torch.as_tensor(invalid,device=images.device)
            plans=self.planner.plan(patch_updates(changed,self.model.patch_embed.proj),stop_after=len(self.prefix))
        else:plans=None
        validation_ms=(perf_counter()-start)*1000
        features,stages,outputs,sources=self._execute(images,context,plans,reason)
        skipped=sum(s.get('reused_windows',0) for s in stages)
        capture=valid and self.config.enabled and self.config.cache_capture!='never'
        capture &= self.config.cache_capture=='always' or bool(self.costs.records)
        capture &= reason not in ('missing_computation_state','m1_context_mismatch','m1_aborted_transaction',
                                 'm1_state_epoch_mismatch','invalid_refresh_contract')
        publication_error=None
        if capture:
            try:self.cache.commit(images,outputs,context,signature,features.stride(),sources)
            except RuntimeError as error:
                publication_error='nonfinite_cache_features' if isinstance(error,NonFiniteCacheError) else 'cache_publication_failed'
                capture=False;self.cache.clear()
        else:self.cache.clear()
        now=context.get('timestamp');old=(previous or {}).get('timestamp')
        diag=dict(mode='incremental' if skipped else 'full_update',full_update_reason=reason,fallback_reason=reason,
            cache_boundary='window',state_interface='source_linked',cacheable_blocks=list(self.prefix),
            global_barrier_block=len(self.prefix),patch_embedding='full',numerical_policy=NUMERICAL_POLICY,
            pointwise_execution='native_full_shapes_when_attention_updates',
            window_count_scope='cached_prefix_output_windows; partial savings cover attention only',
            stages=stages,actual_skipped_windows=skipped,cache_hits=skipped,
            actual_skipped_block_computations=sum(s.get('total_windows',0) for s in stages if s.get('executed_windows')==0),
            recomputed_windows=sum(s.get('executed_windows',0) for s in stages),
            cache_bytes=self.cache.bytes,cache_version=self.cache.version,cache_saved=capture,
            cache_publication_error=publication_error,output_window_provenance=copy.deepcopy(sources),
            cache_context=copy.deepcopy(context),cost_table_status=self.costs.status,
            task_quality_certified=publication_error!='nonfinite_cache_features',m1_transformer_certificate_used=False,
            cache_verification_age=now-old if finite_time(now) and finite_time(old) else None,
            source_age=now-min(t for t in source_times if finite_time(t)) if finite_time(now) and any(finite_time(t) for t in source_times) else None,
            M2_validation_ms=validation_ms,M2_wall_ms=(perf_counter()-start)*1000,
            prefix_window_geometry={i:[self.model.patch_embed.proj.kernel_size[j]*self.model.blocks[i].window_size for j in (0,1)] for i in self.prefix},
            prefix_dependency_footprints=self.planner.input_footprints(*images.shape[-2:]))
        diag.update(details);self.last_diagnostics=diag
        return [features],diag

    def _execute(self,images,context,plans,reason):
        from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
        x=self.model.patch_embed(images)
        if self.model.pos_embed is not None:x=x+get_abs_pos(self.model.pos_embed,self.model.pretrain_use_cls_token,x.shape[1:3])
        stages=[];outputs={};sources={}
        for i,block in enumerate(self.model.blocks):
            if i not in self.prefix:
                x=block(x);stages.append(dict(block=i,mode='full',reason='global_dependency'))
            else:
                size=block.window_size;total=len(x)*((x.shape[1]+size-1)//size)*((x.shape[2]+size-1)//size)
                ids=plans[i].updated.nonzero(as_tuple=False).flatten() if plans is not None else torch.arange(total,device=x.device)
                why=reason
                if len(ids)<total:
                    allowed,why=self.costs.beneficial(cost_key(block,x,i),len(ids))
                    if not allowed:ids=torch.arange(total,device=x.device)
                self.executing_block=i
                x,packed=self.executor.execute(block,x,self.cache.outputs.get(i),ids)
                outputs[i]=packed;computed=set(ids.tolist());per_camera=total//len(images)
                sources[i]=[dict(computation_frame_id=context.get('frame_id'),computation_timestamp=context.get('timestamp'),
                    computation_version=self.cache.version+1,camera_id=context['camera_ids'][j//per_camera],
                    producer_layer=i,dependency_scope='closed_window_prefix',task_quality_certified=True)
                    if j in computed else copy.deepcopy(self.cache.sources[i][j]) for j in range(total)]
                stages.append(dict(block=i,total_windows=total,updated_windows=len(ids),executed_windows=len(ids),
                    reused_windows=total-len(ids),mode='windows',full_update_reason=why,attention_windows=len(ids),
                    pointwise_tokens=len(x)*x.shape[1]*x.shape[2] if len(ids) else 0,
                    pointwise_window_work=total if len(ids) else 0))
            if self.stage_observer is not None:self.stage_observer(i,x)
        return x.permute(0,3,1,2),stages,outputs,sources
