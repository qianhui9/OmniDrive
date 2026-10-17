"""Exact prefix reuse, cheap M1 admission, and cache-preserving full execution."""
from time import perf_counter
import torch
from .config import M2Config
from .cost_model import CostModel, cost_key
from .dependency_planner import DependencyPlanner, patch_updates, exact_pixel_changes
from .m1_adapter import invalidation_mask, cpu_window_opportunity
from .packing import partition
from .scheduler import Scheduler
from .state_cache import StateCache
from .window_executor import WindowExecutor
from .cheap_admission import CheapAdmission
from .camera_executor import CameraExecutor,numeric_details
from .execution_policy import validation_budget,optimistic_window_saving


class IncrementalVisionRuntime:
    def __init__(self, model, config=None):
        self.model=model
        self.config=config if isinstance(config,M2Config) else M2Config(**(config or {}))
        self.planner=DependencyPlanner(model)
        self.prefix=self.planner.cacheable_prefix()
        self.cache=StateCache(self.config.memory_budget_mb*1024**2)
        self.executor=WindowExecutor()
        self.scheduler=Scheduler(self.config,CostModel(self.config.cost_table))
        self.last_diagnostics={}
        self.stage_observer=None
        self.cheap_admission=CheapAdmission()
        self.last_admission={}
        self.camera_executor=CameraExecutor(model,self.config)

    def reset(self):
        self.cache.clear();self.executor.clear()
        self.cheap_admission.clear()
        self.camera_executor.reset()

    def pre_m1_admission(self, images, context):
        """Reject before M1 only; the selected service owns any explicit reset."""
        if not self.config.cheap_gate:
            return dict(reject=False,reason='gate_disabled')
        flat=images.flatten(0,1) if images.ndim==5 else images
        p=self.model.patch_embed.proj
        size=self.model.blocks[0].window_size
        pixel_window=(p.kernel_size[0]*size,p.kernel_size[1]*size)
        if self.cheap_admission.required_bytes(flat,pixel_window)>self.cache.budget_bytes:
            self.cheap_admission.clear()
            self.last_admission=dict(reject=True,reason='gate_memory_budget',sample_acceptance=False)
        else:
            try:
                self.last_admission=self.cheap_admission.check(flat,context,pixel_window)
            except torch.cuda.OutOfMemoryError:
                self.cheap_admission.clear()
                self.last_admission=dict(reject=True,reason='gate_memory_budget',sample_acceptance=False)
        self.last_admission.update(frame_id=context.get('frame_id'),scene_id=context.get('scene_id'))
        return self.last_admission

    def signature(self, context):
        tensors=tuple((n,t.data_ptr(),t._version,tuple(t.shape),str(t.dtype),str(t.device))
                      for n,t in list(self.model.named_parameters())+list(self.model.named_buffers()))
        topology=tuple((type(b),b.window_size,b.use_residual_block,b.attn.flash_attn,
                        b.attn.scale,b.norm1.eps,b.norm2.eps) for b in self.model.blocks)
        p=self.model.patch_embed.proj
        geometry=(p.kernel_size,p.stride,p.padding,p.dilation,self.model.pretrain_use_cls_token)
        return (context.get('execution_signature'),tensors,topology,geometry,numeric_details(self.model),torch.is_autocast_enabled(),
                torch.get_autocast_gpu_dtype(),torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)

    def _capture_reason(self, images, reason, has_opportunity):
        if self.config.cache_capture=='never': return 'capture_disabled'
        if self.config.cache_capture=='adaptive':
            if reason=='no_measured_beneficial_count':return 'capture_not_expected_to_pay'
            if reason in ('insufficient_stable_windows','m1_full_refresh') and not has_opportunity:
                return 'capture_not_expected_to_pay'
            if self.config.mode=='auto' and not self.scheduler.costs.records:
                return (self.scheduler.costs.status if self.scheduler.costs.status in
                        ('cost_schema_mismatch','invalid_cost_table') else 'unprofiled_cost')
        p=self.model.patch_embed.proj
        h,w=images.shape[-2]//p.stride[0],images.shape[-1]//p.stride[1]
        estimate=images.numel()*images.element_size()
        for i in self.prefix:
            s=self.model.blocks[i].window_size
            estimate+=len(images)*((h+s-1)//s*s)*((w+s-1)//s*s)*p.out_channels*images.element_size()
        if 3*estimate>self.cache.budget_bytes: return 'memory_budget'
        if images.is_cuda and 3*estimate>torch.cuda.mem_get_info(images.device)[0]: return 'memory_budget'
        return None

    def _dense(self, images, reason, context=None, capture=False, signature=None):
        # This frame still executes exactly once. Hooks observe its ORIGINAL forward.
        self.cache.clear();self.executor.clear()
        outputs={};handles=[];stages=[]
        if capture:
            for i in self.prefix:
                def save(module, inputs, out, i=i):
                    outputs[i]=partition(out,self.model.blocks[i].window_size)[0].detach()
                handles.append(self.model.blocks[i].register_forward_hook(save))
        if self.stage_observer is not None:
            for i,block in enumerate(self.model.blocks):
                handles.append(block.register_forward_hook(
                    lambda m,inp,out,i=i:self.stage_observer(i,out)))
        try:
            features=self.model(images)
        finally:
            for handle in handles:handle.remove()
        if capture:
            self.cache.commit(images,outputs,context,signature,features[0].stride())
            for i in self.prefix:
                total=outputs[i].shape[0]
                stages.append(dict(block=i,updated_windows=total,total_windows=total,
                    executed_windows=total,reused_windows=0,mode='full',fallback_reason=reason))
        self.last_diagnostics=dict(mode='full',fallback_reason=reason,stages=stages,
            cache_bytes=self.cache.bytes,cache_version=self.cache.version,
            cache_saved=capture,cache_source_frame=context['frame_id'] if capture else None,
            actual_skipped_windows=0,cost_table_status=self.scheduler.costs.status,
            cache_capture_reason=None if capture else (getattr(self,'_capture_skip',None) or reason))
        return features,self.last_diagnostics

    def forward(self,current_images,m1_result=None,m1_state=None,execution_context=None):
        try:
            context=execution_context or {}
            if (self.config.enabled and self.config.cheap_gate and self.config.reuse_boundary=='window'
                    and self.last_admission.get('reject') and
                    self.last_admission.get('frame_id')==context.get('frame_id') and
                    self.last_admission.get('scene_id')==context.get('scene_id')):
                self._capture_skip='capture_not_expected_to_pay'
                return self._dense(current_images,'cheap_input_rejection')
            if self.config.enabled and self.config.reuse_boundary=='camera':
                context=dict(context,necessary_validation_ms=validation_budget(context,m1_result,m1_state))
                value,self.last_diagnostics=self.camera_executor.forward(current_images,
                    context,self.signature(context))
                return value,self.last_diagnostics
            return self._forward(current_images,m1_result,m1_state,execution_context)
        except torch.cuda.OutOfMemoryError:
            self.reset()
        if current_images.is_cuda:torch.cuda.empty_cache()
        return self._dense(current_images,'memory_budget')

    def _forward(self,images,m1_result=None,m1_state=None,execution_context=None):
        context=dict(execution_context or {})
        self._capture_skip=None
        if not self.config.enabled:return self._dense(images,'disabled')
        if any(m.training for m in self.model.modules()) or torch.is_grad_enabled():
            return self._dense(images,'inference_only')
        if not all(k in context for k in ('scene_id','frame_id','camera_ids')):
            return self._dense(images,'missing_execution_context')
        if images.ndim!=4 or len(context['camera_ids'])!=len(images):
            return self._dense(images,'shape_unsupported')
        prefix=self.planner.cacheable_prefix()
        if prefix!=self.prefix:self.reset();self.prefix=prefix
        if not self.prefix:return self._dense(images,'global_dependency')
        p=self.model.patch_embed.proj
        if p.padding!=(0,0) or p.dilation!=(1,1) or p.stride!=p.kernel_size:
            return self._dense(images,'shape_unsupported')
        if self.cache.budget_bytes==0:return self._dense(images,'memory_budget')
        invalid,reason=invalidation_mask(m1_result,m1_state,context,images,device=False,
            rejection_footprint=(*p.kernel_size,self.model.blocks[0].window_size))
        has_opportunity=False
        if invalid is not None:
            opportunities=cpu_window_opportunity(invalid,p,self.model.blocks[0].window_size)
            has_opportunity=bool(opportunities.any())
            if not has_opportunity:reason='insufficient_stable_windows'
            elif (self.config.mode=='auto' and
                  all(self.model.blocks[i].window_size==self.model.blocks[0].window_size for i in self.prefix)
                  and not self.scheduler.could_benefit(int(opportunities.size-opportunities.sum()),self.prefix)):
                reason='no_measured_beneficial_count'
        # Invalid metadata cannot certify a new cache. M1 refresh itself may capture
        # full outputs, without authorizing any reuse on this frame.
        if reason and reason not in ('m1_full_refresh','insufficient_stable_windows','no_measured_beneficial_count'):
            return self._dense(images,reason)
        self._capture_skip=self._capture_reason(images,reason,has_opportunity)
        if reason:
            capture=self._capture_skip is None
            return self._dense(images,reason,context,capture,self.signature(context) if capture else None)
        if self._capture_skip:
            return self._dense(images,self._capture_skip)
        if (self.config.mode=='auto' and m1_result is not None and
                validation_budget(context,m1_result,m1_state)>=optimistic_window_saving(self.scheduler.costs,self.prefix)):
            self._capture_skip='capture_not_expected_to_pay'
            return self._dense(images,'validation_cost_exceeds_window_saving')
        signature=self.signature(context)
        mismatch=self.cache.mismatch(images,context,signature)
        if mismatch:
            return self._dense(images,mismatch,context,True,signature)
        changed=exact_pixel_changes(images,self.cache.images)
        if invalid is not None:changed|=torch.as_tensor(invalid,device=images.device)
        plans=self.planner.plan(patch_updates(changed,p),stop_after=len(self.prefix))
        # First closure is already the common window set for deployed blocks 0/1.
        ids_by_block=[plan.updated.nonzero(as_tuple=False).flatten() for plan in plans]
        from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
        x=self.model.patch_embed(images)
        if self.model.pos_embed is not None:
            x=x+get_abs_pos(self.model.pos_embed,self.model.pretrain_use_cls_token,x.shape[1:3])
        next_outputs={};stages=[]
        for i,block in enumerate(self.model.blocks):
            if i in self.prefix:
                ids=ids_by_block[i];total=plans[i].updated.numel()
                key=cost_key(block,x,i) if self.config.mode=='auto' or self.scheduler.costs.records else ''
                measured=self.scheduler.costs.lookup(key,len(ids))
                strategy=measured.get('packing_backend','spatial') if measured else 'spatial'
                sparse,fallback=self.scheduler.choose(key,len(ids),total)
                if sparse:
                    x,packed=self.executor.execute(block,x,self.cache.outputs[i],ids,strategy=strategy)
                else:
                    x=block(x);packed=partition(x,block.window_size)[0]
                next_outputs[i]=packed.detach()
                stages.append(dict(block=i,updated_windows=len(ids),total_windows=total,
                    executed_windows=len(ids) if sparse else total,reused_windows=total-len(ids) if sparse else 0,
                    mode='packed' if sparse else 'full',fallback_reason=fallback,
                    packing_backend=strategy if sparse else None,
                    source_frame=self.cache.context['frame_id'] if sparse else None,
                    source_version=self.cache.version if sparse else None))
            else:
                x=block(x)
                stages.append(dict(block=i,mode='full',fallback_reason='global_dependency',
                                   global_expansion=not bool(block.window_size)))
            if self.stage_observer is not None:self.stage_observer(i,x)
        result=x.permute(0,3,1,2)
        expected_stride=self.cache.output_stride
        if expected_stride is not None and tuple(result.stride())!=expected_stride:
            aligned=torch.empty_strided(result.shape,expected_stride,dtype=result.dtype,device=result.device)
            aligned.copy_(result);result=aligned
        self.cache.commit(images,next_outputs,context,signature,result.stride())
        skipped=sum(s.get('reused_windows',0) for s in stages)
        self.last_diagnostics=dict(mode='incremental' if skipped else 'full',stages=stages,
            cache_bytes=self.cache.bytes,cache_version=self.cache.version,cache_saved=True,
            cost_table_status=self.scheduler.costs.status,
            actual_skipped_windows=skipped,patch_embedding='full',
            evidence='same_coordinate_exact_input_and_dependency_closure',m1_transformer_certificate_used=False)
        return [result],self.last_diagnostics
