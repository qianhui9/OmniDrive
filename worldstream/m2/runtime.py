"""Exact prefix reuse, cheap M1 admission, and cache-preserving full execution."""
from time import perf_counter
import copy
import math
import torch
from .config import M2Config
from .cost_model import CostModel, cost_key
from .dependency_planner import DependencyPlanner, patch_updates, exact_pixel_changes
from .m1_adapter import invalidation_mask, cpu_window_opportunity, computation_admission
from .packing import partition
from .scheduler import Scheduler
from .state_cache import StateCache, NonFiniteCacheError
from .window_executor import WindowExecutor
from .cheap_admission import CheapAdmission
from .camera_executor import CameraExecutor,numeric_details
from .execution_policy import validation_budget,optimistic_window_saving


def _finite_time(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


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
        self.conditional_executor=None
        if self.config.experimental_conditional_reuse:
            from .conditional_prefix import ConditionalPrefixExecutor
            self.conditional_executor=ConditionalPrefixExecutor(model,self.config)

    def reset(self):
        self.cache.clear();self.executor.clear()
        self.cheap_admission.clear()
        self.camera_executor.reset()
        if self.conditional_executor is not None:self.conditional_executor.reset()

    def forward_from_m1(self, images, manager, context):
        """Read this transaction's defensive snapshot; failed reads run full EVA.

        World identity/geometry is diagnostic, never a pixel equality receipt.
        The existing source-linked adapter and exact comparison own admission.
        """
        if not self.config.enabled or self.config.state_interface!='source_linked':
            result=getattr(manager,'last_result',None)
            # The legacy detector supplies time explicitly for camera reuse.
            # Window reuse also needs the bound transaction time to reject
            # out-of-order frames and report actual source ages.
            if (self.config.enabled and self.config.reuse_boundary=='window'
                    and context.get('timestamp') is None and result is not None
                    and result.scene_id==context.get('scene_id')
                    and result.frame_id==context.get('frame_id')):
                context=dict(context,timestamp=result.timestamp)
            return self.forward(images,result,manager,context)
        read_start=perf_counter()
        try:
            result=manager.export_m2_inputs(expected_context=context)
        except (AttributeError,RuntimeError,ValueError,TypeError,KeyError):
            self.reset()
            output,diagnostics=self._dense(images,'m1_snapshot_unavailable',context)
            diagnostics.update(m1_snapshot_read=False,actual_reused_cameras=0,cache_hits=0)
            return output,diagnostics
        read_ms=(perf_counter()-read_start)*1000
        output,diagnostics=self.forward(images,result,manager,context)
        diagnostics["M1_M2_state_read_ms"]=read_ms
        diagnostics.update(m1_snapshot_read=True,
            m1_snapshot_epoch=result.diagnostics.get('state_epoch'),
            m1_world_state_count=len(result.diagnostics.get('world_state_snapshot',[])),
            world_identity_authorized_pixel_reuse=False)
        return output,diagnostics

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
        graph=tuple((type(module),getattr(module.forward,'__func__',module.forward))
            for module in self.model.modules())
        return (context.get('execution_signature'),self.config.mode,tensors,topology,graph,geometry,numeric_details(self.model),torch.is_autocast_enabled(),
                torch.get_autocast_gpu_dtype(),torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic,
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                torch.are_deterministic_algorithms_enabled())

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
        dense_start=perf_counter()
        if context is None:context=getattr(self,'_execution_context',{})
        self.cache.clear();self.executor.clear()
        outputs={};handles=[];stages=[]
        for i in self.prefix:
            def save(module, inputs, out, i=i):
                size=module.window_size
                b,h,w,_=out.shape
                per_camera=((h+size-1)//size)*((w+size-1)//size)
                total=b*per_camera
                stages.append(dict(block=i,updated_windows=total,total_windows=total,
                    executed_windows=total,reused_windows=0,mode='full',fallback_reason=reason,
                    per_camera={cid:dict(total_windows=per_camera,executed_windows=per_camera,reused_windows=0)
                        for cid in (context or {}).get('camera_ids',[])}))
                if capture:
                    outputs[i]=partition(out,size)[0].detach()
            handles.append(self.model.blocks[i].register_forward_hook(save))
        if self.stage_observer is not None:
            for i,block in enumerate(self.model.blocks):
                handles.append(block.register_forward_hook(
                    lambda m,inp,out,i=i:self.stage_observer(i,out)))
        try:
            features=self.model(images)
        finally:
            for handle in handles:handle.remove()
        publication_error=None
        if capture:
            try:
                self.cache.commit(images,outputs,context,signature,features[0].stride())
            except RuntimeError as error:
                self.cache.clear();self.executor.clear()
                capture=False;publication_error=('nonfinite_cache_features' if isinstance(error,NonFiniteCacheError)
                    else 'cache_publication_failed')
        self.last_diagnostics=dict(mode='full',fallback_reason=reason,stages=stages,
            cache_bytes=self.cache.bytes,cache_version=self.cache.version,
            cache_saved=capture,cache_source_frame=context['frame_id'] if capture else None,
            M2_execution_ms=(perf_counter()-dense_start)*1000,
            actual_skipped_windows=0,cost_table_status=self.scheduler.costs.status,
            task_quality_certified=(self.config.mode=='auto' and self.conditional_executor is None
                and publication_error!='nonfinite_cache_features'),verified_exact_windows=0,
            cache_publication_error=publication_error,
            per_camera_windows={cid:dict(executed_windows=sum(v['per_camera'].get(cid,{}).get('executed_windows',0) for v in stages),reused_windows=0)
                for cid in (context or {}).get('camera_ids',[])},
            cache_capture_reason=None if capture else (getattr(self,'_capture_skip',None) or reason))
        return features,self.last_diagnostics

    def forward(self,current_images,m1_result=None,m1_state=None,execution_context=None,computation_state=None):
        if computation_state is not None:
            from worldstream.m1.state_types import M1Result
            if m1_result is not None:
                raise ValueError('Pass either M1Result or ComputationStateBundle')
            m1_result = M1Result(computation_state.scene_id,computation_state.frame_id,computation_state.timestamp,
                computation_state=computation_state,
                full_refresh_hint=bool(computation_state.diagnostics.get('full_refresh_hint')),
                diagnostics=dict(transaction_status=computation_state.diagnostics.get('transaction_status'),
                    execution_signature=computation_state.execution_signature))
        start = perf_counter()
        self._adapter_diagnostics={}
        self._execution_context=execution_context or {}
        active_cache = (self.conditional_executor.cache if self.conditional_executor is not None else
            self.camera_executor.cache if self.config.reuse_boundary == 'camera' else self.cache)
        previous_context = dict(active_cache.context or {})
        lineage = (active_cache.sources if isinstance(active_cache.sources,list) else
            [s for values in active_cache.sources.values() for s in values])
        computation_times = [s['computation_timestamp'] for s in lineage if _finite_time(s.get('computation_timestamp'))]
        value, diagnostics = self._execute_forward(current_images,m1_result,m1_state,execution_context)
        context = execution_context or {}
        diagnostics.update(self._adapter_diagnostics)
        diagnostics['cache_context'] = {k: copy.deepcopy(context.get(k)) for k in
            ('scene_id','frame_id','timestamp','camera_ids','execution_signature','coordinate_space','state_epoch')}
        diagnostics['state_interface'] = self.config.state_interface
        diagnostics['cache_boundary'] = self.config.reuse_boundary
        if self.config.reuse_boundary == 'window':
            diagnostics['cacheable_blocks'] = list(self.prefix)
            diagnostics['global_barrier_block'] = len(self.prefix) if len(self.prefix)<len(self.model.blocks) else None
            diagnostics['execution_scope'] = 'independent_window_prefix; remaining blocks execute natively'
            diagnostics['partial_window_numerics_require_quality_evidence'] = True
        if diagnostics.get('cache_saved'):
            if self.config.reuse_boundary == 'camera':
                diagnostics.setdefault('output_camera_provenance',copy.deepcopy(active_cache.sources))
            else:
                diagnostics.setdefault('output_window_provenance',copy.deepcopy(active_cache.sources))
                p = self.model.patch_embed.proj
                diagnostics['prefix_window_geometry'] = {i:[p.kernel_size[0]*self.model.blocks[i].window_size,
                    p.kernel_size[1]*self.model.blocks[i].window_size] for i in self.prefix}
                diagnostics['prefix_dependency_footprints'] = self.planner.input_footprints(*current_images.shape[-2:])
        diagnostics['recomputed_windows'] = sum(s.get('executed_windows',0) for s in diagnostics.get('stages',[]))
        diagnostics['cache_hits'] = (diagnostics.get('actual_reused_cameras',0) if self.config.reuse_boundary=='camera'
                                     else diagnostics.get('actual_skipped_windows',0))
        diagnostics['actual_skipped_block_computations'] = sum(
            s.get('reused_windows',0) for s in diagnostics.get('stages',[]))
        diagnostics['window_count_scope'] = 'block_window_pairs_in_cacheable_prefix'
        diagnostics.setdefault('reuse_mode','exact' if diagnostics['cache_hits'] else 'full')
        diagnostics['source_mismatch'] = bool('mismatch' in (diagnostics.get('fallback_reason') or ''))
        now,old = context.get('timestamp'),previous_context.get('timestamp')
        diagnostics['cache_verification_age'] = now-old if _finite_time(now) and _finite_time(old) else None
        diagnostics['source_age'] = now-min(computation_times) if _finite_time(now) and computation_times else None
        diagnostics.setdefault('M2_validation_ms', max(0.,(perf_counter()-start)*1000-
            (diagnostics.get('M2_execution_ms') or 0.)))
        diagnostics['actual_reused_spatial_windows'] = next(
            (s['reused_windows'] for s in diagnostics.get('stages',[]) if 'reused_windows' in s),0)
        diagnostics.setdefault('M2_execution_ms', None)
        bundle = getattr(m1_result,'computation_state',None)
        if bundle is not None:
            from worldstream.m1.mask_codec import decode_mask
            import numpy as np
            try:
                masks = np.stack([decode_mask(bundle.refresh_contract[cid]['reuse_candidate_mask'])
                    for cid in context['camera_ids']])
                diagnostics['candidate_cameras'] = int(masks.any(axis=(1,2)).sum())
                p = self.model.patch_embed.proj
                window = self.model.blocks[0].window_size
                sizes=[self.model.blocks[i].window_size for i in self.prefix]
                diagnostics['candidate_windows'] = sum(int((~cpu_window_opportunity(masks,p,size)).sum()) for size in sizes)
                diagnostics['complete_candidate_windows'] = sum(int(cpu_window_opportunity(~masks,p,size).sum()) for size in sizes)
                for name,key in [('dirty_hint_windows','dirty_hint_mask'),('unknown_windows','unknown_mask')]:
                    values=np.stack([decode_mask(bundle.refresh_contract[cid][key]) for cid in context['camera_ids']])
                    diagnostics[name]=sum(int((~cpu_window_opportunity(values,p,size)).sum()) for size in sizes)
            except (KeyError,ValueError,TypeError):
                diagnostics['candidate_windows'] = None
        diagnostics['M2_wall_ms'] = (perf_counter()-start)*1000
        diagnostics['timing_scope'] = 'host_wall_including_required_validation_and_publication_syncs'
        self.last_diagnostics = diagnostics
        return value, diagnostics

    def _execute_forward(self,current_images,m1_result=None,m1_state=None,execution_context=None):
        try:
            context=execution_context or {}
            if (self.config.enabled and context.get('timestamp') is not None
                    and not isinstance(context['timestamp'],(int,float))):
                self.reset()
                return self._dense(current_images,'invalid_execution_timestamp')
            if (self.config.enabled and self.config.state_interface == 'source_linked'
                    and getattr(m1_result,'diagnostics',{}).get('transaction_status') == 'aborted'):
                self.reset()
                return self._dense(current_images,'m1_aborted_transaction')
            if self.config.enabled and self.conditional_executor is not None:
                value,self.last_diagnostics=self.conditional_executor.forward(current_images,
                    getattr(m1_result,'computation_state',None),context,self.signature(context),
                    getattr(m1_result,'full_refresh_hint',False),self.stage_observer)
                return value,self.last_diagnostics
            if (self.config.enabled and self.config.cheap_gate and self.config.reuse_boundary=='window'
                    and self.last_admission.get('reject') and
                    self.last_admission.get('frame_id')==context.get('frame_id') and
                    self.last_admission.get('scene_id')==context.get('scene_id')):
                self._capture_skip='capture_not_expected_to_pay'
                return self._dense(current_images,'cheap_input_rejection')
            if self.config.enabled and self.config.reuse_boundary=='camera':
                context=dict(context,necessary_validation_ms=validation_budget(context,m1_result,m1_state))
                admission_reason = None
                admission_details = {}
                force_cameras = None
                if self.config.state_interface == 'source_linked':
                    adapter_start=perf_counter()
                    dirty, admission_reason = computation_admission(
                        getattr(m1_result,'computation_state',None),context,current_images,self.camera_executor.cache,
                        getattr(m1_result,'full_refresh_hint',False),
                        selective_camera=self.config.camera_selective_invalidation,
                        separate_computation_age=self.config.experimental_exact_lease_separation,
                        details=admission_details)
                    admission_details['M1_M2_adapter_ms']=(perf_counter()-adapter_start)*1000
                    if dirty is not None and dirty.any():
                        if self.config.camera_selective_invalidation:
                            force_cameras = dirty.any(axis=(1,2)).tolist()
                        else:admission_reason = 'm1_dirty_camera_observation'
                    if admission_reason:
                        self.camera_executor.reset()
                value,self.last_diagnostics=self.camera_executor.forward(current_images,
                    context,self.signature(context),force_recompute_cameras=force_cameras)
                self.last_diagnostics.update(admission_details)
                self.last_diagnostics['global_cache_reset_reason'] = (admission_reason or
                    self.last_diagnostics.get('global_cache_reset_reason'))
                if admission_reason:self.last_diagnostics['fallback_reason'] = admission_reason
                return value,self.last_diagnostics
            return self._forward(current_images,m1_result,m1_state,execution_context)
        except torch.cuda.OutOfMemoryError:
            self.reset()
        if current_images.is_cuda:torch.cuda.empty_cache()
        return self._dense(current_images,'memory_budget')

    def _forward(self,images,m1_result=None,m1_state=None,execution_context=None):
        validation_start = perf_counter()
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
        if not self.prefix:return self._dense(images,self.planner.rejection_reason or 'global_dependency')
        p=self.model.patch_embed.proj
        if p.padding!=(0,0) or p.dilation!=(1,1) or p.stride!=p.kernel_size:
            return self._dense(images,'shape_unsupported')
        if context.get('timestamp') is not None and not _finite_time(context['timestamp']):
            return self._dense(images,'invalid_execution_timestamp')
        if self.cache.budget_bytes==0:return self._dense(images,'memory_budget')
        if self.config.state_interface == 'source_linked':
            adapter_start=perf_counter()
            admission_details={}
            invalid,reason=computation_admission(getattr(m1_result,'computation_state',None),context,images,self.cache,
                getattr(m1_result,'full_refresh_hint',False),selective_camera=True,details=admission_details,
                window_footprints=self.planner.input_footprints(*images.shape[-2:]))
            adapter_ms=(perf_counter()-adapter_start)*1000
            self._adapter_diagnostics=dict(admission_details,M1_M2_adapter_ms=adapter_ms)
        else:
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
        if reason and reason not in ('m1_full_refresh','insufficient_stable_windows','no_measured_beneficial_count',
                                     'no_valid_cache','m1_cache_source_mismatch','m1_computation_unavailable'):
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
        validation_ms = (perf_counter()-validation_start)*1000
        execution_start = perf_counter()
        from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
        x=self.model.patch_embed(images)
        if self.model.pos_embed is not None:
            x=x+get_abs_pos(self.model.pos_embed,self.model.pretrain_use_cls_token,x.shape[1:3])
        next_outputs={};stages=[];next_sources={}
        for i,block in enumerate(self.model.blocks):
            if i in self.prefix:
                ids=ids_by_block[i];total=plans[i].updated.numel()
                key=cost_key(block,x,i) if self.config.mode=='auto' or self.scheduler.costs.records else ''
                measured=self.scheduler.costs.lookup(key,len(ids))
                strategy=measured.get('packing_backend','spatial') if measured else 'spatial'
                sparse,fallback=self.scheduler.choose(key,len(ids),total)
                if (sparse and 0 < len(ids) < total and x.is_cuda and x.dtype==torch.float32
                        and torch.backends.cuda.matmul.allow_tf32):
                    sparse,fallback=False,'uncertified_tf32_partial_window_execution'
                if sparse:
                    x,packed=self.executor.execute(block,x,self.cache.outputs[i],ids,strategy=strategy)
                else:
                    x=block(x);packed=partition(x,block.window_size)[0]
                next_outputs[i]=packed.detach()
                # Metadata is built before atomic cache publication. Computed
                # sources stay immutable when verification advances to this frame.
                computed_ids = (set() if sparse and not len(ids) else
                    set(range(total)) if not sparse or len(ids) == total else set(ids.tolist()))
                next_sources[i] = [dict(computation_frame_id=context['frame_id'],
                    computation_timestamp=context.get('timestamp'), computation_version=self.cache.version+1,
                    camera_id=context['camera_ids'][j//(total//len(images))],
                    producer_layer=i, dependency_scope='closed_window_prefix',
                    task_quality_certified=(not sparse or len(ids)==total or
                        (self.config.mode=='auto' and bool(measured and measured.get('quality_certified')))))
                    if j in computed_ids else copy.deepcopy(self.cache.sources[i][j]) for j in range(total)]
                stages.append(dict(block=i,updated_windows=len(ids),total_windows=total,
                    executed_windows=len(ids) if sparse else total,reused_windows=total-len(ids) if sparse else 0,
                    mode='packed' if sparse else 'full',fallback_reason=fallback,
                    per_camera={cid:dict(total_windows=total//len(images),
                        executed_windows=sum(j in computed_ids for j in range(c* (total//len(images)),(c+1)*(total//len(images)))),
                        reused_windows=sum(j not in computed_ids for j in range(c* (total//len(images)),(c+1)*(total//len(images)))))
                        for c,cid in enumerate(context['camera_ids'])},
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
        saved=True;publication_error=None
        try:
            self.cache.commit(images,next_outputs,context,signature,result.stride(),next_sources)
        except RuntimeError as error:
            self.cache.clear();self.executor.clear()
            saved=False;publication_error=('nonfinite_cache_features' if isinstance(error,NonFiniteCacheError)
                else 'cache_publication_failed')
            for units in next_sources.values():
                for source in units:
                    if source['computation_frame_id']==context['frame_id']:
                        source['computation_version']=None
                        if publication_error=='nonfinite_cache_features':source['task_quality_certified']=False
        skipped=sum(s.get('reused_windows',0) for s in stages)
        self.last_diagnostics=dict(mode='incremental' if skipped else 'full',stages=stages,
            cache_bytes=self.cache.bytes,cache_version=self.cache.version,cache_saved=saved,
            cache_publication_error=publication_error,
            task_quality_certified=self.config.mode=='auto' and publication_error!='nonfinite_cache_features' and all(source.get('task_quality_certified',True)
                for units in next_sources.values() for source in units),
            cost_table_status=self.scheduler.costs.status,
            actual_skipped_windows=skipped,patch_embedding='full',
            output_window_provenance=copy.deepcopy(next_sources),
            verified_exact_windows=sum(plan.updated.numel()-len(ids) for plan,ids in zip(plans,ids_by_block)),
            M2_validation_ms=validation_ms, M2_execution_ms=(perf_counter()-execution_start)*1000,
            evidence='same_coordinate_exact_input_and_dependency_closure',m1_transformer_certificate_used=False)
        if self.config.state_interface == 'source_linked':
            self.last_diagnostics.update(admission_details)
            self.last_diagnostics['M1_M2_adapter_ms']=adapter_ms
        self.last_diagnostics['per_camera_windows']={cid:dict(
            executed_windows=sum(s.get('per_camera',{}).get(cid,{}).get('executed_windows',0) for s in stages),
            reused_windows=sum(s.get('per_camera',{}).get(cid,{}).get('reused_windows',0) for s in stages))
            for cid in context['camera_ids']}
        return [result],self.last_diagnostics
