"""Isolated research prefix executor. No automatic quality qualification.

Exact runtime caches never read these outputs. The entire attention window is
the reuse unit; all blocks after the independent prefix always execute.
"""
import copy
from time import perf_counter
import numpy as np
import torch
from .state_cache import StateCache
from .dependency_planner import DependencyPlanner,exact_pixel_changes,patch_updates
from .packing import partition
from .window_executor import WindowExecutor
from .source_alignment import token_sources,source_windows
from .m1_adapter import computation_admission


class ConditionalPrefixExecutor:
    def __init__(self,model,config):
        self.model,self.config=model,config
        self.cache=StateCache(config.memory_budget_mb*1024**2)
        self.executor=WindowExecutor();self.content=None
        self.prefix=DependencyPlanner(model).cacheable_prefix()

    def reset(self):
        self.cache.clear();self.executor.clear();self.content=None

    def _publish(self,images,content,outputs,context,signature,stride,sources=None):
        owned=content.detach().clone()
        required=images.numel()*images.element_size()+owned.numel()*owned.element_size()+sum(v.numel()*v.element_size() for v in outputs.values())
        if required>self.cache.budget_bytes:return False
        if sources is None:
            sources={i:[dict(computation_frame_id=context['frame_id'],computation_timestamp=context.get('timestamp'),
                computation_version=self.cache.version+1,camera_id=context['camera_ids'][j//(len(values)//len(images))],
                producer_layer=i,dependency_scope='closed_window_prefix',computation_window_index=j,
                reuse_mode='full',alignment_chain=0) for j in range(len(values))] for i,values in outputs.items()}
        self.cache.commit(images,outputs,context,signature,stride,sources)
        self.content=owned
        return True

    def full(self,images,context,signature,reason,observer=None):
        outputs={};content={};hs=[]
        if self.prefix:
            hs.append(self.model.patch_embed.register_forward_hook(lambda m,a,v:content.update(x=v.detach())))
            for i in self.prefix:
                hs.append(self.model.blocks[i].register_forward_hook(
                    lambda m,a,v,i=i:outputs.update({i:partition(v,self.model.blocks[i].window_size)[0].detach()})))
        if observer is not None:
            for i,b in enumerate(self.model.blocks):hs.append(b.register_forward_hook(lambda m,a,v,i=i:observer(i,v)))
        try:value=self.model(images)
        finally:
            for h in hs:h.remove()
        saved=False
        valid_context=(all(k in context for k in ('scene_id','frame_id','camera_ids','timestamp'))
            and len(context['camera_ids'])==len(images) and len(set(context['camera_ids']))==len(images))
        if (self.config.cache_capture!='never' and outputs and 'x' in content and valid_context and not torch.is_grad_enabled()
                and not any(m.training for m in self.model.modules()) and torch.isfinite(images).all()):
            saved=self._publish(images,content['x'],outputs,context,signature,value[0].stride())
        return value,dict(mode='full',reuse_mode='full',fallback_reason=reason,cache_saved=saved,
            cache_version=self.cache.version,cache_bytes=self.cache.bytes+(0 if self.content is None else self.content.numel()*self.content.element_size()),
            actual_skipped_windows=0,conditional_reused_windows=0,actual_skipped_block_computations=0,
            transformer_dependency_certified=False,task_quality_certified=False,stages=[])

    def forward(self,images,bundle,context,signature,force=False,observer=None):
        started=perf_counter()
        if (any(m.training for m in self.model.modules()) or torch.is_grad_enabled()):
            self.reset();return self.full(images,context,signature,'inference_only',observer)
        if not self.prefix or any(self.model.blocks[i].window_size!=self.model.blocks[0].window_size for i in self.prefix):
            return self.full(images,context,signature,'unsupported_conditional_topology',observer)
        if not all(k in context for k in ('scene_id','frame_id','camera_ids','timestamp')):
            self.reset();return self.full(images,context,signature,'missing_execution_context',observer)
        if len(context['camera_ids'])!=len(images) or len(set(context['camera_ids']))!=len(images):
            self.reset();return self.full(images,context,signature,'camera_order_mismatch',observer)
        if self.config.cache_capture=='never':
            self.reset();return self.full(images,context,signature,'capture_disabled',observer)
        if self.config.mode!='exact':
            # There is no qualified conditional cost/quality population yet.
            # An ordinary Exact cost table cannot authorize contextual warping.
            return self.full(images,context,signature,'conditional_auto_unqualified',observer)
        reason=self.cache.mismatch(images,context,signature)
        if reason:
            self.reset();return self.full(images,context,signature,reason,observer)
        dirty,reason=computation_admission(bundle,context,images,self.cache,force)
        if reason:return self.full(images,context,signature,reason,observer)
        if not torch.isfinite(images).all():
            self.reset();return self.full(images,context,signature,'non_finite_input',observer)
        from worldstream.m1.mask_codec import decode_mask
        p=self.model.patch_embed.proj
        if (p.kernel_size!=p.stride or p.padding!=(0,0) or p.dilation!=(1,1)):
            return self.full(images,context,signature,'unsupported_conditional_patch_footprint',observer)
        ws=self.model.blocks[0].window_size
        plans=DependencyPlanner(self.model).plan(patch_updates(exact_pixel_changes(images,self.cache.images),p),stop_after=len(self.prefix))
        exact_ok=(~plans[-1].updated).cpu().numpy()
        exact_ok &= np.array([s.get('reuse_mode')!='experimental_conditional' for s in self.cache.sources[self.prefix[-1]]])
        sources=np.full(len(exact_ok),-1,np.int64)
        source_reports=[];per=len(sources)//len(images)
        for camera,cid in enumerate(context['camera_ids']):
            c=bundle.refresh_contract[cid]
            mapping=token_sources(bundle.observation_links[cid],decode_mask(c['reuse_candidate_mask']),
                decode_mask(c['dirty_hint_mask']),decode_mask(c['unknown_mask']),p.kernel_size,
                self.cache.context['frame_id'],cid,context['frame_id'])
            local=source_windows(mapping,ws)
            sources[camera*per:(camera+1)*per]=np.where(local>=0,local+camera*per,-1)
            source_reports.append(dict(camera=cid,observed_tokens=int((mapping.support_count>0).sum()),
                complete_tokens=int(mapping.complete.sum()),complete_source_windows=int((local>=0).sum())))
        # Dirty is an explicit force, Unknown can pass ONLY the independent
        # same-coordinate full input comparison. Conditional units require full
        # actual observation support and all content residuals below threshold.
        forced=DependencyPlanner(self.model).plan(patch_updates(torch.as_tensor(dirty,device=images.device),p),stop_after=len(self.prefix))[-1].updated.cpu().numpy()
        exact_ok &= ~forced
        conditional=(sources>=0)&~forced&~exact_ok
        if not exact_ok.any() and not conditional.any():
            validation_ms=(perf_counter()-started)*1000
            value,diag=self.full(images,context,signature,'no_complete_source_window',observer)
            diag.update(source_alignment=source_reports,M2_validation_ms=validation_ms)
            return value,diag
        content=self.model.patch_embed(images)
        current_content,_=partition(content,ws);old_content,_=partition(self.content,ws)
        from .packing import partition as pack
        valid,_=pack(torch.ones_like(content[...,:1],dtype=torch.bool),ws)
        proposed=np.flatnonzero(conditional)
        if len(proposed):
            dst=torch.as_tensor(proposed,device=images.device);src=torch.as_tensor(sources[proposed],device=images.device)
            residual=(current_content.index_select(0,dst)-old_content.index_select(0,src)).abs()
            residual=residual.masked_fill(~valid.index_select(0,dst),0).flatten(1).amax(1)
            accepted=(residual<=self.config.conditional_max_patch_error).cpu().numpy()
            conditional[proposed[~accepted]]=False
        sources[exact_ok]=np.flatnonzero(exact_ok)
        sources[~(exact_ok|conditional)]=-1
        updated=np.flatnonzero(sources<0)
        if len(updated)==len(sources):
            # Patch result is already available; avoid computing it again.
            value,diag=self.execute(images,content,context,signature,sources,observer,source_reports,started)
            return value,diag
        return self.execute(images,content,context,signature,sources,observer,source_reports,started,
            exact_windows=int(exact_ok.sum()),conditional_windows=int(conditional.sum()))

    def execute(self,images,content,context,signature,sources,observer=None,reports=None,started=None,
                exact_windows=0,conditional_windows=0):
        from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
        execution_start=perf_counter()
        validation_ms=(execution_start-started)*1000 if started is not None else None
        x=content
        if self.model.pos_embed is not None:x=x+get_abs_pos(self.model.pos_embed,self.model.pretrain_use_cls_token,x.shape[1:3])
        ids=torch.as_tensor(np.flatnonzero(sources<0),device=images.device)
        reused=np.flatnonzero(sources>=0);outputs={};provenance={};stages=[]
        total=len(sources);version=self.cache.version+1
        for i,b in enumerate(self.model.blocks):
            if i in self.prefix:
                # Copy historical state into current layout. This is explicitly
                # position-dependent Approximate reuse for displaced windows;
                # it is never presented as position/RoPE correction or Exact.
                historical=self.cache.outputs[i]
                aligned=historical.clone()
                if len(reused):
                    dst=torch.as_tensor(reused,device=images.device);src=torch.as_tensor(sources[reused],device=images.device)
                    aligned.index_copy_(0,dst,historical.index_select(0,src))
                if len(ids)==total:
                    x=b(x);packed=partition(x,b.window_size)[0]
                else:x,packed=self.executor.execute(b,x,aligned,ids,
                    strategy='spatial' if self.config.conditional_packing_optimization else 'original')
                outputs[i]=packed.detach();provenance[i]=[]
                for j,src in enumerate(sources):
                    if src<0:
                        unit=dict(computation_frame_id=context['frame_id'],computation_timestamp=context['timestamp'],
                            computation_version=version,camera_id=context['camera_ids'][j//(total//len(images))],
                            producer_layer=i,dependency_scope='closed_window_prefix',reuse_mode='full',alignment_chain=0,
                            computation_window_index=j)
                    else:
                        unit=copy.deepcopy(self.cache.sources[i][int(src)])
                        displaced=int(src)!=j
                        unit.update(verification_frame=context['frame_id'],source_window=int(src),current_window=j,
                            reuse_mode='experimental_conditional' if displaced or unit.get('reuse_mode')=='experimental_conditional' else 'exact',
                            alignment_chain=unit.get('alignment_chain',0)+int(displaced))
                    provenance[i].append(unit)
                stages.append(dict(block=i,executed_windows=len(ids),reused_windows=total-len(ids),total_windows=total))
            else:x=b(x)
            if observer is not None:observer(i,x)
        result=x.permute(0,3,1,2)
        if self.cache.output_stride is not None and result.stride()!=self.cache.output_stride:
            owned=torch.empty_strided(result.shape,self.cache.output_stride,dtype=result.dtype,device=result.device);owned.copy_(result);result=owned
        saved=self._publish(images,content,outputs,context,signature,result.stride(),provenance)
        skipped=len(reused)*len(self.prefix)
        diag=dict(mode='experimental_conditional' if conditional_windows else 'incremental' if skipped else 'full',
            reuse_mode='experimental_conditional' if conditional_windows else 'exact' if skipped else 'full',
            fallback_reason=None if skipped else 'content_residual_rejected',stages=stages,
            actual_skipped_windows=skipped,actual_skipped_block_computations=skipped,
            conditional_reused_windows=conditional_windows,verified_exact_windows=exact_windows,
            cache_saved=saved,cache_version=self.cache.version,cache_bytes=self.cache.bytes+self.content.numel()*self.content.element_size(),
            output_window_provenance=copy.deepcopy(provenance),source_alignment=reports or [],
            M2_validation_ms=validation_ms,M2_execution_ms=(perf_counter()-execution_start)*1000,
            transformer_dependency_certified=False,task_quality_certified=False,
            position_context='Unresolved historical ABS/RoPE context for displaced outputs; research only.')
        return [result],diag
