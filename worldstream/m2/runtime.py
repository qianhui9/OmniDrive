"""Strict EVA prefix reuse with a global dependency barrier and bounded GPU state."""
import torch
from .config import M2Config
from .cost_model import CostModel, cost_key
from .dependency_planner import DependencyPlanner, patch_updates, exact_pixel_changes
from .m1_adapter import invalidation_mask
from .packing import partition, window_mask
from .scheduler import Scheduler
from .state_cache import StateCache
from .window_executor import WindowExecutor


class IncrementalVisionRuntime:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config if isinstance(config, M2Config) else M2Config(**(config or {}))
        self.planner = DependencyPlanner(model)
        self.prefix = self.planner.cacheable_prefix()
        self.cache = StateCache(self.config.memory_budget_mb * 1024**2)
        self.executor = WindowExecutor()
        self.scheduler = Scheduler(self.config, CostModel(self.config.cost_table))
        self.last_diagnostics = {}
        self.stage_observer = None  # Experiment-only numerical oracle; disabled during timing.

    def reset(self):
        self.cache.clear()
        self.executor.clear()

    def signature(self, context):
        # Detect replaced tensors as well as in-place optimizer/load_state_dict updates.
        tensors = tuple((name, t.data_ptr(), t._version, tuple(t.shape), str(t.dtype), str(t.device))
                        for name, t in list(self.model.named_parameters()) + list(self.model.named_buffers()))
        topology = tuple((type(b), b.window_size, b.use_residual_block, b.attn.flash_attn)
                         for b in self.model.blocks)
        return (context.get('execution_signature'), tensors, topology, torch.is_autocast_enabled(),
                torch.get_autocast_gpu_dtype(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)

    def _dense(self, images, reason):
        self.reset()
        self.last_diagnostics = dict(mode='full', fallback_reason=reason, stages=[],
                                     cache_bytes=0, actual_skipped_windows=0)
        return self.model(images), self.last_diagnostics

    def forward(self, current_images, m1_result=None, m1_state=None, execution_context=None):
        try:
            return self._forward(current_images, m1_result, m1_state, execution_context)
        except torch.cuda.OutOfMemoryError:
            self.reset()
        # Exit the exception scope before retrying, releasing sparse temporaries.
        if current_images.is_cuda:
            torch.cuda.empty_cache()
        return self._dense(current_images, 'memory_budget')

    def _forward(self, current_images, m1_result=None, m1_state=None, execution_context=None):
        images = current_images
        context = dict(execution_context or {})
        if not self.config.enabled:
            return self._dense(images, 'disabled')
        if any(m.training for m in self.model.modules()) or torch.is_grad_enabled():
            return self._dense(images, 'inference_only')
        if not all(k in context for k in ('scene_id', 'frame_id', 'camera_ids')):
            return self._dense(images, 'missing_execution_context')
        if images.ndim != 4 or len(context['camera_ids']) != len(images):
            return self._dense(images, 'shape_unsupported')
        prefix = self.planner.cacheable_prefix()
        if prefix != self.prefix:
            self.reset()
            self.prefix = prefix
        if not self.prefix:
            return self._dense(images, 'global_dependency')
        p = self.model.patch_embed.proj
        if p.padding != (0, 0) or p.dilation != (1, 1) or p.stride != p.kernel_size:
            return self._dense(images, 'shape_unsupported')
        invalid, reason = invalidation_mask(m1_result, m1_state, context, images)
        if reason:
            return self._dense(images, reason)
        # Reject entirely invalid M1 windows before comparing/copying GPU activations.
        if invalid is not None:
            initial = patch_updates(invalid, p)
            if bool(window_mask(initial, self.model.blocks[0].window_size).all()):
                return self._dense(images, 'insufficient_stable_windows')
        h, w = images.shape[-2]//p.stride[0], images.shape[-1]//p.stride[1]
        c = p.out_channels
        estimate = images.numel()*images.element_size()
        for i in self.prefix:
            size = self.model.blocks[i].window_size
            estimate += len(images)*((h+size-1)//size*size)*((w+size-1)//size*size)*c*images.element_size()
        # Reserve space for input history, cache replacement, gather and temporary scatter.
        if 3*estimate > self.cache.budget_bytes:
            return self._dense(images, 'memory_budget')
        if images.is_cuda:
            free, _ = torch.cuda.mem_get_info(images.device)
            if 3*estimate > free:
                return self._dense(images, 'memory_budget')
        signature = self.signature(context)
        mismatch = self.cache.mismatch(images, context, signature)
        if mismatch:
            self.reset()
        # A previous exact input is the independent equality witness. M1 only restricts it.
        changed = (torch.ones_like(images[:, 0], dtype=torch.bool) if self.cache.images is None
                   else exact_pixel_changes(images, self.cache.images))
        if invalid is not None:
            changed |= invalid
        plans = self.planner.plan(patch_updates(changed, p))
        from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
        x = self.model.patch_embed(images)  # PatchEmbed is always executed, never claimed skipped.
        if self.model.pos_embed is not None:
            x = x + get_abs_pos(self.model.pos_embed, self.model.pretrain_use_cls_token, x.shape[1:3])
        next_outputs, stages = {}, []
        for i, block in enumerate(self.model.blocks):
            if i in self.prefix:
                ids = plans[i].updated.nonzero(as_tuple=False).flatten()
                total = plans[i].updated.numel()
                key = cost_key(block, x) if self.config.mode == 'auto' else ''
                sparse, fallback = self.scheduler.choose(key, len(ids), total, mismatch)
                if sparse:
                    x, packed = self.executor.execute(block, x, self.cache.outputs[i], ids)
                else:
                    x = block(x)
                    packed, _ = partition(x, block.window_size)
                next_outputs[i] = packed.detach()
                stages.append(dict(block=i, updated_windows=len(ids), total_windows=total,
                    executed_windows=len(ids) if sparse else total, reused_windows=total-len(ids) if sparse else 0,
                    mode='packed' if sparse else 'full', fallback_reason=fallback,
                    source_frame=self.cache.context['frame_id'] if sparse else None,
                    source_version=self.cache.version if sparse else None))
            else:
                x = block(x)
                stages.append(dict(block=i, mode='full', fallback_reason=plans[i].reason,
                                   global_expansion=not bool(block.window_size)))
            if self.stage_observer is not None:
                self.stage_observer(i, x)
        self.cache.images = images.detach().clone()
        self.cache.outputs = next_outputs
        self.cache.context = context
        self.cache.signature = signature
        self.cache.version += 1
        skipped = sum(s.get('reused_windows', 0) for s in stages)
        self.last_diagnostics = dict(mode='incremental' if skipped else 'full', stages=stages,
            cache_bytes=self.cache.bytes, cache_version=self.cache.version,
            actual_skipped_windows=skipped, patch_embedding='full',
            evidence='same_coordinate_exact_input_and_dependency_closure',
            m1_transformer_certificate_used=False)
        return [x.permute(0, 3, 1, 2)], self.last_diagnostics
