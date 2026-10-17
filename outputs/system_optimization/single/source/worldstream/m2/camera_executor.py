"""Exact-input camera boundary. Downstream perception and temporal state still run."""
import json
import math
import torch
from .camera_feature_cache import CameraFeatureCache,layout_copy
from .cost_model import block_fingerprint


def numeric_details(model):
    return tuple((getattr(getattr(b.attn,'inner_attn',None),'softmax_scale',None),
        getattr(getattr(b.attn,'inner_attn',None),'fp16_enabled',None),
        getattr(b.attn.inner_attn_ln,'eps',None),getattr(b.mlp.ffn_ln,'eps',None)) for b in model.blocks)


def camera_cost_key(model,images):
    props=torch.cuda.get_device_properties(images.device) if images.is_cuda else None
    return json.dumps(dict(schema=1,boundary='eva_camera',model=block_fingerprint(model),
        shape=list(images.shape),stride=list(images.stride()),dtype=str(images.dtype),
        gpu=(props.name,props.major,props.minor,props.total_memory) if props else 'cpu',
        torch=torch.__version__,cuda=torch.version.cuda,
        flash=[b.attn.flash_attn for b in model.blocks],
        topology=[(b.window_size,b.use_residual_block,b.attn.scale,b.norm1.eps,b.norm2.eps) for b in model.blocks],
        patch=(model.patch_embed.proj.kernel_size,model.patch_embed.proj.stride,
               model.patch_embed.proj.padding,model.patch_embed.proj.dilation),
        pretrain_use_cls_token=model.pretrain_use_cls_token,
        numeric_details=numeric_details(model),
        cudnn=[torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic],
        autocast=torch.is_autocast_enabled(),autocast_dtype=str(torch.get_autocast_gpu_dtype()),
        tf32=[torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32]),sort_keys=True)


class CameraExecutor:
    def __init__(self,model,config):
        self.model=model;self.config=config
        self.cache=CameraFeatureCache(config.memory_budget_mb*1024**2)
        self.records=[];self.table_status='unprofiled_camera_cost'
        if config.camera_cost_table:
            try:
                with open(config.camera_cost_table) as f:table=json.load(f)
                if table['schema_version']!=1:raise ValueError('schema')
                records=table['records']
                if not isinstance(records,list):raise ValueError('records must be a list')
                for r in records:
                    if not isinstance(r,dict):raise ValueError('invalid record')
                    json.loads(r['key'])
                    for key in ('changed_cameras','reuse_wall_ms','full_wall_ms','reuse_gpu_ms','full_gpu_ms'):
                        if not isinstance(r[key],(float,int)) or not math.isfinite(r[key]) or r[key]<0:
                            raise ValueError('invalid measured cost')
                self.records=records;self.table_status='ready'
            except (OSError,ValueError,KeyError,TypeError):self.table_status='invalid_camera_cost_table'

    def reset(self):self.cache.clear()

    def _save(self,images,features,context,signature,changed):
        needed=(images.numel()*images.element_size()+features.numel()*features.element_size())
        if needed*3>self.cache.budget_bytes:
            self.reset();return False
        try:
            self.cache.publish(images,features,context,signature,changed)
        except torch.cuda.OutOfMemoryError:
            # The current output already exists. Drop state, return that output,
            # and do not run another entire EVA merely because saving it failed.
            self.reset()
            return False
        return True

    def forward(self,images,context,signature):
        cache=self.cache;n=len(images)
        if self.config.cache_capture=='never':
            self.reset()
            return self.model(images),dict(mode='full',fallback_reason='capture_disabled',
                actual_reused_cameras=0,cache_saved=False,cache_bytes=0)
        valid_context=all(k in context for k in ('scene_id','frame_id','camera_ids'))
        supported=(type(self.model).__name__=='EVAViT' and type(self.model).__module__.endswith('eva_vit') and
            images.ndim==4 and valid_context and len(context['camera_ids'])==n and len(set(context['camera_ids']))==n and
            not any(m.training for m in self.model.modules()) and not torch.is_grad_enabled())
        if not supported:
            self.reset();return self.model(images),dict(mode='full',fallback_reason='unsupported_camera_contract',actual_reused_cameras=0)
        if context.get('timestamp') is not None and not math.isfinite(context['timestamp']):
            self.reset()
            return self.model(images),dict(mode='full',fallback_reason='invalid_camera_timestamp',
                actual_reused_cameras=0,cache_saved=False,cache_bytes=0)
        mismatch=cache.mismatch(images,context,signature)
        now=context.get('timestamp');previous=(cache.context or {}).get('timestamp')
        if now is not None and (not math.isfinite(now) or (previous is not None and now<=previous)):
            mismatch='camera_timestamp_not_increasing'
        if mismatch:
            self.reset();out=self.model(images)
            saved=self._save(images,out[0],context,signature,[True]*n)
            return out,dict(mode='full',fallback_reason=mismatch,actual_reused_cameras=0,cache_saved=saved,
                            cache_version=cache.version,cache_bytes=cache.bytes)
        # A fast rejection per camera avoids reading every pixel on moving images.
        sample=images[:,:,::128,::128];old=cache.images[:,:,::128,::128]
        possible=((sample==old)&torch.isfinite(sample)).flatten(1).all(1)
        possible_ids=possible.nonzero(as_tuple=False).flatten()
        changed=torch.ones(n,dtype=torch.bool,device=images.device)
        if len(possible_ids):
            current=images.index_select(0,possible_ids);previous=cache.images.index_select(0,possible_ids)
            equal=((current==previous)&torch.isfinite(current)).flatten(1).all(1)
            changed[possible_ids]=~equal
        ids=changed.nonzero(as_tuple=False).flatten();count=len(ids)
        reason=None
        if count==n:reason='no_exact_camera'
        elif self.config.mode=='auto':
            key=camera_cost_key(self.model,images)
            record=next((r for r in self.records if r.get('key')==key and r.get('changed_cameras')==count),None)
            if record is None:reason='unprofiled_camera_cost'
            elif record.get('quality_certified') is not True:reason='camera_downstream_quality_unverified'
            elif record['reuse_wall_ms']>=record['full_wall_ms'] or record['reuse_gpu_ms']>=record['full_gpu_ms']:
                reason='camera_cost_not_beneficial'
            elif record['full_wall_ms']-record['reuse_wall_ms']<=context.get('necessary_validation_ms',0.):
                reason='camera_validation_cost_not_beneficial'
        source_version=cache.version;source_frames=[s['computation_frame'] for s in cache.sources]
        if reason:
            out=self.model(images);computed=[True]*n
        else:
            result=layout_copy(cache.features)
            if count:
                fresh=self.model(images.index_select(0,ids))[0]
                result.index_copy_(0,ids,fresh)
            out=[result];computed=changed.cpu().tolist()
        saved=self._save(images,out[0],context,signature,computed)
        return out,dict(mode='full' if reason else 'camera_reuse',fallback_reason=reason,
            actual_reused_cameras=0 if reason else n-count,recomputed_cameras=n if reason else count,
            camera_ids=list(context['camera_ids']),source_version=source_version,
            source_computation_frames=source_frames,cache_version=cache.version,cache_bytes=cache.bytes,
            output_camera_provenance=[dict(s) for s in cache.sources],
            cache_saved=saved,feature_stride=list(out[0].stride()),
            actual_skipped_windows=0,evidence='finite_same_coordinate_full_camera_input_equality',
            task_quality_certified=self.config.mode=='auto' and reason is None)
