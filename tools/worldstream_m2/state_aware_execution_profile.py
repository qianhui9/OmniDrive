"""Real EVA prefix work and GPU kernels; unqualified controlled-source costs."""
import hashlib
import json
from collections import Counter
import numpy as np
import torch
from worldstream.m1 import M1Config
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.cost_model import cost_key
from worldstream.m2.profiler import measure,errors
from tools.worldstream_m1.source_state_refinement_replay import preprocessed_frame
from tools.worldstream_m1.evidence_experiments import load_inputs
from tools.worldstream_m1.io_utils import write_json
from .state_aware_feasibility import ROOT
from .common import backbone


@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    import cv2
    cv2.setNumThreads(1)
    from mmcv import Config
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py')
    specs,infos,_,_=load_inputs('development');token=specs[0]['samples'][0]
    frame,base=preprocessed_frame(infos[token],cfg,M1Config.load('configs/worldstream_m1_engineering_frozen.yaml'))
    base=base.cuda();current=base.clone()
    current[0,:,:256,:256]=base[0,:,:256,256:512];current[0,:,:256,256:512]=base[0,:,:256,:256]
    model=backbone();reference=model(current)[0];content=model.patch_embed(current)
    context=dict(scene_id=frame.scene_id,frame_id=token,timestamp=frame.timestamp,camera_ids=list(frame.cameras),
        coordinate_space='preprocessed_image',execution_signature=frame.execution_signature,state_epoch=0)
    later=dict(context,frame_id='controlled_layout_next',timestamp=frame.timestamp+.5)
    full=measure(lambda:model(current),warmup=3,repeats=10);records=[];kernel=[]
    observed_work=[];hs=[]
    for i,b in enumerate(model.blocks):
        hs.append(b.attn.register_forward_pre_hook(lambda m,a,i=i:observed_work.append(dict(block=i,shape=list(a[0].shape)))))
    for optimized in (False,True):
        runtime=IncrementalVisionRuntime(model,dict(mode='exact',reuse_boundary='window',state_interface='source_linked',
            experimental_conditional_reuse=True,conditional_packing_optimization=optimized))
        ex=runtime.conditional_executor;ex.full(base,context,runtime.signature(context),'profile_seed')
        saved=(ex.cache.images,ex.cache.outputs,ex.cache.context,ex.cache.signature,ex.cache.output_stride,ex.cache.sources,ex.cache.version,ex.content)
        total=len(ex.cache.outputs[0]);key=cost_key(model.blocks[0],content,0)
        for updated in (0,1,6,18,36,total):
            sources=np.arange(total,dtype=np.int64);sources[0],sources[1]=1,0
            if updated:sources[-updated:]=-1
            def execute():
                ex.cache.images,ex.cache.outputs,ex.cache.context,ex.cache.signature,ex.cache.output_stride,ex.cache.sources,ex.cache.version,ex.content=saved
                # Real Patch Embedding and cache publication included. Source
                # mapping is an explicit offline intervention, never M1 input.
                return ex.execute(current,model.patch_embed(current),later,runtime.signature(later),sources,
                    exact_windows=max(0,total-updated-2),conditional_windows=2 if updated<total else 0)
            timing=measure(execute,warmup=3,repeats=10)
            observed_work.clear();value,diag=execute();torch.cuda.synchronize()
            records.append(dict(key=key,updated=updated,total=total,optimization=optimized,timing=timing,
                full=full,feature_error=errors(value[0],reference),actual_GPU_work=list(observed_work),runtime=diag,
                task_quality_certified=False,auto_eligible=False,
                boundary='Two original independent prefix blocks, all global/tail blocks execute',
                source='Known controlled RGB permutation; M1 admission bypassed for research sensitivity only',
                alignment_and_validation_ms=None,alignment_and_validation_status='Not measured by lower-level bypass; see full integration M2_validation_ms'))
            if updated in (1,36):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True,profile_memory=True) as prof:
                    execute();torch.cuda.synchronize()
                path=ROOT/'profile_output'/f'conditional_opt{int(optimized)}_updated{updated}.json'
                prof.export_chrome_trace(str(path))
                events=[e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
                kernel.append(dict(optimization=optimized,updated=updated,trace=str(path),device_events=len(events),
                    status='measured' if events else 'CUDA_device_trace_unavailable',
                    device_duration_ms=sum(e.time_range.elapsed_us() for e in events)/1000 if events else None,
                    event_names=dict(Counter(e.name for e in events)),
                    host_events={e.key:dict(calls=e.count,self_cpu_ms=e.self_cpu_time_total/1000,cpu_ms=e.cpu_time_total/1000)
                        for e in prof.key_averages() if any(term in e.key.lower() for term in ('cuda','copy','index','gather','scatter'))},
                    interpretation='Device event duration sum; distinct from synchronized wall and GPU intervals. Includes memcpy device events if present.'))
            write_json(ROOT/'conditional_execution_profile.json',dict(status='running',records=records,kernel_profiles=kernel))
    for h in hs:h.remove()
    digest=hashlib.sha256()
    for name,value in list(model.named_parameters())+list(model.named_buffers()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    witness=torch.randn(16,16);torch.backends.cuda.matmul.allow_tf32=False
    delta=float(((witness.cuda()@witness.cuda()).cpu()-witness@witness).abs().max());assert delta<1e-4
    write_json(ROOT/'conditional_execution_profile.json',dict(status='completed',records=records,kernel_profiles=kernel,
        full_backbone_weight_sha256=digest.hexdigest(),fresh_kernel_witness_max_abs=delta,
        image_signature=dict(shape=list(current.shape),stride=list(current.stride()),dtype=str(current.dtype),device=str(current.device)),
        position_mechanism=dict(absolute_position_shape=list(model.pos_embed.shape) if model.pos_embed is not None else None,
            blocks=[dict(block=i,window=b.window_size,attention_class=type(b.attn).__name__,
                RoPE_class=type(b.attn.rope).__name__,
                RoPE_frequency_shape=list(b.attn.rope.freqs_cos.shape) if getattr(b.attn.rope,'freqs_cos',None) is not None else None)
                for i,b in enumerate(model.blocks)],
            interpretation='Whole-window translation preserves local window coordinate order, but ABS contaminates Q/K/V and residuals. Identical local RoPE grids do not undo historical ABS/attention/MLP context. Global attention is per camera; later heads fuse cameras and temporal state.'),
        GPU=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
        qualified_conditional_records=[],production_eligible=False))


if __name__=='__main__':main()
