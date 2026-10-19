"""Frozen-M1 / actual EVA feasibility before changing the M2 runtime."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
from time import perf_counter
import cv2
import numpy as np
import torch
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1.mask_codec import decode_mask
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.approximate_probe import align_patch_features
from worldstream.m2.profiler import measure,errors
from worldstream.m2.packing import partition,SpatialWindowGather,unpartition
from worldstream.m2.window_executor import WindowExecutor,packed_forward
from worldstream.m2.cost_model import cost_key
from tools.worldstream_m1.source_state_refinement_replay import preprocessed_frame
from tools.worldstream_m1.evidence_experiments import load_inputs
from tools.worldstream_m1.io_utils import write_json
from .common import backbone
from .final_validation_cuda import CPUS,PYTHON
from .source_state_refinement_cuda_batch import wait_idle

ROOT=Path('outputs/module2_state_aware_incremental_reuse')


def freeze():
    if (ROOT/'baseline/manifest.json').exists():raise RuntimeError('Preserve prior baseline; choose a fresh output root before reproduction')
    ROOT.mkdir(exist_ok=True);(ROOT/'baseline').mkdir(exist_ok=True);(ROOT/'profile_output').mkdir(exist_ok=True)
    files=[*Path('worldstream').rglob('*.py'),Path('configs/worldstream_m1_engineering_frozen.yaml')]
    pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write_json(ROOT/'baseline/manifest.json',dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sha256=pins))
    with tarfile.open(ROOT/'baseline/source.tar.gz','w:gz') as tar:
        for p in files:tar.add(p)
    write_json(ROOT/'m1_frozen_contract_audit.json',dict(configuration='configs/worldstream_m1_engineering_frozen.yaml',
        config=M1Config.load('configs/worldstream_m1_engineering_frozen.yaml').to_dict(),
        source_sha256={p:v for p,v in pins.items() if p.startswith('worldstream/m1/') or p.startswith('configs/')},
        engineering_interface='frozen',candidate_quality='unqualified_research_only',M1_modified=False))


@torch.no_grad()
def main():
    freeze();gpu=wait_idle(ROOT);os.environ['CUDA_VISIBLE_DEVICES']=str(gpu['index'])
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(1901)
    torch.backends.cuda.matmul.allow_tf32=False
    witness=torch.randn(16,16);error=float(((witness.cuda()@witness.cuda()).cpu()-witness@witness).abs().max());assert error<1e-4
    write_json(ROOT/'profile_output/kernel_witness.json',dict(gpu=gpu,torch=torch.__version__,cuda=torch.version.cuda,error=error))
    from mmcv import Config
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py')
    specs,infos,outputs,_=load_inputs('development');specs=specs[:8]
    model=backbone();settings=M1Config.load('configs/worldstream_m1_engineering_frozen.yaml')
    topology=dict(patch=repr(model.patch_embed),absolute_position=model.pos_embed is not None,
        absolute_position_shape=list(model.pos_embed.shape) if model.pos_embed is not None else None,
        blocks=[dict(index=i,window=b.window_size,residual=b.use_residual_block,
            heads=b.attn.num_heads,flash=b.attn.flash_attn,RoPE=repr(b.attn.rope)) for i,b in enumerate(model.blocks)],
        global_boundary='First non-window block or cross-window residual; all later blocks are reexecuted',
        content_boundary='patch_embed output before current absolute position',
        contextual_boundary='block 0/1 include absolute-position residuals, window RoPE, attention softmax and MLP; warped output is approximate only',
        downstream='Final EVA -> original Detection/Map and temporal Memory -> original LLM/Planning; no downstream state reused')
    write_json(ROOT/'eva_dependency_analysis.json',topology)
    rows=[];all_components={};first=True
    for spec in specs:
        manager=WorldStateManager(settings)
        runtime=IncrementalVisionRuntime(model,dict(mode='auto',reuse_boundary='window',cache_capture='always',state_interface='source_linked'))
        previous=None
        for token in spec['samples'][:12]:
            frame,images=preprocessed_frame(infos[token],cfg,settings);images=images.cuda()
            start=perf_counter();pre=manager.precheck(frame);m1_ms=(perf_counter()-start)*1000
            observed={};handles=[]
            for name,module in [('patch',model.patch_embed),*[(f'block_{i}',model.blocks[i]) for i in (0,1,2,len(model.blocks)-1)]]:
                handles.append(module.register_forward_hook(lambda m,a,v,name=name:observed.update({name:v.detach()})))
            context=dict(scene_id=frame.scene_id,frame_id=token,timestamp=frame.timestamp,
                camera_ids=list(frame.cameras),coordinate_space='preprocessed_image',execution_signature=frame.execution_signature,state_epoch=pre.computation_state.state_epoch)
            torch.cuda.synchronize();start=perf_counter();features,receipt=runtime.forward(images,m1_result=pre,m1_state=manager,execution_context=context)
            torch.cuda.synchronize();m2_ms=(perf_counter()-start)*1000
            for h in handles:h.remove()
            if first:
                # Real six-camera data, original modules; event intervals are not kernel sums.
                records={};hs=[];starts={}
                modules=[('patch',model.patch_embed),*[(f'block_{i}',b) for i,b in enumerate(model.blocks)]]
                for name,module in modules:
                    def before(m,a,name=name):
                        s=torch.cuda.Event(enable_timing=True);s.record();starts[name]=s
                    def after(m,a,v,name=name):
                        e=torch.cuda.Event(enable_timing=True);e.record();records.setdefault(name,[]).append((starts[name],e))
                    hs.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
                for _ in range(3):model(images)
                records.clear()
                for _ in range(10):model(images)
                torch.cuda.synchronize()
                for h in hs:h.remove()
                all_components={n:[a.elapsed_time(b) for a,b in rs] for n,rs in records.items()}
                full=measure(lambda:model(images),warmup=3,repeats=10)
                def tail():
                    x=observed['block_1']
                    for b in model.blocks[2:]:x=b(x)
                    return x
                ideal=measure(tail,warmup=3,repeats=10)
                patch=measure(lambda:model.patch_embed(images),warmup=3,repeats=10)
                write_json(ROOT/'reuse_feasibility_upper_bound.json',dict(full_EVA=full,ideal_tail_after_block1=ideal,
                    ideal_saving_wall_ms=full['wall_ms']-ideal['wall_ms'],patch=patch,component_GPU_intervals_ms=all_components,
                    interpretation='Impossible current-input oracle excludes M1, validation, alignment, packing and cache. Not actual reuse or permission.'))
                irregular_profile(model,images,observed)
                first=False
            probes=[]
            for ci,cid in enumerate(frame.cameras):
                bundle=pre.computation_state;link=bundle.observation_links[cid]
                candidate=decode_mask(bundle.refresh_contract[cid]['reuse_candidate_mask'])
                e=manager.last_evidence.get(cid);stats=dict(camera=cid,links=len(link['source_points']),candidate_pixels=int(candidate.sum()),layers={})
                if previous is not None and e is not None:
                    start=perf_counter()
                    for layer in ('patch','block_0','block_1'):
                        aligned,report=align_patch_features(previous[layer][ci],observed[layer][ci],e.dense_source_map,candidate)
                        stats['layers'][layer]=report
                    torch.cuda.synchronize();stats['alignment_wall_ms']=(perf_counter()-start)*1000
                    stats['contextual_probe_only']=True
                probes.append(stats)
            out=copy.deepcopy(outputs[token]);out.execution_receipt=receipt
            start=perf_counter();manager.commit(frame,out,pre);commit_ms=(perf_counter()-start)*1000
            rows.append(dict(scene=frame.scene_id,frame=token,timestamp=frame.timestamp,M1_precheck_ms=m1_ms,M1_commit_ms=commit_ms,
                M2_full_wall_ms=m2_ms,receipt=receipt,probes=probes))
            previous={k:v.clone() for k,v in observed.items()}
            print(spec['scene_id'],token,'probe',sum(v['layers'].get('patch',{}).get('accepted_patches',0) for v in probes),flush=True)
            write_json(ROOT/'source_aligned_feature_analysis.json',dict(status='running',rows=rows,population='exploratory_reused_population'))
    write_json(ROOT/'source_aligned_feature_analysis.json',dict(status='completed',rows=rows,population='exploratory_reused_population',
        scope='Legacy full-supported patch probe before production changes; contextual warp is deliberately diagnostic only.'))
    write_json(ROOT/'feasibility_complete.json',dict(status='completed',frames=len(rows)))


def irregular_profile(model,images,observed):
    from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
    x=observed['patch']
    if model.pos_embed is not None:x=x+get_abs_pos(model.pos_embed,model.pretrain_use_cls_token,x.shape[1:3])
    b=model.blocks[0];cached,padded=partition(observed['block_0'],b.window_size);total=len(cached)
    ex=WindowExecutor();records=[]
    for count in (0,1,6,18,36,total):
        ids=torch.linspace(0,total-1,count,device=images.device).long() if count else torch.empty(0,device=images.device,dtype=torch.long)
        for strategy in ('original','spatial'):
            timing=measure(lambda:ex.execute(b,x,cached,ids,strategy=strategy),warmup=3,repeats=10)
            records.append(dict(updated=count,total=total,strategy=strategy,timing=timing,key=cost_key(b,x,0),
                input='Real current tensor with same-input oracle cached outputs; controlled executor workload, not natural eligibility',quality_certified=False))
    ids=torch.arange(total//2,device=images.device);spatial=SpatialWindowGather()
    selected,valid=spatial.gather(x,b.window_size,ids)
    norm=b.norm1(selected)*valid;attn=b.attn(norm);residual=selected+b.drop_path(attn)
    values=dict(gather=measure(lambda:spatial.gather(x,b.window_size,ids),repeats=10),
        attention=measure(lambda:b.attn(norm),repeats=10),MLP=measure(lambda:b.mlp(b.norm2(residual)),repeats=10),
        cache_clone=measure(lambda:cached.clone(),repeats=10),
        index_construction=measure(lambda:torch.arange(total//2,device=images.device),repeats=10),
        scatter=measure(lambda:cached.clone().index_copy_(0,ids,selected),repeats=10))
    write_json(ROOT/'gpu_irregular_execution_profile.json',dict(baseline_records=records,stages=values,
        timing='CUDA event intervals include host launch gaps; cache clone and scatter records overlap by design and must not be summed.',
        GPU=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda))


if __name__=='__main__':main()
