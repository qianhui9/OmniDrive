"""Measured source-token continuity, including partial-but-ineligible tokens."""
import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import torch
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1.mask_codec import decode_mask
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.source_alignment import token_sources,source_windows,probe_features
from worldstream.m2.m1_adapter import computation_admission
from worldstream.m2.profiler import measure
from tools.worldstream_m1.source_state_refinement_replay import preprocessed_frame
from tools.worldstream_m1.evidence_experiments import load_inputs
from tools.worldstream_m1.io_utils import write_json
from .common import backbone
from .source_state_refinement_cuda_batch import wait_idle
from .state_aware_feasibility import ROOT


def token_errors(previous,current,mapping):
    """Per observed token errors, offline diagnostics; never admission input."""
    import torch.nn.functional as F
    h,w,_=current.shape
    coords=torch.as_tensor(mapping.source_coordinates,device=current.device,dtype=current.dtype)
    grid=torch.stack((2*(coords[...,0]+.5)/w-1,2*(coords[...,1]+.5)/h-1),-1)
    warped=F.grid_sample(previous.permute(2,0,1)[None],grid[None],mode='bilinear',align_corners=False)[0].permute(1,2,0)
    valid=torch.as_tensor((mapping.support_count>0)&~mapping.conflict,device=current.device)
    valid&=(coords[...,0]>=0)&(coords[...,0]<=w-1)&(coords[...,1]>=0)&(coords[...,1]<=h-1)
    valid&=torch.isfinite(warped).all(-1)
    ids=valid.flatten().nonzero().flatten();delta=(warped-current).float()[valid];ref=current.float()[valid]
    return dict(current_token_indices=ids.cpu().numpy(),source_token_coordinates=coords[valid].cpu().numpy(),
        max_abs=delta.abs().amax(-1).cpu().numpy(),L2=delta.norm(dim=-1).cpu().numpy(),
        relative_l2=(delta.norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-12)).cpu().numpy())


@torch.no_grad()
def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT));parser.add_argument('--frame-stride',type=int,default=1)
    parser.add_argument('--scenes',type=int,default=8)
    args=parser.parse_args()
    if args.frame_stride<1 or args.scenes<1:raise ValueError('Positive stride and scenes required')
    ROOT=Path(args.output);ROOT.mkdir(parents=True,exist_ok=True)
    gpu=wait_idle(ROOT);os.environ['CUDA_VISIBLE_DEVICES']=str(gpu['index'])
    torch.set_num_threads(1);cv2.setNumThreads(1)
    from mmcv import Config
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py')
    specs,infos,outputs,_=load_inputs('development');settings=M1Config.load('configs/worldstream_m1_engineering_frozen.yaml')
    model=backbone();rows=[];adapter_checks=[]
    write_json(ROOT/'eva_position_metadata.json',dict(
        absolute_position_shape=list(model.pos_embed.shape) if model.pos_embed is not None else None,
        blocks=[dict(block=i,window=b.window_size,attention_class=type(b.attn).__name__,
            RoPE_class=type(b.attn.rope).__name__,
            RoPE_frequency_shape=list(b.attn.rope.freqs_cos.shape) if getattr(b.attn.rope,'freqs_cos',None) is not None else None)
            for i,b in enumerate(model.blocks)],
        interpretation='Whole-window translation preserves local window coordinates. Identical local RoPE grids do not undo historical ABS/attention/MLP context. Global attention spans each camera; heads fuse cameras and temporal memory.'))
    details=ROOT/'token_error_details';details.mkdir(exist_ok=False)
    scene_meta={s['token']:s for s in json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())}
    for spec in specs[:args.scenes]:
        m=WorldStateManager(settings);runtime=IncrementalVisionRuntime(model,dict(state_interface='source_linked',reuse_boundary='window',cache_capture='always'))
        previous=None;previous_frame=None
        for token in spec['samples'][:12:args.frame_stride]:
            frame,images=preprocessed_frame(infos[token],cfg,settings);images=images.cuda();pre=m.precheck(frame)
            observed={};handles=[]
            for name,module in [('patch',model.patch_embed),*[(f'block_{i}',model.blocks[i]) for i in (0,1,2,len(model.blocks)-1)]]:
                handles.append(module.register_forward_hook(lambda m,a,v,name=name:observed.update({name:v.detach()})))
            context=dict(scene_id=frame.scene_id,frame_id=token,timestamp=frame.timestamp,camera_ids=list(frame.cameras),
                coordinate_space='preprocessed_image',execution_signature=frame.execution_signature,state_epoch=pre.computation_state.state_epoch)
            start=perf_counter()
            _,adapter_reason=computation_admission(pre.computation_state,context,images,runtime.cache,pre.full_refresh_hint)
            adapter_checks.append(dict(frame=token,scene=frame.scene_id,wall_ms=(perf_counter()-start)*1000,
                reason=adapter_reason,cache_version=runtime.cache.version,
                role='Read-only diagnostic using actual M1 bundle and actual GPU cache; not an automatic execution qualification'))
            _,receipt=runtime.forward(images,m1_result=pre,m1_state=m,execution_context=context)
            for h in handles:h.remove()
            if not (ROOT/'prefix_only_profile.json').exists():
                from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
                x=observed['patch']
                if model.pos_embed is not None:x=x+get_abs_pos(model.pos_embed,model.pretrain_use_cls_token,x.shape[1:3])
                timing=measure(lambda:model.blocks[1](model.blocks[0](x)),warmup=3,repeats=10)
                write_json(ROOT/'prefix_only_profile.json',dict(timing=timing,input_shape=list(x.shape),input_stride=list(x.stride()),dtype=str(x.dtype),
                    scope='Only original block 0 + block 1, excludes PE and ABS addition. Actual prefix execution; no reuse qualification.'))
            token_details={}
            for ci,cid in enumerate(frame.cameras):
                bundle=pre.computation_state;c=bundle.refresh_contract[cid];link=bundle.observation_links[cid]
                start=perf_counter();mapping=token_sources(link,decode_mask(c['reuse_candidate_mask']),decode_mask(c['dirty_hint_mask']),
                    decode_mask(c['unknown_mask']),model.patch_embed.proj.kernel_size,previous_frame,cid,token)
                windows=source_windows(mapping,model.blocks[0].window_size);host=(perf_counter()-start)*1000
                gray=cv2.cvtColor(frame.cameras[cid].image,cv2.COLOR_RGB2GRAY)
                ph,pw=model.patch_embed.proj.kernel_size;gh,gw=mapping.complete.shape
                patches=gray[:gh*ph,:gw*pw].reshape(gh,ph,gw,pw)
                texture=patches.std(axis=(1,3))
                row=dict(scene=frame.scene_id,frame=token,camera=cid,source_frame=previous_frame,
                    timestamp=frame.timestamp,camera_timestamp=frame.cameras[cid].timestamp,
                    scene_name=scene_meta[frame.scene_id]['name'],scene_description=scene_meta[frame.scene_id]['description'],
                    ego_yaw_rad=float(np.arctan2(frame.ego_to_world[1,0],frame.ego_to_world[0,0])),
                    ego_translation=frame.ego_to_world[:3,3],
                    patch_gray_std=dict(p05=float(np.percentile(texture,5)),p50=float(np.median(texture)),p95=float(np.percentile(texture,95)),
                        zero_variance_patches=int((texture==0).sum()),scope='Observed preprocessed RGB; diagnostic only, never admission evidence'),
                    computation_sources=[s for s in bundle.source_records if s['source_camera_id']==cid and s.get('computation_available')],
                    observed_tokens=int((mapping.support_count>0).sum()),complete_tokens=int(mapping.complete.sum()),
                    windows=int((windows>=0).sum()),conflicting_tokens=int(mapping.conflict.sum()),mapping_host_ms=host,
                    support_ratios=mapping.support_ratio.ravel(),source_token_indices=mapping.source_indices.ravel(),
                    source_token_coordinates=mapping.source_coordinates.reshape(-1,2),
                    dirty_ratios=mapping.dirty_ratio.ravel(),unknown_ratios=mapping.unknown_ratio.ravel(),
                    complete_mask=mapping.complete.ravel(),conflict_mask=mapping.conflict.ravel(),
                    source_window_indices=windows,detail_archive=str(details/(token+'.npz')),layers={})
                ws=model.blocks[0].window_size
                if previous is None:
                    rgb=frame.cameras[cid].image[:gh*ph,:gw*pw]
                    rgb=np.ascontiguousarray(rgb.reshape(gh,ph,gw,pw,3).transpose(0,2,1,3,4)).reshape(gh*gw,-1)
                    count=len(np.unique(rgb.view(np.dtype((np.void,rgb.shape[1]))).ravel()))
                    row['repeated_RGB_patches']=int(gh*gw-count)
                row['legal_source_token_counts_per_window']=[int(mapping.complete[y:y+ws,x:x+ws].sum())
                    for y in range(0,mapping.complete.shape[0],ws) for x in range(0,mapping.complete.shape[1],ws)]
                for field in ('source_indices','source_coordinates','support_count','support_ratio','translation_error_px','conflict','complete','dirty_ratio','unknown_ratio'):
                    token_details[cid+'.mapping.'+field]=getattr(mapping,field)
                if previous is not None:
                    torch.cuda.synchronize();start=perf_counter()
                    for layer in ('patch','block_0','block_1','block_2',f'block_{len(model.blocks)-1}'):
                        report=probe_features(previous[layer][ci],observed[layer][ci],mapping)
                        report['deliberate_wrong_source_control']=probe_features(previous[layer][ci].roll(1,1),observed[layer][ci],mapping)
                        for key,value in token_errors(previous[layer][ci],observed[layer][ci],mapping).items():
                            token_details[cid+'.'+layer+'.'+key]=value
                        row['layers'][layer]=report
                    torch.cuda.synchronize();row['probe_wall_ms']=(perf_counter()-start)*1000
                rows.append(row)
            np.savez_compressed(details/(token+'.npz'),**token_details)
            out=copy.deepcopy(outputs[token]);out.execution_receipt=receipt;m.commit(frame,out,pre)
            previous={k:v.clone() for k,v in observed.items()};previous_frame=token
            print(spec['scene_id'],token,'aligned',sum(r['windows'] for r in rows[-6:]),flush=True)
    write_json(ROOT/'aligned_probe_raw.json',dict(status='completed',population='exploratory_reused_population',rows=rows,
        adapter_checks=adapter_checks,
        frame_stride=args.frame_stride,
        patch_size=model.patch_embed.proj.kernel_size,window_size=model.blocks[0].window_size,
        role='All current features actually computed for diagnostic comparison. No time counted as saved.',
        per_token='token_error_details/*.npz: source/current coordinates, real support, completeness/conflict, translation residual and per-layer maxAbs/L2/relativeL2; inf translation residual means unsupported, not a JSON finite score.',
        timing='probe_wall_ms includes instrumented per-token diagnostic/copy work; not a cost qualification. Mapping CPU and complete VLM runtime costs are separate.',
        source_association='Measured M1 correspondence; not independently certified physical identity',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path('worldstream/m1').glob('*.py'),*Path('worldstream/m2').glob('*.py'),Path(__file__)]}))
    write_json(details/'completion.json',dict(status='completed',frames=len({r['frame'] for r in rows}),
        archives=len(list(details.glob('*.npz'))),mapping_units='Current row-major token grid; source coordinates in token x/y; displacement pixels = (source_xy - current_xy) * patch width/height'))


if __name__=='__main__':main()
