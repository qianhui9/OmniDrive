"""Offline full-forward intervention on the fixed 8-scene/96-frame population.

Fresh full current block-1 outputs supply every unsupported token. This is an
unimplementable oracle, not a cache policy. All oracle execution is charged and
no partial token is authorized to skip the original window attention.
"""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from tools.worldstream_m1.io_utils import write_json
from tools.worldstream_m2.source_state_refinement_cuda_batch import wait_idle

ROOT=Path(os.environ.get('OMNIDRIVE_TASK_SERVING_OUTPUT','outputs/m2_freeze_m3_task_serving'))
OLD=Path('outputs/module2_state_aware_incremental_reuse')


def funnel():
    raw=json.loads((OLD/'aligned_probe_raw.json').read_text());rows=[];counts=np.zeros(8,np.int64);pixel_count=0;observed=0
    for r in raw['rows']:
        a=np.load(OLD/'token_error_details'/(r['frame']+'.npz'));prefix=r['camera']+'.mapping.'
        support=a[prefix+'support_count'];conflict=a[prefix+'conflict'];complete=a[prefix+'complete'];error=a[prefix+'translation_error_px'];src=a[prefix+'source_indices']
        pixel_count+=int(support.sum());observed+=int((support>0).sum());full=support==256;clear=full&~conflict
        displaced=clear&(error<=.5)&(src>=0)&(src<1600)
        for y in range(0,40,16):
            for x in range(0,40,16):
                sl=np.s_[y:min(y+16,40),x:min(x+16,40)];s=support[sl];c=complete[sl]
                # Eight stages retain the identical current window IDs; earlier
                # pixel/patch counts also retain their own explicit units.
                passed=[bool((s>0).any()),bool(full[sl].any()),bool(clear[sl].any()),bool(displaced[sl].any()),bool(c.all()),False,False,False]
                if passed[4]:
                    from worldstream.m2.source_alignment import SourceTokenMap,source_windows
                    m=SourceTokenMap(*[a[prefix+n] for n in ('source_indices','source_coordinates','support_count','support_ratio','translation_error_px','conflict','complete','dirty_ratio','unknown_ratio')],r['camera'],r['source_frame'],r['frame'])
                    passed[5]=bool(source_windows(m,16)[(y//16)*3+x//16]>=0)
                    # Content/task/cost admission is explicitly unqualified, even
                    # if a future replay provides a structurally complete window.
                counts+=np.asarray(passed)
                rows.append(dict(scene=r['scene'],frame=r['frame'],camera=r['camera'],source_frame=r['source_frame'],window=[y//16,x//16],linked_pixels=int(s.sum()),pixels=int(s.size*256),patches=int(s.size),fully_supported_patches=int(full[sl].sum()),conflict_free_patches=int(clear[sl].sum()),displacement_valid_patches=int(displaced[sl].sum()),complete_patches=int(c.sum()),stages=passed,rejection_reason=next((n for n,ok in zip(['no_source_pixels','partial_patch_or_window_support','source_conflict','displacement_or_bounds','dirty_unknown_or_incomplete_window','attention_dependency_or_padding','content_residual_unqualified','task_quality_or_cost_unqualified'],passed) if not ok),None)))
    names=['actual_source_linked_pixels','fully_supported_patches','conflict_free_patches','displacement_valid_patches','fully_supported_windows','attention_dependency_valid_windows','content_residual_valid_windows','quality_and_cost_qualified_windows']
    patch_counts=dict(fully_supported=sum(r['fully_supported_patches'] for r in rows),conflict_free=sum(r['conflict_free_patches'] for r in rows),displacement_valid=sum(r['displacement_valid_patches'] for r in rows),complete=sum(r['complete_patches'] for r in rows))
    totals=[pixel_count,patch_counts['fully_supported'],patch_counts['conflict_free'],patch_counts['displacement_valid'],int(counts[4]),int(counts[5]),int(counts[6]),int(counts[7])]
    units=['pixel','patch','patch','patch','window','window','window','window'];all_patches=sum(r['patches'] for r in rows);all_pixels=sum(r['pixels'] for r in rows)
    from collections import Counter
    stages=[]
    for i,(n,c,u) in enumerate(zip(names,totals,units)):
        total=all_pixels if u=='pixel' else all_patches if u=='patch' else len(rows)
        stages.append(dict(stage=n,count=c,unit=u,denominator=total,fraction_population=c/total,fraction_previous_same_unit_stage=c/totals[i-1] if i and units[i-1]==u and totals[i-1] else None,window_candidate_groups_remaining=int(counts[i]),fraction_window_groups=float(counts[i]/len(rows))))
    return dict(patch_rejection_counts=dict(partial_support=observed-patch_counts['fully_supported'],conflict=patch_counts['fully_supported']-patch_counts['conflict_free'],displacement_or_bounds=patch_counts['conflict_free']-patch_counts['displacement_valid']),window_rejection_reasons=dict(Counter(r['rejection_reason'] for r in rows)),population='exploratory_reused_population',source=str(OLD/'aligned_probe_raw.json'),source_sha256=hashlib.sha256((OLD/'aligned_probe_raw.json').read_bytes()).hexdigest(),scenes=len({r['scene'] for r in raw['rows']}),frames=len({r['frame'] for r in raw['rows']}),cameras=len(raw['rows']),actual_source_linked_pixels=pixel_count,observed_patches=observed,patch_counts=patch_counts,window_denominator=len(rows),stages=stages,rows=rows,limitations='Stage 1 pixels are unique eligible link centres retained by the frozen mapping; conflicting duplicate pixels were already excluded there. Pixel and patch totals are not window totals. Same fixed windows remain the denominator; no population switch.')


@torch.no_grad()
def main():
    dest=ROOT/'m2_oracle';dest.mkdir(exist_ok=False)
    (dest/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    gpu=wait_idle(dest);os.environ['CUDA_VISIBLE_DEVICES']=str(gpu['index'])
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    import cv2
    cv2.setNumThreads(1)
    from mmcv import Config
    from worldstream.m1 import M1Config
    from tools.worldstream_m1.evidence_experiments import load_inputs
    from tools.worldstream_m1.source_state_refinement_replay import preprocessed_frame
    from tools.worldstream_m2.common import backbone
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py');settings=M1Config.load('configs/worldstream_m1_engineering_frozen.yaml')
    specs,infos,_,_=load_inputs('development');model=backbone();raw=json.loads((OLD/'aligned_probe_raw.json').read_text());by_frame={}
    for r in raw['rows']:by_frame.setdefault(r['frame'],[]).append(r)
    captured={};h=model.blocks[1].register_forward_hook(lambda m,a,v:captured.update(x=v.detach()))
    rows=[]
    for spec in specs[:8]:
        previous=None;previous_frame=None
        for token in spec['samples'][:12]:
            frame,images=preprocessed_frame(infos[token],cfg,settings);images=images.cuda();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            if not (dest/'prefix_only_profile.json').exists():
                from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
                embedding=model.patch_embed(images)
                if model.pos_embed is not None:embedding=embedding+get_abs_pos(model.pos_embed,model.pretrain_use_cls_token,embedding.shape[1:3])
                for _ in range(3):model.blocks[1](model.blocks[0](embedding))
                samples=[]
                for _ in range(20):
                    torch.cuda.synchronize();tick=perf_counter();a.record();model.blocks[1](model.blocks[0](embedding));b.record();b.synchronize()
                    samples.append(dict(wall_ms=(perf_counter()-tick)*1000,GPU_ms=a.elapsed_time(b)))
                write_json(dest/'prefix_only_profile.json',dict(samples=samples,scope='Only actual original blocks 0/1 on actual current input; excludes PE/ABS; impossible complete-prefix-deletion upper bound, not partial-token speed estimation',input_shape=list(embedding.shape),dtype=str(embedding.dtype),online_qualified_saving_ms=0))
                del embedding

            torch.cuda.synchronize();tick=perf_counter();a.record();full=model(images)[0];b.record();b.synchronize();full_wall=(perf_counter()-tick)*1000;full_gpu=a.elapsed_time(b)
            current=captured['x'];hybrid=current.clone();a_np=np.load(OLD/'token_error_details'/(token+'.npz'));used=0
            if previous is not None:
                for ci,r in enumerate(by_frame[token]):
                    assert r['camera']==list(frame.cameras)[ci] and r['source_frame']==previous_frame
                    p=r['camera']+'.mapping.';valid=a_np[p+'complete'];ids=a_np[p+'source_indices'];mask=torch.as_tensor(valid,device=images.device)
                    source=torch.as_tensor(ids[valid],device=images.device);hybrid[ci][mask]=previous[ci].reshape(-1,current.shape[-1])[source];used+=int(valid.sum())
            delta=(hybrid-current).float();prefix_error=float(delta.abs().max());a.record();tick=perf_counter()
            x=hybrid
            for block in model.blocks[2:]:x=block(x)
            actual=x.permute(0,3,1,2);b.record();b.synchronize();oracle_wall=(perf_counter()-tick)*1000;oracle_gpu=a.elapsed_time(b)
            error=(actual-full).float();row=dict(scene=frame.scene_id,frame=token,source_frame=previous_frame,full_wall_ms=full_wall,full_GPU_ms=full_gpu,oracle_tail_wall_ms=oracle_wall,oracle_tail_GPU_ms=oracle_gpu,oracle_actual_execution_wall_lower_bound_ms=full_wall+oracle_wall,fully_supported_spliced_tokens=used,total_tokens=9600,ideal_token_update_fraction=1-used/9600,legal_window_update_fraction=1.,prefix_max_abs=prefix_error,final_feature_max_abs=float(error.abs().max()),final_feature_relative_l2=float(error.norm()/full.float().norm().clamp_min(1e-12)),online_admission=False,full_system_quality_certificate=False)
            rows.append(row)
            with (dest/'frames.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('ORACLE',len(rows),token,used,row['final_feature_max_abs'],flush=True)
            previous=current.clone();previous_frame=token
    h.remove();write_json(ROOT/'m2_admission_funnel.json',dict(funnel(),oracle=dict(source=str(dest/'frames.jsonl'),fresh_frames=len(rows),role='Offline current Full Forward supplies unsupported tokens; cannot be used by online admission; splice errors measured by actual original tail 2..23',online_enabled=False)))
    write_json(dest/'status.json',dict(status='completed',frames=len(rows),weight_checkpoint='ckpts/iter_10548.pth',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

if __name__=='__main__':main()
