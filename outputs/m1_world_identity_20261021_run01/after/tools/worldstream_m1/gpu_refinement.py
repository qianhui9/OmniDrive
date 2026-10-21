"""Repeated real full-model equality/cost checks plus a separate offline feature oracle pass."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter
import cv2
import numpy as np
from .io_utils import write_json
from .model_smoke import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/module1_refinement/gpu')
    p.add_argument('--scene',default='01452fbfbf4543af8acdfd3e8a1ee806')
    p.add_argument('--frames',type=int,default=8)
    p.add_argument('--warmup',type=int,default=1); p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--skip-feature-oracle',action='store_true')
    p.add_argument('--config',default='configs/worldstream_m1_refinement.yaml')
    p.add_argument('--reference-config',default='configs/worldstream_m1_refinement.yaml')
    p.add_argument('--reference-root',help='Optimization root containing the pre-change baseline_source snapshot')
    args=p.parse_args(); root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    if (root/'status.json').exists(): raise ValueError('Choose fresh output to protect completed/partial run')
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('GPU device nodes unavailable in this process; use authorized host execution')
    cv2.setNumThreads(1); torch.set_num_threads(1)
    metadata=dict(args=vars(args),torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                  source='Newly Executed GPU Inference',checkpoint='ckpts/iter_10548.pth')
    write_json(root/'status.json',dict(status='loading',**metadata))
    from mmcv import Config
    from mmcv.parallel import collate,scatter
    from mmcv.runner import load_checkpoint,wrap_fp16_model
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    from worldstream.m1 import M1Config,WorldStateManager
    from .evaluate_feature_oracle import FeatureOracle
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py')
    cfg.model.m1_enable=False; cfg.model.train_cfg=None; cfg.model.pretrained=None; cfg.data.test.test_mode=True
    dataset=build_dataset(cfg.data.test)
    indices=[i for i,v in enumerate(dataset.data_infos) if v['scene_token']==args.scene][:args.frames]
    if len(indices)<2: raise ValueError('Need sequential scene')
    metadata['samples']=[dataset.data_infos[i]['token'] for i in indices]
    model=build_model(cfg.model,test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'): wrap_fp16_model(model)
    load_checkpoint(model,'ckpts/iter_10548.pth',map_location='cpu'); model.cuda().eval()
    settings=M1Config.load(args.config)
    reference_factory=None
    modes=('baseline','old_strict','new_full')
    if args.reference_root:
        from .optimize_balanced import reference
        reference_factory=reference(Path(args.reference_root))
        modes=('baseline','reference','balanced')
    metadata['m1_config']=settings.to_dict()
    baseline=None; runs=[]; comparisons=[]
    # Each repetition has a fresh language directory so original LLM execution runs.
    for mode in (modes if args.skip_feature_oracle else modes+('feature_oracle',)):
        reps=1 if mode=='feature_oracle' else args.warmup+args.repeats
        for rep in range(reps):
            random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            model.test_flag=False
            config=replace(settings,temporal_visibility=mode!='old_strict',local_support=mode!='old_strict',
                log_path=str(root/f'{mode}_{rep:02d}.jsonl'))
            if mode=='reference':
                original_settings=replace(M1Config.load(args.reference_config),
                    log_path=str(root/f'{mode}_{rep:02d}.jsonl'))
                model.m1_manager=reference_factory(original_settings)
            else:
                model.m1_manager=None if mode=='baseline' else WorldStateManager(config)
            model.save_path=str(root/f'{mode}_{rep:02d}'/'language')+'/'
            oracle=FeatureOracle(model,root) if mode=='feature_oracle' else None
            backbone=[]; hook=model.img_backbone.register_forward_hook(lambda m,i,o:backbone.append(digest(o)))
            outputs=[]; memories=[]; frame_times=[]; m1_times=[]
            torch.cuda.reset_peak_memory_stats(); start=perf_counter()
            with torch.no_grad():
                for index in indices:
                    batch=scatter(collate([dataset[index]],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                    torch.cuda.synchronize(); tick=perf_counter()
                    output=model(return_loss=False,rescale=True,**batch)
                    torch.cuda.synchronize(); frame_times.append((perf_counter()-tick)*1000)
                    outputs.append(digest(output))
                    memories.append({name:{k:digest(v) for k,v in vars(head).items() if k.startswith('memory_') or k=='sample_time'}
                                     for name,head in [('det',model.pts_bbox_head),('map',model.map_head)]})
                    if model.m1_manager:
                        m1_times.append(dict(model.m1_manager.last_result.diagnostics['timings_ms'],**model.m1_manager.last_commit_diagnostics))
            hook.remove()
            if oracle: oracle.close()
            current=dict(outputs=outputs,memory=memories,backbone=backbone)
            if baseline is None: baseline=current; write_json(root/'baseline_digests.json',baseline)
            equal={k:baseline[k]==current[k] for k in current}
            comparisons.append(dict(mode=mode,repeat=rep,**equal))
            if not all(equal.values()): raise AssertionError('Full-model equality failed: '+str(equal))
            record=dict(mode=mode,repeat=rep,warmup=mode!='feature_oracle' and rep<args.warmup,
                frame_wall_ms=frame_times,m1_timings=m1_times,total_wall_ms=(perf_counter()-start)*1000,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            runs.append(record); write_json(root/'runs.json',runs); write_json(root/'comparisons.json',comparisons)
            print(mode,rep,'passed','frame ms',np.round(frame_times,2).tolist(),flush=True)
    summary={}
    for mode in modes:
        xs=[t for r in runs if r['mode']==mode and not r['warmup'] for t in r['frame_wall_ms']]
        ms=[t['m1_total_including_adapters_ms'] for r in runs if r['mode']==mode and not r['warmup'] for t in r['m1_timings']]
        summary[mode]=dict(frame_p50_p95_ms=np.percentile(xs,[50,95]).tolist(),samples=len(xs),
            m1_p50_p95_ms=np.percentile(ms,[50,95]).tolist() if ms else None)
    write_json(root/'results.json',dict(metadata=metadata,runs=runs,summary=summary,comparisons=comparisons,
        observe_only_equal=all(all(c[k] for k in ('outputs','memory','backbone')) for c in comparisons),
        limitations='Sequential mode order, not a speedup benchmark. Feature-oracle pass excluded from latency samples; LLM path always executes.'))
    write_json(root/'status.json',dict(status='passed',**metadata))

if __name__=='__main__': main()
