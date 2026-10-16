"""Balanced actual formal Auto vs full VLM for explicitly current-output requests."""
import copy
import json
import random
from dataclasses import replace
from time import perf_counter
import numpy as np
import torch
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m1 import M1Config,WorldStateManager
from tools.worldstream_m2.refine_integration import planning_metric
from .common import ROOT,write,arrays,memory,compare,quality_pass


@torch.no_grad()
def run(model,dataset,real=False):
    from mmcv.parallel import scatter,collate
    protocol=json.load(open(ROOT/'protocol.json'))
    specs=protocol['real_scenes'][:2] if real else protocol['controlled_scenes']
    refs={};runs=[];quality=[];feature=[]
    extract=model.extract_img_feat
    def capture(img):
        result=extract(img);feature.append(result.detach());return result
    model.extract_img_feat=capture
    settings=replace(M1Config.load('configs/worldstream_m1_coverage.yaml'),logging=False)
    modes=['baseline','camera_auto_m1'] if real else ['baseline','camera_auto','camera_auto_m1']
    case='balanced_real' if real else 'balanced_repeat'
    result_name='balanced_real_camera_results.json' if real else 'balanced_camera_results.json'
    for rep in range(3):
        for si,spec in enumerate(specs):
            order=modes[(rep+si)%len(modes):]+modes[:(rep+si)%len(modes)]
            # First warm replay must create numerical oracles before Auto comparison.
            if rep==0:order=list(modes)
            for mode in order:
                random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
                model.test_flag=False
                model.m1_manager=WorldStateManager(settings) if mode=='camera_auto_m1' else None
                model.m2_runtime=(IncrementalVisionRuntime(model.img_backbone,dict(mode='auto',reuse_boundary='camera',
                    cheap_gate=mode=='camera_auto_m1',camera_cost_table=str(ROOT/'camera_costs.json'))) if mode!='baseline' else None)
                dest=ROOT/f'{case}_{mode}_{rep}_{spec["scene"]}'
                if dest.exists():raise ValueError('Fresh balanced language path required')
                model.save_path=str(dest/'language')+'/'
                frames=[];base=None;torch.cuda.reset_peak_memory_stats()
                for ordinal,index in enumerate(spec['indices'][:4]):
                    batch=scatter(collate([dataset[index]],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                    image=batch['img'][0]
                    if not real:
                        if base is None:base=image.clone()
                        image.copy_(base)
                    feature.clear()
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize();tick=perf_counter();a.record()
                    output=model(return_loss=False,rescale=True,**batch)
                    b.record();torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
                    text=output[0].get('text_out');assert text
                    current=dict(feature=arrays(feature.pop()),output=arrays(output),memory=memory(model),
                        text=text,planning=planning_metric(text,dataset.data_infos[index]))
                    key=(spec['scene'],ordinal)
                    if mode=='baseline' and key not in refs:refs[key]=current
                    if mode!='baseline':
                        q=compare(current,refs[key]);q['frozen_targets_pass']=quality_pass(q)
                        q.update(scene=spec['scene'],ordinal=ordinal,repeat=rep,mode=mode)
                        quality.append(q)
                    frames.append(dict(ordinal=ordinal,index=index,complete_wall_ms=wall,gpu_interval_ms=a.elapsed_time(b),
                        necessary_validation_ms=((model.m2_runtime.camera_executor.cache.context or {}).get('necessary_validation_ms')
                            if model.m2_runtime else None),
                        runtime=copy.deepcopy(model.m2_diagnostics) if model.m2_runtime else {}))
                runs.append(dict(scene=spec['scene'],mode=mode,repeat=rep,warmup=rep==0,frames=frames,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved()))
                write(ROOT/result_name,dict(runs=runs,quality=quality,
                    service='camera_auto: current-output, M1 absent; camera_auto_m1: continuous full M1. Both include current heads/Memory/LLM and validation-budget admission',
                    order='one warmup replay; two measured replays rotate all three modes across scene/replay',
                    kind='actual real moving images' if real else 'controlled repeated images with real changing pose/time, not real driving coverage'))
                print(case,mode,rep,spec['scene'][:8],flush=True)
    model.extract_img_feat=extract;model.m2_runtime=None
    passed=bool(quality) and all(q['frozen_targets_pass'] and q['feature']['max_abs']==0 and
        q['memory']['max_abs']==0 and not q['memory']['shape_mismatches'] and
        q['output']['max_abs']==0 and not q['output']['shape_mismatches'] for q in quality)
    if not real:
        table=json.load(open(ROOT/'camera_costs.json'))
        table['final_balanced_full_model_pass']=passed
        if not passed:
            for record in table['records']:record['quality_certified']=False
        write(ROOT/'camera_costs.json',table)

