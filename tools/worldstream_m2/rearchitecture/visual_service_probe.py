"""Isolated real visual prefix: stop at language entry, never claim full VLM timing."""
import copy
from dataclasses import replace
import json
from time import perf_counter
import numpy as np
import torch
from worldstream.m1 import WorldStateManager,M1Config
from worldstream.m2 import IncrementalVisionRuntime
from .common import ROOT,write,reference,stats


class VisualPrefixComplete(Exception):pass


@torch.no_grad()
def run(model,dataset):
    from mmcv.parallel import collate,scatter
    specs=json.load(open(ROOT/'protocol.json'))['real_scenes'][:2]
    settings=replace(M1Config.load('configs/worldstream_m1_coverage.yaml'),logging=False)
    modes=['baseline','original','gate_maintain','gate_reset','gate_output','camera_auto']
    records=[];generate=model.lm_head.generate
    def stop_at_language(*args,**kw):
        # Detection/Map/query assembly and the M1 observer have already completed.
        # No LLM computation is timed or claimed in this component-only probe.
        raise VisualPrefixComplete()
    model.lm_head.generate=stop_at_language
    for rep in range(2):
        for si,spec in enumerate(specs):
            order=modes[(rep+si)%6:]+modes[:(rep+si)%6]
            for mode in order:
                model.test_flag=False
                model.m1_manager=None if mode=='baseline' else WorldStateManager(settings)
                if mode=='baseline':model.m2_runtime=None
                elif mode=='original':model.m2_runtime=reference().IncrementalVisionRuntime(model.img_backbone,dict(mode='exact'))
                else:
                    service={'gate_maintain':'maintain','gate_reset':'reset_on_reject'}.get(mode,'output_only')
                    model.m2_runtime=IncrementalVisionRuntime(model.img_backbone,dict(mode='auto',cheap_gate=True,
                        m1_service=service,reuse_boundary='camera' if mode=='camera_auto' else 'window',
                        cost_table='outputs/module2_refinement/break_even_results.json',
                        camera_cost_table=str(ROOT/'camera_costs.json')))
                model.save_path=str(ROOT/f'visual_prefix_no_language_{mode}_{rep}_{spec["scene"]}')+'/'
                frames=[]
                for ordinal,index in enumerate(spec['indices']):
                    batch=scatter(collate([dataset[index]],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize();tick=perf_counter();a.record()
                    try:model(return_loss=False,rescale=True,**batch)
                    except VisualPrefixComplete:pass
                    else:raise AssertionError('Visual probe must reach the actual language entry')
                    b.record();torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
                    frames.append(dict(index=index,ordinal=ordinal,wall_ms=wall,gpu_interval_ms=a.elapsed_time(b),
                        m1_ms=model.m1_manager.last_commit_diagnostics.get('m1_total_including_adapters_ms',0.) if model.m1_manager else 0.,
                        runtime=copy.deepcopy(model.m2_diagnostics) if model.m2_runtime else {}))
                records.append(dict(scene=spec['scene'],mode=mode,repeat=rep,frames=frames))
    model.lm_head.generate=generate;model.m1_manager=None;model.m2_runtime=None
    summary={}
    for mode in modes:
        frames=[f for r in records if r['mode']==mode for f in r['frames']]
        summary[mode]=dict(wall_ms=stats([f['wall_ms'] for f in frames]),gpu_interval_ms=stats([f['gpu_interval_ms'] for f in frames]),
            m1_ms=stats([f['m1_ms'] for f in frames]),frames=len(frames))
    write(ROOT/'visual_service_results.json',dict(runs=records,summary=summary,
        scope='separate component-only visual prefix through current heads, world-state commit and query concatenation; exits at actual LLM entry. NOT complete VLM latency.',
        protocol='two real scenes, four chronological frames each, two replays with rotated mode order; loaded model already warmed by full VLM experiments',
        includes='validation, cache, planning, GPU, sync, current heads and query assembly; scatter/diagnostics outside timer'))
    print('Real visual-service prefix probe completed',flush=True)
