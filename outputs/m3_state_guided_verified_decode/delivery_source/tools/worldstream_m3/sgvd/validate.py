"""Full-model frozen-service regression and the exact-target stop gate.

This runner intentionally has no unqualified SGVD execution route. If the real
parallel target differs from native serial sampling, stop before installing a
generation loop, retain the evidence, and evaluate the bounded KV fallback.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter
import numpy as np
import torch
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m3 import M3Runtime
from worldstream.m3.visual_state import base_model
from tools.worldstream_m1.io_utils import write_json
from tools.worldstream_m2.rearchitecture.common import arrays,memory,compare,quality_pass
from tools.worldstream_m2.refine_integration import planning_metric
from tools.worldstream_m3.decode_optimization.profile import load,ForwardTimer
from .probe import append,measure
from .summarize_probe import summarize
from . import numerical,alternative


@torch.no_grad()
def service(model,dataset,root):
    from mmcv.parallel import collate,scatter
    dest=root/'service';dest.mkdir(parents=True,exist_ok=False)
    modes=['original','full_maintain','frozen_exact_m2','m3_native']
    specs=json.loads((root/'phase1/protocol.json').read_text())['specs']
    # Fixed, bounded full-model quality population; phase1 keeps the longer
    # 0/1/2/4 multi-frame/gap history census separately.
    populations=[('planning',specs),('native_multi',specs[:1])]
    write_json(dest/'protocol.json',dict(modes=modes,populations=populations,frames=2,
        independent_seeds=[123,223,323],warmup='first seed is cold-inclusive, separately reported',
        population='exploratory_reused_population',same_service='M1 full maintenance for all non-Original arms',
        omitted_arms='Experimental SGVD is conditional on exact target, cost and task admission; '
            'old CUDA Graph negative baseline is provenance-linked without re-capture'))
    base=base_model(model.lm_head);native=model.lm_head.generate;generated=[];features=[];generation_profiles=[]
    from functools import wraps
    @wraps(native)
    def generate(*a,**kw):
        forward=base.forward;first_decode=[None]
        @wraps(forward)
        def observe(*args,**kwargs):
            if (kwargs.get('past_key_values') is not None and kwargs.get('inputs_embeds') is None
                    and first_decode[0] is None):first_decode[0]=perf_counter()
            return forward(*args,**kwargs)
        base.forward=observe;tick=perf_counter()
        try:
            out=native(*a,**kw)
            torch.cuda.synchronize()
            end=perf_counter()
        finally:base.forward=forward
        generated.append(out.cpu().tolist())
        generation_profiles.append(dict(generation_wall_ms=(end-tick)*1000,
            decode_span_wall_ms=(end-first_decode[0])*1000 if first_decode[0] is not None else 0.))
        return out
    model.lm_head.generate=generate
    original_extract=model.extract_img_feat
    def extract(*a,**kw):
        out=original_extract(*a,**kw);features.append(out.detach());return out
    model.extract_img_feat=extract
    checks=[];runs=[]
    for population,groups in populations:
        for transform in dataset.pipeline.transforms:
            if hasattr(transform,'load_type'):transform.load_type=['planning'] if population=='planning' else ['planning','conv']
        random.seed(0)
        samples={i:dataset[i] for _,seq in groups for i in seq[:2]}
        for scene,seq in groups:
            seq=seq[:2]
            for rep in range(3):
                refs={}
                rest=modes[1:];rotation=rep%len(rest);order=['original']+rest[rotation:]+rest[:rotation]
                for mode in order:
                    model.test_flag=False;model.m1_manager=None;model.m2_runtime=None;model.m3_runtime=None
                    gc.collect();torch.cuda.empty_cache()
                    if mode!='original':model.m1_manager=WorldStateManager(M1Config.load('configs/worldstream_m1_engineering_frozen.yaml'))
                    if mode in ('frozen_exact_m2','m3_native'):
                        model.m2_runtime=IncrementalVisionRuntime(model.img_backbone,dict(mode='auto',reuse_boundary='window',
                            state_interface='source_linked',cache_capture='always',cost_table='outputs/module2_refinement/break_even_results.json',
                            experimental_conditional_reuse=False))
                    if mode=='m3_native':model.m3_runtime=M3Runtime(dict(mode='full',decode_cuda_graph=False))
                    model.save_path=str(dest/'language'/f'{population}_{scene}_{rep}_{mode}')+'/'
                    for ordinal,index in enumerate(seq):
                        seed=123+100*rep+ordinal
                        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                        batch=scatter(collate([copy.deepcopy(samples[index])],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                        features.clear();generated.clear();generation_profiles.clear();torch.cuda.reset_peak_memory_stats()
                        with ForwardTimer(base) as timer:
                            out,timing=measure(lambda:model(return_loss=False,rescale=True,**batch))
                            calls=timer.collect()
                        actual=dict(output=arrays(out),feature=arrays(features[-1]),memory=memory(model),
                            text=out[0]['text_out'],planning=planning_metric(out[0]['text_out'],dataset.data_infos[index]))
                        ids=copy.deepcopy(generated)
                        if mode=='original':refs[ordinal]=(actual,ids);q=None
                        else:
                            ref,ref_ids=refs[ordinal];q=compare(actual,ref)
                            q.update(frozen_targets_pass=quality_pass(q),token_ids_equal=ids==ref_ids,
                                temporal_memory_exact=q['memory']['max_abs']==0 and not q['memory']['shape_mismatches'],
                                feature_exact=np.array_equal(actual['feature'],ref['feature']),
                                mode=mode,population=population,scene=scene,ordinal=ordinal,repeat=rep)
                            checks.append(q);append(dest/'quality.jsonl',q)
                        row=dict(mode=mode,population=population,scene=scene,frame=dataset.data_infos[index]['token'],
                            ordinal=ordinal,cold=ordinal==0,repeat=rep,seed=seed,**timing,invocations=calls,
                            generated_ids=ids,generation_profiles=copy.deepcopy(generation_profiles),
                            questions=[v['Q'] for v in actual['text']],text=actual['text'],
                            planning=actual['planning'],quality_pass=q['frozen_targets_pass'] if q else None,
                            peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved(),
                            m1=copy.deepcopy(model.m1_manager.last_commit_diagnostics) if model.m1_manager else {},
                            m2=copy.deepcopy(model.m2_runtime.last_diagnostics) if model.m2_runtime else {},
                            m3=copy.deepcopy(getattr(model,'m3_diagnostics',{})) if model.m3_runtime else {})
                        append(dest/'runs.jsonl',row);runs.append(row)
                        raw=dest/'raw_outputs';raw.mkdir(exist_ok=True)
                        torch.save(dict(output=actual['output'],memory=actual['memory'],planning=actual['planning'],
                            text=actual['text'],feature_sha256=hashlib.sha256(actual['feature'].tobytes()).hexdigest()),
                            raw/f'{population}_{scene}_{rep}_{ordinal}_{mode}.pt')
                        print(population,scene[:8],rep,mode,ordinal,round(timing['wall_ms'],2),None if q is None else q['frozen_targets_pass'],flush=True)
    write_json(dest/'status.json',dict(completed=True,requests=len(runs),quality_checks=len(checks),
        frozen_quality_pass=all(q['frozen_targets_pass'] for q in checks)))
    model.lm_head.generate=native;model.extract_img_feat=original_extract
    return runs,checks


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/m3_state_guided_verified_decode')
    p.add_argument('--phase',choices=['numerical','service','all'],default='all');args=p.parse_args();root=Path(args.output)
    gate=summarize(root)
    if not gate['phase1_completed']:raise ValueError('Finish phase1 before any dependent execution')
    dest=root/('validation_'+args.phase);dest.mkdir(parents=True,exist_ok=False)
    sources=[*Path('worldstream').rglob('*.py'),*Path('tools/worldstream_m3/sgvd').glob('*.py'),
        *Path('tests/worldstream_m3').glob('*.py'),Path('configs/worldstream_m1_engineering_frozen.yaml')]
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    write_json(dest/'source_manifest.json',hashes)
    for f in sources:
        target=dest/'source'/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(f.read_bytes())
    model,dataset=load(dest)
    snapshots=torch.load(root/'phase1/carrier_snapshots.pt',map_location='cpu')
    if args.phase in ('numerical','all'):
        result=numerical.run(model,snapshots,root)
        write_json(root/'exact_target_gate.json',dict(phase1=gate,
            parallel_native_target_bitwise_equal=result['parallel_native_target_bitwise_equal'],
            production_enabled=False,
            decision='stop_no_exact_target_certificate' if not result['parallel_native_target_bitwise_equal'] else 'checked_positions_only_not_global_certificate',
            max_target_total_variation=result['max_target_total_variation']))
        if not result['parallel_native_target_bitwise_equal']:
            alternative.run(model,snapshots,root)
    if args.phase in ('service','all'):service(model,dataset,root)
    write_json(dest/'status.json',dict(completed=True,
        sources_unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items())))


if __name__=='__main__':main()
