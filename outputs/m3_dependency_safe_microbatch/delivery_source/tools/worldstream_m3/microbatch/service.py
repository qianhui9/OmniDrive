"""Complete original OmniDrive service; new sampling paths require explicit research config."""
import argparse
import copy
from functools import wraps
import gc
import hashlib
from pathlib import Path
import random
from time import perf_counter
import numpy as np
import torch
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m3 import M3Runtime
from worldstream.m3.visual_state import base_model
from worldstream.m3.sampling_microbatch import request_seed
from tools.worldstream_m2.rearchitecture.common import arrays,memory,compare,quality_pass
from tools.worldstream_m2.refine_integration import planning_metric
from .common import ROOT,OLD,load,pin,read,write_json,append,measure,SAMPLING,ForwardTimer


class ObservedRuntime(M3Runtime):
    def generate(self,*a,**kw):
        out,diag=super().generate(*a,**kw)
        self.last_ids=[t.detach().cpu().tolist() for t in out]
        self.last_diag=copy.deepcopy(diag)
        return out,diag


class IsolatedSerial(ObservedRuntime):
    def __init__(self,seed):super().__init__(dict(mode='full'));self.seed=seed
    @torch.no_grad()
    def generate(self,lm_head,input_ids_list,vision_carriers,context=None,**kw):
        generation={k:kw[k] for k in SAMPLING};answers=[];rows=[]
        # This comparator has the identical request-seed assignment as batching.
        # Fork protects the original shared RNG contract of other service arms.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for i,ids in enumerate(input_ids_list):
                seed=request_seed(self.seed,context,i);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                out=lm_head.generate(inputs=ids[None] if ids.ndim==1 else ids,images=vision_carriers,**generation)
                answers.append(out);rows.append(dict(request_id=i,seed=seed,
                    final_rng_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()))
        self.last_ids=[t.cpu().tolist() for t in answers];self.last_diag=dict(mode='isolated_serial_native',requests=rows)
        return answers,self.last_diag


class GreedyControl(ObservedRuntime):
    def generate(self,*a,**kw):
        kw['do_sample']=False;kw.pop('temperature',None);kw.pop('top_p',None)
        return super().generate(*a,**kw)


@torch.no_grad()
def run(model,dataset,root):
    from mmcv.parallel import collate,scatter
    dest=root/'service';dest.mkdir(exist_ok=False)
    modes=['original','full_maintain','m3_native','isolated_serial','microbatch2','microbatch4','greedy_control']
    specs=read(OLD/'phase1/protocol.json')['specs']
    protocol=dict(modes=modes,specs=specs,populations=['planning','native_multi'],
        native_multi_scenes=1,native_multi_seeds=[123,223,323],planning_seeds=[123],frames_per_scene=2,
        sampling=SAMPLING,population='exploratory_reused_population',
        greedy_control='Explicit independent engineering arm, do_sample=False; never a sampling speedup/quality certificate',
        rng_comparator='isolated_serial uses original native generate per request, same request_seed mapping as microbatch',
        ordering='Original and isolated serial reference first; remaining arms rotated per repeat',
        inputs='All original questions for each frame, same prepared samples, full six-camera model; no duplicated or truncated questions',
        single_request='Planning uses ordinary fallback, cannot inherit batch throughput',
        optimization='Fixed groups 2/4 only; no unmeasured length grouping or Batch=8 runtime')
    write_json(dest/'protocol.json',protocol)
    base=base_model(model.lm_head);native=model.lm_head.generate;generated=[];language=[];features=[];timeline=[];frame_start=[0.]
    @wraps(native)
    def generate(*a,**kw):
        tick=perf_counter();out=native(*a,**kw);torch.cuda.synchronize()
        generated.append(out.cpu().tolist());language.append(dict(question_index=len(generated)-1,
            generation_wall_ms=(perf_counter()-tick)*1000,complete_from_frame_start_ms=(perf_counter()-frame_start[0])*1000))
        return out
    model.lm_head.generate=generate
    extract=model.extract_img_feat
    def images(*a,**kw):
        out=extract(*a,**kw);features.append(out.detach());timeline.append(dict(stage='vision_submit_end',time_ms=(perf_counter()-frame_start[0])*1000));return out
    model.extract_img_feat=images
    hooks=[]
    for label,module in [('detection_temporal',model.pts_bbox_head),('map_temporal',model.map_head)]:
        def hook(m,a,o,label=label):timeline.append(dict(stage=label+'_submit_end',time_ms=(perf_counter()-frame_start[0])*1000))
        hooks.append(module.register_forward_hook(hook))
    checks=[];runs=[]
    try:
        for population in ('planning','native_multi'):
            for transform in dataset.pipeline.transforms:
                if hasattr(transform,'load_type'):transform.load_type=['planning'] if population=='planning' else ['planning','conv']
            groups=specs if population=='planning' else specs[:1]
            random.seed(0);samples={i:dataset[i] for _,seq in groups for i in seq[:2]}
            for scene,seq in groups:
                for rep in range(1 if population=='planning' else 3):
                    current_modes=modes[:3]+['microbatch4'] if population=='planning' else modes
                    rest=[m for m in current_modes if m not in ('original','isolated_serial')];rot=rep%len(rest)
                    order=['original']+(['isolated_serial'] if 'isolated_serial' in current_modes else [])+rest[rot:]+rest[:rot]
                    references={};isolated={}
                    for mode in order:
                        model.test_flag=False;model.m1_manager=None;model.m2_runtime=None;model.m3_runtime=None
                        gc.collect();torch.cuda.empty_cache()
                        if mode!='original':
                            model.m1_manager=WorldStateManager(M1Config.load('configs/worldstream_m1_engineering_frozen.yaml'))
                            original_commit=model.m1_manager.commit
                            @wraps(original_commit)
                            def commit(*a,_native=original_commit,**kw):
                                start=perf_counter();timeline.append(dict(stage='commit_start',time_ms=(start-frame_start[0])*1000))
                                result=_native(*a,**kw)
                                timeline.append(dict(stage='commit_end',time_ms=(perf_counter()-frame_start[0])*1000,wall_ms=(perf_counter()-start)*1000))
                                return result
                            model.m1_manager.commit=commit
                        if mode not in ('original','full_maintain'):
                            model.m2_runtime=IncrementalVisionRuntime(model.img_backbone,dict(mode='auto',reuse_boundary='window',
                                state_interface='source_linked',cache_capture='always',cost_table='outputs/module2_refinement/break_even_results.json',
                                experimental_conditional_reuse=False))
                        experiment_seed=123+100*rep
                        if mode=='m3_native':model.m3_runtime=ObservedRuntime(dict(mode='full'))
                        if mode=='isolated_serial':model.m3_runtime=IsolatedSerial(experiment_seed)
                        if mode in ('microbatch2','microbatch4'):
                            model.m3_runtime=ObservedRuntime(dict(mode='full',offline_experiment=True,experimental_rng_isolated_microbatch=True,
                                sampling_microbatch_size=int(mode[-1]),sampling_microbatch_seed=experiment_seed))
                        if mode=='greedy_control':
                            model.m3_runtime=GreedyControl(dict(mode='exact',offline_experiment=True,batched_decode=True))
                        model.save_path=str(dest/'language'/f'{population}_{scene}_{rep}_{mode}')+'/'
                        for ordinal,index in enumerate(seq[:2]):
                            seed=experiment_seed+ordinal
                            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                            batch=scatter(collate([copy.deepcopy(samples[index])],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                            generated.clear();language.clear();features.clear();timeline.clear();torch.cuda.reset_peak_memory_stats()
                            with ForwardTimer(base) as timer:
                                frame_start[0]=perf_counter()
                                out,timing=measure(lambda:model(return_loss=False,rescale=True,**batch));calls=timer.collect()
                            actual=dict(output=arrays(out),feature=arrays(features[-1]),memory=memory(model),text=out[0]['text_out'],
                                planning=planning_metric(out[0]['text_out'],dataset.data_infos[index]))
                            ids=copy.deepcopy(model.m3_runtime.last_ids if model.m3_runtime else generated)
                            diag=copy.deepcopy(model.m3_runtime.last_diag if model.m3_runtime else {})
                            if mode=='original':references[ordinal]=(actual,ids);q=None
                            else:
                                ref,refids=references[ordinal];q=compare(actual,ref)
                                q.update(frozen_targets_pass=quality_pass(q),token_ids_equal=ids==refids,
                                    perception_memory_exact=(q['memory']['max_abs']==0 and not q['memory']['shape_mismatches'] and
                                        np.array_equal(actual['feature'],ref['feature'])),
                                    mode=mode,population=population,scene=scene,repeat=rep,ordinal=ordinal)
                                if ordinal in isolated and mode.startswith('microbatch'):
                                    iref,iids=isolated[ordinal];iq=compare(actual,iref)
                                    q['isolated_serial']=dict(frozen_targets_pass=quality_pass(iq),token_ids_equal=ids==iids,
                                        text_equal=iq['text_equal'],planning_equal=iq['planning_equal'])
                                checks.append(q);append(dest/'quality.jsonl',q)
                            if mode=='isolated_serial':isolated[ordinal]=(actual,ids)
                            row=dict(mode=mode,population=population,scene=scene,frame=dataset.data_infos[index]['token'],
                                ordinal=ordinal,cold=ordinal==0,repeat=rep,seed=seed,**timing,invocations=calls,
                                generated_ids=ids,questions=[v['Q'] for v in actual['text']],text=actual['text'],planning=actual['planning'],
                                quality_pass=q['frozen_targets_pass'] if q else None,language_timings=copy.deepcopy(language),
                                m3=diag,timeline=copy.deepcopy(timeline),
                                m1=copy.deepcopy(model.m1_manager.last_commit_diagnostics) if model.m1_manager else {},
                                m2=copy.deepcopy(model.m2_runtime.last_diagnostics) if model.m2_runtime else {},
                                peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved())
                            append(dest/'runs.jsonl',row);runs.append(row)
                            raw=dest/'raw_outputs';raw.mkdir(exist_ok=True)
                            torch.save(dict(output=actual['output'],memory=actual['memory'],text=actual['text'],planning=actual['planning'],
                                feature_sha256=hashlib.sha256(actual['feature'].tobytes()).hexdigest()),
                                raw/f'{population}_{scene}_{rep}_{ordinal}_{mode}.pt')
                            print(population,scene[:8],rep,mode,ordinal,round(timing['wall_ms'],2),None if q is None else q['frozen_targets_pass'],flush=True)
    finally:
        model.lm_head.generate=native;model.extract_img_feat=extract
        for h in hooks:h.remove()
    write_json(dest/'status.json',dict(completed=True,requests=len(runs),quality_checks=len(checks),
        frozen_native_checks_pass=all(q['frozen_targets_pass'] for q in checks if q['mode'] in ('full_maintain','m3_native')),
        microbatch_quality_pass=all(q['frozen_targets_pass'] for q in checks if q['mode'].startswith('microbatch'))))
    return runs,checks


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT));args=p.parse_args();root=Path(args.output)
    gate=read(root/'microbatch_gpu_feasibility.json')
    if not gate['opportunity'] or not gate['task_independence']:raise ValueError('No opportunity; stop prototype/service work')
    hashes=pin(root/'validation');model,dataset=load(root/'validation')
    run(model,dataset,root)
    from .fallback import run as fallback
    fallback(model,root)
    write_json(root/'validation/status.json',dict(completed=True,
        sources_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())))


if __name__=='__main__':main()
