"""Fresh synchronized full-model profile; nested intervals are not additive.

Instrumentation stays in this runner. Every actual prefill/decode invocation
has its own CUDA/wall timer; percentile subtraction is never used.
"""
import argparse
import gc
import copy
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import random
import resource
from time import perf_counter
import numpy as np
import torch
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m3 import M3Runtime
from worldstream.m3.visual_state import base_model
from tools.worldstream_m2.rearchitecture.integration import load
from tools.worldstream_m2.rearchitecture.common import arrays, memory, compare, quality_pass
from tools.worldstream_m2.refine_integration import planning_metric
from tools.worldstream_m1.io_utils import write_json

ROOT=Path('outputs/m2_freeze_m3_task_serving')


class InvocationTimer:
    def __init__(self):self.rows=[];self.originals=[]
    def wrap(self,obj,attr,label,dynamic=None,observer=None):
        native=getattr(obj,attr)
        @wraps(native)
        def call(*args,**kw):
            name=dynamic(args,kw) if dynamic else label
            torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            tick=perf_counter();a.record()
            try:out=native(*args,**kw)
            finally:b.record();b.synchronize()
            row=dict(stage=name,wall_ms=(perf_counter()-tick)*1000,gpu_ms=a.elapsed_time(b))
            if observer:row.update(observer(args,kw,out))
            self.rows.append(row);return out
        self.originals.append((obj,attr,native));setattr(obj,attr,call)
    def restore(self):
        for o,a,f in reversed(self.originals):setattr(o,a,f)
    def collect(self):
        result={}
        for r in self.rows:
            d=result.setdefault(r['stage'],dict(wall_ms=0.,gpu_ms=0.,calls=0))
            for k in ('wall_ms','gpu_ms'):d[k]+=r[k]
            d['calls']+=1
        return result


def stage(args,kw):
    # A suffix prefill may have length one. KV presence alone is insufficient;
    # native suffix entry uses inputs_embeds, decode uses generated input_ids.
    return 'prefill' if kw.get('inputs_embeds') is not None or kw.get('past_key_values') is None else 'decode'


class GreedyRuntime(M3Runtime):
    def generate(self,*a,**kw):
        kw.update(do_sample=False);kw.pop('temperature',None);kw.pop('top_p',None)
        out,diag=super().generate(*a,**kw)
        self.last_generated_ids=[x.detach().cpu().tolist() for x in out]
        return out,diag


@torch.no_grad()
def run(args):
    dest=ROOT/args.phase;dest.mkdir(parents=True,exist_ok=False)
    from tools.worldstream_m2.source_state_refinement_cuda_batch import wait_idle
    import os
    selected=wait_idle(dest,None);os.environ['CUDA_VISIBLE_DEVICES']=str(selected['index'])
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    import cv2
    cv2.setNumThreads(1)
    model,dataset=load();base=base_model(model.lm_head)
    source_files=[*Path('worldstream').rglob('*.py'),Path(__file__)]
    write_json(dest/'source_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    write_json(dest/'environment.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),checkpoint='ckpts/iter_10548.pth',phase=args.phase,timers='synchronized CUDA/wall per invocation; overlapping intervals not additive'))
    groups=json.loads(Path('outputs/module1_final_validation_refinement/replay/protocol.json').read_text())['scenes'][:args.scenes]
    specs=[]
    for g in groups:
        ids=sorted([i for i,v in enumerate(dataset.data_infos) if v['scene_token']==g['scene_id']],key=lambda i:dataset.data_infos[i]['timestamp'])
        start=next(j for j,i in enumerate(ids) if dataset.data_infos[i]['token']==g['samples'][0]);specs.append((g['scene_id'],ids[start:start+args.frames]))
    modes=['original'] if args.phase=='initial_profile' else ['original','full_maintain','frozen_exact_m2','exact_prefix','evidence_shadow','experimental_direct','direct_prefix_fallback']
    populations=['natural_planning','native_multi','identical_requests'] if args.phase!='initial_profile' else ['natural_planning','native_multi']
    timer=InvocationTimer();captures=[];carriers=[];queries=[];generation_ids=[]
    def lm_observe(a,k,o):
        t=k.get('inputs_embeds');t=t if t is not None else k.get('input_ids')
        if t is None and a:t=a[0]
        return dict(input_length=int(t.shape[1]) if t is not None else None,has_past=k.get('past_key_values') is not None)
    timer.wrap(base,'forward','llm',stage,lm_observe)
    def expanded(a,k,o):
        embeddings=o[4];ids=a[0] if a else k.get('input_ids')
        return dict(prompt_tokens=int(ids.numel()),expanded_tokens=int(embeddings.shape[1]) if embeddings is not None else int(ids.numel()))
    timer.wrap(base,'prepare_inputs_labels_for_multimodal','multimodal_expansion',observer=expanded)
    timer.wrap(model,'extract_img_feat','vision',observer=lambda a,k,o:(captures.append(o.detach()) or {}))
    for name,head in [('det',model.pts_bbox_head),('map',model.map_head)]:
        timer.wrap(head,'forward',name+'_head')
        for attr in ('pre_update_memory','post_update_memory'):timer.wrap(head,attr,name+'_'+attr)
    native=inspect.getclosurevars(model.lm_head.generate).nonlocals['generate']
    def generate(*a,**kw):
        kw.update(do_sample=False);kw.pop('temperature',None);kw.pop('top_p',None)
        carriers.append(kw['images'].detach());out=native(*a,**kw);generation_ids.append(out.detach().cpu().tolist());return out
    model.lm_head.generate=generate;timer.wrap(model.lm_head,'generate','generation')
    pts=model.simple_test_pts;all_native=[]
    def pts_entry(img_metas,**data):
        from mmcv.parallel import DataContainer
        meta=img_metas[0];qs=list(meta['vlm_labels'].data);ids=list(data['input_ids'][0]);all_native.append(dict(frame=meta['sample_idx'],questions=qs,prompt_lengths=[int(x.numel()) for x in ids]))
        if current_population[0]=='identical_requests':take=[0,0]
        elif current_population[0]=='native_multi':take=list(range(min(args.questions,len(qs))))
        else:take=[0]
        queries.extend(qs[i] for i in take);request_ids[:]=[ids[i].detach().cpu() for i in take];meta['vlm_labels']=DataContainer([qs[i] for i in take],cpu_only=True);data['input_ids'][0]=[ids[i] for i in take]
        return pts(img_metas,**data)
    model.simple_test_pts=pts_entry
    runs=[];quality=[];refs={};current_population=[''];snapshots=[];request_ids=[]
    from mmcv.parallel import collate,scatter
    try:
        for population in populations:
            current_population[0]=population
            for transform in dataset.pipeline.transforms:
                if hasattr(transform,'load_type'):transform.load_type=['planning','short','conv'] if population=='native_multi' else ['planning']
            random.seed(0);samples={i:dataset[i] for _,ids in specs for i in ids}
            for scene,indices in specs:
                for rep in range(args.repeats+args.warmup):
                    for mode in modes:
                        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
                        model.test_flag=False;model.m1_manager=None;model.m2_runtime=None;model.m3_runtime=None
                        gc.collect();torch.cuda.empty_cache()
                        if mode!='original':
                            model.m1_manager=WorldStateManager(M1Config.load('configs/worldstream_m1_engineering_frozen.yaml'))
                            for attr in ('precheck_with_observation','commit'):
                                timer.wrap(model.m1_manager,attr,'M1_'+attr)
                                timer.originals.pop()
                        if mode not in ('original','full_maintain'):
                            model.m2_runtime=IncrementalVisionRuntime(model.img_backbone,dict(mode='auto',reuse_boundary='window',state_interface='source_linked',cache_capture='always',cost_table='outputs/module2_refinement/break_even_results.json',experimental_conditional_reuse=False))
                        if mode in ('exact_prefix','evidence_shadow','experimental_direct','direct_prefix_fallback'):
                            cfg=dict(mode='exact',offline_experiment=True)
                            if mode!='exact_prefix':cfg.update(task_evidence_enabled=True,shadow_evaluation=mode=='evidence_shadow',experimental_direct_query=mode!='evidence_shadow',direct_query_prefix_fallback=mode=='direct_prefix_fallback')
                            model.m3_runtime=GreedyRuntime(cfg)
                            timer.wrap(model.m3_runtime,'generate','M3_context_and_generation');timer.originals.pop()
                        model.save_path=str(dest/'language'/f'{population}_{scene}_{mode}_{rep}')+'/'
                        torch.cuda.reset_peak_memory_stats()
                        for ordinal,index in enumerate(indices):
                            batch=scatter(collate([copy.deepcopy(samples[index])],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                            captures.clear();carriers.clear();queries.clear();request_ids.clear();generation_ids.clear();timer.rows.clear()
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();tick=perf_counter();a.record()
                            out=model(return_loss=False,rescale=True,**batch);b.record();b.synchronize();elapsed=(perf_counter()-tick)*1000
                            actual=dict(output=arrays(out),feature=arrays(captures[-1]),memory=memory(model),text=out[0].get('text_out'),planning=planning_metric(out[0].get('text_out'),dataset.data_infos[index]))
                            key=(population,scene,rep,ordinal)
                            if mode=='original':refs[key]=actual
                            else:
                                q=compare(actual,refs[key]);q.update(mode=mode,population=population,scene=scene,repeat=rep,ordinal=ordinal,frozen_targets_pass=quality_pass(q));quality.append(q)
                            row=dict(population=population,population_label='exploratory_reused_population',scene=scene,frame=dataset.data_infos[index]['token'],timestamp=dataset.data_infos[index]['timestamp']/1e6,mode=mode,repeat=rep,warmup=rep<args.warmup,ordinal=ordinal,E2E_wall_ms=elapsed,E2E_GPU_ms=a.elapsed_time(b),stages=timer.collect(),invocations=copy.deepcopy(timer.rows),questions=queries.copy(),generated_ids=getattr(model.m3_runtime,'last_generated_ids',generation_ids) if model.m3_runtime else generation_ids,prompt_lengths=[int(x.numel()) for x in request_ids],text=actual['text'],planning=actual['planning'],m2=copy.deepcopy(model.m2_runtime.last_diagnostics) if model.m2_runtime else {},m3=copy.deepcopy(getattr(model,'m3_diagnostics',{})) if model.m3_runtime else {},m1_commit=copy.deepcopy(model.m1_manager.last_commit_diagnostics) if model.m1_manager else {},peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved(),cpu_rss_high_water_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                            runs.append(row)
                            with (dest/'runs.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
                            if mode=='original' and rep==0 and ordinal==0 and carriers:
                                snapshots.append(dict(population=population,scene=scene,frame=row['frame'],carriers=carriers[0].cpu(),ids=[x.clone() for x in request_ids],questions=queries.copy()))
                            print(population,mode,scene,rep,ordinal,round(elapsed,2),len(queries),flush=True)
                    write_json(dest/'quality.json',quality)
        if args.phase!='initial_profile':
            from .task_serving_recovery import recovery_probe
            model.m1_manager=None;model.m2_runtime=None;model.m3_runtime=None
            # Restore wrappers before the independently synchronized recovery probe.
            timer.restore();model.simple_test_pts=pts
            from .task_serving_checks import run_checks
            run_checks(model,snapshots,dest/'prefix_checks')
            recovery_probe(model,dataset,specs[0][1][:3],dest/'recovery')
        write_json(dest/'native_questions.json',all_native)
        torch.save(snapshots,dest/'carrier_snapshots.pt')
        write_json(dest/'status.json',dict(status='completed',requests=len(runs),quality_checks=len(quality)))
    finally:timer.restore()


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',default='initial_profile');p.add_argument('--scenes',type=int,default=2);p.add_argument('--frames',type=int,default=4);p.add_argument('--questions',type=int,default=3);p.add_argument('--repeats',type=int,default=1);p.add_argument('--warmup',type=int,default=0);run(p.parse_args())

if __name__=='__main__':main()
