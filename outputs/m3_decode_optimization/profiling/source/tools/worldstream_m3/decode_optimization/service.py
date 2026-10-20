"""Paired original full-model continuous nuScenes requests, unchanged sampling."""
import copy
import gc
import hashlib
import json
import random
import resource
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m3 import M3Runtime
from worldstream.m3.visual_state import base_model
from tools.worldstream_m1.io_utils import write_json
from tools.worldstream_m2.rearchitecture.common import arrays, memory, compare, quality_pass
from tools.worldstream_m2.refine_integration import planning_metric
from .profile import ROOT, SAMPLING, ForwardTimer, pin, load

MODES=['original','full_maintain','frozen_exact_m2','m3_native','decode_graph','legacy_prefix','guarded_prefix']


class CapturingRuntime(M3Runtime):
    def generate(self,*a,**kw):
        out,diag=super().generate(*a,**kw)
        self.last_ids=[t.cpu().tolist() for t in out]
        return out,diag


@torch.no_grad()
def main():
    dest=ROOT/'service';hashes=pin(dest);model,dataset=load(dest);base=base_model(model.lm_head)
    from mmcv.parallel import collate,scatter,DataContainer
    specs=[]
    for group in json.loads(Path('outputs/module1_final_validation_refinement/replay/protocol.json').read_text())['scenes'][:2]:
        seq=sorted([i for i,v in enumerate(dataset.data_infos) if v['scene_token']==group['scene_id']],key=lambda i:dataset.data_infos[i]['timestamp'])
        start=next(j for j,i in enumerate(seq) if dataset.data_infos[i]['token']==group['samples'][0])
        specs.append((group['scene_id'],seq[start:start+2]))
    write_json(dest/'protocol.json',dict(modes=MODES,specs=specs,populations=['planning','native_multi'],warmup=1,repeats=2,
        sampling=SAMPLING,order='Original first for each paired repeat; other arms rotate by repeat',
        timers='deferred per-forward CUDA events and CPU submission; request boundary synchronize',
        native_generator='original base.generate with all injected active LoRA; PEFT wrapper compatibility failure archived',
        protected='M1/M2/TaskEvidence/weights/heads unchanged; experimental conditional false'))
    features=[];carrier=[];ids_now=[];qs_now=[];gen_ids=[]
    original_extract=model.extract_img_feat
    def extract(*a,**k):
        out=original_extract(*a,**k);features.append(out.detach());return out
    model.extract_img_feat=extract
    native_generate=model.lm_head.generate
    def generate(*a,**k):
        carrier[:]=[k['images'].detach()]
        out=native_generate(*a,**k);gen_ids.append(out.cpu().tolist());return out
    model.lm_head.generate=generate
    pts=model.simple_test_pts
    def pts_entry(img_metas,**data):
        meta=img_metas[0];qs=list(meta['vlm_labels'].data);ids=list(data['input_ids'][0]);take=list(range(min(3,len(qs)))) if population=='native_multi' else [0]
        ids_now[:]=[ids[i].cpu() for i in take];qs_now[:]=[qs[i] for i in take]
        meta['vlm_labels']=DataContainer(qs_now.copy(),cpu_only=True);data['input_ids'][0]=[ids[i] for i in take]
        return pts(img_metas,**data)
    model.simple_test_pts=pts_entry
    # Runtime objects persist across replays; reset drops request KV, while the
    # decode executor owns only shape/weight graphs (no token/history KV).
    runtimes={m:CapturingRuntime(dict(mode='full' if m in ('m3_native','decode_graph') else 'exact',
        offline_experiment=True,decode_cuda_graph=m=='decode_graph',
        identical_prefill_only=m=='guarded_prefix')) for m in MODES[3:]}
    refs={};snapshots=[];checks=[];runs=[]
    for population in ('planning','native_multi'):
        for transform in dataset.pipeline.transforms:
            if hasattr(transform,'load_type'):transform.load_type=['planning','short','conv'] if population=='native_multi' else ['planning']
        random.seed(0);samples={i:dataset[i] for _,seq in specs for i in seq}
        for scene,seq in specs:
            for rep in range(3):
                rest=MODES[1:];rotation=rep%len(rest);order=['original']+rest[rotation:]+rest[:rotation]
                for mode in order:
                    model.test_flag=False;model.m1_manager=None;model.m2_runtime=None;model.m3_runtime=None
                    gc.collect();torch.cuda.empty_cache()
                    if mode!='original':model.m1_manager=WorldStateManager(M1Config.load('configs/worldstream_m1_engineering_frozen.yaml'))
                    if mode not in ('original','full_maintain'):
                        model.m2_runtime=IncrementalVisionRuntime(model.img_backbone,dict(mode='auto',reuse_boundary='window',state_interface='source_linked',cache_capture='always',cost_table='outputs/module2_refinement/break_even_results.json',experimental_conditional_reuse=False))
                    if mode in runtimes:
                        model.m3_runtime=runtimes[mode];model.m3_runtime.reset()
                    model.save_path=str(dest/'language'/f'{population}_{scene}_{rep}_{mode}')+'/'
                    for ordinal,index in enumerate(seq):
                        random.seed(123+ordinal);np.random.seed(123+ordinal);torch.manual_seed(123+ordinal);torch.cuda.manual_seed_all(123+ordinal)
                        batch=scatter(collate([copy.deepcopy(samples[index])],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                        features.clear();carrier.clear();gen_ids.clear();torch.cuda.reset_peak_memory_stats()
                        with ForwardTimer(base) as timer:
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize();a.record();tick=perf_counter()
                            out=model(return_loss=False,rescale=True,**batch)
                            b.record();b.synchronize();wall=(perf_counter()-tick)*1000
                            invocations=timer.collect()
                        actual=dict(output=arrays(out),feature=arrays(features[-1]),memory=memory(model),text=out[0].get('text_out'),planning=planning_metric(out[0].get('text_out'),dataset.data_infos[index]))
                        generated=getattr(model.m3_runtime,'last_ids',gen_ids) if model.m3_runtime else gen_ids
                        key=(population,scene,rep,ordinal)
                        q=None
                        if mode=='original':refs[key]=(actual,generated)
                        else:
                            ref,refids=refs[key];q=compare(actual,ref);q.update(mode=mode,population=population,scene=scene,repeat=rep,ordinal=ordinal,
                                frozen_targets_pass=quality_pass(q),token_ids_equal=generated==refids,warmup=rep==0)
                            checks.append(q)
                            with (dest/'quality.jsonl').open('a') as f:f.write(json.dumps(q,default=str)+'\n')
                        row=dict(mode=mode,population=population,scene=scene,frame=dataset.data_infos[index]['token'],repeat=rep,ordinal=ordinal,warmup=rep==0,
                            wall_ms=wall,gpu_interval_ms=a.elapsed_time(b),invocations=invocations,generated_ids=generated,questions=qs_now.copy(),
                            text=actual['text'],planning=actual['planning'],peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved(),
                            cpu_rss_high_water_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            quality_pass=q['frozen_targets_pass'] if q else None,m3=copy.deepcopy(getattr(model,'m3_diagnostics',{})) if model.m3_runtime else {},
                            m2=copy.deepcopy(model.m2_runtime.last_diagnostics) if model.m2_runtime else {})
                        runs.append(row)
                        with (dest/'runs.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
                        raw=dest/'raw_outputs';raw.mkdir(exist_ok=True)
                        torch.save(dict(output=actual['output'],memory=actual['memory'],text=actual['text'],planning=actual['planning'],feature_sha256=hashlib.sha256(actual['feature'].tobytes()).hexdigest()),raw/f'{population}_{scene}_{rep}_{ordinal}_{mode}.pt')
                        if mode=='original' and rep==0 and population=='native_multi':
                            snapshots.append(dict(scene=scene,frame=row['frame'],ordinal=ordinal,ids=[x.clone() for x in ids_now],questions=qs_now.copy(),carriers=carrier[0].cpu()))
                            torch.save(snapshots,dest/'carrier_snapshots.pt')
                        print(population,scene,rep,mode,ordinal,round(wall,2),None if q is None else q['frozen_targets_pass'],flush=True)
    write_json(dest/'status.json',dict(completed=True,requests=len(runs),quality_checks=len(checks),
        sources_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())))


if __name__=='__main__':main()
