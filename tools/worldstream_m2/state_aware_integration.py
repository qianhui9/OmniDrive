"""Original VLM, frozen M1, matched window boundary; three separate populations."""
import copy
import hashlib
import json
from pathlib import Path
import random
import resource
from time import perf_counter
import numpy as np
import torch
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.source_alignment import token_sources,probe_features
from worldstream.m1.mask_codec import decode_mask
from tools.worldstream_m1.io_utils import write_json
from .rearchitecture.integration import load,PipelineTimer
from .rearchitecture.common import arrays,memory,compare,quality_pass
from .refine_integration import planning_metric
from .refinement_common import error_location
from .state_aware_feasibility import ROOT


def layer_error(actual,baseline,window=16):
    """CPU diagnostics after synchronized request timer; retain spatial errors."""
    report=error_location(torch.from_numpy(actual),torch.from_numpy(baseline))
    delta=actual.astype(np.float32)-baseline.astype(np.float32)
    squared=np.square(delta).sum(-1);reference=np.square(baseline.astype(np.float32)).sum(-1)
    maxima=np.abs(delta).max(-1);tokens=np.sqrt(squared)
    report.update(token_error_p50=float(np.percentile(tokens,50)),token_error_p95=float(np.percentile(tokens,95)),windows=[])
    for camera in range(len(actual)):
        for y in range(0,actual.shape[1],window):
            for x in range(0,actual.shape[2],window):
                sl=(camera,slice(y,y+window),slice(x,x+window))
                report['windows'].append(dict(camera=camera,row=y//window,col=x//window,
                    max_abs=float(maxima[sl].max()),relative_l2=float(np.sqrt(squared[sl].sum()/max(float(reference[sl].sum()),1e-24)))))
    return report


class FullFeatureProbe(IncrementalVisionRuntime):
    def _execute_forward(self,images,m1_result=None,m1_state=None,execution_context=None):
        c=execution_context or {};ex=self.conditional_executor;old=ex.content;previous=copy.copy(ex.cache.context)
        value,diag=ex.full(images,c,self.signature(c),'diagnostic_probe_full_execution',self.stage_observer)
        bundle=getattr(m1_result,'computation_state',None);reports=[]
        if old is not None and previous and bundle is not None and not m1_result.full_refresh_hint:
            for i,cid in enumerate(c['camera_ids']):
                contract=bundle.refresh_contract[cid]
                mapping=token_sources(bundle.observation_links[cid],decode_mask(contract['reuse_candidate_mask']),
                    decode_mask(contract['dirty_hint_mask']),decode_mask(contract['unknown_mask']),self.model.patch_embed.proj.kernel_size,
                    previous['frame_id'],cid,c['frame_id'])
                reports.append(probe_features(old[i],ex.content[i],mapping))
        diag['feature_probe']=reports;self.last_diagnostics=diag;return value,diag


class UnqualifiedLayoutIntervention(IncrementalVisionRuntime):
    """Controlled-source oracle stress test, NEVER M1-admitted or Auto eligible."""
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.ordinal=0
    def _execute_forward(self,images,m1_result=None,m1_state=None,execution_context=None):
        ex=self.conditional_executor;c=execution_context or {};ordinal=self.ordinal;self.ordinal+=1
        if ordinal in (0,6,11) or ex.cache.context is None or (m1_result is not None and m1_result.full_refresh_hint):
            value,diag=ex.full(images,c,self.signature(c),'controlled_full_refresh',self.stage_observer)
        else:
            n=len(ex.cache.outputs[0]);sources=np.arange(n,dtype=np.int64)
            # Controlled image source: actual first-camera top two complete
            # windows are swapped every frame. Not an online M1 observation.
            sources[0],sources[1]=1,0
            content=self.model.patch_embed(images)
            value,diag=ex.execute(images,content,c,self.signature(c),sources,self.stage_observer,
                reports=[dict(control='known_input_permutation_oracle',M1_admitted=False)],conditional_windows=2,exact_windows=n-2)
        diag.update(reuse_mode='experimental_conditional' if diag['actual_skipped_windows'] else 'full',
            evidence='controlled_known_input_source_intervention_not_M1_admission',production_eligible=False)
        self.last_diagnostics=diag;return value,diag


@torch.no_grad()
def run(model,dataset,case,scenes=2,frames=8,warmup=1,repeats=1):
    dest=ROOT/'gpu_validation'/case;dest.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(Path('outputs/module1_final_validation_refinement/replay/protocol.json').read_text())
    groups={}
    for spec in protocol['scenes'][:scenes]:
        ids=[i for i,v in enumerate(dataset.data_infos) if v['scene_token']==spec['scene_id']]
        ids.sort(key=lambda i:dataset.data_infos[i]['timestamp'])
        start=next(j for j,i in enumerate(ids) if dataset.data_infos[i]['token']==spec['samples'][0]);groups[spec['scene_id']]=ids[start:start+frames]
    modes=['original','full_maintain','existing_exact','source_probe','conditional','conditional_optimized']
    if case=='local':modes+=['unqualified_layout_intervention']
    write_json(dest/'protocol.json',dict(case=case,groups=groups,modes=modes,warmup=warmup,repeats=repeats,
        M1='configs/worldstream_m1_engineering_frozen.yaml',cost_table='outputs/module2_refinement/break_even_results.json',
        quality='unchanged 1cm detection/map, parsed planning and text equality; raw + matched memory retained',
        population='exploratory_reused_population',local_control='base RGB or top two 256x256 windows swapped; full prefix refresh at ordinals 0/6/11',
        source_control='Layout-intervention source map is explicit input-permutation oracle, not M1 evidence; it cannot qualify production.'))
    files=[*Path('worldstream/m1').glob('*.py'),*Path('worldstream/m2').glob('*.py'),Path(__file__),Path('configs/worldstream_m1_engineering_frozen.yaml')]
    write_json(dest/'source_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    refs={};runs=[];quality=[];captured=[];stages={};work=[]
    extract=model.extract_img_feat
    timer=PipelineTimer()
    def observed_extract(*args,**kw):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        tick=perf_counter();a.record();values=extract(*args,**kw);b.record()
        timer.records.append(('visual_service',a,b,(perf_counter()-tick)*1000))
        captured.append(values.detach());return values
    model.extract_img_feat=observed_extract
    hs=[]
    hs.append(model.img_backbone.patch_embed.register_forward_hook(lambda m,a,v:stages.update(patch=v.detach())))
    for i,b in enumerate(model.img_backbone.blocks):
        if i in (0,1,2,len(model.img_backbone.blocks)-1):hs.append(b.register_forward_hook(lambda m,a,v,i=i:stages.update({str(i):v.detach()})))
        hs.append(b.attn.register_forward_pre_hook(lambda m,a,i=i:work.append(dict(block=i,component='attention',shape=list(a[0].shape)))))
        hs.append(b.mlp.register_forward_pre_hook(lambda m,a,i=i:work.append(dict(block=i,component='MLP',shape=list(a[0].shape)))))
    original_forwards=[]
    modules=[('patch_embedding',model.img_backbone.patch_embed),('detection_and_memory',model.pts_bbox_head),
        ('map_and_memory',model.map_head),('LLM_decoder',model.lm_head.get_model())]
    for i,b in enumerate(model.img_backbone.blocks):
        modules.extend([(f'block_{i}',b),(f'attention_{i}',b.attn),(f'MLP_{i}',b.mlp)])
    for name,module in modules:
        original_forwards.append((module,module.forward));timer.wrap(module,name)
    generate=model.lm_head.generate
    def observed_generate(*args,**kw):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        tick=perf_counter();a.record();value=generate(*args,**kw);b.record()
        timer.records.append(('LLM_generation',a,b,(perf_counter()-tick)*1000));return value
    model.lm_head.generate=observed_generate
    try:
        for rep in range(warmup+repeats):
            for scene,ids in groups.items():
                arms=modes[1:];order=['original']+arms[rep%len(arms):]+arms[:rep%len(arms)]
                for mode in order:
                    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
                    model.test_flag=False;model.m1_manager=None;model.m2_runtime=None
                    if mode!='original':model.m1_manager=WorldStateManager(M1Config.load('configs/worldstream_m1_engineering_frozen.yaml'))
                    if mode not in ('original','full_maintain'):
                        cfg=dict(mode='auto' if mode=='existing_exact' else 'exact',reuse_boundary='window',
                            state_interface='source_linked',cache_capture='always',cost_table='outputs/module2_refinement/break_even_results.json',
                            experimental_conditional_reuse=mode!='existing_exact',conditional_packing_optimization=mode=='conditional_optimized')
                        cls=FullFeatureProbe if mode=='source_probe' else UnqualifiedLayoutIntervention if mode=='unqualified_layout_intervention' else IncrementalVisionRuntime
                        model.m2_runtime=cls(model.img_backbone,cfg)
                        def observe(i,x):
                            if i in (0,1,2,23):stages[str(i)]=x.detach()
                        model.m2_runtime.stage_observer=observe
                    model.save_path=str(dest/f'{scene}_{mode}_{rep}'/'language')+'/'
                    sequence=[];base=None;torch.cuda.reset_peak_memory_stats()
                    for ordinal,index in enumerate(ids):
                        from mmcv.parallel import collate,scatter
                        batch=scatter(collate([dataset[index]],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                        if case!='natural':
                            if base is None:base=batch['img'][0].clone()
                            batch['img'][0].copy_(base)
                            if case=='local' and ordinal%2 and ordinal not in (6,11):
                                batch['img'][0][0,0,:,:256,:256]=base[0,0,:,:256,256:512]
                                batch['img'][0][0,0,:,:256,256:512]=base[0,0,:,:256,:256]
                        if case=='local' and ordinal in (6,11) and model.m2_runtime is not None:
                            # Prefix refresh alone does not reset downstream temporal memory.
                            model.m2_runtime.reset()
                        captured.clear();stages.clear();work.clear();timer.records.clear()
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize();tick=perf_counter();a.record();out=model(return_loss=False,rescale=True,**batch);b.record();torch.cuda.synchronize()
                        wall=(perf_counter()-tick)*1000
                        current=dict(output=arrays(out),feature=arrays(captured[-1]),memory=memory(model),text=out[0].get('text_out'),
                            planning=planning_metric(out[0].get('text_out'),dataset.data_infos[index]),stages={k:arrays(v) for k,v in stages.items()})
                        if mode=='original':refs[scene,rep,ordinal]=current
                        else:
                            q=compare(current,refs[scene,rep,ordinal]);q.update(mode=mode,scene=scene,repeat=rep,ordinal=ordinal,frozen_targets_pass=quality_pass(q))
                            q['layers']={k:layer_error(v,refs[scene,rep,ordinal]['stages'][k]) for k,v in current['stages'].items()}
                            quality.append(q)
                        diag=copy.deepcopy(model.m2_runtime.last_diagnostics) if model.m2_runtime else {}
                        sequence.append(dict(frame=dataset.data_infos[index]['token'],ordinal=ordinal,E2E_wall_ms=wall,E2E_GPU_ms=a.elapsed_time(b),
                            runtime=diag,M1_time_ms=(model.m1_manager.last_commit_diagnostics.get('m1_total_including_adapters_ms',0) if model.m1_manager else 0),
                            m1_commit=copy.deepcopy(model.m1_manager.last_commit_diagnostics) if model.m1_manager else None,
                            m1_anchor_ids=list(model.m1_manager.store.anchors) if model.m1_manager else None,
                            m1_computation_state=(dict(source_records=copy.deepcopy(model.m1_manager.last_result.computation_state.source_records)) if model.m1_manager else None),
                            pipeline=timer.collect(),actual_GPU_work=copy.deepcopy(work),
                            current_model_outputs=dict(text=current['text'],planning=current['planning'])))
                        folder=dest/'memory'/f'{scene}_{mode}_{rep}';folder.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(folder/f'{ordinal}.npz',**{f'{branch}.{name}':v for branch,values in current['memory'].items() for name,v in values.items() if isinstance(v,np.ndarray)})
                        print(case,mode,scene,rep,ordinal,round(wall,2),diag.get('reuse_mode'),diag.get('actual_skipped_windows'),flush=True)
                    runs.append(dict(scene=scene,mode=mode,repeat=rep,warmup=rep<warmup,frames=sequence,
                        peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved(),cpu_rss_high_water_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                    write_json(dest/'runs.json',runs);write_json(dest/'quality.json',quality)
        # Controlled failures are evidence, not an exception to discard the run.
        write_json(dest/'status.json',dict(status='completed',quality_pass=all(q['frozen_targets_pass'] for q in quality),
            quality_by_mode={m:dict(checks=sum(q['mode']==m for q in quality),passed=sum(q['mode']==m and q['frozen_targets_pass'] for q in quality)) for m in modes if m!='original'}))
    finally:
        model.extract_img_feat=extract
        model.lm_head.generate=generate
        for h in hs:h.remove()
        for module,original in original_forwards:module.forward=original


def main():
    model,dataset=load()
    for case,scenes,frames,warmup in (('natural',2,8,1),('repeat',1,8,1),('local',1,12,0)):
        run(model,dataset,case,scenes,frames,warmup)


if __name__=='__main__':
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    import cv2
    cv2.setNumThreads(1);main()
