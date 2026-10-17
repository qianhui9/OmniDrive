"""Paired real continuous OmniDrive refinement protocol; no answer-file reuse."""
import argparse
import copy
from contextlib import nullcontext
from dataclasses import replace
import gc
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import subprocess
from functools import wraps
import sys
from time import perf_counter
import numpy as np
import torch
from worldstream.m3 import M3Runtime,expand_input
from worldstream.m3.visual_state import base_model
from worldstream.m3.numerics import precision_probe
from .experiment import Timer,write,ground_truth,chosen_questions,typed_digest


def monitor_gpu(function):
    @wraps(function)
    def wrapped(args):
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        log=path.parent/'logs'/f'{path.name}_gpu.csv';log.parent.mkdir(exist_ok=True)
        with log.open('w') as handle:
            process=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                '--format=csv,noheader,nounits','--loop-ms=1000'],stdout=handle,stderr=subprocess.STDOUT)
            try:return function(args)
            finally:
                process.terminate();process.wait(timeout=10)
    return wrapped


class RecordedRuntime:
    """No instance-bound closure or hidden retention of a previous runtime."""
    def __init__(self,runtime,recorded,sampling=False,legacy=False,policy='native'):
        self.inner=runtime;self.recorded=recorded;self.config=runtime.config;self.sampling=sampling;self.legacy=legacy
        self.policy=policy
    def reset(self):self.inner.reset()
    def generate(self,*a,**kw):
        if not self.sampling:
            kw.update(do_sample=False);kw.pop('temperature',None);kw.pop('top_p',None)
        if self.legacy:
            kw.pop('verified_state',None);kw.pop('tokenizer',None)
        lm=kw.get('lm_head',a[0] if a else None)
        with precision_probe(base_model(lm),self.policy):
            answers,diag=self.inner.generate(*a,**kw)
        self.recorded.extend(x.detach() for x in answers)
        return answers,diag


def legacy_runtime():
    path=Path('outputs/module3/clean_multi_20261016_220113/source/worldstream/m3')
    spec=importlib.util.spec_from_file_location('m3_frozen_legacy',path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.M3Runtime


def scenes_for(dataset):
    scenes=copy.deepcopy(json.loads(Path('outputs/module3/clean_multi_20261016_220113/protocol.json').read_text())['scenes'])
    for s in scenes:s['split']='calibration'
    used={s['scene'] for s in scenes};groups={}
    for i,info in enumerate(dataset.data_infos):groups.setdefault(info['scene_token'],[]).append(i)
    for scene,ids in sorted(groups.items()):
        if scene in used or len(ids)<4:continue
        ids=sorted(ids,key=lambda i:dataset.data_infos[i]['timestamp'])
        info=dataset.data_infos[ids[0]]
        scenes.append(dict(scene=scene,scene_name=info.get('scene_name'),category='independent',
            indices=ids[:2],description=info.get('description'),split='heldout'))
        if len(scenes)==8:break
    return scenes


def snapshot_source(dest):
    manifest={}
    paths=list(Path('worldstream/m3').glob('*.py'))+list(Path('tools/worldstream_m3').glob('*.py'))
    paths+=[Path('projects/mmdet3d_plugin/models/detectors/petr3d.py')]
    for p in paths:
        target=dest/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    write(dest/'source_manifest.json',manifest)


def freeze_sample_questions(sample,frozen):
    """Replay only native question IDs/text, never historical image/carrier data.
    Run on prepared CPU samples before scatter and all service timing.
    """
    from mmcv.parallel import DataContainer
    assert len(sample['input_ids'])==len(sample['img_metas'])==1
    sample['input_ids']=[DataContainer([x.detach().clone() for x in frozen['ids']],stack=False)]
    sample['img_metas'][0].data['vlm_labels']=DataContainer(list(frozen['questions']),stack=False)
    return sample


@monitor_gpu
@torch.no_grad()
def run(args):
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False);snapshot_source(dest)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    import cv2
    cv2.setNumThreads(1)
    from tools.worldstream_m2.rearchitecture.integration import load
    from worldstream.m1 import M1Config,WorldStateManager
    from worldstream.m2 import IncrementalVisionRuntime
    from mmcv.parallel import scatter,collate,DataContainer
    model,dataset=load();base=base_model(model.lm_head);Legacy=legacy_runtime()
    from .refinement_probes import install_service_clock
    install_service_clock(model)
    native=inspect.getclosurevars(model.lm_head.generate).nonlocals['generate']
    model.lm_head.generate=native
    if args.probes:
        from .refinement_probes import run as run_probes
        run_probes(model,dest/'controlled_probes')
    for transform in dataset.pipeline.transforms:
        if hasattr(transform,'load_type'):transform.load_type=['planning'] if args.single else ['planning','short','conv']
    scenes=scenes_for(dataset)
    if args.single:scenes=[scenes[0],scenes[-1]]
    if args.smoke:scenes=[scenes[0]]
    if args.numeric_only:scenes=[next(s for s in scenes if 5489 in s['indices'])]
    modes=args.modes
    write(dest/'protocol.json',dict(scenes=scenes,modes=modes,warmup=args.warmup,repeats=args.repeats,
        frames=2,questions=1 if args.single else 5,max_new_tokens=320,sampling=args.sampling,precision=args.precision,
        frozen_quality=dict(full_token_equality=True,task_quality_required=True,cost_benefit_required=True),
        selection='Original question text only, two heldout scenes excluded from candidate choice',
        state_policy='precommit history snapshot; unavailable -> full carrier; no fabricated context'))
    write(dest/'environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),argv=vars(args)))
    # Reuse prepared CPU samples outside all timing regions.
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    samples={i:dataset[i] for s in scenes for i in s['indices'][:2]}
    frozen_prompts={};frozen_manifest={}
    for scene in scenes:
        if scene['split']!='calibration':continue
        for index in scene['indices'][:2]:
            token=dataset.data_infos[index]['token']
            path=Path('outputs/module3/clean_multi_20261016_220113/carriers')/(token+'.pt')
            old=torch.load(path,map_location='cpu')
            # Only these fields are retained from the earlier real run.
            frozen=dict(ids=old['ids'],questions=old['questions'],selected_indices=old['selected_indices'])
            freeze_sample_questions(samples[index],frozen);frozen_prompts[token]=frozen
            frozen_manifest[token]=dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                selected_indices=frozen['selected_indices'],historical_carriers_used=False)
    write(dest/'frozen_questions.json',frozen_manifest)
    recorded=[];current={};snapshots={}
    def record_generate(*a,**kw):
        if not args.sampling:
            kw.update(do_sample=False);kw.pop('temperature',None);kw.pop('top_p',None)
        token=current['token']
        if token not in snapshots:
            # Owned CPU capture outside the timed service pass is performed
            # separately by the post-frame snapshot flag below.
            snapshots[token]=dict(scene=current['scene'],carriers=kw['images'].detach(),
                ids=[x.detach() for x in current['all_ids']],questions=current['all_questions'],
                selected_indices=current['selected_indices'],ground_truth=current['ground_truth'])
        answer=native(*a,**kw)
        # The runtime adapter records its complete ordered return once. A
        # runtime-native fallback must not be logged a second time here.
        if model.m3_runtime is None:recorded.append(answer.detach())
        return answer
    model.lm_head.generate=record_generate
    pts=model.simple_test_pts
    def entry(metas,**data):
        meta=metas[0];all_questions=list(meta['vlm_labels'].data);all_ids=list(data['input_ids'][0])
        token=meta['sample_idx']
        picked=([0] if args.single else frozen_prompts[token]['selected_indices']) if token in frozen_prompts else chosen_questions(all_questions,5)
        current.clear();current.update(token=meta['sample_idx'],scene=meta['scene_token'],all_ids=all_ids,
            all_questions=all_questions,selected_indices=picked,questions=[all_questions[i] for i in picked],
            ground_truth=ground_truth(meta['sample_idx']))
        meta['vlm_labels']=DataContainer(current['questions'],cpu_only=True);data['input_ids'][0]=[all_ids[i] for i in picked]
        return pts(metas,**data)
    model.simple_test_pts=entry
    timer=Timer();timer.wrap(base,'llm',dynamic=True)
    timer.wrap_method(base,'prepare_inputs_labels_for_multimodal','multimodal_preparation')
    timer.wrap(model.img_backbone,'eva');timer.wrap(model.pts_bbox_head,'detection');timer.wrap(model.map_head,'map')
    timer.wrap(model.pts_bbox_head.output_projection,'object_projection');timer.wrap(model.map_head.output_projection,'map_projection')
    initial_versions=[(n,p._version) for n,p in model.named_parameters()]
    (dest/'carriers').mkdir()
    for rep in range(args.warmup+args.repeats):
        for si,scene in enumerate(scenes):
            offset=(rep+si)%len(modes);ordered=modes[offset:]+modes[:offset]
            for mode in ordered:
                if model.m3_runtime is not None:model.m3_runtime.reset()
                model.m1_manager=None;model.m2_runtime=None;model.m3_runtime=None;model.test_flag=False;gc.collect()
                # State-context candidates must pay the full real M1 service.
                if mode in ('state_only','state_full','m123','m12'):
                    model.m1_manager=WorldStateManager(replace(M1Config.load('configs/worldstream_m1_coverage.yaml'),
                        log_path=str(dest/'m1_state.jsonl')))
                if mode in ('m123','m12'):
                    model.m2_runtime=IncrementalVisionRuntime(model.img_backbone,dict(mode='auto',reuse_boundary='camera',
                        cheap_gate=True,m1_service='maintain',memory_budget_mb=512,
                        camera_cost_table='outputs/module2_rearchitecture/camera_costs.json'))
                if mode not in ('baseline','m12'):
                    config=dict(mode='exact',offline_experiment=True)
                    runtime_class=M3Runtime
                    if mode=='legacy_exact':runtime_class=Legacy
                    if mode=='legacy_exact':pass
                    elif mode=='compact':config.update(mode='compact')
                    elif mode in ('state_only','state_full'):config.update(mode='compact',state_context='only' if mode=='state_only' else 'augment')
                    else:config.update(borrow_prefix=True)
                    if mode=='common_prefix_seq':config.update(common_prefix_seed=True)
                    if mode in ('batched','refined_combined','m123'):config.update(batched_decode=True)
                    policy=mode.removeprefix('precision_') if mode.startswith('precision_') else args.precision if mode in ('refined_exact','refined_combined','m123') else 'native'
                    model.m3_runtime=RecordedRuntime(runtime_class(config),recorded,args.sampling,legacy=mode=='legacy_exact',policy=policy)
                policy=args.precision if mode in ('refined_exact','refined_combined','m123') else 'native'
                if mode.startswith('precision_'):policy=mode.removeprefix('precision_')
                with nullcontext():
                    for fi,index in enumerate(scene['indices'][:2]):
                        model.save_path=str(dest/f'answers/r{rep}_{si}_{mode}_{fi}')+'/'
                        batch=scatter(collate([copy.deepcopy(samples[index])],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
                        timer.events.clear();recorded.clear();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=perf_counter()
                        output=model(return_loss=False,rescale=True,**batch)
                        torch.cuda.synchronize();wall=(perf_counter()-tick)*1000;components=timer.collect()
                        snapshot=getattr(model,'m3_verified_state',None)
                        row=dict(repeat=rep,warmup=rep<args.warmup,mode=mode,scene=scene['scene'],category=scene['category'],
                            split=scene['split'],index=index,token=current['token'],cold_sequence=fi==0,wall_ms=wall,components=components,
                            questions=current['questions'],ground_truth=current['ground_truth'],answers=output[0]['text_out'],
                            generated_ids=[x.cpu().tolist() for x in recorded],diagnostics=copy.deepcopy(getattr(model,'m3_diagnostics',{})) if model.m3_runtime else {},
                            m1=copy.deepcopy(model.m1_manager.last_commit_diagnostics) if model.m1_manager else {},
                            m2=copy.deepcopy(getattr(model,'m2_diagnostics',{})) if model.m2_runtime else {},
                            state_snapshot=dict(diagnostics=snapshot.diagnostics,records=snapshot.records) if snapshot else None,
                            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),precision=policy)
                        if args.single:
                            memory={name:{k:v for k,v in vars(head).items() if k.startswith('memory_') or k=='sample_time'}
                                for name,head in [('det',model.pts_bbox_head),('map',model.map_head)]}
                            row['perception_digest']=typed_digest(dict(boxes=output[0]['pts_bbox'],lanes=output[0]['lane_results'],memory=memory))
                        with (dest/'frames.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
                        for token,snap in list(snapshots.items()):
                            if isinstance(snap['carriers'],torch.Tensor) and snap['carriers'].is_cuda:
                                snap['carriers']=snap['carriers'].cpu().clone();snap['ids']=[x.cpu().clone() for x in snap['ids']]
                                torch.save(snap,dest/'carriers'/f'{token}.pt')
                        print('FRAME',rep,mode,index,scene['split'],round(wall,2),flush=True)
    assert initial_versions==[(n,p._version) for n,p in model.named_parameters()]
    write(dest/'completion.json',dict(complete=True,weights_versions_unchanged=True,
        expected_rows=(args.warmup+args.repeats)*sum(min(2,len(s['indices'])) for s in scenes)*len(modes)))
    if model.m3_runtime is not None:model.m3_runtime.reset()
    model.m3_runtime=None;model.m1_manager=None;model.m2_runtime=None;gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--precision',default='native')
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--single',action='store_true');p.add_argument('--sampling',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--probes',action='store_true')
    p.add_argument('--numeric-only',action='store_true')
    p.add_argument('--modes',nargs='+',default=['baseline','legacy_exact','refined_exact','compact','state_only','state_full','batched','refined_combined','m123','m12'])
    run(p.parse_args())
