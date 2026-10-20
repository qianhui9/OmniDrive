"""Full native service first; separate profiler/witness replays, no reuse path."""
import argparse
from contextlib import ExitStack
from functools import wraps
import json
import os
from pathlib import Path
import random
import subprocess
from time import perf_counter

import numpy as np
import torch
from tools.worldstream_m123.stateful_opportunity.common import (
    SAMPLING, diagnostic_json, manifest, protected_files, replace_method, sha256, write)
from tools.worldstream_m123.stateful_opportunity.gpu_audit import load_native, wait_idle, state_snapshot
from .profile import Scopes, aggregate_events, aggregate_trace, install_scopes


def seed():
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)


def lm_stage(a,k):
    return 'llm_prefill' if k.get('inputs_embeds') is not None and k['inputs_embeds'].shape[1]>1 else 'llm_decode'


def service_scopes(model, stack):
    from worldstream.m3.visual_state import base_model
    scopes=Scopes(stack,ranges=False)
    for module,label in [(model.img_backbone,'eva'),(model.pts_bbox_head,'detection'),(model.map_head,'map')]:
        scopes.wrap(module,'forward',label)
    scopes.wrap(base_model(model.lm_head),'forward',lm_stage)
    scopes.wrap(model.lm_head,'generate','llm_generation')
    return scopes


class Witness:
    """Whole-subgraph input equality, with known source, never similarity."""
    def __init__(self, model, stack):
        self.current={};self.previous=None;self.rows=[];self.state=[]
        for name,head in [('detection',model.pts_bbox_head),('map',model.map_head)]:
            def capture(module,args,name=name):
                for key,value in zip(('query','current_visual','query_pos','current_pos','mask','temporal_memory','temporal_pos'),args):
                    if isinstance(value,torch.Tensor):self.current[name+'/'+key]=value.detach()
            h=head.transformer.register_forward_pre_hook(capture);stack.callback(h.remove)
        def image(module,args):self.current['rgb']=args[0].detach()
        h=model.img_backbone.register_forward_pre_hook(image);stack.callback(h.remove)
        from worldstream.m3.visual_state import base_model
        generate=model.lm_head.generate
        @wraps(generate)
        def generation(*a,**kw):
            self.current['llm/native_token_ids']=kw['inputs'].detach()
            self.current['llm/rng_cpu']=torch.get_rng_state()
            self.current['llm/rng_cuda']=torch.cuda.get_rng_state()
            return generate(*a,**kw)
        stack.enter_context(replace_method(model.lm_head,'generate',generation))
        base=base_model(model.lm_head);forward=base.forward
        @wraps(forward)
        def lm(*a,**kw):
            prefill=kw.get('inputs_embeds') is not None and kw['inputs_embeds'].shape[1]>1
            if prefill:
                for key in ('inputs_embeds','position_ids','attention_mask'):
                    value=kw.get(key)
                    if isinstance(value,torch.Tensor):self.current['llm/'+key]=value.detach()
            output=forward(*a,**kw)
            if prefill:
                self.current['llm/layer0_key']=output.past_key_values[0][0].detach()
                self.current['llm/layer0_value']=output.past_key_values[0][1].detach()
                self.current['llm/last_logits']=output.logits[:,-1:].detach()
            return output
        stack.enter_context(replace_method(base,'forward',lm))
        precheck=model.m1_manager.precheck_with_observation
        def freeze(frame,*a,**kw):
            result=precheck(frame,*a,**kw)
            _,state=state_snapshot(model.m1_manager,frame)
            self.state.append(diagnostic_json(state))
            return result
        stack.enter_context(replace_method(model.m1_manager,'precheck_with_observation',freeze))

    def finish(self,context):
        checks={};tick=perf_counter()
        old=self.previous
        legal=bool(old and old['context']['scene']==context['scene'] and
            0<context['timestamp']-old['context']['timestamp']<=1. and
            old['context']['cameras']==context['cameras'] and
            old['context']['signature']==context['signature'] and old['context']['epoch']==context['epoch'])
        if legal:
            for name,value in self.current.items():
                previous=old['values'].get(name)
                shape=previous is not None and previous.shape==value.shape and previous.dtype==value.dtype
                checks[name]=dict(shape_equal=shape,exact_equal=bool(shape and torch.equal(value,previous)),
                    current_shape=list(value.shape),dtype=str(value.dtype))
        verify_ms=(perf_counter()-tick)*1000
        tick=perf_counter()
        owned={k:v.detach().clone() for k,v in self.current.items()}
        torch.cuda.synchronize()
        copy_ms=(perf_counter()-tick)*1000
        self.rows.append(dict(**context,source_frame=old['context']['frame'] if old else None,
            history_legal=legal,dependencies=checks,whole_input_verification_wall_ms=verify_ms,
            diagnostic_snapshot_copy_wall_ms=copy_ms,
            diagnostic_snapshot_bytes=sum(v.numel()*v.element_size() for v in owned.values()),
            actual_reuse=0,decision='Observation only; no executor attached'))
        self.previous=dict(context=dict(context),values=owned)
        self.current.clear()


def frozen_groups(dataset,scenes,frames):
    old=json.loads(Path('outputs/stateful_cross_frame_opportunity/gpu_audit02/protocol.json').read_text())['groups']
    index={v['token']:i for i,v in enumerate(dataset.data_infos)}
    groups=[]
    for group in old[:scenes]:
        tokens=group['tokens'][:frames]
        ordered=sorted((v for v in dataset.data_infos if v['scene_token']==group['scene']),key=lambda v:v['timestamp'])
        start=next(i for i,v in enumerate(ordered) if v['token']==tokens[0])
        if [v['token'] for v in ordered[start:start+len(tokens)]]!=tokens:raise ValueError('Not consecutive real frames')
        groups.append(dict(group,tokens=tokens,timestamps=group['timestamps'][:len(tokens)],indices=[index[t] for t in tokens]))
    return groups


@torch.no_grad()
def run(args):
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    write(dest/'status.json',dict(status='running',actual_reuse=0))
    before=manifest(protected_files());write(dest/'protected_before.json',before)
    old_outputs=manifest([*Path('outputs/stateful_cross_frame_opportunity').glob('*.json'),
        *Path('outputs/stateful_cross_frame_opportunity').glob('*.md'),
        Path('outputs/stateful_cross_frame_opportunity/gpu_audit02/regression.xml')])
    write(dest/'historical_output_manifest.json',old_outputs)
    paths=[*Path('tools/worldstream_m123/large_boundary').glob('*.py'),
        *Path('tools/worldstream_m123/stateful_opportunity').glob('*.py')]
    source=manifest(paths);write(dest/'source_manifest.json',source)
    for path in paths:
        target=dest/'source'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    if {4,5}&os.sched_getaffinity(0):raise ValueError('Faulty CPUs must be excluded')
    torch.set_num_threads(1)
    import cv2,transformers
    cv2.setNumThreads(1);wait_idle();seed()
    print('HASH_ORIGINAL_CHECKPOINT',flush=True)
    write(dest/'environment.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        python=os.sys.executable,torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(0),cpu_affinity=sorted(os.sched_getaffinity(0)),seed=0,
        checkpoint_sha256=sha256('ckpts/iter_10548.pth'),
        tokenizer=manifest(Path('ckpts/pretrain_qformer').glob('token*')),sampling=SAMPLING,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,argv=vars(args)))
    print('LOAD_NATIVE_MODEL',flush=True)
    model,dataset=load_native()
    from worldstream.m1 import M1Config,WorldStateManager
    from worldstream.m3.visual_state import base_model
    from tools.worldstream_m3.experiment import typed_digest
    from tools.worldstream_m2.rearchitecture.common import memory
    from mmcv.parallel import collate,scatter
    base=base_model(model.lm_head)
    versions=[(n,p.data_ptr(),p._version) for n,p in model.named_parameters()]
    groups=frozen_groups(dataset,args.scenes,args.frames)
    write(dest/'protocol.json',dict(groups=groups,population='Exploratory',original_task='planning',
        task_classifier_label='global_safety',native_multi_cross_frame_requalification='Not Run',
        dataset=dict(path=dataset.ann_file,sha256=sha256(dataset.ann_file)),
        service_modes=['Native Original','Persistent Full'],active_modes='Not Run',
        warmup='Cold ordinal 0 reported separately; steady paired set excludes ordinal 0',
        order='Rotate two complete scene replays by scene; reset native Memory/M1/RNG for each replay',
        service_timing='Input loading/preprocessing/collate/scatter + complete model + native answer write + final GPU sync. Output digests and memory copy outside timing. Lightweight identical coarse CUDA event scopes retained in both modes.',
        profiler='Separate replay, two selected real requests; record_shapes=False, profile_memory=True; traces and event attribution preserved.',
        quality='Native/Persistent token+output+memory guards only; Stateful Active task quality Not Run',
        config='projects/configs/OmniDrive/worldstream_m1_coverage_observe.py',sampling=SAMPLING))
    write(dest/'effective_generation.json',dict(defaults=base.generation_config.to_dict(),overrides=SAMPLING))
    tokens=[];generate=model.lm_head.generate
    @wraps(generate)
    def record(*a,**kw):
        if any(kw.get(k)!=v for k,v in SAMPLING.items()):raise ValueError('Original generation changed')
        output=generate(*a,**kw);tokens.append(output.detach());return output
    service=[];profiles=[];guards={};state_rows=[];witness_rows=[]
    def configure(mode,path):
        seed();model.test_flag=False;model.m2_runtime=model.m3_runtime=None
        model.m1_manager=WorldStateManager(M1Config.load('configs/worldstream_m1_engineering_frozen.yaml')) if mode=='Persistent Full' else None
        model.save_path=str(path)+'/'
    def request(index):
        return scatter(collate([dataset[index]],samples_per_gpu=1),[0])[0]
    def output_guard(output):
        return diagnostic_json(dict(output=typed_digest(output),memory=typed_digest(memory(model)),
            token_ids=[v.cpu().tolist() for v in tokens]))
    with replace_method(model.lm_head,'generate',record):
        for si,group in enumerate(groups):
            modes=['Native Original','Persistent Full'] if si%2==0 else ['Persistent Full','Native Original']
            for mode in modes:
                configure(mode,dest/'language'/'service'/str(si)/mode)
                with ExitStack() as stack:
                    scopes=service_scopes(model,stack)
                    for ordinal,index in enumerate(group['indices']):
                        tokens.clear();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=perf_counter()
                        batch=request(index);torch.cuda.synchronize();input_ms=(perf_counter()-tick)*1000
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        a.record();model_tick=perf_counter();output=model(return_loss=False,rescale=True,**batch)
                        b.record();torch.cuda.synchronize();model_ms=(perf_counter()-model_tick)*1000
                        wall=(perf_counter()-tick)*1000
                        service.append(dict(scene=group['scene'],frame=group['tokens'][ordinal],ordinal=ordinal,
                            mode=mode,cold=ordinal==0,request_wall_ms=wall,model_wall_ms=model_ms,input_ms=input_ms,
                            model_cuda_event_interval_ms=a.elapsed_time(b),scopes=scopes.collect(),
                            m1=diagnostic_json(model.m1_manager.last_commit_diagnostics) if model.m1_manager else None,
                            m1_precheck=diagnostic_json(model.m1_manager.last_result.diagnostics['timings_ms']) if model.m1_manager else None,
                            m2=dict(enabled=False,maintenance_ms=0,actual_reuse=0),m3=dict(enabled=False,maintenance_ms=0,actual_reuse=0),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),actual_skipped_forwards=0))
                        key=group['scene']+'/'+group['tokens'][ordinal]
                        guards.setdefault(key,{})[mode]=output_guard(output)
                        write(dest/'service_frames.json',service)
                        write(dest/'service_output_guards.json',guards)
                        print('SERVICE',si,mode,ordinal,round(wall,2),flush=True)
        write(dest/'phase.json',dict(status='service_completed_profile_next',frames=len(service)))
        # Profiling/witness subset is separate from the measured service population.
        for gi,group in enumerate(groups[:2]):
            configure('Persistent Full',dest/'language'/'profiler'/str(gi))
            with ExitStack() as stack:
                witness=Witness(model,stack)
                scopes=install_scopes(model,stack)
                for ordinal,index in enumerate(group['indices'][:4]):
                    tokens.clear();batch=request(index);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                    selected=ordinal==(1 if gi==0 else 3)
                    tick=perf_counter()
                    if selected:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False,profile_memory=True,with_stack=False) as prof:
                            with torch.autograd.profiler.record_function('lb::request'):
                                output=model(return_loss=False,rescale=True,**batch)
                                torch.cuda.synchronize()
                            profiled_wall=(perf_counter()-tick)*1000
                        teardown=(perf_counter()-tick)*1000-profiled_wall
                        trace=dest/'traces'/f'{gi}_{ordinal}.json';trace.parent.mkdir(exist_ok=True)
                        prof.export_chrome_trace(str(trace))
                        reference=aggregate_events(prof.events())
                        summary=aggregate_trace(trace,reference)
                        profiles.append(dict(scene=group['scene'],frame=group['tokens'][ordinal],ordinal=ordinal,
                            profiled_model_wall_ms=profiled_wall,profiler_teardown_wall_ms=teardown,
                            trace=str(trace),summary=summary,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                        write(dest/'profiles.json',profiles)
                    else:
                        output=model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
                    intervals=scopes.collect()
                    if selected:profiles[-1]['scopes']=intervals
                    state=witness.state[-1]
                    witness.finish(dict(scene=group['scene'],frame=group['tokens'][ordinal],ordinal=ordinal,
                        timestamp=state['timestamp'],cameras=list(state['camera_timestamps']),
                        epoch=state['epoch'],signature=state['signature']))
                    profile_guard=output_guard(output)
                    service_guard=guards[group['scene']+'/'+group['tokens'][ordinal]]['Persistent Full']
                    witness.rows[-1]['service_outputs_bitwise_equal']=profile_guard==service_guard
                    write(dest/'profiles.json',profiles)
                    print('WITNESS_PROFILE',gi,ordinal,'profile',selected,flush=True)
                witness_rows.extend(witness.rows);state_rows.extend(witness.state)
                write(dest/'dependency_witnesses.json',witness_rows);write(dest/'m1_states.json',state_rows)
    after=manifest(protected_files())
    write(dest/'protected_after.json',after)
    write(dest/'status.json',dict(status='completed',service_requests=len(service),scenes=len(groups),
        profile_requests=len(profiles),witness_requests=len(witness_rows),actual_reuse=0,
        protected_source_unchanged=before==after,historical_outputs_unchanged=old_outputs==manifest([Path(p) for p in old_outputs]),
        parameters_unchanged=versions==[(n,p.data_ptr(),p._version) for n,p in model.named_parameters()],
        audit_source_unchanged=source==manifest(paths),hooks_restored=True,
        gpu_accounting_valid=all(p['summary']['accounting_valid'] for p in profiles),
        attribution_complete=all(p['summary']['attribution_complete'] for p in profiles)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--scenes',type=int,default=8);p.add_argument('--frames',type=int,default=12)
    args=p.parse_args()
    if not 2<=args.scenes<=8 or not 4<=args.frames<=12:p.error('Use 2..8 frozen scenes and 4..12 consecutive frames')
    try:run(args)
    except Exception as error:
        status=Path(args.output)/'status.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='running':
            write(status,dict(status='failed',error=repr(error),actual_reuse=0,production_go=False))
        raise


if __name__=='__main__':main()
