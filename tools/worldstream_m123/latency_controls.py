"""Identical-state M1 replay and fixed-token decoder controls, benchmark only."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pickle
import types
from time import perf_counter
import numpy as np
import torch
from tools.worldstream_m123.benchmark.metrics import distribution
from tools.worldstream_m123.benchmark.common import write


class ReplayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Captured immutable metadata is reconstructed as owned containers in
        # this offline replay only. Production snapshot mutability is unchanged.
        if module == 'worldstream.m1.readonly_snapshot':
            if name == 'ReadOnlyDict': return dict
            if name == 'ReadOnlyList': return list
        return super().find_class(module, name)


def load_capture(path):
    replay_pickle = types.ModuleType('replay_pickle')
    replay_pickle.__dict__.update(pickle.__dict__)
    replay_pickle.Unpickler = ReplayUnpickler
    # This pinned Torch build overrides the supplied pickle_module in load().
    # Use its ZIP reader explicitly for our own trusted benchmark captures.
    with torch.serialization._open_zipfile_reader(str(path)) as reader:
        return torch.serialization._load(reader, 'cpu', replay_pickle, encoding='utf-8')


def before_package(root, component):
    name='worldstream._latency_before_'+component
    path=root/('before_'+component)
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def clean(value):
    from dataclasses import asdict,is_dataclass
    if is_dataclass(value):return clean(asdict(value))
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return repr(value)
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items() if not (str(k).endswith('_ms') or k in ('timings_ms',))}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    return value


def m1_control(root):
    import cv2
    cv2.setNumThreads(1)
    from worldstream.m1 import WorldStateManager
    from worldstream.m1.state_store import StateStore
    old=before_package(root,'m1')
    oldstore=sys.modules[old.__name__+'.state_store'].StateStore
    records=[]
    for path in sorted((root/'after_profile/m1_controls').glob('*.pt')):
        saved=load_capture(path)
        for repeat in range(3):
            outcomes={}
            for label,cls,storecls in ([('before',old.WorldStateManager,oldstore),('after',WorldStateManager,StateStore)] if repeat%2==0 else [('after',WorldStateManager,StateStore),('before',old.WorldStateManager,oldstore)]):
                manager=object.__new__(cls);manager.__dict__.update(copy.deepcopy(saved['state']))
                manager.store.__class__=storecls
                frame=copy.deepcopy(saved['frame']);outputs=copy.deepcopy(saved['outputs'])
                tick=perf_counter();result=manager.precheck(frame);pre=(perf_counter()-tick)*1000
                context=dict(scene_id=frame.scene_id,frame_id=frame.frame_id,timestamp=frame.timestamp,
                    state_epoch=manager.store.generation,execution_signature=frame.execution_signature)
                tick=perf_counter();snapshot=manager.export_m2_inputs(expected_context=context,computation_only=True);export=(perf_counter()-tick)*1000
                snapshot=clean(snapshot)
                tick=perf_counter();manager.commit(frame,outputs,result);commit=(perf_counter()-tick)*1000
                outcomes[label]=dict(precheck_ms=pre,snapshot_ms=export,commit_ms=commit,total_ms=pre+export+commit,
                    snapshot=snapshot,anchors=clean(manager.store.anchors),
                    associations=clean(manager.last_commit_diagnostics['association_pairs']))
            checks={key:outcomes['before'][key]==outcomes['after'][key] for key in ('snapshot','anchors','associations')}
            if not all(checks.values()):
                write(root/'m1_control_failure.json',dict(frame=path.stem,checks=checks,before=outcomes['before'],after=outcomes['after']))
                raise ValueError('M1 semantic control differs: '+str(checks))
            records.append(dict(frame_id=path.stem,repeat=repeat,checks=checks,
                **{k:{kk:vv for kk,vv in v.items() if kk.endswith('_ms')} for k,v in outcomes.items()}))
    if not records:raise ValueError('No captured real M1 states')
    summary={label:{key:distribution(r[label][key] for r in records) for key in ('precheck_ms','snapshot_ms','commit_ms','total_ms')} for label in ('before','after')}
    write(root/'m1_same_state_control.json',dict(records=records,summary=summary,all_semantics_equal=True,
        scope='Same actual frame, committed state, perception outputs and M2 receipt; alternating three repeats; diagnostic CPU replay, not E2E timing'))
    print('M1 CONTROL', {k:v['total_ms']['mean'] for k,v in summary.items()},flush=True)


@torch.no_grad()
def decode_control(root):
    import importlib
    importlib.import_module('projects.mmdet3d_plugin')
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    wait_gpu()
    from tools.worldstream_m123.benchmark.worker import effective_config,load_model
    from worldstream.m3.visual_state import expand_input,base_model
    from worldstream.m3.prefill_executor import GenerationFacade
    old=before_package(root,'m3')
    oldfacade=sys.modules[old.__name__+'.prefill_executor'].GenerationFacade
    dest=root/'decode_control';dest.mkdir(exist_ok=False)
    cfg=effective_config('B3','language',dest)
    model=load_model(cfg,'ckpts/iter_10548.pth',dest)
    base=base_model(model.lm_head)
    candidates=[]
    for p in (root/'before_profile/m3_inputs').glob('*.pt'):
        saved=load_capture(p);candidates.append((saved['generated_ids'][0].numel(),p,saved))
    _,path,saved=max(candidates,key=lambda r:r[0])
    ids=saved['ids'][0].cuda();carriers=saved['carriers'].cuda()
    expanded=expand_input(model.lm_head,ids,carriers,carrier_layout=saved['layout'])
    prompt=base(inputs_embeds=expanded.embeddings,position_ids=expanded.position_ids,attention_mask=expanded.attention_mask,use_cache=True,return_dict=True)
    initial=prompt.past_key_values
    versions=tuple(t._version for layer in initial for t in layer)
    tokens=saved['generated_ids'][0].cuda();steps=min(32,tokens.shape[-1]-2)
    if steps<2:raise ValueError('Insufficient real generated tokens for a fixed decoder control')
    measurements=[];reference=None
    for repeat in range(6):
        for label,cls in ([('before',oldfacade),('after',GenerationFacade)] if repeat%2==0 else [('after',GenerationFacade),('before',oldfacade)]):
            facade=cls(base,expanded,lambda:prompt);facade.first=False
            kv=initial;events=[];logits=[];torch.cuda.synchronize();start=perf_counter()
            for i in range(steps):
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2));a.record()
                mask=torch.cat((expanded.attention_mask,torch.ones((1,i+1),device=ids.device,dtype=expanded.attention_mask.dtype)),dim=-1)
                inputs=facade.prepare_inputs_for_generation(tokens[:,:i+2],past_key_values=kv,attention_mask=mask)
                output=facade(**inputs);kv=output.past_key_values;logits.append(output.logits[:,-1].clone())
                b.record();events.append((a,b))
            torch.cuda.synchronize();wall=(perf_counter()-start)*1000
            if tuple(t._version for layer in initial for t in layer)!=versions:raise ValueError('Initial KV was mutated')
            values=torch.stack(logits).cpu()
            if reference is None:reference=values
            elif not torch.equal(reference,values):raise ValueError('Fixed-token decode logits changed')
            if repeat:measurements.append(dict(kind=label,repeat=repeat,wall_ms=wall,token_ms=[a.elapsed_time(b) for a,b in events]))
    # One extra sampled trace is excluded from measurements.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        kv=initial
        for i in range(steps):
            mask=torch.cat((expanded.attention_mask,torch.ones((1,i+1),device=ids.device,dtype=expanded.attention_mask.dtype)),dim=-1)
            output=base(input_ids=tokens[:,i+1:i+2],position_ids=(expanded.attention_mask.long().sum(-1,keepdim=True)+i),attention_mask=mask,past_key_values=kv,use_cache=True,return_dict=True);kv=output.past_key_values
    prof.export_chrome_trace(str(dest/'decode_cuda.json'))
    write(dest/'summary.json',dict(captured_source=str(path),steps=steps,repeats=5,same_token_sequence=True,same_initial_kv=True,logits_bitwise_equal=True,
        dtype=str(expanded.embeddings.dtype),records=measurements,
        summary={label:dict(total_wall_ms=distribution(r['wall_ms'] for r in measurements if r['kind']==label),
            per_token_cuda_span_ms=distribution(t for r in measurements if r['kind']==label for t in r['token_ms'])) for label in ('before','after')},
        scope='Teacher-forced captured actual tokens; same full prefill KV, weights and dtype; HF sampling unchanged in serving; event spans include host gaps'))
    print('FIXED DECODE',steps,'steps, logits equal',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--stage',choices=['m1','decode'],required=True)
    a=p.parse_args()
    if a.stage=='m1':m1_control(Path(a.root))
    else:
        from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
        with gpu_job_lock():decode_control(Path(a.root))
