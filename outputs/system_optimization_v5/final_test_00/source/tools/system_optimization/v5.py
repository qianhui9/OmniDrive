"""Frozen v5 feasibility experiment; no approximate input is enabled online."""
import argparse
import copy
import hashlib
import inspect
from functools import wraps
import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
from time import perf_counter

from .audit import write

ROOT = Path(os.environ.get('OMNIDRIVE_V5_OUTPUT','outputs/system_optimization_v5'))


class ProbeCapacityError(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def freeze():
    path = ROOT/'experiment_protocol.json'
    if path.exists():
        raise FileExistsError('Never overwrite a frozen protocol')
    infos = pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl', 'rb'))['infos']
    previous = json.loads(Path('outputs/system_optimization_v4/protocol.json').read_text())
    excluded = set(previous['excluded_scenes'])
    excluded.update(s['scene'] for specs in previous['selections'].values() for s in specs)
    # All observed v1-v4 run protocols, including module development data.
    for root in ('outputs/system_optimization', 'outputs/system_optimization_v2',
                 'outputs/system_optimization_v3', 'outputs/module2_rearchitecture',
                 'outputs/module2_refinement', 'outputs/module1_refinement'):
        for file in Path(root).glob('**/protocol.json'):
            if 'source' in file.parts: continue
            def visit(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in ('scene', 'scene_id', 'scene_token') and isinstance(item, str):
                            excluded.add(item)
                        else: visit(item)
                elif isinstance(value, list):
                    for item in value: visit(item)
            visit(json.loads(file.read_text()))
    groups = {}
    for info in infos: groups.setdefault(info['scene_token'], []).append(info)
    pools = {city: [s for s, rows in sorted(groups.items()) if s not in excluded and
                   ('boston' if rows[0]['location'].startswith('boston') else 'singapore') == city]
             for city in ('boston', 'singapore')}
    selections = {}
    for ordinal, split in enumerate(('calibration', 'heldout', 'final_test')):
        chosen = sorted(s for pool in pools.values() for s in pool[3*ordinal:3*ordinal+3])
        if len(chosen) != 6: raise ValueError('Insufficient untouched scenes')
        selections[split] = [dict(scene=s, location=groups[s][0]['location'],
            tokens=[r['token'] for r in sorted(groups[s], key=lambda r:r['timestamp'])[:4]]) for s in chosen]
    for name in ('logs', 'traces', 'figures'): (ROOT/name).mkdir(parents=True, exist_ok=True)
    write(path, dict(frozen_before_outcomes=True, selections=selections, excluded_scenes=sorted(excluded),
        selection='Sorted unused scenes; 3 per city per split, first 4 consecutive keyframes',
        generation=previous['generation'], physical_risk=previous['physical_risk'],
        task_quality=previous['task_quality'], performance=previous['performance'],
        compact_quality=dict(ade_delta_max_m=0., fde_delta_max_m=0., invalid_rate_delta_max=0.,
            require_every_scene_noninferior=True, token_equality='diagnostic only',
            failures='Invalid output cannot disappear from paired quality denominator'),
        compact_representation='Final short format: all current >=0.3 objects class/xy integer metres; all >=0.3 lane first/middle/last xy; ego telemetry; approximate, no GT',
        modes=['full','current_compact','compact_full'], cpu_affinity='0-3,6-31',
        checkpoint='ckpts/iter_10548.pth', frames_per_scene=4, repeats=1, warmup=0,
        online_enable=False, history_consumption='Not run unless A and B pass',
        fact_accuracy='Source and current unique offline GT identity; requested current ego sign correct; unknown unscorable',
        fact_calibration='Scene maxima of absolute x/y localization residuals, split conformal 95%; no detector-score conversion',
        no_tuning_after_heldout=True, qa_scope='Fixed seed stratified review list; automated labels stay Unknown',
        final_scope='Local nuScenes validation scene holdout; not official hidden test'))
    write(ROOT/'compact_representation_amendment.json',dict(frozen_before_any_independent_outcomes=True,
        reason='Fresh reproduction of final v5 short format; original calibration failures remain in original v5 directory',
        representation='All native >=0.3 objects class+xy integer metres; lanes first/middle/last xy; ego telemetry',
        quality_gates_scenes_generation_unchanged=True))
    fact_source=Path('outputs/system_optimization_v4/task_fact_protocol.json')
    fact_protocol=json.loads(fact_source.read_text())
    fact_protocol.update(frozen_before_v5_fact_oracle=True,online_consumption_allowed=False,
        source_protocol_sha256=sha256(fact_source),splits='v5 experiment_protocol.json disjoint scenes',
        direction='Both current ego x/y signs match unique source/current offline instance, current sensor support and visibility=4; zero boundaries cannot certify',
        certification='Persist annotation-free candidate/certification before Oracle; freeze calibration artifact before independent Oracle score',
        interval='Joint xy square from scene max absolute residual; rank ceil((n+1)*.95); unbounded if rank>n',
        population_limitation='2m offline association censors source error; diagnostic calibration only')
    write(ROOT/'relative_direction_protocol.json',fact_protocol)
    # Preserve v1-v4 reports, protocols, and certificates using a content manifest.
    protected = {}
    for root in ('outputs/system_optimization','outputs/system_optimization_v2',
                 'outputs/system_optimization_v3','outputs/system_optimization_v4'):
        for file in Path(root).glob('*'):
            if file.is_file(): protected[str(file)] = sha256(file)
    write(ROOT/'preserved_artifacts.json', protected)


def gpu():
    import cv2
    import numpy as np
    import torch
    import transformers
    from mmcv.parallel import collate, scatter
    from tools.worldstream_m2.rearchitecture.integration import load
    from tools.worldstream_m3.qa_analysis import planning_quality
    from worldstream.m3.visual_state import base_model
    from .experiment import seed
    from .compact_context import current_context, probe_inputs
    from .v3 import environment

    if set(os.sched_getaffinity(0)) & {4,5}: raise RuntimeError('Forbidden CPUs')
    torch.set_num_threads(1); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    protocol=json.loads((ROOT/'experiment_protocol.json').read_text())
    expected=json.loads(Path('outputs/system_optimization_v3/model_manifest.json').read_text())['sha256']
    before=sha256(protocol['checkpoint'])
    if before != expected: raise AssertionError('Checkpoint changed')
    write(ROOT/'model_manifest.json',dict(checkpoint_sha256=before,lora='embedded original',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__))
    write(ROOT/'environment_before.json',environment())
    model,dataset=load()
    # integration.load wraps native generate in a greedy adapter; use its exact
    # captured original callable, as the existing single-question runner does.
    native=inspect.getclosurevars(model.lm_head.generate).nonlocals['generate']
    model.m1_manager=None;model.m2_runtime=None;model.m3_runtime=None
    model.stream_runtime=None
    for transform in dataset.pipeline.transforms:
        if hasattr(transform,'load_type'): transform.load_type=['planning']
    indices={info['token']:i for i,info in enumerate(dataset.data_infos)}
    captured={};timings=[]
    base=base_model(model.lm_head)
    original_forward=base.forward
    @wraps(original_forward)
    def timed_forward(*args,**kwargs):
        tensor=kwargs.get('inputs_embeds')
        if tensor is None: tensor=kwargs.get('input_ids')
        if tensor is None and args: tensor=args[0]
        label='prefill' if kwargs.get('past_key_values') is None else 'decode'
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();tick=perf_counter();a.record()
        out=original_forward(*args,**kwargs)
        b.record();torch.cuda.synchronize()
        timings.append(dict(stage=label,wall_ms=(perf_counter()-tick)*1000,
                            gpu_ms=a.elapsed_time(b),tokens=int(tensor.shape[1])))
        return out
    base.forward=timed_forward
    old_pts=model.simple_test_pts
    def pts(metas,_m1_observer=None,**data):
        def observer(boxes,lanes,features):
            captured.update(boxes=boxes,lanes=lanes,ego=data.get('can_bus',torch.empty(0)))
            if _m1_observer is not None: _m1_observer(boxes,lanes,features)
        return old_pts(metas,_m1_observer=observer,**data)
    model.simple_test_pts=pts
    def generate(*args,**kwargs):
        if args: raise ValueError('Expected native keyword input')
        tick=perf_counter()
        text='' if captured['mode']=='full' else current_context(captured['boxes'],captured['lanes'],captured['ego'])
        ids,images=probe_inputs(kwargs['inputs'],kwargs['images'],model.tokenizer,text,captured['mode'])
        captured.update(context=text,context_ms=(perf_counter()-tick)*1000,
            original_input_tokens=int(kwargs['inputs'].shape[1]),input_tokens=int(ids.shape[1]),
            carrier_tokens=int(kwargs['images'].shape[1]),
            effective_tokens=int(ids.shape[1])+(int(images.shape[1])-1 if images is not None else 0))
        if captured['effective_tokens']+kwargs.get('max_new_tokens',320)>base.config.max_position_embeddings:
            raise ProbeCapacityError('Frozen representation plus unchanged answer budget exceeds native position capacity')
        kwargs.update(inputs=ids,images=images)
        # Same native generator/LoRA/tokenizer/sampling and answer budget.
        output=native(**kwargs)
        captured['generated_ids']=output.detach().cpu().tolist()
        return output
    model.lm_head.generate=generate
    rows_path=ROOT/'traces/compact_frames.jsonl'
    if rows_path.exists(): raise FileExistsError('Use a fresh run; never overwrite measurements')
    source={str(p):sha256(p) for folder in ('worldstream','tools/system_optimization') for p in Path(folder).rglob('*.py')}
    write(ROOT/'source_manifest.json',source)
    with rows_path.open('x') as handle, torch.no_grad():
        for split,specs in protocol['selections'].items():
            for ordinal,spec in enumerate(specs):
                modes=protocol['modes']; modes=modes[ordinal%3:]+modes[:ordinal%3]
                for mode in modes:
                    model.test_flag=False
                    model.pts_bbox_head.reset_memory();model.map_head.reset_memory()
                    model.save_path=str(ROOT/'language_fixed_format'/split/spec['scene']/mode)+'/'
                    for frame_ordinal,token in enumerate(spec['tokens']):
                        index=indices[token]
                        seed(index);captured.clear();captured['mode']=mode;timings.clear()
                        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=perf_counter()
                        sample=dataset[index]
                        batch=scatter(collate([sample],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                        preparation_ms=(perf_counter()-tick)*1000
                        seed(index+10000)
                        failure=None
                        try:
                            output=model(return_loss=False,rescale=True,**batch)
                        except (ProbeCapacityError,torch.cuda.OutOfMemoryError) as error:
                            failure=dict(type=type(error).__name__,reason=str(error))
                            torch.cuda.empty_cache()
                        torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
                        answer=output[0]['text_out'][0] if failure is None else None
                        # Offline GT scoring only AFTER generation/timer.
                        row=dict(split=split,scene=spec['scene'],location=spec['location'],sample_token=token,
                            ordinal=frame_ordinal,mode=mode,e2e_wall_ms=wall,input_preparation_ms=preparation_ms,
                            context_construction_ms=captured['context_ms'],context=captured['context'],
                            input_tokens=captured['input_tokens'],original_input_tokens=captured['original_input_tokens'],
                            carrier_tokens=captured['carrier_tokens'],effective_multimodal_tokens=captured['effective_tokens'],
                            generated_ids=captured.get('generated_ids'),answer=answer,failure=failure,
                            planning=planning_quality(' '.join(answer['A']),dataset.data_infos[index]) if answer else dict(status='not_generated_capacity_failure',ade_m=None,fde_m=None),
                            stages=copy.deepcopy(timings),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                            cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                            evidence_scope='real_gpu_instrumented_offline_approximate_probe',online_approved=False,
                            complete_vlm_timing=failure is None)
                        handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()
                        print(split,spec['scene'][:8],mode,frame_ordinal,'tokens',row['effective_multimodal_tokens'],
                              'ms',round(wall,1),'planning',row['planning']['status'],flush=True)
                write(ROOT/'progress.json',dict(split=split,scene=spec['scene']))
    write(ROOT/'environment_after.json',environment())
    after=sha256(protocol['checkpoint'])
    write(ROOT/'weights_preservation.json',dict(before=before,after=after,unchanged=before==after))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    if args.freeze:freeze()
    else:
        if (ROOT/'traces/compact_frames.jsonl').exists():
            raise FileExistsError('Existing measurements; use OMNIDRIVE_V5_OUTPUT with a fresh directory')
        write(ROOT/'gpu_status.json',dict(status='running'))
        try:gpu()
        except BaseException as error:
            write(ROOT/'gpu_status.json',dict(status='failed',error=repr(error)));raise
        else:write(ROOT/'gpu_status.json',dict(status='completed'))
