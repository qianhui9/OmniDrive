"""Fresh real-weight VLM ablations, paired by frame and generation contract.

Examples (OmniDrive environment, >=19500 MiB free for full batch experiments):
 python -m tools.system_optimization.experiment --output outputs/system_optimization/single --workload single
Never writes any previous module's artifacts. All answers are really generated.
"""
import argparse
import copy
from dataclasses import replace
import gc
import hashlib
import inspect
from functools import wraps
import json
from pathlib import Path
import pickle
import random
import resource
import subprocess
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch
from worldstream.runtime import UnifiedRuntime
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m3 import M3Runtime
from .audit import write


MODES = ['original', 'm1', 'm2', 'm3', 'm12', 'm13', 'm23', 'm123',
         'unified_persistent', 'unified_output']
RUN_HOOKS = {}  # Optional audit callbacks, always outside measured inference.


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


class RecordedM3(M3Runtime):
    def __init__(self, config, sink):
        super().__init__(config)
        self.sink = sink

    def generate(self, *args, **kwargs):
        self.sink.in_m3 = True
        try:
            output, diag = super().generate(*args, **kwargs)
        finally:
            self.sink.in_m3 = False
        # Outside the coordinator's timed language computation; still inside
        # complete service wall, identically charged to all methods.
        self.sink.ids = [x.detach().cpu().tolist() for x in output]
        return output, diag


def snapshot(dest):
    manifest = {}
    files = list(Path('worldstream').rglob('*.py'))+list(Path('tools/system_optimization').glob('*.py'))
    files += [Path('projects/mmdet3d_plugin/models/detectors/petr3d.py')]
    for p in files:
        target = dest/'source'/p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
        manifest[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    write(dest/'source_manifest.json', manifest)


def configure(model, mode, sink, dest, budget, refresh, profile):
    previous=getattr(model,'stream_runtime',None)
    if previous is not None and hasattr(previous,'close'):previous.close()
    model.m1_manager = model.m2_runtime = model.m3_runtime = None
    persistent = mode in ('m1', 'm12', 'm13', 'm123', 'unified_persistent', 'state_only', 'state_full')
    if persistent:
        settings = replace(M1Config.load('configs/worldstream_m1_coverage.yaml'),
            logging=True, log_path=str(dest/'m1_state.jsonl'), full_refresh_interval=refresh)
        model.m1_manager = WorldStateManager(settings)
    if mode in ('m2', 'm12', 'm23', 'm123', 'unified_persistent', 'unified_output'):
        model.m2_runtime = IncrementalVisionRuntime(model.img_backbone, dict(mode='auto',
            memory_budget_mb=budget, reuse_boundary='camera', cheap_gate=True, m1_service='maintain',
            camera_cost_table='outputs/module2_rearchitecture/camera_costs.json'))
    if mode in ('m3', 'm13', 'm23', 'm123', 'unified_persistent', 'unified_output',
                'prefix', 'batch', 'adaptive', 'state_only', 'state_full', 'compact'):
        config = dict(mode='auto', memory_budget_mb=budget)
        if mode in ('prefix', 'batch', 'adaptive'):
            config.update(mode='exact', offline_experiment=True, borrow_prefix=True,
                common_prefix_seed=mode=='prefix', batched_decode=mode!='prefix',
                adaptive_batch=mode=='adaptive', max_batch_size=3)
        if mode in ('state_only', 'state_full', 'compact'):
            config.update(mode='compact',offline_experiment=True,
                state_context={'state_only':'only','state_full':'augment','compact':'none'}[mode])
        model.m3_runtime = RecordedM3(config, sink)
    model.stream_runtime = UnifiedRuntime(dict(service='persistent_state' if persistent else 'current_output',
        model_version='iter_10548.pth', lora_version='embedded_original_lora_iter_10548',
        memory_budget_mb=budget*2, cost_aware=mode.startswith('unified'), profile_stages=profile,
        trace_path=str(dest/'cross_module_trace.jsonl')))
    model.test_flag = False
    model.pts_bbox_head.reset_memory(); model.map_head.reset_memory()
    model._m2_context = None
    model.m2_diagnostics = {}; model.m3_diagnostics = {}
    gc.collect()


@torch.no_grad()
def run(args):
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    snapshot(dest)
    if args.lifecycle_probe is None:
        args.lifecycle_probe=args.workload=='single' and args.frames>=16
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    import cv2
    cv2.setNumThreads(1)
    write(dest/'status.json', dict(status='loading_model'))
    from tools.worldstream_m2.rearchitecture.integration import load
    from tools.worldstream_m3.experiment import chosen_questions, typed_digest, ground_truth
    from tools.worldstream_m3.refinement import freeze_sample_questions
    from tools.worldstream_m3.qa_analysis import planning_quality
    from tools.worldstream_m3.llm_probe import quality as qa_quality
    from tools.worldstream_m3.offline_reference import resolve_reference
    from tools.worldstream_m2.refinement_common import arrays, numeric_diff
    from tools.worldstream_m2.rearchitecture.common import memory
    from mmcv.parallel import scatter, collate
    model, dataset = load()
    versions = [(n,p._version,p.dtype) for n,p in model.named_parameters()]
    native = inspect.getclosurevars(model.lm_head.generate).nonlocals['generate']
    sink = SimpleNamespace(ids=[], in_m3=False)
    def generate(*a, **kw):
        if args.workload != 'single':
            kw.update(do_sample=False); kw.pop('temperature', None); kw.pop('top_p', None)
        result = native(*a, **kw)
        if args.decode_probe and not hasattr(sink, 'probe_input'):
            sink.probe_input = dict(ids=kw['inputs'].detach(), images=kw['images'].detach())
        if not sink.in_m3:
            sink.ids.append(result.detach().cpu().tolist())
        return result
    model.lm_head.generate = generate
    # M3 receives the identical workload generation contract at its entry too.
    original_m3 = RecordedM3.generate
    def controlled_m3(self, *a, **kw):
        if args.workload != 'single':
            kw.update(do_sample=False); kw.pop('temperature', None); kw.pop('top_p', None)
        return original_m3(self, *a, **kw)
    RecordedM3.generate = controlled_m3
    for transform in dataset.pipeline.transforms:
        if hasattr(transform, 'load_type'):
            transform.load_type = ['planning'] if args.workload == 'single' else ['planning', 'short', 'conv']
    groups = {}
    for i, info in enumerate(dataset.data_infos):
        groups.setdefault(info['scene_token'], []).append(i)
    scenes = []
    for si, (scene, indices) in enumerate(sorted(groups.items())[args.scene_start:args.scene_start+args.scenes]):
        indices = sorted(indices, key=lambda i: dataset.data_infos[i]['timestamp'])
        indices = indices[::args.frame_stride][:args.frames]
        if len(indices) < args.frames:
            raise ValueError('Requested sequence exceeds available unique frames')
        from .identity import validate_sequence
        identity=validate_sequence([dataset.data_infos[i] for i in indices],scene)
        scenes.append(dict(scene=scene, indices=indices, split=(getattr(args,'scene_splits',None) or {}).get(scene,'calibration' if si%2==0 else 'heldout'),
            tokens=identity['sample_tokens'],timestamps_s=identity['timestamps_s']))
    if not scenes:
        raise ValueError('No scenes selected')
    # Freeze native question IDs before any method's results or ground truth.
    frozen = {}
    for scene in scenes:
        for index in scene['indices']:
            seed(index)
            sample = dataset[index]
            questions = sample['img_metas'][0].data['vlm_labels'].data
            ids = sample['input_ids'][0].data
            chosen = chosen_questions(questions, 1 if args.workload=='single' else 5)
            frozen[index] = dict(questions=[questions[i] for i in chosen], ids=[ids[i] for i in chosen])
    write(dest/'questions.json', {str(i): dict(questions=x['questions'], ids=[t.tolist() for t in x['ids']]) for i,x in frozen.items()})
    write(dest/'protocol.json', dict(scenes=scenes, args=vars(args), generation=dict(
        do_sample=args.workload=='single', temperature=.1, top_p=.75, max_new_tokens=320, num_beams=1),
        timing='Preprocess/disk/collate/scatter plus synchronized VLM inference; generated-ID CPU capture and native/runtime I/O included; post-timer bulk tensor audits, digests/snapshots and reference I/O excluded',
        quality='Frozen complete token equality; perception and temporal tensors compared with original dtype; no thresholds relaxed',
        independence='Scene indices 96+ selected from sorted local val scenes, not official hidden test; no global certificate emitted'))
    write(dest/'environment.json', dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        nvidia_smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout))
    captured = {}
    extract_owned='extract_img_feat' in vars(model)
    original_extract=model.extract_img_feat
    @wraps(original_extract)
    def capture_extract(*a,**kw):
        result=original_extract(*a,**kw)
        captured['feature']=result.detach()
        return result
    model.extract_img_feat=capture_extract
    (dest/'references').mkdir()
    modes = args.modes
    for rep in range(args.warmup+args.repeats):
        for si, scene in enumerate(scenes):
            # Establish native reference separately; never treat this extra
            # inference as a new unique frame or include it in speed statistics.
            order = modes[(rep+si)%len(modes):]+modes[:(rep+si)%len(modes)]
            for mode in ['reference']+order:
                configure(model, 'original' if mode=='reference' else mode, sink, dest,
                          args.cache_budget_mb, args.refresh_interval, args.profile_stages)
                for ordinal, index in enumerate(scene['indices']):
                    seed(index)
                    prep_tick = perf_counter()
                    sample = freeze_sample_questions(dataset[index], frozen[index])
                    batch = scatter(collate([sample], samples_per_gpu=1), [torch.cuda.current_device()])[0]
                    torch.cuda.synchronize()
                    prep_ms = (perf_counter()-prep_tick)*1000
                    seed(index+10000)
                    sink.ids = []
                    if RUN_HOOKS.get('before'):RUN_HOOKS['before'](model,mode,rep,index,sink,dest)
                    # Fresh path: existing language files can never skip generation.
                    model.save_path = str(dest/'language'/f'{rep}_{si}_{mode}')+'/'
                    model.stream_runtime.input_preparation_wall_ms = prep_ms
                    torch.cuda.reset_peak_memory_stats()
                    tick = perf_counter()
                    output = model(return_loss=False, rescale=True, **batch)
                    torch.cuda.synchronize()
                    wall = (perf_counter()-tick)*1000+prep_ms
                    record = model.stream_runtime.last_record
                    record.e2e_wall_ms = wall  # includes outer native forward_test overhead
                    record.memory.update(gpu_peak_allocated=torch.cuda.max_memory_allocated(),
                        gpu_peak_reserved=torch.cuda.max_memory_reserved(),
                        cpu_rss_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                    current = dict(output=arrays(output), feature=arrays(captured.pop('feature')),
                        memory=memory(model), ids=sink.ids)
                    reference_path = dest/'references'/f'{rep}_{index}.pkl'
                    if mode=='reference':
                        with reference_path.open('wb') as handle:
                            pickle.dump(current, handle)
                        continue
                    with reference_path.open('rb') as handle:
                        reference = pickle.load(handle)
                    equality = sink.ids==reference['ids']
                    diff = {key: numeric_diff(current[key], reference[key]) for key in ('output','feature','memory')}
                    # numeric_diff compares numerical leaves; output text is
                    # checked independently through complete generated IDs.
                    record.quality = dict(exact_token_equality=equality,
                        token_count_matches=len(sink.ids)==len(frozen[index]['questions']),
                        numerical=diff, exact_perception_memory_digest=all(
                            typed_digest(current[k])==typed_digest(reference[k]) for k in ('feature','memory')))
                    tasks = []
                    for question, answer in zip(frozen[index]['questions'], output[0]['text_out']):
                        text = ' '.join(answer['A'])
                        gt, origin = resolve_reference(dataset.data_infos[index]['token'], question,
                            ground_truth(dataset.data_infos[index]['token']))
                        tasks.append(dict(question=question, answer=text, ground_truth=gt, ground_truth_source=origin,
                            text_overlap=qa_quality(text,gt) if gt is not None else None,
                            planning=planning_quality(text,dataset.data_infos[index]) if 'planning trajectory' in question.lower() else None))
                    row = dict(mode=mode, workload=args.workload, repeat=rep, warmup=rep<args.warmup,
                        split=scene['split'], ordinal=ordinal, cold=ordinal==0, index=index,
                        frame=record.to_dict(), generated_ids=sink.ids, tasks=tasks)
                    if RUN_HOOKS.get('after'):RUN_HOOKS['after'](model,row,sink,dest)
                    with (dest/'frames.jsonl').open('a') as handle:
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    print(args.workload, mode, rep, si, ordinal, round(wall,2), 'token_equal', equality, flush=True)
                    write(dest/'status.json', dict(status='running', mode=mode, repeat=rep, scene=si, ordinal=ordinal))
                del current
    if args.trace:
        # Profiler is a separate full generation, not a timing-table sample.
        configure(model, 'original', sink, dest, args.cache_budget_mb, args.refresh_interval, True)
        index = scenes[0]['indices'][0]
        model.save_path = str(dest/'profile_language')+'/'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            with torch.autograd.profiler.record_function('stream/input_preparation'):
                seed(index)
                batch = scatter(collate([freeze_sample_questions(dataset[index],frozen[index])],samples_per_gpu=1),[0])[0]
            model(return_loss=False,rescale=True,**batch)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(dest/'pipeline_trace.json'))
    if extract_owned:model.extract_img_feat=original_extract
    else:delattr(model,'extract_img_feat')
    if args.lifecycle_probe:
        from .lifecycle_probe import run as lifecycle_probe
        lifecycle_probe(model,dataset,frozen,scenes[0]['indices'],sink,dest/'lifecycle_probe',
            args,configure,seed,RecordedM3)
    if args.decode_probe:
        from .decode_probe import run as decode_probe
        decode_probe(model, native, sink.probe_input, dest/'decode_probe')
    unchanged = versions == [(n,p._version,p.dtype) for n,p in model.named_parameters()]
    assert unchanged, 'Model parameters changed during inference'
    write(dest/'status.json', dict(status='completed', weights_unchanged=unchanged))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--workload', choices=['single','multi'], default='single')
    parser.add_argument('--scenes', type=int, default=2)
    parser.add_argument('--scene-start', type=int, default=96)
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--cache-budget-mb', type=int, default=512)
    parser.add_argument('--refresh-interval', type=int, default=10)
    parser.add_argument('--modes', nargs='+', default=MODES, choices=MODES+['prefix','batch','adaptive','state_only','state_full','compact'])
    parser.add_argument('--profile-stages', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--decode-probe', action='store_true')
    parser.add_argument('--lifecycle-probe', action=argparse.BooleanOptionalAction, default=None,
        help='Controlled real-cache post-head failure/recovery; enabled by default for long single-question streams')
    args=parser.parse_args()
    preexisting_output=Path(args.output).exists()
    try:
        run(args)
    except Exception as error:
        if not preexisting_output and (Path(args.output)/'source_manifest.json').exists():
            write(Path(args.output)/'status.json',dict(status='failed',error=type(error).__name__+': '+str(error)))
        raise
