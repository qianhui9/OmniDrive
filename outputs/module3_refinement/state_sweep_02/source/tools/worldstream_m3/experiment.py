"""Real weights, real nuScenes, unchanged question/GT, unique language paths.

Run in OmniDrive: python -m tools.worldstream_m3.experiment --output outputs/module3/run01
No output-file skip is used, even for the original baseline.
"""
import argparse
import copy
import gc
from dataclasses import replace
from functools import wraps
import hashlib
import inspect
import json
import random
import subprocess
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch

from worldstream.m3 import M3Runtime, CarrierLayout, expand_input
from worldstream.m3.visual_state import base_model
from worldstream.m3.prefix_matcher import exact_lcp
from worldstream.m3.task_conditioner import task_type


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def stats(xs):
    return dict(p50=float(np.percentile(xs, 50)), p95=float(np.percentile(xs, 95)), count=len(xs)) if xs else None


class Timer:
    def __init__(self):
        self.events = []

    def wrap(self, module, name, dynamic=False):
        original = module.forward
        @wraps(original)
        def forward(*a, **kw):
            label = name
            if dynamic:
                x = kw.get('inputs_embeds')
                if x is None: x = kw.get('input_ids')
                if x is None and a: x = a[0]
                label = 'prefill' if kw.get('inputs_embeds') is not None or kw.get('past_key_values') is None else 'decode'
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); tick = perf_counter()
            with torch.autograd.profiler.record_function('m3_probe_' + label):
                result = original(*a, **kw)
            end.record()
            self.events.append((label, start, end, (perf_counter()-tick)*1000))
            return result
        module.forward = forward

    def collect(self):
        output = {}
        for label, a, b, wall in self.events:
            r = output.setdefault(label, dict(gpu_ms=0., cpu_enqueue_ms=0., calls=0))
            r['gpu_ms'] += a.elapsed_time(b); r['cpu_enqueue_ms'] += wall; r['calls'] += 1
        self.events.clear()
        return output

    def wrap_method(self, obj, attribute, name):
        original = getattr(obj, attribute)
        @wraps(original)
        def method(*a, **kw):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); tick = perf_counter()
            with torch.autograd.profiler.record_function('m3_probe_' + name):
                output = original(*a, **kw)
            end.record()
            self.events.append((name, start, end, (perf_counter()-tick)*1000))
            return output
        setattr(obj, attribute, method)


class ExperimentRuntime(M3Runtime):
    """A class override avoids an instance -> closure -> bound-method GPU cycle."""
    def __init__(self, config, sampling, recorded):
        super().__init__(config)
        self.sampling = sampling
        self.recorded = recorded

    def generate(self,*a,**kw):
        if not self.sampling:
            kw.update(do_sample=False);kw.pop('temperature',None);kw.pop('top_p',None)
        answers,diagnostics=super().generate(*a,**kw)
        self.recorded.extend(x.detach().cpu().tolist() for x in answers)
        return answers,diagnostics


def ground_truth(token):
    result = {}
    for folder in ('conv/val', 'vqa/val'):
        path = Path('data/nuscenes') / folder / (token+'.json')
        if path.exists():
            for pair in json.loads(path.read_text()):
                result[pair['question']] = pair['answer']
    return result


def typed_digest(value):
    if hasattr(value,'tensor'):value=value.tensor
    if isinstance(value,torch.Tensor):
        tensor=value.detach().cpu().contiguous()
        return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value,np.ndarray):
        return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,dict):return {k:typed_digest(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [typed_digest(v) for v in value]
    return value


def chosen_questions(questions, limit):
    # Deterministic task coverage from question text only, never from ground truth.
    selected = [0]
    for task in ('object_relation', 'map', 'global_safety', 'unknown'):
        found = next((i for i, q in enumerate(questions) if i not in selected and task_type(q) == task), None)
        if found is not None: selected.append(found)
    selected += [i for i in range(len(questions)) if i not in selected]
    return sorted(selected[:limit]) if limit else list(range(len(questions)))


@torch.no_grad()
def run(args):
    dest = Path(args.output)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Output must be fresh; old answers cannot skip inference')
    dest.mkdir(parents=True, exist_ok=True)
    files = list(Path('worldstream/m3').glob('*.py')) + list(Path('tools/worldstream_m3').glob('*.py'))
    files += [Path('projects/mmdet3d_plugin/models/detectors/petr3d.py')]
    source = {}
    for path in files:
        target = dest/'source'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        source[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(dest/'source_manifest.json', source)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    import cv2
    cv2.setNumThreads(1)
    import transformers, peft
    write(dest/'environment.json', dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
        gpu=torch.cuda.get_device_name(0), argv=vars(args),timer_scope='whole_causal_lm_forward'))
    from tools.worldstream_m2.rearchitecture.integration import load
    model, dataset = load()
    print('MODEL_LOADED', flush=True)
    for transform in dataset.pipeline.transforms:
        if hasattr(transform, 'load_type'):
            transform.load_type = ['planning'] if args.deployment_single else ['planning', 'short', 'conv']
    protocol = json.load(open('outputs/module2_rearchitecture/protocol.json'))['real_scenes'][:args.scenes]
    samples = {i: dataset[i] for scene in protocol for i in scene['indices'][:args.frames]}
    write(dest/'protocol.json', dict(scenes=protocol, warmup=args.warmup, repeats=args.repeats,
        question_limit=args.questions, selection='deterministic coverage from question text only',
        quality=dict(greedy_token_ids_exact=True, compact_online=False, frozen_before_runs=True,
                     gt_metrics='token F1, exact normalized answer, numeric list error; diagnostics only, not safety certification'),
        modes=args.modes, generation=dict(do_sample=args.sampling, max_new_tokens=320),
        caveat='Original deployment uses sampling; greedy used for controlled equivalence. Sampling separately tested.'))
    timer = Timer(); timer.wrap(base_model(model.lm_head), 'llm', dynamic=True)
    timer.wrap_method(base_model(model.lm_head), 'prepare_inputs_labels_for_multimodal', 'multimodal_preparation')
    timer.wrap(model.pts_bbox_head.output_projection, 'object_projection')
    timer.wrap(model.map_head.output_projection, 'map_projection')
    timer.wrap(model.img_backbone, 'eva'); timer.wrap(model.pts_bbox_head, 'detection'); timer.wrap(model.map_head, 'map')
    # The older M2 loader wraps generate in greedy(). Unwrap it once, so the
    # separate sampling correctness probe really compares native sampling.
    native_generate = inspect.getclosurevars(model.lm_head.generate).nonlocals['generate']
    generated_ids = []
    def record_generate(*a, **kw):
        if not args.sampling:
            kw.update(do_sample=False);kw.pop('temperature',None);kw.pop('top_p',None)
        output = native_generate(*a, **kw)
        generated_ids.append(output.detach().cpu().tolist())
        return output
    model.lm_head.generate = record_generate
    pts_original = model.simple_test_pts
    audit_rows, snapshots, current_info = [], {}, {}
    audit_enabled = [True]

    def pts_entry(img_metas, **data):
        from mmcv.parallel import DataContainer
        meta = img_metas[0]
        all_questions = list(meta['vlm_labels'].data)
        all_ids = list(data['input_ids'][0])
        chosen = chosen_questions(all_questions, args.questions)
        meta['vlm_labels'] = DataContainer([all_questions[i] for i in chosen], cpu_only=True)
        data['input_ids'][0] = [all_ids[i] for i in chosen]
        current_info.clear(); current_info.update(token=meta['sample_idx'], scene=meta['scene_token'],
            questions=[all_questions[i] for i in chosen], all_questions=all_questions, selected_indices=chosen,
            ground_truth=ground_truth(meta['sample_idx']), all_ids=all_ids)
        return pts_original(img_metas, **data)
    model.simple_test_pts = pts_entry
    # Audit at actual LLM entry, before any modifications to the carrier.
    original_record = model.lm_head.generate
    def audited_generate(*a, **kw):
        token = current_info['token']
        if audit_enabled[0] and token not in snapshots:
            carriers = kw['images']
            all_x = [expand_input(model.lm_head, ids, carriers) for ids in current_info['all_ids']]
            pairwise = [exact_lcp(x, y) for i, x in enumerate(all_x) for y in all_x[i+1:]]
            previous = next((v for v in reversed(list(snapshots.values())) if v['scene'] == current_info['scene']), None)
            cross = None
            if previous:
                prev = expand_input(model.lm_head, previous['ids'][0].to(carriers.device), previous['carriers'].to(carriers.device))
                cross = exact_lcp(prev, all_x[0])
            audit_rows.append(dict(token=token, scene=current_info['scene'], actual_questions=len(all_x),
                selected_indices=current_info['selected_indices'], questions=current_info['all_questions'],
                text_tokens=[int((ids != -200).sum()) for ids in current_info['all_ids']],
                image_positions=[(ids == -200).nonzero().flatten().tolist() for ids in current_info['all_ids']],
                expanded_tokens=[x.length for x in all_x], visual_tokens=int(all_x[0].visual_mask.sum()),
                pairwise_exact_lcp=pairwise, cross_frame_first_question_lcp=cross))
            snapshot = dict(scene=current_info['scene'], carriers=carriers.detach().cpu(),
                ids=[x.detach().cpu() for x in current_info['all_ids']], questions=current_info['all_questions'],
                ground_truth=current_info['ground_truth'], selected_indices=current_info['selected_indices'])
            snapshots[token] = snapshot
            torch.save(snapshot, dest/'carriers'/f'{token}.pt')
            write(dest/'input_audit.json', audit_rows)
        return original_record(*a, **kw)
    model.lm_head.generate = audited_generate
    (dest/'carriers').mkdir()
    from mmcv.parallel import scatter, collate
    rows = []
    from worldstream.m1 import M1Config, WorldStateManager
    from worldstream.m2 import IncrementalVisionRuntime
    for rep in range(args.warmup + args.repeats):
        for si, scene in enumerate(protocol):
            modes = list(args.modes)
            offset = (rep + si) % len(modes); modes = modes[offset:] + modes[:offset]
            for mode in modes:
                model.test_flag = False
                if model.m3_runtime is not None:
                    model.m3_runtime.reset()
                model.m1_manager = None; model.m2_runtime = None; model.m3_runtime = None
                gc.collect()
                if mode in ('m12', 'm123'):
                    model.m1_manager = WorldStateManager(replace(M1Config.load('configs/worldstream_m1_coverage.yaml'),
                        log_path=str(dest/'m1_state.jsonl')))
                    model.m2_runtime = IncrementalVisionRuntime(model.img_backbone, dict(mode='auto', reuse_boundary='camera',
                        cheap_gate=True, m1_service='maintain', memory_budget_mb=512,
                        camera_cost_table='outputs/module2_rearchitecture/camera_costs.json'))
                if mode in ('compact', 'exact', 'combined', 'm123'):
                    runtime = ExperimentRuntime(dict(mode='exact' if mode in ('exact', 'm123') else mode,
                        selector='task', keep_ratio=.5, offline_experiment=True, cross_frame=False),
                        args.sampling,generated_ids)
                    model.m3_runtime = runtime
                for fi, index in enumerate(scene['indices'][:args.frames]):
                    # Prior run paths are never reused, including warmup and trace.
                    model.save_path = str(dest/f'answers/r{rep}_{si}_{mode}_{fi}')+'/'
                    batch = scatter(collate([copy.deepcopy(samples[index])], samples_per_gpu=1), [torch.cuda.current_device()])[0]
                    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
                    timer.events.clear(); generated_ids.clear()
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    tick = perf_counter()
                    output = model(return_loss=False, rescale=True, **batch)
                    torch.cuda.synchronize(); wall = (perf_counter()-tick)*1000
                    components = timer.collect()
                    row = dict(repeat=rep, warmup=rep < args.warmup, mode=mode, scene=scene['scene'], category=scene['category'],
                        index=index, token=current_info['token'], cold_sequence=fi == 0, wall_ms=wall,
                        components=components, questions=current_info['questions'], ground_truth=current_info['ground_truth'],
                        answers=output[0]['text_out'], generated_ids=copy.deepcopy(generated_ids),
                        diagnostics=copy.deepcopy(getattr(model, 'm3_diagnostics', {})) if model.m3_runtime else {},
                        m1=copy.deepcopy(model.m1_manager.last_commit_diagnostics) if model.m1_manager else {},
                        m2=copy.deepcopy(getattr(model, 'm2_diagnostics', {})) if model.m2_runtime else {},
                        peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
                    if args.perception:
                        memory = {name:{k:v for k,v in vars(head).items() if k.startswith('memory_') or k=='sample_time'}
                            for name,head in [('det',model.pts_bbox_head),('map',model.map_head)]}
                        row['perception_digest'] = typed_digest(dict(boxes=output[0]['pts_bbox'],lanes=output[0]['lane_results'],memory=memory))
                        target=dest/f'perception/r{rep}_{si}_{mode}_{fi}.pt';target.parent.mkdir(exist_ok=True)
                        # CPU owned values preserve original dtype; timing already stopped.
                        def own(v):
                            if hasattr(v,'tensor'):return v.tensor.detach().cpu().clone()
                            if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
                            if isinstance(v,dict):return {k:own(z) for k,z in v.items()}
                            if isinstance(v,(list,tuple)):return [own(z) for z in v]
                            return copy.deepcopy(v)
                        torch.save(own(dict(boxes=output[0]['pts_bbox'],lanes=output[0]['lane_results'],memory=memory)),target)
                    # Only JSON primitives from diagnostic objects are retained by default conversion.
                    row['m2'] = {k: v for k, v in row['m2'].items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                    rows.append(row)
                    with (dest/'frames.jsonl').open('a') as handle:
                        handle.write(json.dumps(row, default=str)+'\n')
                    print('FRAME', rep, mode, index, round(wall, 2), flush=True)
                audit_enabled[0] = False if len(snapshots) >= len(samples) else True
    write(dest/'input_audit.json', audit_rows)
    print('E2E_COMPLETE', len(rows), flush=True)
    model.m1_manager = None; model.m2_runtime = None; model.m3_runtime = None
    model.lm_head.generate = native_generate
    # Paired single-LLM comparison on real carriers, no temporal head replay.
    from .llm_probe import run as llm_probe
    if not args.skip_probes:
        llm_probe(model.lm_head, model.tokenizer, snapshots, dest, args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='outputs/module3/run01')
    p.add_argument('--scenes', type=int, default=6)
    p.add_argument('--frames', type=int, default=2)
    p.add_argument('--questions', type=int, default=5)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--skip-probes',action='store_true')
    p.add_argument('--deployment-single',action='store_true')
    p.add_argument('--sampling',action='store_true')
    p.add_argument('--perception',action='store_true')
    p.add_argument('--modes', nargs='+', default=['baseline', 'm12', 'compact', 'exact', 'combined', 'm123'])
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
