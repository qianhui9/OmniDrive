"""External observers. GT is consumed only after the native commit returns."""
import copy
from collections import Counter
from functools import wraps
from time import perf_counter

import torch

from tools.worldstream_m1.identity_metrics import ContinuityMetrics, possible_instances
from tools.worldstream_m1.identity_metrics import strict_labels
from .common import diagnostic_json
from .metrics import ratio


class IdentityScorer:
    def __init__(self, annotations):
        self.annotations = annotations
        self.counts = Counter()
        self.scene = None
        self.previous = {}
        self.continuity = ContinuityMetrics()
        self.finished = Counter()
        self.scene_frames = 0

    def update(self, frame, anchors, commit):
        if self.scene != frame.scene_id:
            self.finished.update(self.continuity.counts)
            self.scene, self.previous = frame.scene_id, {}
            self.continuity = ContinuityMetrics()
            self.scene_frames = 0
        annotations = self.annotations.get(frame.frame_id, [])
        labels = strict_labels(anchors, annotations)
        selected, ids = commit['selected_prediction_indices'], commit['committed_anchor_ids']
        current = {aid: labels[index] for aid, index in zip(ids, selected) if index in labels}
        eligible = set(self.previous.values()) & set(current.values())
        edges = []
        # The original scorer's candidate denominator is current selected
        # predictions on non-initial frames; preserve that protocol explicitly.
        if self.scene_frames:
            self.counts['candidates'] += len(selected)
        self.counts['eligible'] += len(eligible)
        for aid, index in commit['association_pairs']:
            old, new = self.previous.get(aid), labels.get(index)
            known = old is not None and new is not None
            status = 'correct' if known and old == new else 'wrong' if known else 'unknown'
            self.counts[status] += 1
            self.counts['accepted'] += 1
            self.counts['scored'] += int(known)
            edges.append(dict(anchor_id=aid, index=index, previous_gt=old, current_gt=new, status=status))
        self.continuity.update(annotations, {v: k for k, v in current.items()},
                               possible_instances(anchors, annotations))
        self.counts['frames'] += 1
        self.scene_frames += 1
        self.previous = current
        return edges

    def summary(self):
        c = {k: self.counts[k] for k in ('frames', 'candidates', 'eligible', 'accepted',
                                        'scored', 'correct', 'wrong', 'unknown')}
        continuity = {k: self.finished[k]+v for k, v in self.continuity.counts.items()}
        return dict(counts=c, continuity=continuity,
            association_precision=ratio(c['correct'], c['scored']),
            wrong_association_rate=ratio(c['wrong'], c['scored']),
            effective_recall=ratio(c['correct'], c['eligible']),
            association_coverage=ratio(c['accepted'], c['candidates']),
            alignment_precision=None, alignment_recall=None,
            alignment_numerator=0, alignment_denominator=0,
            scope='Offline partial Oracle: unique class/size/1m matching; not standard MOT. GT never enters online decisions.',
            candidate_definition='selected predictions on scene frames after the first',
            alignment_status='NOT_AVAILABLE: no independently scored alignment denominator')


class Observer:
    def __init__(self, model):
        self.model = model
        self.events, self.native_generations, self.commits, self.reuses = [], [], [], []
        self.backbone_cameras = 0
        self.window_attention_batches = Counter()
        self.prefill_generations = []
        self.handles = []
        if hasattr(model.img_backbone, 'register_forward_pre_hook'):
            # Wrapping forward changes the topology identity that M2 verifies.
            # Read-only hooks retain the original executable graph.
            stack = []
            def start_backbone(module, args):
                self.backbone_cameras += len(args[0])
                event = torch.cuda.Event(enable_timing=True); event.record()
                stack.append((event, perf_counter()))
            def end_backbone(module, args, result):
                event, tick = stack.pop()
                end = torch.cuda.Event(enable_timing=True); end.record()
                self.events.append(('eva_native', event, end, (perf_counter()-tick)*1000))
            self.handles += [model.img_backbone.register_forward_pre_hook(start_backbone),
                             model.img_backbone.register_forward_hook(end_backbone)]
            runtime = model.m2_runtime
            if runtime is not None and runtime.config.reuse_boundary == 'window':
                for i in runtime.prefix:
                    attention=model.img_backbone.blocks[i].attn
                    if attention.flash_attn:
                        def observe_windows(module,args,i=i):
                            self.window_attention_batches[i]+=len(args[0])
                        self.handles.append(attention.inner_attn.register_forward_pre_hook(observe_windows))
                if not all(model.img_backbone.blocks[i].attn.flash_attn for i in runtime.prefix):
                    runtime.executor.attention_observer=lambda count:self.window_attention_batches.update({runtime.executing_block:count})
        else:
            self.wrap(model.img_backbone, 'forward', 'eva_native')
        self.wrap(model.pts_bbox_head, 'forward', 'detection')
        self.wrap(model.map_head, 'forward', 'map')
        self.wrap(model.lm_head, 'generate', 'native_generation', generation=True)
        from worldstream.m3.visual_state import base_model
        self.wrap(base_model(model.lm_head), 'forward', 'llm', dynamic=True)
        if model.m3_runtime is not None:
            self.wrap(model.m3_runtime, 'generate', 'm3_auto_total')
        if model.m1_manager is not None:
            # The deployed detector calls precheck_with_observation directly.
            # Wrapping the shared implementation also covers precheck(), once.
            self.wrap(model.m1_manager, 'precheck_with_observation', 'm1_precheck')
            native_commit = model.m1_manager.commit

            @wraps(native_commit)
            def commit(frame, outputs, result):
                value = native_commit(frame, outputs, result)
                # Keep references to immutable observation arrays; copy diagnostic
                # dictionaries after the full request, outside its measured time.
                self.commits.append((frame, outputs.anchors, result))
                return value
            model.m1_manager.commit = commit
            self.wrap(model.m1_manager, 'commit', 'm1_commit')
        if model.m2_runtime is not None:
            # Executor calls also cover partial windows that do not call block.forward.
            # These wrappers are analysis-only and leave the executable topology intact.
            self.wrap(model.m2_runtime.executor, 'execute', 'm2_window_block', window=True)
            self.observe_module(model.img_backbone.patch_embed, 'm2_patch_embedding')
            for i, block in enumerate(model.img_backbone.blocks):
                if i not in model.m2_runtime.prefix:
                    self.observe_module(block, f'm2_suffix_block_{i}')
            original=model.m2_runtime.forward_from_m1
            @wraps(original)
            def windows(*args,_original=original,**kwargs):
                result,diag=_original(*args,**kwargs)
                if diag.get('actual_skipped_windows',0):
                    self.reuses.append((args[0],result[0],copy.deepcopy(diag)))
                return result,diag
            model.m2_runtime.forward_from_m1=windows
            self.wrap(model.m2_runtime,'forward_from_m1','m2_total')
        if model.m3_runtime is not None:
            executor=model.m3_runtime.executor
            original=executor.generate
            @wraps(original)
            def prefill(*args,_original=original,**kwargs):
                start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2));start.record()
                tick=perf_counter();result,diag=_original(*args,**kwargs);wall=(perf_counter()-tick)*1000;end.record()
                output=result.sequences if hasattr(result,'sequences') else result
                self.prefill_generations.append(dict(output=output,host_wall_ms=wall,
                    input_tokens=diag['input_tokens'],generated_tokens=output.shape[-1]-1,
                    start=start,end=end))
                self.events.append(('prefill_executor_generate',start,end,wall))
                return result,diag
            executor.generate=prefill

    def observe_module(self, module, label):
        stack = []
        def start(module, args):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            stack.append((event, perf_counter()))
        def end(module, args, output):
            event, tick = stack.pop()
            stop = torch.cuda.Event(enable_timing=True)
            stop.record()
            self.events.append((label, event, stop, (perf_counter()-tick)*1000))
        self.handles.extend([module.register_forward_pre_hook(start), module.register_forward_hook(end)])

    def wrap(self, obj, attribute, name, dynamic=False, generation=False, window=False):
        original = getattr(obj, attribute)

        @wraps(original)
        def call(*args, **kwargs):
            label = name
            if window:
                label = f'm2_window_block_{self.model.m2_runtime.executing_block}'
            if dynamic:
                label = 'prefill' if kwargs.get('inputs_embeds') is not None or kwargs.get('past_key_values') is None else 'decode'
            if name == 'eva_native':
                self.backbone_cameras += len(args[0])
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            tick = perf_counter()
            result = original(*args, **kwargs)
            elapsed = (perf_counter()-tick)*1000
            end.record()
            self.events.append((label, start, end, elapsed))
            if generation:
                ids = kwargs.get('inputs')
                carriers = kwargs.get('images')
                prompt_tokens = ids.numel() + int((ids == -200).sum()) * (carriers.shape[1]-1)
                self.native_generations.append(dict(output=result, host_wall_ms=elapsed,
                    input_tokens=prompt_tokens, output_sequence_tokens=result.shape[-1],
                    # The deployed Llava wrapper passes inputs_embeds, so HF
                    # starts returned sequences with one synthetic BOS token.
                    generated_tokens=result.shape[-1]-1, start=start, end=end))
            return result
        setattr(obj, attribute, call)

    def clear(self):
        self.events.clear()
        self.native_generations.clear()
        self.prefill_generations.clear()
        self.commits.clear()
        self.reuses.clear()
        self.backbone_cameras = 0
        self.window_attention_batches.clear()

    def collect(self):
        stages = {}
        for label, start, end, wall in self.events:
            s = stages.setdefault(label, dict(calls=0, gpu_span_ms=0., host_wall_ms=0.))
            s['calls'] += 1
            s['gpu_span_ms'] += start.elapsed_time(end)
            s['host_wall_ms'] += wall
        return stages


def m1_record(model, observation):
    frame, anchors, precheck = observation
    manager = model.m1_manager
    return frame, anchors, diagnostic_json(dict(precheck=precheck.to_dict(),
        commit=copy.deepcopy(manager.last_commit_diagnostics), committed_state=manager.export_state_audit()))
