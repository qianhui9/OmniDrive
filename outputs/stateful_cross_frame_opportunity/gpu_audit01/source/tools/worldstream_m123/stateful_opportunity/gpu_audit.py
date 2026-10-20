"""Native continuous six-camera inference with temporary observation hooks.

No cached result is substituted into the native model. Additional forwards are
explicit P0 cost/numerical diagnostics, never counted as avoided computation.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from functools import wraps
import json
import os
from pathlib import Path
import random
import subprocess
from time import perf_counter, sleep

import numpy as np
import torch
import torch.nn.functional as F

from .common import (AFFINITY, SAMPLING, Context, history_rejection, manifest,
                     protected_files, replace_method, sha256, stats, write)


def timed(call, warmup=2, repeats=7):
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state()
    for _ in range(warmup):
        result = call()
        del result
    walls, gpus = [], []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        tick = perf_counter()
        a.record()
        result = call()
        b.record()
        torch.cuda.synchronize()
        walls.append((perf_counter() - tick) * 1000)
        gpus.append(a.elapsed_time(b))
        del result
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(gpu_rng, torch.cuda.get_rng_state()):
        raise ValueError('Diagnostic changed native RNG stream')
    return dict(wall_ms=stats(walls), gpu_interval_ms=stats(gpus),
                wall_samples_ms=walls, gpu_samples_ms=gpus,
                warmup=warmup, repeats=repeats, rng_unchanged=True)


def wait_idle():
    """Observe and wait; never signal another GPU process."""
    while True:
        value = subprocess.check_output(['nvidia-smi', '--id=0',
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        memory, utilization = [int(v.strip()) for v in value.strip().split(',')]
        if memory < 500 and utilization < 10:
            return
        print('WAIT_GPU', memory, 'MiB', utilization, '%', flush=True)
        sleep(30)


class TemporalObserver:
    """Track known FIFO source indices, excluding uninitialised memory slots."""
    def __init__(self, model, current, stack):
        self.current = current
        self.previous = {}
        self.rows = []
        self.probes = []
        self.measured = set()
        self.pending = {}
        weights = {}
        for branch, head in [('detection', model.pts_bbox_head), ('map', model.map_head)]:
            for i, layer in enumerate(head.transformer.query_decoder._layers):
                # The installed decoder has self-attention then cross-attention.
                attention = layer.transformer_layers[0].attn
                if not isinstance(attention, torch.nn.MultiheadAttention) or not attention._qkv_same_embed_dim:
                    raise ValueError('Unsupported attention implementation; no dependency inference')
                key = (branch, i)
                h = attention.register_forward_pre_hook(self.before(key, head))
                stack.callback(h.remove)
                weights[id(attention.in_proj_weight)] = (key, attention)
        native_projection = F._in_projection_packed
        def projection_observer(q, k, v, w, b=None):
            result = native_projection(q, k, v, w, b)
            if id(w) in weights:
                key, attention = weights[id(w)]
                self.projected(key, attention)(result[2].transpose(0, 1))
            return result
        stack.enter_context(replace_method(F, '_in_projection_packed', projection_observer))

    def before(self, key, head):
        def hook(module, args):
            q, k, v = args[:3]
            old = self.previous.get(key)
            nq = q.shape[1]
            prop = getattr(head, 'num_propagated', 0) if key[0] == 'detection' else 0
            real = max(0, min(self.current['ordinal'] * head.topk_proposals, head.memory_len) - prop)
            start = head.topk_proposals
            end = min(real, start + (old['real'] if old else 0))
            lifecycle = history_rejection(old['context'] if old else None,
                                          self.current['context'], 1.)
            n = max(0, end - start) if lifecycle is None else 0
            tick = perf_counter()
            same_v = same_k = 0
            if n:
                old_slice = slice(old['nq'], old['nq'] + n)
                cur_slice = slice(nq + start, nq + end)
                same_v = int((v[:, cur_slice] == old['v'][:, old_slice]).all(-1).sum())
                same_k = int((k[:, cur_slice] == old['k'][:, old_slice]).all(-1).sum())
            row = dict(branch=key[0], layer=key[1], frame=self.current['frame'],
                       source_frame=old['context'].frame if old else None,
                       scene=self.current['scene'], ordinal=self.current['ordinal'],
                       real_historical_memory_rows=real, retained_source_rows=n,
                       identical_value_input_rows=same_v, identical_key_input_rows=same_k,
                       input_shape=list(v.shape), dtype=str(v.dtype),
                       value_stride=list(v.stride()), state_epoch=self.current['context'].epoch,
                       execution_signature=self.current['context'].signature,
                       weight_version=module.in_proj_weight._version,
                       raw_memory_origin_frames=dict(Counter(
                           self.current.get('scene_tokens', [])[self.current['ordinal']-1-(prop+j)//head.topk_proposals]
                           for j in range(start, start+n))) if self.current.get('scene_tokens') else {},
                       lifecycle_rejection=lifecycle, verification_wall_ms=(perf_counter()-tick)*1000,
                       native_temporal_memory=True, new_execution_skips=0)
            self.rows.append(row)
            self.pending[key] = (q, k, v, nq, real, start, end, old, row)
        return hook

    def projected(self, key, attention):
        def hook(projected_v):
            _, k, v, nq, real, start, end, old, row = self.pending.pop(key)
            n = row['retained_source_rows']
            measurement = (self.current['scene'], key)
            if n and row['identical_value_input_rows'] == n and measurement not in self.measured:
                self.measured.add(measurement)
                w = attention.in_proj_weight.chunk(3)[2]
                bias = attention.in_proj_bias.chunk(3)[2] if attention.in_proj_bias is not None else None
                versions = (attention.in_proj_weight._version,
                            attention.in_proj_bias._version if attention.in_proj_bias is not None else None)
                # Source value and its native projection are retained from t-1.
                historical = old['projected_v'][:, old['nq']:old['nq']+n]
                def candidate():
                    if versions != (attention.in_proj_weight._version,
                                    attention.in_proj_bias._version if attention.in_proj_bias is not None else None):
                        raise ValueError('Changed weights')
                    if not torch.equal(v[:, nq+start:nq+end], old['v'][:, old['nq']:old['nq']+n]):
                        return F.linear(v, w, bias)
                    fresh = F.linear(torch.cat((v[:, :nq+start], v[:, nq+end:]), 1), w, bias)
                    return torch.cat((fresh[:, :nq+start], historical, fresh[:, nq+start:]), 1)
                dense = F.linear(v, w, bias)
                proposed = candidate()
                error = float((dense - proposed).abs().max())
                full = timed(lambda: F.linear(v, w, bias))
                partial = timed(candidate)
                self.probes.append(dict(branch=key[0], layer=key[1], scene=self.current['scene'],
                    frame=self.current['frame'], source_frame=old['context'].frame,
                    reused_value_rows=n, full_rows=v.shape[1], full=full, candidate=partial,
                    bitwise_equal=bool(torch.equal(dense, proposed)), max_abs=error,
                    measured_net_wall_ms=full['wall_ms']['p50']-partial['wall_ms']['p50'],
                    measured_gpu_interval_delta_ms=full['gpu_interval_ms']['p50']-partial['gpu_interval_ms']['p50'],
                    dependency='V_linear_only: exact retained native value + layer weights/bias; not K/Q/attention/head',
                    maintenance='Read-only native input/projection references retained. Source equality, weight versions, fresh-row gather, GEMM, output concatenation included; M1 and broader certification not included.',
                    actual_runtime_skips=0, diagnostic_only=True))
                del dense, proposed
            # Native concatenations/projections are out of place. Retaining them
            # avoids inventing a second state store or mutating model memory.
            self.previous[key] = dict(v=v.detach(), k=k.detach(), nq=nq, real=real,
                                      projected_v=projected_v.detach(), context=self.current['context'])
        return hook


def load_native():
    # Deliberately do not call old integration.load(): it forces greedy decoding.
    from mmcv import Config
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin  # noqa: F401
    cfg = Config.fromfile('projects/configs/OmniDrive/worldstream_m1_coverage_observe.py')
    cfg.model.m1_enable = False
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):
        wrap_fp16_model(model)
    load_checkpoint(model, 'ckpts/iter_10548.pth', map_location='cpu')
    return model.cuda().eval(), dataset


def state_snapshot(manager, frame):
    from worldstream.m3.world_state import freeze_verified_state
    from worldstream.m1.mask_codec import decode_mask
    state = freeze_verified_state(manager, frame)
    result = manager.last_result
    bundle = result.computation_state
    cameras = {}
    if bundle:
        for cid, contract in bundle.refresh_contract.items():
            cameras[cid] = {name: int(decode_mask(contract[name]).sum()) for name in
                            ('reuse_candidate_mask', 'dirty_hint_mask', 'unknown_mask')}
    anchors = []
    for aid, a in manager.store.anchors.items():
        certification = a.physical_certification
        anchors.append(dict(anchor_id=aid, source_frame=a.source_frame_id,
            source_timestamp=a.source_timestamp, source_epoch=a.source_state_epoch,
            source_signature=a.source_execution_signature, source_cameras=sorted(a.observation_support),
            evidence_origin=a.evidence_origin, observation_valid=aid in result.valid_anchor_ids,
            physical_certified=bool(certification and certification.certified)))
    features = [dict(feature_id=fid, source_frame=f.source_frame_id,
        source_timestamp=f.computation_timestamp, source_epoch=f.source_state_epoch,
        version=f.computation_version, signature=f.execution_signature,
        dependency_scope=f.dependency_scope, cameras=f.source_camera_ids)
        for fid, f in manager.store.features.items()]
    return state, dict(scene=frame.scene_id, frame=frame.frame_id, timestamp=frame.timestamp,
        epoch=manager.store.generation, signature=frame.execution_signature,
        camera_timestamps={cid: c.timestamp for cid, c in frame.cameras.items()},
        anchors=anchors, features=features, camera_pixel_census=cameras,
        verified_state=asdict(state), source_records=bundle.source_records if bundle else [],
        observation_valid_anchors=len(result.valid_anchor_ids),
        physically_certified_anchors=len(result.physically_certified_anchor_ids),
        transformer_equivalence_certified=False, task_input_sufficiency_certified=False,
        full_refresh=result.full_refresh_hint)


@torch.no_grad()
def run(args):
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    # A partial run always stays explicitly incomplete, including OOM.
    write(dest/'status.json', dict(status='running', actual_skips=0))
    before = manifest(protected_files())
    write(dest/'protected_before.json', before)
    tool_paths = sorted(Path('tools/worldstream_m123/stateful_opportunity').glob('*.py'))
    source = manifest(tool_paths)
    write(dest/'source_manifest.json', source)
    for path in tool_paths:
        target = dest/'source'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    torch.set_num_threads(1)
    import cv2
    import transformers
    cv2.setNumThreads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if {4, 5} & os.sched_getaffinity(0):
        raise ValueError('Faulty CPUs 4/5 must be excluded by taskset')
    wait_idle()
    torch.cuda.manual_seed_all(args.seed)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,uuid', '--format=csv,noheader'], text=True)
    print('HASH_CHECKPOINT_AND_RECORD_ENVIRONMENT', flush=True)
    write(dest/'environment.json', dict(torch=torch.__version__, transformers=transformers.__version__,
        python=os.sys.executable, cpu_affinity=sorted(os.sched_getaffinity(0)), gpu=gpu,
        cuda=torch.version.cuda, seed=args.seed, argv=vars(args), sampling=SAMPLING,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        checkpoint=dict(path='ckpts/iter_10548.pth', sha256=sha256('ckpts/iter_10548.pth')),
        tokenizer=manifest(Path('ckpts/pretrain_qformer').glob('token*')),
        backend=dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                     tf32_cudnn=torch.backends.cudnn.allow_tf32)))
    from worldstream.m1 import M1Config, WorldStateManager
    from worldstream.m3.visual_state import base_model, expand_input, execution_signature
    from worldstream.m3.prefix_kv_cache import PrefixKVCache, clone_kv
    from worldstream.m3.world_state import query_world_state
    from worldstream.m3.task_conditioner import task_type
    from mmcv.parallel import collate, scatter
    print('LOAD_ORIGINAL_MODEL', flush=True)
    model, dataset = load_native()
    print('MODEL_LOADED_NATIVE_SAMPLING', flush=True)
    parameters = [(n, p.data_ptr(), p._version) for n, p in model.named_parameters()]
    specs = json.loads(Path('outputs/module1_final_validation_refinement/replay/protocol.json').read_text())['scenes'][:args.scenes]
    index = {v['token']: i for i, v in enumerate(dataset.data_infos)}
    groups = []
    for spec in specs:
        tokens = spec['samples'][:args.frames]
        if len(tokens) != args.frames:
            raise ValueError('Insufficient frozen continuous frames')
        infos = [dataset.data_infos[index[t]] for t in tokens]
        if any(v['scene_token'] != spec['scene_id'] for v in infos):
            raise ValueError('Scene provenance mismatch')
        ordered = sorted((v for v in dataset.data_infos if v['scene_token'] == spec['scene_id']), key=lambda v: v['timestamp'])
        first = next(i for i, v in enumerate(ordered) if v['token'] == tokens[0])
        if [v['token'] for v in ordered[first:first+len(tokens)]] != tokens:
            raise ValueError('Population is not natural consecutive frames')
        groups.append(dict(scene=spec['scene_id'], tokens=tokens,
                           timestamps=[v['timestamp']/1e6 for v in infos]))
    write(dest/'protocol.json', dict(groups=groups, population='Exploratory', seed=args.seed,
        native_load_types=['planning'], model_config='projects/configs/OmniDrive/worldstream_m1_coverage_observe.py',
        config_sha256=sha256('projects/configs/OmniDrive/worldstream_m1_coverage_observe.py'),
        dataset_manifest=dict(annotation_file=dataset.ann_file, sha256=sha256(dataset.ann_file)),
        sampling=SAMPLING, execution='native full + frozen M1, M2/M3 disabled; diagnostic-only probes',
        timing='Instrumented audit wall includes observer work. Not a P3 performance run.',
        added_forwards='P0 linear/prefill microprobes only; excluded from savings',
        thresholds='Frozen M1/M2/M3 quality and numerical contracts unchanged'))
    frames, questions, states, kv_probes = [], [], [], []
    current = {}
    base = base_model(model.lm_head)
    cache = PrefixKVCache(512*1024**2)
    previous_context = None
    frozen_state = None
    active_expanded = None
    prefix = 0
    in_native = False
    original_generate = model.lm_head.generate
    original_forward = base.forward
    captured_logits = None
    native_prefill_calls = native_decode_calls = 0
    with ExitStack() as stack:
        temporal = TemporalObserver(model, current, stack)
        @wraps(original_forward)
        def forward_observer(*a, **kw):
            nonlocal captured_logits, native_prefill_calls, native_decode_calls
            output = original_forward(*a, **kw)
            if in_native:
                if kw.get('inputs_embeds') is not None and kw['inputs_embeds'].shape[1] > 1:
                    native_prefill_calls += 1
                    captured_logits = output.logits[:, -1:].detach().clone()
                    if prefix and current['ordinal'] == 1 and cache.entry is not None:
                        old = cache.entry
                        inherited = clone_kv(old.kv, prefix)
                        x = active_expanded
                        def full():
                            return original_forward(inputs_embeds=x.embeddings, position_ids=x.position_ids,
                                attention_mask=x.attention_mask, use_cache=True, return_dict=True)
                        def partial():
                            return original_forward(inputs_embeds=x.embeddings[:, prefix:],
                                position_ids=x.position_ids[:, prefix:], attention_mask=x.attention_mask,
                                past_key_values=clone_kv(inherited, prefix), use_cache=True, return_dict=True)
                        proposed = partial()
                        full_logits = output.logits[:, -1:]
                        same = torch.equal(full_logits, proposed.logits[:, -1:])
                        delta = float((full_logits-proposed.logits[:, -1:]).abs().max())
                        ptime, ftime = timed(partial, repeats=3), timed(full, repeats=3)
                        kv_probes.append(dict(scene=current['scene'], source_frame=old.expanded.source[1],
                            frame=current['frame'], exact_input_prefix_tokens=prefix,
                            input_tokens=x.length, full=ftime, suffix=ptime,
                            last_logits_bitwise_equal=same, last_logits_max_abs=delta,
                            distribution_equivalence_certified=False, generated_with_reuse=False,
                            forward_calls_avoided=0, prefill_token_work_potentially_avoided=prefix,
                            decode_calls_avoided=0,
                            optimistic_prefill_wall_delta_ms=ftime['wall_ms']['p50']-ptime['wall_ms']['p50'],
                            optimistic_prefill_gpu_delta_ms=ftime['gpu_interval_ms']['p50']-ptime['gpu_interval_ms']['p50'],
                            excludes='signature, exact match, full cache publication, expansion; not E2E',
                            diagnostic_only=True))
                        del proposed, inherited
                    tick = perf_counter()
                    cache.put(active_expanded, output.past_key_values, output.logits)
                    torch.cuda.synchronize()
                    current.setdefault('cache_capture_wall_ms', []).append((perf_counter()-tick)*1000)
                else:
                    native_decode_calls += 1
            return output
        stack.enter_context(replace_method(base, 'forward', forward_observer))
        @wraps(original_generate)
        def generate_observer(*a, **kw):
            nonlocal active_expanded, prefix, in_native, previous_context
            for key, value in SAMPLING.items():
                if kw.get(key) != value:
                    raise ValueError('Native sampling contract changed: '+key)
            ids = kw['inputs']
            tick = perf_counter()
            sig = execution_signature(model.lm_head, 'iter_10548.pth')
            signature_ms = (perf_counter()-tick)*1000
            ctx = current['context']
            rejection = history_rejection(previous_context, ctx, 1.)
            if rejection:
                cache.reset()
            tick = perf_counter()
            active_expanded = expand_input(model.lm_head, ids, kw['images'],
                scope=(ctx.scene, ctx.cameras), signature=sig,
                source=(ctx.scene, ctx.frame, ctx.timestamp, len(current['questions'])))
            torch.cuda.synchronize()
            expansion_ms = (perf_counter()-tick)*1000
            tick = perf_counter()
            prefix = cache.match(active_expanded)
            match_ms = (perf_counter()-tick)*1000
            question = current['question_texts'][len(current['questions'])]
            query = query_world_state(frozen_state, question, dict(scene_id=ctx.scene,
                frame_id=ctx.frame, timestamp=ctx.timestamp, state_epoch=ctx.epoch,
                state_execution_signature=ctx.signature))
            first_visual = int(active_expanded.visual_mask[0].nonzero()[0])
            row = dict(scene=ctx.scene, frame=ctx.frame, question=question, task=task_type(question),
                source_frame=cache.entry.expanded.source[1] if cache.entry else None,
                cross_frame_source=bool(cache.entry and cache.entry.expanded.source[1] != ctx.frame),
                expanded_tokens=active_expanded.length, first_visual_position=first_visual,
                visual_tokens=int(active_expanded.visual_mask.sum()), actual_exact_lcp=prefix,
                exact_visual_prefix_tokens=int(active_expanded.visual_mask[:, :prefix].sum()),
                lifecycle_rejection=rejection, state_query=query,
                task_evidence_complete=False, lawful_executable_KV_tokens=0,
                signature_wall_ms=signature_ms, expansion_wall_ms=expansion_ms,
                match_wall_ms=match_ms, actually_reused_tokens=0,
                prefix_is_computational_candidate=prefix >= 32,
                qualification='input equality only; original sampling distribution/numerical/cost gates not inferred')
            current['questions'].append(row)
            # Only native generate consumes the service RNG stream. Diagnostic
            # prefill is eval-only; timed() verifies its CPU/CUDA RNG invariance.
            in_native = True
            try:
                output = original_generate(*a, **kw)
            finally:
                in_native = False
            row['generated_ids'] = output.detach().cpu().tolist()
            previous_context = ctx
            return output
        stack.enter_context(replace_method(model.lm_head, 'generate', generate_observer))
        pts = model.simple_test_pts
        def pts_observer(img_metas, **data):
            current['question_texts'] = list(img_metas[0]['vlm_labels'].data)
            return pts(img_metas, **data)
        stack.enter_context(replace_method(model, 'simple_test_pts', pts_observer))
        for group in groups:
            model.test_flag = False
            model.m2_runtime = model.m3_runtime = None
            model.m1_manager = WorldStateManager(M1Config.load('configs/worldstream_m1_engineering_frozen.yaml'))
            temporal.previous.clear(); cache.reset(); previous_context = None
            manager = model.m1_manager
            precheck = manager.precheck_with_observation
            def precheck_observer(frame, *a, **kw):
                nonlocal frozen_state
                result = precheck(frame, *a, **kw)
                frozen_state, state = state_snapshot(manager, frame)
                states.append(state)
                current['context'] = Context(frame.scene_id, frame.frame_id, frame.timestamp,
                    tuple(frame.cameras), manager.store.generation, frame.execution_signature)
                return result
            with replace_method(manager, 'precheck_with_observation', precheck_observer):
                model.save_path = str(dest/'language'/group['scene'])+'/'
                for ordinal, token in enumerate(group['tokens']):
                    current.clear()
                    current.update(scene=group['scene'], frame=token, ordinal=ordinal, questions=[], scene_tokens=group['tokens'])
                    batch = scatter(collate([dataset[index[token]]], samples_per_gpu=1), [0])[0]
                    native_prefill_calls = native_decode_calls = 0
                    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); tick = perf_counter()
                    output = model(return_loss=False, rescale=True, **batch)
                    torch.cuda.synchronize()
                    wall = (perf_counter()-tick)*1000
                    from tools.worldstream_m3.experiment import typed_digest
                    from tools.worldstream_m2.rearchitecture.common import memory
                    frames.append(dict(scene=group['scene'], frame=token, ordinal=ordinal,
                        audit_wall_ms=wall, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        native_prefill_calls=native_prefill_calls, native_decode_calls=native_decode_calls,
                        cache_capture_wall_ms=current.get('cache_capture_wall_ms', []),
                        m1=manager.last_commit_diagnostics, output=typed_digest(output),
                        temporal_memory=typed_digest(memory(model)),
                        original_text=output[0]['text_out'], actual_skips=0))
                    questions.extend(current['questions'])
                    write(dest/'frames.json', frames); write(dest/'questions.json', questions)
                    write(dest/'states.json', states); write(dest/'temporal_rows.json', temporal.rows)
                    write(dest/'temporal_cost_probes.json', temporal.probes); write(dest/'kv_cost_probes.json', kv_probes)
                    print('AUDIT', group['scene'], ordinal, token, round(wall, 2),
                          'LCP', [q['actual_exact_lcp'] for q in current['questions']], flush=True)
    after = manifest(protected_files())
    write(dest/'protected_after.json', after)
    write(dest/'status.json', dict(status='completed', frames=len(frames), scenes=len(groups),
        actual_skips=0, protected_source_unchanged=before == after,
        weights_versions_unchanged=parameters == [(n,p.data_ptr(),p._version) for n,p in model.named_parameters()],
        sampling=SAMPLING, hooks_restored=True,
        source_unchanged=source == manifest(tool_paths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scenes', type=int, default=8)
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.scenes <= 8 or not 2 <= args.frames <= 12:
        parser.error('Use 1..8 frozen exploratory scenes and 2..12 consecutive frames')
    try:
        run(args)
    except Exception as error:
        dest = Path(args.output)
        # Never overwrite a prior completed experiment, including accidental reruns.
        status = dest/'status.json'
        if status.exists() and json.loads(status.read_text()).get('status') == 'running':
            write(status, dict(status='failed', error=repr(error), actual_skips=0,
                               production_authorized=False))
        raise


if __name__ == '__main__':
    main()
