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

from .common import (AFFINITY, SAMPLING, Context, diagnostic_json, history_rejection, manifest,
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
            valid_full = min(self.current['ordinal'] * head.topk_proposals, head.memory_len)
            real = max(0, valid_full - prop)
            # A retained row can move from the previous propagated QUERY slice
            # into today's temporal MEMORY slice. Layer-0 values are unchanged;
            # later-layer propagated queries have been updated and must be
            # compared, never assumed equivalent. In previous V the propagated
            # query suffix and temporal memory are contiguous.
            start = max(0, head.topk_proposals - prop)
            end = min(real, head.topk_proposals + (old['valid_full'] if old else 0) - prop)
            lifecycle = history_rejection(old['context'] if old else None,
                                          self.current['context'], 1.)
            n = max(0, end - start) if lifecycle is None else 0
            tick = perf_counter()
            same_v = same_k = 0
            old_start = old['nq'] + start - head.topk_proposals if old else 0
            if n:
                old_slice = slice(old_start, old_start + n)
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
            self.pending[key] = (q, k, v, nq, real, valid_full, start, end, old_start, old, row)
        return hook

    def projected(self, key, attention):
        def hook(projected_v):
            _, k, v, nq, real, valid_full, start, end, old_start, old, row = self.pending.pop(key)
            n = row['retained_source_rows']
            measurement = (self.current['scene'], key)
            if n and row['identical_value_input_rows'] == n and measurement not in self.measured:
                self.measured.add(measurement)
                w = attention.in_proj_weight.chunk(3)[2]
                bias = attention.in_proj_bias.chunk(3)[2] if attention.in_proj_bias is not None else None
                versions = (attention.in_proj_weight._version,
                            attention.in_proj_bias._version if attention.in_proj_bias is not None else None)
                # Source value and its native projection are retained from t-1.
                historical = old['projected_v'][:, old_start:old_start+n]
                def candidate():
                    if versions != (attention.in_proj_weight._version,
                                    attention.in_proj_bias._version if attention.in_proj_bias is not None else None):
                        raise ValueError('Changed weights')
                    if not torch.equal(v[:, nq+start:nq+end], old['v'][:, old_start:old_start+n]):
                        return F.linear(v, w, bias)
                    fresh = F.linear(torch.cat((v[:, :nq+start], v[:, nq+end:]), 1), w, bias)
                    return torch.cat((fresh[:, :nq+start], historical, fresh[:, nq+start:]), 1)
                dense = F.linear(v, w, bias)
                proposed = candidate()
                error = float((dense - proposed).abs().max())
                changed = (dense != proposed).nonzero()
                full = timed(lambda: F.linear(v, w, bias))
                partial = timed(candidate)
                self.probes.append(dict(branch=key[0], layer=key[1], scene=self.current['scene'],
                    frame=self.current['frame'], source_frame=old['context'].frame,
                    reused_value_rows=n, full_rows=v.shape[1], full=full, candidate=partial,
                    bitwise_equal=bool(torch.equal(dense, proposed)), max_abs=error,
                    first_differing_output_index=changed[0].tolist() if changed.numel() else None,
                    dense_vs_native_projection_max_abs=float((dense-projected_v).abs().max()),
                    reused_rows_projection_max_abs=float((dense[:,nq+start:nq+end]-historical).abs().max()),
                    fresh_rows_projection_max_abs=float(torch.cat((
                        (dense-proposed)[:,:nq+start],(dense-proposed)[:,nq+end:]),1).abs().max()),
                    measured_net_wall_ms=full['wall_ms']['p50']-partial['wall_ms']['p50'],
                    measured_gpu_interval_delta_ms=full['gpu_interval_ms']['p50']-partial['gpu_interval_ms']['p50'],
                    dependency='V_linear_only: exact retained native value + layer weights/bias; not K/Q/attention/head',
                    maintenance='Read-only native input/projection references retained. Source equality, weight versions, fresh-row gather, GEMM, output concatenation included; M1 and broader certification not included.',
                    actual_runtime_skips=0, diagnostic_only=True))
                del dense, proposed
            # Native concatenations/projections are out of place. Retaining them
            # avoids inventing a second state store or mutating model memory.
            self.previous[key] = dict(v=v.detach(), k=k.detach(), nq=nq, real=real, valid_full=valid_full,
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
