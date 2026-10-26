"""Optional full-model M2 candidate measurements, outside request timing.

Uses the exact pending M1 snapshot and immutable source cache of a measured
request. Replays never commit M1 or advance production history.
"""
import copy
from functools import wraps
import hashlib
import math
from pathlib import Path

import torch

from worldstream.m2.cost_model import cost_key, block_fingerprint, SCHEMA_VERSION
from worldstream.m2.window_executor import NUMERICAL_POLICY
from .window_validate import paired_measure, difference
from tools.worldstream_m123.benchmark.common import write, sha256


class FullModelEvidence:
    def __init__(self, runtime, repeats=3, warmup=1):
        self.runtime = runtime
        self.repeats, self.warmup = repeats, warmup
        self.cases, self.measurements = [], []
        self.pending = None
        self.inputs = {}
        original = runtime._forward
        execute = runtime.executor.execute

        @wraps(original)
        def forward(images, m1_result, execution_context, computation_state, offline=None, window_targets=None):
            self.inputs = {}
            source = copy.copy(runtime.cache)
            capture_decision = copy.deepcopy(runtime._last_capture_decision)
            result = original(images, m1_result, execution_context, computation_state, offline, window_targets)
            self.pending = (images, m1_result, copy.deepcopy(execution_context), computation_state,
                offline, window_targets, source, capture_decision, dict(self.inputs), result)
            return result

        @wraps(execute)
        def observe(block, x, cached, ids):
            self.inputs[id(block)] = (x, cached, ids)
            return execute(block, x, cached, ids)

        runtime._forward = forward
        runtime.executor.execute = observe
        self.forward, self.execute = original, execute

    def clear(self):
        self.pending = None
        self.inputs.clear()

    @torch.no_grad()
    def collect(self, frame_id):
        if self.pending is None: raise ValueError('Missing actual M2 call')
        images, snapshot, context, state, offline, targets, source, capture_decision, inputs, result = self.pending
        if offline != 'D': raise ValueError('Full-model cost collection requires offline D')
        rt = self.runtime
        saved = (rt.cache, rt.last_diagnostics, rt._last_capture_decision, rt.executing_block)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        def candidate():
            rt.cache = copy.copy(source)
            rt._last_capture_decision = copy.deepcopy(capture_decision)
            return self.forward(images, snapshot, context, state, offline, targets)
        try:
            (replayed, native), inc, full = paired_measure(candidate,
                lambda: rt.model(images)[0], self.repeats, self.warmup)
            delta = difference(replayed[0][0], native)
            if not all(math.isfinite(delta[k]) for k in ('max_abs','mean_abs')):
                raise ValueError('Nonfinite full-model candidate feature error')
            measured_diag = result[1]
            for key in ('actual_skipped_windows','recomputed_windows'):
                if replayed[1][key] != measured_diag[key]: raise ValueError('Replay changed actual window plan')
            case = 'full_model:'+str(frame_id)
            stages = [s for s in measured_diag['stages'] if 'total_windows' in s]
            self.cases.append(dict(case=case, frame_id=str(frame_id), scene_id=context['scene_id'],
                difference=delta, stages=copy.deepcopy(stages), incremental=inc, native=full))
            for i in rt.prefix:
                block = rt.model.blocks[i]
                x, cached, ids = inputs[id(block)]
                if len(ids) >= next(s['total_windows'] for s in stages if s['block']==i): continue
                (partial, dense), a, b = paired_measure(lambda:self.execute(block,x,cached,ids),
                    lambda:block(x), self.repeats, self.warmup)
                local = difference(partial[0], dense)
                if not all(math.isfinite(local[k]) for k in ('max_abs','mean_abs')):
                    raise ValueError('Nonfinite candidate block error')
                key = cost_key(block,x,i,model=rt.model,refresh_policy=rt.config.refresh_signature())
                stage = next(s for s in stages if s['block']==i)
                if stage['cost_key_sha256'] != hashlib.sha256(key.encode()).hexdigest():
                    raise ValueError('Measured cost identity differs from actual model call')
                self.measurements.append(dict(key=key, block=i, updated=len(ids),
                    numerical_verified=True, numerical_validation_scope='Finite measured error; task/limit assessment pending',
                    quality_certified=False, quality_rejection_reason='downstream_quality_unverified',
                    incremental_gpu_ms=a['gpu_span_ms'], full_gpu_ms=b['gpu_span_ms'],
                    incremental_wall_ms=a['wall_ms'], full_wall_ms=b['wall_ms'],
                    pipeline_incremental_wall_ms=inc['wall_ms'],pipeline_full_wall_ms=full['wall_ms'],
                    pipeline_incremental_gpu_ms=inc['gpu_span_ms'],pipeline_full_gpu_ms=full['gpu_span_ms'],
                    difference=local, evidence_cases=[case],block_incremental=a,block_full=b))
        finally:
            rt.cache, rt.last_diagnostics, rt._last_capture_decision, rt.executing_block = saved
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
            self.clear()

    def finish(self, output, checkpoint_sha256):
        output = Path(output)
        validation = output/'m2_full_model_validation.json'
        write(validation,dict(validation_kind='wsgrr_quantitative_v1',
            backbone_fingerprint=block_fingerprint(self.runtime.model),numerical_policy=NUMERICAL_POLICY,
            checkpoint_sha256=checkpoint_sha256,natural=self.cases,controlled_microbenchmarks=[],
            provenance='Live full-model M1 Precheck snapshot; real candidate heads and VLM',
            timing_scope='Post-request replay of M2 visual path on identical immutable source; M1 commit and task heads excluded',
            measurement_repeats=self.repeats,warmup=self.warmup))
        groups = {}
        for row in self.measurements: groups.setdefault((row['key'],row['updated']),[]).append(row)
        records=[]
        for group in groups.values():
            record=dict(max(group,key=lambda r:r['incremental_wall_ms']/max(r['full_wall_ms'],1e-9)))
            record.update(measurements=group,evidence_cases=[c for r in group for c in r['evidence_cases']],
                numerical_evidence=dict(path=validation.name,sha256=sha256(validation)))
            records.append(record)
        write(output/'m2_window_costs_provisional.json',dict(schema_version=SCHEMA_VERSION,
            records=records,validation_sha256=sha256(validation),
            backbone_fingerprint=block_fingerprint(self.runtime.model),
            formal_quality_policy=dict(approved_counts=[]),production_ready=False))
