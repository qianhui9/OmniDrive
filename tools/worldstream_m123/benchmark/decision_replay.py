"""Read-only planner trace and CPU counterfactuals; never executes model KV.

Replay calls the production planner on its observed prefix statistics, with
the observed scheduler overhead held fixed. Removing a signal means neutral
validity / zero M2 refresh and age, not changing policy thresholds.
"""
import copy
import hashlib
from dataclasses import asdict
from functools import wraps
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from worldstream.m3.config import M3Config
from worldstream.m3.reuse_planner import ReusePlanner
from worldstream.m3.scheduler import Scheduler


class DecisionTrace:
    def __init__(self, runtime):
        self.rows = []
        self.pending = None
        planner = runtime.planner
        original_choose = runtime.scheduler.choose

        @wraps(original_choose)
        def choose(candidates, context, overhead_ms):
            if self.pending is not None:
                self.pending.update(candidates=copy.deepcopy(candidates),
                    scheduler_context=copy.deepcopy(context), overhead_ms=overhead_ms,
                    policy_compatible=runtime.scheduler.compatible(context))
            return original_choose(candidates, context, overhead_ms)

        runtime.scheduler.choose = choose
        original_plan = planner.plan

        @wraps(original_plan)
        def plan(question, expanded, cache, history, context, m2_diagnostics=None, overhead_ms=0.):
            entry = cache.entry
            row = dict(question=question, context=copy.deepcopy(context),
                history=copy.deepcopy(history), m2=copy.deepcopy(m2_diagnostics),
                input_tokens=expanded.length, dtype=str(expanded.embeddings.dtype),
                source=list(expanded.source), layout_signature=expanded.layout_signature,
                visual_group_spans=expanded.visual_group_spans, config=asdict(runtime.config),
                cache=None if entry is None else dict(source=list(entry.expanded.source),
                    root_frame_index=entry.root_frame_index, approx_reuse_depth=entry.approx_reuse_depth,
                    quality_risk=entry.quality_risk, risk_profile=entry.risk_profile,
                    computed_source=list(entry.computed_source), input_tokens=entry.expanded.length,
                    frame_index=entry.frame_index,source_state_epoch=entry.expanded.scope[2] if len(entry.expanded.scope)>2 else None,
                    kv_tensor_versions=[[t._version for t in layer] for layer in entry.kv],
                    storage_revision_sha256=hashlib.sha256(repr(cache.revision).encode()).hexdigest()))
            self.pending = row
            try:
                decision, details = original_plan(question, expanded, cache, history, context,
                    m2_diagnostics, overhead_ms=overhead_ms)
            finally:
                self.pending = None
            row.update(decision=decision.diagnostics(), details=copy.deepcopy(details),
                cache_validated=cache.entry is not None)
            if cache.entry is not None and not details.get('block_changes'):
                # Diagnostic only, AFTER the actual planner made its decision.
                # Never include traced runs in performance claims.
                from worldstream.m3.prefix_matcher import prefix_statistics
                exact,legal,blocks=prefix_statistics(cache.entry.expanded,expanded,runtime.config.block_size)
                row['offline_full_statistics']=dict(exact=exact,legal=legal,blocks=blocks)
            if 'candidates' in row and hasattr(runtime.scheduler,'explain'):
                row['candidate_evaluations']=runtime.scheduler.explain(row['candidates'],row['scheduler_context'],row['overhead_ms'])
            self.rows.append(row)
            return decision, details

        planner.plan = plan

    def clear(self):
        self.rows.clear()


def replay(trace, policy_path, variant='actual'):
    """Fresh planner, no GPU/cache mutation; every replay preserves policy bytes."""
    cfg = dict(trace['config'], reuse_policy=str(policy_path))
    scheduler = Scheduler(M3Config(**cfg))
    native_choose = scheduler.choose
    scheduler.choose = lambda candidates, context, overhead: native_choose(
        candidates, context, trace.get('overhead_ms', 0.))
    history = copy.deepcopy(trace['history'] or {})
    m2 = copy.deepcopy(trace['m2'] or {})
    context = dict(trace['context'])
    if variant == 'without_m1':
        history['state_valid'] = True
    elif variant == 'without_m2':
        m2 = dict(execution_completed=True,
            cache_context={k: context[k] for k in ('scene_id', 'frame_id', 'timestamp')},
            stages=[dict(total_windows=1, executed_windows=0, cache_age_frames=0)])
    elif variant != 'actual':
        raise ValueError('Unknown replay variant: '+variant)
    dtype = getattr(torch, trace['dtype'].removeprefix('torch.'))
    expanded = NS(length=trace['input_tokens'], source=tuple(trace['source']),
        embeddings=NS(dtype=dtype), layout_signature=trace['layout_signature'],
        visual_group_spans=trace['visual_group_spans'])
    saved = trace['cache']
    entry = None
    if saved is not None and trace['cache_validated']:
        entry = NS(**{k: v for k, v in saved.items() if k not in ('source', 'input_tokens')},
            expanded=NS(source=tuple(saved['source']), length=saved['input_tokens']))
    cache = NS(validate_storage=lambda: entry, revision=0)
    details = trace['details']
    statistics = (details.get('matched_prefix_tokens', 0),
        details.get('structural_prefix_tokens', 0), details.get('block_changes', []))
    # Prefix statistics and revisions are already measured on real inputs.
    # Only proof construction is stubbed; no such proof is ever executed.
    with patch('worldstream.m3.reuse_planner.prefix_statistics', return_value=statistics), \
            patch('worldstream.m3.reuse_planner.input_revision', return_value='offline-only'):
        decision, result = ReusePlanner(scheduler).plan(trace['question'], expanded, cache,
            history, context, m2)
    return dict(decision=decision.diagnostics(), candidate_prefix_tokens=result.get('candidate_prefix_tokens', []))


def compare_trace(trace, policy_path):
    actual = replay(trace, policy_path)
    keys = ('reuse_length', 'reuse_type', 'decision_reason')
    matched = all(actual['decision'][k] == trace['decision'][k] for k in keys)
    return dict(scene_id=trace['context']['scene_id'], frame_id=trace['context']['frame_id'],
        question=trace['question'], profile=trace['details']['task_profile'],
        baseline_reproduced=matched, actual=actual,
        without_m1=replay(trace, policy_path, 'without_m1'),
        without_m2=replay(trace, policy_path, 'without_m2'))
