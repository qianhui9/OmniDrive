"""Rebuild findings from completed P0 evidence, without promoting old results."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .common import counts, manifest, protected_files, sha256, source_evidence, stats, write


def read(root, name):
    return json.loads((root/name).read_text())


def validate_run(root):
    status = read(root, 'status.json')
    if status.get('status') != 'completed':
        raise ValueError('Incomplete/failed audit cannot support a final census')
    for key in ('protected_source_unchanged', 'weights_versions_unchanged', 'hooks_restored', 'source_unchanged'):
        if status.get(key) is not True:
            raise ValueError('Integrity check failed: '+key)
    protocol = read(root, 'protocol.json')
    expected = [(g['scene'], t) for g in protocol['groups'] for t in g['tokens']]
    frames, states = read(root, 'frames.json'), read(root, 'states.json')
    if [(f['scene'], f['frame']) for f in frames] != expected:
        raise ValueError('Missing, reordered or duplicate frame records')
    if [(f['scene'], f['frame']) for f in states] != expected:
        raise ValueError('M1 state population differs from native frames')
    questions = read(root, 'questions.json')
    frame_keys = set(expected)
    if not questions or any((q['scene'], q['frame']) not in frame_keys for q in questions):
        raise ValueError('Question provenance mismatch')
    if any(len(s['camera_timestamps']) != 6 for s in states):
        raise ValueError('Not six-camera data')
    if any(f['actual_skips'] != 0 for f in frames):
        raise ValueError('P0 observer must not execute reuse')
    if len({(q['scene'], q['frame']) for q in questions}) != len(expected):
        raise ValueError('Some frames did not execute their native questions')
    if sum(f['native_prefill_calls'] for f in frames) != len(questions):
        raise ValueError('Native prefill count does not match question population')
    return protocol, frames, states, questions


def candidate_profitability(temporal, kv, questions, frames):
    """Microbenchmarks reject unprofitable plans; never certify service speed."""
    temporal_groups = {}
    for p in temporal:
        temporal_groups.setdefault((p['scene'], p['branch']), []).append(p)
    per_branch = {}
    for (scene, branch), rows in temporal_groups.items():
        per_branch.setdefault(branch, []).append(dict(scene=scene, measured_layers=len(rows),
            net_wall_ms=sum(p['measured_net_wall_ms'] for p in rows),
            gpu_interval_delta_ms=sum(p['measured_gpu_interval_delta_ms'] for p in rows),
            numerical_equal=all(p['bitwise_equal'] for p in rows)))
    qindex = {(q['scene'], q['frame']): q for q in questions}
    findex = {(f['scene'], f['frame']): f for f in frames}
    prefix_rows = []
    for p in kv:
        key = (p['scene'], p['frame'])
        q, f = qindex[key], findex[key]
        overhead = q['signature_wall_ms'] + q['match_wall_ms'] + sum(f['cache_capture_wall_ms'])
        prefix_rows.append(dict(scene=p['scene'], frame=p['frame'],
            prefix_tokens=p['exact_input_prefix_tokens'],
            optimistic_prefill_wall_delta_ms=p['optimistic_prefill_wall_delta_ms'],
            measured_validation_publication_wall_ms=overhead,
            optimistic_net_wall_ms=p['optimistic_prefill_wall_delta_ms']-overhead,
            last_logits_bitwise_equal=p['last_logits_bitwise_equal'],
            last_logits_max_abs=p['last_logits_max_abs']))
    # Stability across measured scene units, not a favourable single noisy timing.
    temporal_go = {b: bool(rows) and all(r['net_wall_ms'] > 0 for r in rows)
                   for b, rows in per_branch.items()}
    kv_go = bool(prefix_rows) and all(r['optimistic_net_wall_ms'] > 0 for r in prefix_rows)
    return dict(temporal_by_scene=per_branch, prefix_by_scene=prefix_rows,
        temporal_positive_in_all_measured_scenes=temporal_go,
        prefix_positive_in_all_measured_scenes=kv_go,
        any_profitable_candidate=any(temporal_go.values()) or kv_go,
        interpretation='Diagnostic serial CPU wall. CUDA event intervals include launch gaps, are not kernel sums. Summed layer deltas describe the same candidate plan, not additive E2E speedup. No production qualification.')


def opportunity_gate(evidence):
    """Research admission, never a production certificate or a local cost vote.

    Unknown/missing evidence is a rejection. State evidence must match the
    candidate's claim (tensor computation provenance is distinct from physical
    object identity). A positive microbenchmark alone cannot admit research.
    """
    required = ('historical_sources_legal', 'mathematical_closure',
        'state_evidence_qualified', 'avoids_native_repeated_work',
        'net_cost_probe_positive', 'whole_service_upper_bound_worthwhile',
        'validation_plan_executable')
    missing = [name for name in required if evidence.get(name) is not True]
    values = [evidence.get(name) for name in
              ('optimistic_gpu_savings_upper_ms','minimum_required_savings_ms')]
    valid = all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0 for v in values)
    scope_positive = valid and values[0] > values[1]
    if not scope_positive:
        missing.append('insufficient_compute_scope_or_unmeasured_service_cost')
    return dict(mathematical_closure=evidence.get('mathematical_closure') is True,
        state_evidence_qualified=evidence.get('state_evidence_qualified') is True,
        performance_opportunity_go=bool(scope_positive and evidence.get('net_cost_probe_positive') is True
            and evidence.get('whole_service_upper_bound_worthwhile') is True),
        research_go=not missing, production_go=False, stop_reasons=missing,
        active_task_quality='Not Run', active_e2e='Not Run', sampling_distribution='Not Certified')
