"""Read-only logical payload accounting; never clone model state or read GT."""
from dataclasses import fields, is_dataclass
import math
import sys
import numpy as np

MIB = 1024 ** 2


def payload_bytes(value):
    """Python owned graph + unique ndarray/tensor storage, excluding model/config.

    Logical retained payload estimate, not allocator residency or process RSS.
    Only explicitly selected module cache/state roots are traversed. Tensor
    storage metadata is inspected without transferring features to the host.
    """
    seen, storages = set(), set()

    def visit(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        size = sys.getsizeof(obj)
        if isinstance(obj, np.ndarray):
            root = obj
            while isinstance(root.base, np.ndarray):
                root = root.base
            key = ('numpy', root.__array_interface__['data'][0])
            extra = 0 if key in storages else root.nbytes
            storages.add(key)
            # getsizeof owning arrays already includes their data allocation.
            return size - (obj.nbytes if obj.flags.owndata else 0) + extra
        if hasattr(obj, 'storage') and hasattr(obj, 'element_size'):
            storage = obj.untyped_storage() if hasattr(obj, 'untyped_storage') else obj.storage()
            key = ('torch', str(obj.device), storage.data_ptr())
            extra = 0 if key in storages else (storage.nbytes() if hasattr(storage, 'nbytes')
                                              else storage.size() * obj.element_size())
            storages.add(key)
            return size + extra
        if isinstance(obj, dict):
            return size + sum(visit(k) + visit(v) for k, v in obj.items())
        if isinstance(obj, (tuple, list, set, frozenset)):
            return size + sum(visit(v) for v in obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            return size + visit(vars(obj))
        return size

    return visit(value)


def distribution(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    return dict(count=len(values), mean=float(np.mean(values)) if values else None,
                p95=float(np.percentile(values, 95)) if values else None,
                maximum=max(values) if values else None,
                histogram=dict(zip(('0', '(0,0.5]', '(0.5,1]', '(1,2]', '(2,4]', '(4,8]', '>8'),
                                   [sum(v == 0 for v in values)] +
                                   [sum(a < v <= b for v in values) for a, b in
                                    ((0, .5), (.5, 1), (1, 2), (2, 4), (4, 8), (8, math.inf))])))


def collect(model, timestamp, frame_times):
    manager, visual, language = model.m1_manager, model.m2_runtime, model.m3_runtime
    store, cache, memory = manager.store, visual.cache, language.answer_memory
    anchors = list(store.anchors.values())
    precheck = manager.last_result
    commit = manager.last_commit_diagnostics
    diag = model.m2_diagnostics
    requests = model.m3_diagnostics['requests']
    ages = lambda name: [timestamp - getattr(a, name) for a in anchors if getattr(a, name) is not None]
    pairs = len(commit.get('association_pairs', []))
    selected = len(commit.get('selected_prediction_indices', []))
    considered = len(precheck.valid_anchor_ids) + len(precheck.invalid_anchor_ids)
    world = dict(state_size_mb=payload_bytes({k: v for k, v in vars(store).items() if k != 'config'}) / MIB,
        captured_tensor_mb=store.tensor_bytes / MIB,
        object_anchors=sum(a.anchor_type == 'object' for a in anchors),
        map_anchors=sum(a.anchor_type == 'map' for a in anchors),
        total_anchors=len(anchors), features=len(store.features),
        state_epoch=store.generation, state_version=store.commit_count,
        record_storage_age_s=ages('track_birth_timestamp'),
        source_observation_age_s=ages('source_timestamp'),
        last_actual_observation_age_s=ages('last_actual_observation_timestamp'),
        last_verified_age_s=ages('last_verified_timestamp'),
        # Age alone does not certify physical validity or object identity.
        updates=len(commit.get('committed_anchor_ids', [])), commit_events=1,
        association_pairs=pairs, association_predictions=selected,
        association_pair_records=commit.get('association_pairs', []),
        association_identity_error_rate_percent=None,
        association_success_percent=100 * pairs / selected if selected else None,
        unmatched_predictions=selected - pairs,
        invalidated_anchors=len(precheck.invalid_anchor_ids), considered_anchors=considered,
        invalidated_ratio=len(precheck.invalid_anchor_ids) / considered if considered else None,
        ambiguous_association_count=None,
        expired_anchors=len(precheck.diagnostics.get('expired_anchor_ids', [])),
        ogsm_ms=commit.get('m1_total_including_adapters_ms'))
    for label in ('record_storage_age_s', 'source_observation_age_s', 'last_actual_observation_age_s', 'last_verified_age_s'):
        world[label + '_distribution'] = distribution(world[label])
    # Small integer age tensors only; no visual feature transfer. Sources give
    # original computation time, rather than last cache verification time.
    window_ages = [int(v) for tensor in cache.window_ages.values() for v in tensor.detach().cpu().reshape(-1).tolist()]
    reused_ages, reused_seconds = [], []
    for stage in diag['per_block']:
        computed = set(stage['executed_window_ids'])
        block_ages = cache.window_ages[stage['block']].detach().cpu().reshape(-1).tolist()
        for i, source in enumerate(cache.sources[stage['block']]):
            if i not in computed:
                reused_ages.append(block_ages[i])
                reused_seconds.append(timestamp - source['computation_timestamp'])
    anchor = cache.global_anchor
    refreshed = not diag['approx_global_residual_reuse']
    vision = dict(cache_size_mb=payload_bytes({k: getattr(cache, k) for k in
        ('images', 'block0_input', 'outputs', 'context', 'sources', 'region_states',
         'window_ages', 'descriptors', 'global_anchor')}) / MIB,
        tensor_cache_size_mb=cache.bytes / MIB, cached_window_count=len(window_ages),
        cache_version=cache.version, window_age_frames=window_ages,
        reused_window_age_frames=reused_ages, reused_window_age_s=reused_seconds,
        window_age_distribution=distribution(window_ages), reuse_age_distribution=distribution(reused_ages),
        total_windows=diag['total_windows'], reused_windows=diag['reused_windows'],
        window_reuse_percent=100 * diag['reused_windows'] / diag['total_windows'] if diag['total_windows'] else None,
        max_reuse_frames=visual.config.max_reuse_frames,
        windows_at_or_over_reuse_limit=sum(v >= visual.config.max_reuse_frames for v in window_ages),
        global_anchor_age_frames=anchor.age if anchor else None,
        global_anchor_age_s=timestamp - anchor.timestamp if anchor else None,
        global_anchor_generation=anchor.generation if anchor else None,
        global_anchor_refreshed=refreshed, global_anchor_interval=visual.config.global_anchor_interval,
        full_global_refresh=refreshed, global_feature_drift=diag.get('global_feature_drift'),
        incremental_visual_execution=diag['mode'] == 'incremental', visual_ms=diag.get('M2_total_ms'))
    proposed = sum(r.get('draft_candidate_tokens', 0) for r in requests)
    accepted = sum(r.get('draft_accepted_tokens', 0) for r in requests)
    records = list(memory.records.values())
    selected_ages = [timestamp - frame_times[r['draft_source_frame']] for r in requests
                     if r.get('draft_source_frame') in frame_times]
    semantics = dict(cache_size_mb=payload_bytes(dict(records=memory.records, acceptance=memory.acceptance,
        performance=memory.performance)) / MIB,
        cache_entries=len(records), candidate_token_count=sum(len(r.tokens) for r in records),
        prompt_token_count=sum(len(r.prompt_ids) for r in records),
        max_entries=memory.max_entries, max_age_frames=memory.max_age_frames, max_age_s=memory.max_age_s,
        candidate_age_s=[timestamp - r.timestamp for r in records],
        candidate_age_frames=[language._frame_index - r.frame_index for r in records],
        selected_candidate_age_s=selected_ages,
        proposed_tokens=proposed, accepted_tokens=accepted,
        token_acceptance_percent=100 * accepted / proposed if proposed else None,
        rejected_tokens=proposed - accepted,
        token_validation_failure_percent=100 * (proposed - accepted) / proposed if proposed else None,
        # There is no published complete per-entry validation failure counter.
        candidate_entry_validation_failure_percent=None,
        queries=len(requests), direct_answers=sum(bool(r.get('llm_skipped')) for r in requests),
        direct_answer_percent=100 * sum(bool(r.get('llm_skipped')) for r in requests) / len(requests) if requests else None,
        language_ms=model.m3_diagnostics.get('complete_m3_wall_ms'),
        request_candidate_ages=[dict(source_frame=r.get('draft_source_frame'),
            source_age_s=timestamp - frame_times[r['draft_source_frame']] if r.get('draft_source_frame') in frame_times else None,
            proposed_tokens=r.get('draft_candidate_tokens', 0), accepted_tokens=r.get('draft_accepted_tokens', 0)) for r in requests])
    semantics['candidate_age_distribution'] = distribution(semantics['candidate_age_s'])
    return dict(ogsm=world, davp=vision, qcse=semantics)
