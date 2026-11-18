"""Experiment-only DAVP policies, all executing the real WindowExecutor.

Full DAVP is inherited unchanged. Simple alternatives compute the entire global
suffix every frame. No cached tensor is silently substituted for computed work.
"""
import math
import numpy as np
import torch
from worldstream.m1.mask_codec import encode_mask
from worldstream.m1.state_types import ComputationStateBundle
from worldstream.m2.runtime import IncrementalVisionRuntime
from worldstream.m2.dependency_planner import StagePlan, patch_updates
from worldstream.m2.packing import window_mask
from worldstream.m2.region_refresh import pooled_descriptors, cosine_drift, region_tokens


def source_bundle(runtime, images, context, masks=None, horizon=4.0):
    """Bind experimental evidence to the actual preceding computation cache.

    This is an explicitly weaker *experiment* observation contract. It carries
    no production observation certificates or invented computation records.
    The inherited runtime still checks cache lineage, ages and tensor integrity.
    """
    cache = runtime.cache
    previous = cache.context
    bundle = ComputationStateBundle(context['scene_id'], context['frame_id'],
        context['timestamp'], context['state_epoch'], context['execution_signature'])
    bundle.diagnostics = dict(transaction_status='pending',
        cache_available=previous is not None, source_cache_version=cache.version,
        cache_frame_id=previous['frame_id'] if previous else None,
        observed_source_frame_id=previous['frame_id'] if previous else None,
        full_refresh_hint=previous is None, experimental_evidence=True)
    h, w = images.shape[-2:]
    for ci, cid in enumerate(context['camera_ids']):
        dirty = np.ones((h, w), bool) if masks is None else np.asarray(masks[ci], bool)
        # Camera/image dimensions in the shared deployed pipeline include padding.
        if dirty.shape != (h, w):
            padded = np.ones((h, w), bool)
            padded[:dirty.shape[0], :dirty.shape[1]] = dirty
            dirty = padded
        bundle.refresh_contract[cid] = dict(coordinate_space='preprocessed_image',
            dirty_hint_mask=encode_mask(dirty), reuse_candidate_mask=encode_mask(~dirty),
            unknown_mask=encode_mask(np.zeros((h, w), bool)))
        bundle.observation_links[cid] = dict(source_frame_id=previous['frame_id'] if previous else None,
            source_camera_id=cid, source_timestamp=previous['timestamp'] if previous else None,
            validity_lease=dict(source_valid=previous is not None, max_observation_age=horizon))
    return bundle


def budget_selection(scores, ages, budget, max_age):
    """Top-K with deadline reservations; never exceeds the per-camera budget.

    Reserve slots before starvation can make the hard maximum age infeasible.
    Remaining slots use the requested periodic/feature ordering, without an age
    term in their ordinary scores. Ages are incremented before selection.
    """
    n = len(scores)
    if budget < 1 or budget * max_age < n:
        raise ValueError('Refresh budget/max age cannot cover the camera window grid')
    budget = min(budget, n)
    oldest = sorted(range(n), key=lambda i: (-ages[i], i))
    required = 0
    for steps in range(1, max_age + 1):
        due = sum(a >= max_age - steps + 1 for a in ages)
        required = max(required, due - budget * (steps - 1))
    if required > budget:
        raise RuntimeError('Window deadline infeasible; refusing unbounded reuse')
    selected = oldest[:max(0, required)]
    candidates = sorted((i for i in range(n) if i not in selected), key=lambda i: (-scores[i], i))
    return selected + candidates[:budget-len(selected)]


class ExperimentalVision(IncrementalVisionRuntime):
    def __init__(self, backbone, config, policy, manager=None):
        super().__init__(backbone, config)
        self.policy = policy
        self.manager = manager

    def forward_from_m1(self, images, manager, context):
        context = dict(context)
        context.setdefault('execution_signature', 'ablation-image-only-v1')
        context.setdefault('state_epoch', 0)
        if self.policy in ('periodic', 'feature_drift'):
            # Deliberately do not read M1 events, masks, TTLs or snapshots.
            bundle = source_bundle(self, images, context, horizon=1.e6)
        elif self.policy == 'image_difference':
            current = images.detach().float()
            previous = self.cache.images
            masks = None if previous is None else ((current-previous.float()).abs().mean(1) > .18).cpu().numpy()
            bundle = source_bundle(self, images, context, masks, horizon=1.e6)
        else:
            bundle = manager.export_m2_inputs(expected_context=context, computation_only=True).computation_state
        result, diag = self.forward(images, execution_context=context, computation_state=bundle)
        diag.update(experimental_policy=self.policy, dependency_closure=self.policy != 'region_mapping',
                    world_state_consumed=self.policy not in ('periodic', 'feature_drift', 'image_difference'))
        return result, diag

    def _plan_refresh(self, x0, bundle, context, uncertainty, mandatory):
        if self.policy == 'full':
            return super()._plan_refresh(x0, bundle, context, uncertainty, mandatory)
        size = self.model.blocks[0].window_size
        desc = pooled_descriptors(x0, size)
        drift = cosine_drift(desc, self.cache.descriptors)
        ages = self.cache.window_ages[0] + 1
        count = len(desc) // len(x0)
        selected = torch.zeros(len(desc), dtype=torch.bool, device=x0.device)
        if self.policy in ('periodic', 'feature_drift'):
            values, age_values = drift.cpu().tolist(), ages.cpu().tolist()
            period = math.ceil(count/self.config.ordinary_budget)
            for c in range(len(x0)):
                offset = c*count
                # Fixed round-robin phase: each window refreshes once per period.
                scores = ([float((j//self.config.ordinary_budget) == self.cache.version % period)
                           for j in range(count)] if self.policy == 'periodic' else values[offset:offset+count])
                ids = budget_selection(scores, age_values[offset:offset+count],
                    self.config.ordinary_budget, self.config.max_reuse_frames)
                selected[torch.tensor([offset+j for j in ids], device=x0.device)] = True
        else:
            if self.policy == 'image_difference':
                from worldstream.m1.mask_codec import decode_mask
                pixels = np.stack([decode_mask(bundle.refresh_contract[c]['dirty_hint_mask']) for c in context['camera_ids']])
                tokens = patch_updates(torch.as_tensor(pixels, device=x0.device), self.model.patch_embed.proj)
            else:
                tokens = torch.zeros(x0.shape[:3], dtype=torch.bool, device=x0.device)
                # Direct mapping of changed/uncertain regions, including disappearance
                # footprints from the preceding committed state.
                for cid in context['camera_ids']:
                    ci = context['camera_ids'].index(cid)
                    from worldstream.m1.mask_codec import decode_mask
                    contract = bundle.refresh_contract[cid]
                    dirty = decode_mask(contract['dirty_hint_mask']) | decode_mask(contract['unknown_mask'])
                    masks = torch.zeros((len(x0), *dirty.shape), dtype=torch.bool, device=x0.device)
                    masks[ci] = torch.as_tensor(dirty, device=x0.device)
                    tokens |= patch_updates(masks, self.model.patch_embed.proj)
                for row in bundle.uncertainty_regions:
                    if row.get('scope') == 'camera':
                        tokens[context['camera_ids'].index(row['camera_id'])] = True
                    else:
                        tokens |= region_tokens(row['bbox_xyxy'], context['camera_ids'].index(row['camera_id']),
                            x0.shape[:3], self.model.patch_embed.proj, self.cache.images.shape[-2:], device=x0.device)
                old = {(r['camera_id'], r['track_id']): r for r in self.cache.region_states}
                now = {(r['camera_id'], r['track_id']): r for r in bundle.region_states}
                event_ids = {r['track_id'] for r in bundle.region_events}
                for key in set(old) | set(now):
                    a, b = old.get(key), now.get(key)
                    if (a is None or b is None or key[1] in event_ids or
                        a.get('state_signature') != b.get('state_signature') or
                        max(abs(x-y) for x,y in zip(a['bbox_xyxy'],b['bbox_xyxy'])) >= self.config.movement_threshold_px):
                        for row in (a,b):
                            if row:
                                tokens |= region_tokens(row['bbox_xyxy'], context['camera_ids'].index(row['camera_id']),
                                    x0.shape[:3], self.model.patch_embed.proj, self.cache.images.shape[-2:], device=x0.device)
            tokens |= mandatory
            selected |= window_mask(tokens, size)
            selected |= ages >= self.config.max_reuse_frames
        ids = selected.nonzero(as_tuple=False).flatten().cpu().tolist()
        refresh = dict(mandatory_window_ids=ids, ordinary_window_ids=[], mandatory_windows=len(ids),
                       ordinary_windows=0, policy=self.policy)
        if self.policy == 'region_mapping':
            # Map the original token region separately to every prefix block.
            # Never feed a preceding block's expanded mask into the next block.
            plans = [StagePlan(i, self.model.blocks[i].window_size,
                window_mask(tokens | self.planner.windows_to_tokens(selected, x0.shape[:3], size),
                            self.model.blocks[i].window_size), tokens, 'direct_region_mapping') for i in self.prefix]
        else:
            plans = self.planner.plan_windows(selected, x0.shape[:3], stop_after=len(self.prefix))
        return plans, desc, list(bundle.region_states), refresh

    def _update_global(self, x, context, signature):
        if self.policy in ('full', 'image_difference'):
            return super()._update_global(x, context, signature)
        # The three simple alternatives do not inherit DAVP's global residual reuse.
        # The shared helper executes its ordinary full suffix when no anchor exists.
        from worldstream.m2.global_residual import update_global
        import copy
        cold = copy.copy(self.cache)
        cold.global_anchor = None
        return update_global(x, self.model.blocks, self.prefix, cold, context,
                             signature, self.config, self.stage_observer)
