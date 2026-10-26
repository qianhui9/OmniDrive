"""Historical CPU selection ablations; all GPU execution uses serving M2.

D is an alias of the formal algorithm, including cache signatures and versions.
A/B/C and matched controls live here, outside the serving selection routine.
"""
import copy
import types
import numpy as np
import torch
from worldstream.m2.dependency_planner import exact_token_changes,patch_updates
from worldstream.m2.region_refresh import pooled_descriptors,select_regions


def evaluate_from_m1(runtime,images,manager,context,variant,targets):
    if variant=='D':return runtime.forward_from_m1(images,manager,context)
    snapshot=manager.export_m2_inputs(expected_context=context,computation_only=True)
    return evaluate(runtime,images,variant,snapshot,context,None,targets)


def evaluate(runtime,images,variant,snapshot,context,state,targets):
    if variant not in ('A','B','C','D','NoM1Region','NoM1RegionMatched'):
        raise ValueError('Unknown offline ablation')
    if variant=='NoM1RegionMatched':
        if not isinstance(targets,list) or len(targets)!=len(images) or any(type(v) is not int or v<0 for v in targets):
            raise ValueError('Explicit per-camera offline total targets required')
    elif targets is not None:raise ValueError('Total targets belong only to the offline matched-compute control')
    if variant=='D':return runtime._forward(images,snapshot,context,state)
    def plan(rt,x0,bundle,ctx,uncertainty,mandatory):
        if variant=='A':
            from worldstream.m1.mask_codec import decode_mask
            dirty=torch.as_tensor(np.stack([decode_mask(bundle.refresh_contract[c]['dirty_hint_mask']) for c in ctx['camera_ids']]),device=x0.device)
            mandatory |= patch_updates(dirty,rt.model.patch_embed.proj)
            changed=exact_token_changes(x0,rt.cache.block0_input)|mandatory
            return (rt.planner.plan(changed,stop_after=len(rt.prefix)),
                pooled_descriptors(x0,rt.model.blocks[0].window_size),copy.deepcopy(bundle.region_states),
                dict(exact_token_update_ratio=float(changed.float().mean())))
        for i in rt.prefix:
            expired=rt.cache.window_ages[i]+1 >= rt.config.max_reuse_frames
            mandatory |= rt.planner.windows_to_tokens(expired,x0.shape[:3],rt.model.blocks[i].window_size)
        selected,descriptors,regions,refresh=select_regions(x0,rt.cache,bundle,ctx,
            rt.model.patch_embed.proj,rt.model.blocks[0].window_size,rt.config,
            uncertainty,mandatory,events=variant not in ('B','NoM1Region','NoM1RegionMatched'),
            budget=variant not in ('B','C'),window_targets=targets)
        return rt.planner.plan_windows(selected,x0.shape[:3],stop_after=len(rt.prefix)),descriptors,regions,refresh
    original=runtime._plan_refresh
    runtime._plan_refresh=types.MethodType(plan,runtime)
    try:
        output,diag=runtime._forward(images,snapshot,context,state)
        diag.update(offline_experiment=True,offline_variant=variant)
        if variant=='A':diag['change_detection']='exact_block0_input'
        return output,diag
    finally:runtime._plan_refresh=original
