"""Source-bound refresh contracts; unknown pixels require independent comparison."""
import copy
import numpy as np
import pytest
import torch
from worldstream.m1.state_types import ComputationStateBundle
from worldstream.m1.mask_codec import encode_mask


def cold(runtime,image,context):
    bundle=ComputationStateBundle(context['scene_id'],context['frame_id'],context['timestamp'],
        context['state_epoch'],context['execution_signature'],diagnostics=dict(full_refresh_hint=True))
    return runtime.forward(image,execution_context=context,computation_state=bundle)


def bundle_for(runtime,context):
    cache=runtime.cache;h,w=cache.images.shape[-2:]
    bundle=ComputationStateBundle(context['scene_id'],context['frame_id'],context['timestamp'],
        context['state_epoch'],context['execution_signature'],diagnostics=dict(
            cache_frame_id=cache.context['frame_id'],source_cache_version=cache.version,
            observed_source_frame_id=cache.context['frame_id'],cache_available=True,transaction_status='pending'))
    for cid in context['camera_ids']:
        bundle.observation_links[cid]=dict(source_frame_id=cache.context['frame_id'],source_camera_id=cid,
            source_timestamp=cache.context['timestamp'],validity_lease=dict(source_valid=True,max_observation_age=1.))
        bundle.refresh_contract[cid]=dict(coordinate_space='preprocessed_image',
            dirty_hint_mask=encode_mask(np.zeros((h,w),bool)),reuse_candidate_mask=encode_mask(np.zeros((h,w),bool)),
            unknown_mask=encode_mask(np.ones((h,w),bool)))
    return bundle
