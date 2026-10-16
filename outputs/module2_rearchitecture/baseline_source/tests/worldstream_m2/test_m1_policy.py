import numpy as np
import torch
from worldstream.m1 import M1Result
from worldstream.m1.mask_codec import encode_mask
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.cost_model import CostModel
from worldstream.m2.m1_adapter import invalidation_mask


def result(context, mask):
    return M1Result(context['scene_id'], context['frame_id'], 1.,
        invalidation_regions={cid: {'exact_invalidation_mask': encode_mask(mask),
                                   'coordinate_space':'preprocessed_image'} for cid in context['camera_ids']},
        diagnostics={'execution_signature': context['execution_signature'], 'transformer_dependency_certified': False})


def test_geometry_mismatch(context):
    x = torch.zeros(2, 3, 40, 40)
    r = result(context, np.zeros((36, 64), bool))
    mask, reason = invalidation_mask(r, None, context, x)
    assert mask is None and reason == 'preprocessing_geometry_unavailable'


@torch.no_grad()
def test_unknown_never_reused(eva, context):
    r = IncrementalVisionRuntime(eva, dict(mode='exact'))
    x = torch.randn(2, 3, 40, 40)
    r.forward(x, execution_context=context)
    c = dict(context, frame_id='1')
    y, d = r.forward(x, result(c, np.ones((40, 40), bool)), execution_context=c)
    assert d['fallback_reason'] == 'insufficient_stable_windows'
    assert torch.equal(y[0], eva(x)[0]) and r.cache.bytes == 0


@torch.no_grad()
def test_m1_false_certificate_requires_exact_input(eva, context):
    r = IncrementalVisionRuntime(eva, dict(mode='exact'))
    x = torch.randn(2, 3, 40, 40)
    r.forward(x, execution_context=context)
    c = dict(context, frame_id='1')
    x = x + 1  # M1 claims support everywhere; independent input check rejects it.
    y, d = r.forward(x, result(c, np.zeros((40, 40), bool)), execution_context=c)
    assert not d['actual_skipped_windows']
    assert torch.equal(y[0], eva(x)[0])


def test_cost_policy():
    costs = CostModel()
    costs.records = [dict(key='k', updated=2, incremental_gpu_ms=2., full_gpu_ms=1.,
                          incremental_wall_ms=.5, full_wall_ms=1., quality_certified=True)]
    assert costs.beneficial('k', 2) == (False, 'estimated_cost_not_beneficial')
    assert costs.beneficial('other', 2) == (False, 'unprofiled_cost')


def test_missing_cost_table(tmp_path):
    costs=CostModel(str(tmp_path/'missing.json'))
    assert costs.beneficial('k',0)==(False,'unprofiled_cost')


def test_uncertified_quality_cannot_auto_execute():
    costs=CostModel()
    costs.records=[dict(key='k',updated=1,incremental_gpu_ms=.1,full_gpu_ms=1,
                        incremental_wall_ms=.1,full_wall_ms=1)]
    assert costs.beneficial('k',1)==(False,'numerical_quality_unverified')


def test_raw_region_rejected_even_with_matching_shape(context):
    r=result(context,np.zeros((40,40),bool))
    r.invalidation_regions['front']['coordinate_space']='resized_raw_image'
    assert invalidation_mask(r,None,context,torch.zeros(2,3,40,40))[1]=='preprocessing_geometry_unavailable'
