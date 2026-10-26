"""Expired computation or observation hints expand only affected dependencies."""
import pytest
import torch
from .test_source_linked_admission import cold,bundle_for


@torch.no_grad()
@pytest.mark.parametrize('fault',['expired_observation','dirty_source_record'])
def test_camera_observation_expiry_does_not_skip_invalid_sources(exact_runtime,context,fault):
    rt=exact_runtime;x=torch.randn(2,3,40,40);cold(rt,x,context)
    ctx=dict(context,frame_id='1',timestamp=1.1);b=bundle_for(rt,ctx)
    if fault=='expired_observation':b.observation_links['front']['validity_lease']['source_valid']=False
    else:
        b.source_records.append(dict(feature_id='eva_prefix:0:0',source_camera_id='front',
            computation_available=True,computation_timestamp=1.,computation_frame_id='forged',
            computation_version=1,computation_version_namespace='m2_window_execution'))
    if fault=='dirty_source_record':
        with pytest.raises(ValueError,match='m1_computation_source_mismatch'):
            rt.forward(x,execution_context=ctx,computation_state=b)
    else:
        out,d=rt.forward(x,execution_context=ctx,computation_state=b)
        assert d['actual_skipped_windows']==18
        assert torch.allclose(out[0],rt.model(x)[0],atol=3e-6,rtol=3e-6)


@torch.no_grad()
def test_time_reversal_never_reuses(exact_runtime,context):
    rt=exact_runtime;x=torch.randn(2,3,40,40);cold(rt,x,context)
    ctx=dict(context,frame_id='1',timestamp=.9);b=bundle_for(rt,ctx)
    with pytest.raises(ValueError,match='frame_order_mismatch'):
        rt.forward(x,execution_context=ctx,computation_state=b)
