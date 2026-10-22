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
    out,d=rt.forward(x,execution_context=ctx,computation_state=b)
    assert d['actual_skipped_windows']==(18 if fault=='expired_observation' else 0)
    assert torch.equal(out[0],rt.model(x)[0])


@torch.no_grad()
def test_time_reversal_never_reuses(exact_runtime,context):
    rt=exact_runtime;x=torch.randn(2,3,40,40);cold(rt,x,context)
    ctx=dict(context,frame_id='1',timestamp=.9);b=bundle_for(rt,ctx)
    out,d=rt.forward(x,execution_context=ctx,computation_state=b)
    assert d['actual_skipped_windows']==0 and torch.equal(out[0],rt.model(x)[0])
