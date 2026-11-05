"""Calibration identity covers original parameters, numeric policy and input layout."""
import pytest
import torch
from .test_source_linked_admission import cold,bundle_for




@torch.no_grad()
@pytest.mark.parametrize('channels_last',[False,True])
def test_output_layout_and_global_residual_formula(admitted_runtime,context,channels_last):
    rt=admitted_runtime;x=torch.randn(2,3,40,40)
    if channels_last:x=x.to(memory_format=torch.channels_last)
    cold(rt,x,context);anchor=rt.cache.global_anchor;x[0,:,0,0]+=.1
    prefix={}
    rt.stage_observer=lambda i,value:prefix.update({i:value.clone()})
    ctx=dict(context,frame_id='1',timestamp=1.1)
    out,d=rt.forward(x,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    reference=rt.model(x)[0]
    assert out[0].stride()==reference.stride()
    assert out[0].shape==reference.shape and out[0].dtype==reference.dtype
    assert torch.equal(out[0],(prefix[rt.prefix[-1]]+anchor.residual).permute(0,3,1,2))
    assert all(s['mode']=='approx_global_residual_reuse' for s in d['stages'][2:])
