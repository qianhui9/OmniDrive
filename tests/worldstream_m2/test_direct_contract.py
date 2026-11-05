"""Direct serving has no cost dependency and force-refresh cannot waive provenance."""
import pytest
import torch
from worldstream.m2 import IncrementalVisionRuntime,M2Config
from .test_source_linked_admission import cold,bundle_for


@pytest.mark.parametrize('field,value',[('cost_table','missing.json'),('cache_capture','always')])
def test_retired_serving_config_fields_fail_explicitly(field,value):
    with pytest.raises(TypeError):M2Config(**{field:value})




@torch.no_grad()
@pytest.mark.parametrize('fault',['source','aborted','mask'])
def test_forced_full_refresh_cannot_hide_illegal_state(eva,context,fault):
    rt=IncrementalVisionRuntime(eva);image=torch.randn(2,3,40,40);cold(rt,image,context)
    ctx=dict(context,frame_id='1',timestamp=1.1);bundle=bundle_for(rt,ctx)
    bundle.diagnostics['full_refresh_hint']=True
    if fault=='source':bundle.diagnostics['source_cache_version']+=1
    if fault=='aborted':bundle.diagnostics['transaction_status']='aborted'
    if fault=='mask':bundle.refresh_contract['front']['dirty_hint_mask']={}
    calls=[];rt.executor.execute=lambda *a: calls.append(a)
    with pytest.raises(ValueError,match='m1_|invalid_refresh_contract'):
        rt.forward(image,execution_context=ctx,computation_state=bundle)
    assert calls==[] and rt.cache.version==1
    assert not rt.last_diagnostics['execution_completed']
