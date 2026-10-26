"""Direct serving has no cost dependency and force-refresh cannot waive provenance."""
import pytest
import torch
from worldstream.m2 import IncrementalVisionRuntime,M2Config
from .test_source_linked_admission import cold,bundle_for


@pytest.mark.parametrize('field,value',[('cost_table','missing.json'),('cache_capture','always')])
def test_retired_serving_config_fields_fail_explicitly(field,value):
    with pytest.raises(TypeError):M2Config(**{field:value})


@torch.no_grad()
def test_serving_never_constructs_or_queries_cost_model(eva,context,monkeypatch):
    from worldstream.m2.cost_model import CostModel
    def fail(*a,**kw):raise AssertionError('Serving queried the offline cost model')
    for method in ('__init__','beneficial','has_beneficial','assessment'):
        monkeypatch.setattr(CostModel,method,fail)
    rt=IncrementalVisionRuntime(eva);image=torch.randn(2,3,40,40)
    _,first=cold(rt,image,context)
    ctx=dict(context,frame_id='1',timestamp=1.1)
    _,second=rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert first['cache_saved'] and first['executed_windows']==36
    assert second['cache_saved'] and second['actual_skipped_windows']==24
    assert second['cache_version']==2
    assert all(s['planned_window_ids']==s['executed_window_ids'] for s in second['per_block'])


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
