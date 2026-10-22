"""Window computation ages remain distinct from a fresh verification time."""
from worldstream.m2.state_cache import CachePublicationError
import pytest
import torch
from .test_source_linked_admission import cold,bundle_for


@torch.no_grad()
def test_unchanged_window_keeps_actual_computation_source(admitted_runtime,context):
    rt=admitted_runtime;x=torch.randn(2,3,40,40);cold(rt,x,context)
    ctx=dict(context,frame_id='1',timestamp=1.1);x[0,:,0,0]+=.1
    _,d=rt.forward(x,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    source=d['output_window_provenance'][0]
    assert source[0]['computation_frame_id']=='1'
    assert source[1]['computation_frame_id']=='0' and source[1]['computation_timestamp']==1.
    assert source[1]['computation_version']==1
    assert d['cache_context']['frame_id']=='1' and rt.cache.version==2


@torch.no_grad()
@pytest.mark.parametrize('field',['camera_id','producer_layer','computation_version','computation_frame_id'])
def test_external_lineage_mutation_forces_all_window_update(admitted_runtime,context,field):
    rt=admitted_runtime;x=torch.randn(2,3,40,40);cold(rt,x,context)
    rt.cache.sources[0][0][field]='forged'
    ctx=dict(context,frame_id='1',timestamp=1.1)
    out,d=rt.forward(x,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert d['actual_skipped_windows']==0
    assert torch.equal(out[0],rt.model(x)[0])


@torch.no_grad()
def test_failed_publication_preserves_output_lineage(admitted_runtime,context,monkeypatch):
    rt=admitted_runtime;x=torch.randn(2,3,40,40);cold(rt,x,context)
    def fail(*a,**kw):raise CachePublicationError('injected publication failure')
    monkeypatch.setattr(rt.cache,'commit',fail)
    ctx=dict(context,frame_id='1',timestamp=1.1)
    _,d=rt.forward(x,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert not d['cache_saved'] and not rt.cache.bytes and d['actual_skipped_windows']==36
    assert all(s['computation_frame_id']=='0' for units in d['output_window_provenance'].values() for s in units)
