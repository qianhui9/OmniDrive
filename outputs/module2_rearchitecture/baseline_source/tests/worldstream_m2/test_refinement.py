import json
import numpy as np
import pytest
import torch
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.cost_model import CostModel,cost_key
from worldstream.m2.m1_adapter import cpu_window_opportunity
from worldstream.m2.packing import partition,SpatialWindowGather,window_mask
from worldstream.m2.dependency_planner import patch_updates
from .test_m1_policy import result


@torch.no_grad()
def test_full_sparse_full_sparse_lifecycle(eva,context):
    r=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    x=torch.randn(2,3,40,40)
    calls=[]
    hook=eva.blocks[0].register_forward_hook(lambda *a:calls.append(1))
    captured=[]
    for i,kind in enumerate(['refresh','sparse','full','sparse']):
        c=dict(context,frame_id=str(i))
        m=result(c,np.zeros((40,40),bool))
        if kind=='refresh':m.full_refresh_hint=True
        if kind=='full':
            for v in m.invalidation_regions.values():
                from worldstream.m1.mask_codec import encode_mask
                v['exact_invalidation_mask']=encode_mask(np.ones((40,40),bool))
        x=x.clone();x[0,:,i,i]+=.1
        before=len(calls)
        y,d=r.forward(x,m,execution_context=c)
        assert len(calls)-before == (0 if kind=='sparse' else 1)
        assert d['cache_saved'] and r.cache.context['frame_id']==str(i)
        assert d['actual_skipped_windows']>0 if kind=='sparse' else d['actual_skipped_windows']==0
        captured.append({k:v.clone() for k,v in r.cache.outputs.items()})
        hook.remove()
        assert torch.allclose(y[0],eva(x)[0],atol=3e-6,rtol=3e-6)
        hook=eva.blocks[0].register_forward_hook(lambda *a:calls.append(1))
    hook.remove()
    assert r.cache.version==4
    assert not torch.equal(captured[0][0],captured[2][0])


@torch.no_grad()
def test_old_cache_references_immutable(eva):
    from worldstream.m2.window_executor import WindowExecutor
    x=torch.randn(2,10,10,32)
    cache=partition(eva.blocks[0](x),4)[0]
    old=cache.clone();x[0,0,0]+=.3
    ex=WindowExecutor()
    out,new=ex.execute(eva.blocks[0],x,cache,torch.tensor([0,17]))
    assert torch.equal(cache,old) and new.data_ptr()!=cache.data_ptr()
    assert torch.allclose(out,eva.blocks[0](x),atol=2e-6,rtol=2e-6)


def test_cpu_admission_matches_gpu_footprints(eva):
    rng=np.random.default_rng(4)
    for density in (0,.01,.5,1):
        mask=rng.random((2,40,40))<density
        cpu=cpu_window_opportunity(mask,eva.patch_embed.proj,4)
        gpu=~window_mask(patch_updates(torch.tensor(mask),eva.patch_embed.proj),4)
        assert np.array_equal(cpu.ravel(),gpu.numpy())


@torch.no_grad()
def test_spatial_gather_noncontiguous_padding(eva):
    x=torch.randn(2,32,10,10).permute(0,2,3,1)
    ids=torch.tensor([17,0,9,8])
    gather=SpatialWindowGather()
    actual,valid=gather.gather(x,4,ids)
    expected=partition(x,4)[0].index_select(0,ids)
    assert torch.equal(actual,expected)
    pointer=gather.valid.data_ptr()
    gather.gather(x,4,ids[:1]);assert gather.valid.data_ptr()==pointer
    assert valid.dtype==torch.bool


def test_costs_are_block_and_checkpoint_specific(eva,tmp_path):
    x=torch.randn(2,10,10,32)
    a=cost_key(eva.blocks[0],x,0)
    assert a!=cost_key(eva.blocks[0],x,1)
    with torch.no_grad():eva.blocks[0].norm1.weight.add_(.1)
    assert a!=cost_key(eva.blocks[0],x,0)
    path=tmp_path/'table.json'
    path.write_text(json.dumps(dict(records=[dict(key=a,updated=1)])))
    old=CostModel(str(path));assert old.status=='cost_schema_mismatch' and old.records==[]


@torch.no_grad()
def test_zero_opportunity_never_plans_or_gathers(eva,context,monkeypatch):
    r=IncrementalVisionRuntime(eva,dict(mode='exact'))
    def forbidden(*a,**k):raise AssertionError('GPU sparse planning must not run')
    monkeypatch.setattr(r.planner,'plan',forbidden)
    monkeypatch.setattr(r.executor,'execute',forbidden)
    x=torch.randn(2,3,40,40)
    _,d=r.forward(x,result(context,np.ones((40,40),bool)),execution_context=context)
    assert d['cache_capture_reason']=='capture_not_expected_to_pay'
    assert not d['cache_saved']


@torch.no_grad()
def test_camera_context_is_snapshot_not_alias(eva,context):
    r=IncrementalVisionRuntime(eva,dict(mode='exact'))
    x=torch.randn(2,3,40,40)
    r.forward(x,execution_context=context)
    context['camera_ids'].reverse()
    _,d=r.forward(x,execution_context=dict(context,frame_id='1'))
    assert d['fallback_reason']=='execution_signature_mismatch'


@torch.no_grad()
def test_rejection_samples_never_accept_unknown_footprint(eva,context):
    from worldstream.m2.m1_adapter import invalidation_mask
    x=torch.zeros(2,3,40,40)
    mask=np.zeros((40,40),bool);mask[1,1]=True
    invalid,reason=invalidation_mask(result(context,mask),None,context,x,device=False,
                                     rejection_footprint=(4,4,4))
    assert reason is None and invalid[0,1,1]
    assert not cpu_window_opportunity(invalid,eva.patch_embed.proj,4)[0,0,0]


@torch.no_grad()
def test_cost_lower_bound_rejects_before_gpu_plan(eva,context,monkeypatch):
    r=IncrementalVisionRuntime(eva,dict(mode='auto'))
    r.scheduler.costs.records=[dict(block=0,key='unrelated',updated=0,quality_certified=True,
        incremental_gpu_ms=.1,full_gpu_ms=1.,incremental_wall_ms=.1,full_wall_ms=1.)]
    def forbidden(*a,**kw):raise AssertionError('No useful measured count: do not launch planner')
    monkeypatch.setattr(r.planner,'plan',forbidden)
    mask=np.ones((40,40),bool);mask[:16,:16]=False
    _,d=r.forward(torch.randn(2,3,40,40),result(context,mask),execution_context=context)
    assert d['fallback_reason']=='no_measured_beneficial_count' and not d['cache_saved']


def test_provisional_cost_table_cannot_approve_downstream_quality(eva,tmp_path):
    key=cost_key(eva.blocks[0],torch.randn(2,10,10,32),0)
    path=tmp_path/'provisional.json'
    path.write_text(json.dumps(dict(schema_version=2,records=[dict(key=key,block=0,updated=1,
        incremental_gpu_ms=.1,full_gpu_ms=1.,incremental_wall_ms=.1,full_wall_ms=1.,quality_certified=True)])))
    costs=CostModel(str(path))
    assert costs.beneficial(key,1)==(False,'downstream_quality_unverified')


@pytest.mark.parametrize('channels_last',[False,True])
@torch.no_grad()
def test_incremental_output_preserves_oracle_memory_layout(eva,context,channels_last):
    x=torch.randn(2,3,40,40)
    if channels_last:x=x.to(memory_format=torch.channels_last)
    r=IncrementalVisionRuntime(eva,dict(mode='exact'))
    r.forward(x,execution_context=context)
    y,d=r.forward(x,execution_context=dict(context,frame_id='1'))
    oracle=eva(x)[0]
    assert y[0].stride()==oracle.stride()
    assert torch.allclose(y[0],oracle,atol=2e-6,rtol=2e-6)
