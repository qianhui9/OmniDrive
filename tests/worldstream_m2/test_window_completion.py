"""Window execution owns canonical activations and rejects stale calibration."""
import json
import pytest
import torch

from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.cost_model import CostModel, cost_key
from worldstream.m2.packing import partition
from worldstream.m2.state_cache import StateCache
from worldstream.m2.window_executor import WindowExecutor


@torch.no_grad()
def test_partial_window_cache_padding_is_canonical(eva):
    block=eva.blocks[0]
    block.norm1.bias.fill_(.3)
    block.mlp.w3.bias.fill_(.7)
    x=torch.randn(2,10,10,32)
    cache=partition(block(x),4)[0]
    x[0,9,9]+=.1
    out,packed=WindowExecutor().execute(block,x,cache,torch.tensor([8]))
    reference=partition(block(x),4)[0]
    assert torch.allclose(out,block(x),atol=3e-6,rtol=3e-6)
    assert torch.allclose(packed,reference,atol=3e-6,rtol=3e-6)
    assert torch.count_nonzero(packed[8,2:])==0
    assert torch.count_nonzero(packed[8,:,2:])==0


@torch.no_grad()
def test_single_window_zero_update_does_not_alias_source(eva):
    block=eva.blocks[0]
    x=torch.randn(1,4,4,32)
    cache=partition(block(x),4)[0]
    saved=cache.clone()
    out,_=WindowExecutor().execute(block,x,cache,torch.empty(0,dtype=torch.long))
    out.zero_()
    assert torch.equal(cache,saved)


@torch.no_grad()
def test_window_executor_checks_contract_before_zero_update(eva):
    block=eva.blocks[0];x=torch.randn(1,4,4,32)
    cache=partition(block(x),4)[0]
    block.train()
    with pytest.raises(ValueError,match='inference-only'):
        WindowExecutor().execute(block,x,cache,torch.empty(0,dtype=torch.long))
    block.eval()
    with pytest.raises(ValueError,match='feature grid'):
        WindowExecutor().execute(block,x,cache[:0],torch.empty(0,dtype=torch.long))


def test_window_cache_owns_outputs_and_rejects_over_budget_atomically():
    cache=StateCache(2000);images=torch.ones(1,3,4,4);out={0:torch.ones(1,4,4,8)}
    ctx=dict(frame_id='0',camera_ids=['front'])
    cache.commit(images,out,ctx,'signature')
    out[0].zero_();out.clear()
    assert cache.outputs[0].count_nonzero()==128
    old=(cache.images,cache.outputs,cache.context,cache.version)
    with pytest.raises(RuntimeError,match='budget'):
        cache.commit(images,{0:torch.ones(1000)},ctx,'signature')
    assert (cache.images,cache.outputs,cache.context,cache.version)==old


@torch.no_grad()
@pytest.mark.parametrize('target',['images','outputs'])
def test_window_cache_mutation_forces_native_refresh(eva,context,target):
    runtime=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    x=torch.randn(2,3,40,40)
    runtime.forward(x,execution_context=context)
    if target=='images':runtime.cache.images.add_(1)
    else:runtime.cache.outputs[0].zero_()
    out,d=runtime.forward(x,execution_context=dict(context,frame_id='1'))
    assert d['fallback_reason']=='cache_storage_modified'
    assert d['actual_skipped_windows']==0
    assert torch.equal(out[0],eva(x)[0])


def test_window_cost_key_includes_inner_normalization(eva):
    x=torch.randn(2,10,10,32)
    before=cost_key(eva.blocks[0],x,0)
    eva.blocks[0].mlp.ffn_ln.eps*=2
    assert before!=cost_key(eva.blocks[0],x,0)


@pytest.mark.parametrize('fault',['nan','negative','bool_count','wrong_block','string_quality'])
def test_invalid_window_costs_cannot_admit(eva,tmp_path,fault):
    key=cost_key(eva.blocks[0],torch.zeros(2,10,10,32),0)
    row=dict(key=key,block=0,updated=1,incremental_gpu_ms=.1,full_gpu_ms=1.,
             incremental_wall_ms=.1,full_wall_ms=1.,quality_certified=True)
    if fault=='nan':row['full_wall_ms']=float('nan')
    if fault=='negative':row['incremental_wall_ms']=-1.
    if fault=='bool_count':row['updated']=True
    if fault=='wrong_block':row['block']=1
    if fault=='string_quality':row['quality_certified']='false'
    path=tmp_path/'cost.json'
    path.write_text(json.dumps(dict(schema_version=2,records=[row],formal_quality_policy=dict(approved_counts=[1]))))
    costs=CostModel(str(path))
    assert costs.beneficial(key,1)[0] is False
    if fault!='string_quality':assert costs.status=='invalid_cost_table'


@torch.no_grad()
def test_benchmark_hooks_preserve_window_topology_and_observe_real_batches(eva,context,monkeypatch):
    from types import SimpleNamespace as NS
    from tools.worldstream_m123.benchmark.observe import Observer
    from worldstream.m3 import visual_state
    class Event:
        def __init__(self,**kw):pass
        def record(self):pass
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(visual_state,'base_model',lambda m:m)
    runtime=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    noop=lambda *a,**k:None
    model=NS(img_backbone=eva,pts_bbox_head=NS(forward=noop),map_head=NS(forward=noop),
        lm_head=NS(forward=noop,generate=noop),m1_manager=None,m2_runtime=runtime,m3_runtime=None)
    observer=Observer(model)
    assert runtime.planner.cacheable_prefix()==[0,1]
    image=torch.randn(2,3,40,40)
    runtime.forward(image,execution_context=context)
    assert sum(observer.window_attention_batches.values())==36
    observer.clear();image[0,:,0,0]+=.1
    _,diag=runtime.forward(image,execution_context=dict(context,frame_id='1'))
    assert diag['actual_skipped_windows']==34
    assert sum(observer.window_attention_batches.values())==2
    assert observer.backbone_cameras==0


@torch.no_grad()
@pytest.mark.parametrize('approved',[True,False])
def test_window_auto_requires_matching_cost_and_quality_before_packing(eva,context,tmp_path,approved):
    from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
    image=torch.randn(2,3,40,40)
    changed=image.clone();changed[0,:,0,0]+=.1
    probe=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    probe.forward(image,execution_context=context)
    inputs={}
    probe.stage_observer=lambda i,out:inputs.update({i+1:out.detach()})
    probe.forward(changed,execution_context=dict(context,frame_id='1'))
    x=eva.patch_embed(changed)
    inputs[0]=x+get_abs_pos(eva.pos_embed,eva.pretrain_use_cls_token,x.shape[1:3])
    # Synthetic costs are fixtures for admission, never measured evidence.
    records=[dict(key=cost_key(eva.blocks[i],inputs[i],i),block=i,updated=1,
        incremental_gpu_ms=.1,full_gpu_ms=1.,incremental_wall_ms=.1,full_wall_ms=1.,
        quality_certified=True) for i in probe.prefix]
    path=tmp_path/'synthetic_unit_test_costs.json'
    path.write_text(json.dumps(dict(schema_version=2,records=records,
        formal_quality_policy=dict(approved_counts=[1] if approved else []))))
    runtime=IncrementalVisionRuntime(eva,dict(mode='auto',cache_capture='always',cost_table=str(path)))
    runtime.forward(image,execution_context=context)
    out,d=runtime.forward(changed,execution_context=dict(context,frame_id='1'))
    assert d['actual_skipped_windows']==(34 if approved else 0)
    assert all(s['mode']==('packed' if approved else 'full') for s in d['stages'][:2])
    assert torch.allclose(out[0],eva(changed)[0],atol=3e-6,rtol=3e-6)


@torch.no_grad()
def test_legacy_window_model_adapter_carries_bound_m1_time(eva,context):
    from worldstream.m1 import WorldStateManager
    from .test_live_m1_snapshot import frame_for
    manager=WorldStateManager()
    ctx=dict(context,timestamp=100.)
    image=torch.zeros(2,3,40,40,dtype=torch.uint8)
    manager.precheck(frame_for(image,ctx))
    runtime=IncrementalVisionRuntime(eva,dict(mode='exact',cache_capture='always'))
    _,d=runtime.forward_from_m1(image.float(),manager,context)
    assert d['cache_context']['timestamp']==100.
    assert runtime.cache.context['timestamp']==100.
