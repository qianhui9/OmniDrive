"""Real M1 transactions and D3 execution; no synthetic production approval."""
import copy
import json

import numpy as np
import pytest
import torch

from worldstream.m1 import M1Config, WorldStateManager, FullOutputs
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.cost_model import CostModel, SCHEMA_VERSION, cost_key
from worldstream.m2.dependency_planner import block0_input
from worldstream.m2.state_cache import CachePublicationError
from .test_live_m1_snapshot import frame_for


def runtime_and_manager(eva, tmp_path):
    path=tmp_path/'empty_costs.json'
    path.write_text(json.dumps(dict(schema_version=SCHEMA_VERSION,records=[],formal_quality_policy=dict(approved_counts=[]))))
    runtime=IncrementalVisionRuntime(eva)
    manager=WorldStateManager(M1Config.load('configs/worldstream_m1_final.yaml'))
    return runtime,manager


def step(runtime,manager,images,context,i,offline=False):
    ctx=dict(context,frame_id=str(i),timestamp=100.+i*.1)
    frame=frame_for(images,ctx)
    precheck=manager.precheck(frame)
    ctx['state_epoch']=precheck.diagnostics['state_epoch']
    if offline:
        result,diag=runtime.evaluate_offline_from_m1(images.float()/255,manager,ctx,variant='D')
    else:
        result,diag=runtime.forward_from_m1(images.float()/255,manager,ctx)
    manager.commit(frame,FullOutputs([],execution_receipt=diag),precheck)
    return result,diag,ctx


@torch.no_grad()
def test_formal_initialization_and_live_m1_reuse(eva,context,tmp_path):
    rt,manager=runtime_and_manager(eva,tmp_path)
    image=torch.randint(0,256,(2,3,40,40),dtype=torch.uint8)
    first,d0,_=step(rt,manager,image,context,0)
    assert d0['cache_saved'] and rt.cache.version==1
    assert torch.equal(first[0],eva(image.float()/255)[0])
    source=copy.copy(rt.cache)
    _,d1,ctx=step(rt,manager,image,context,1)
    assert d1['full_update_reason'] is None and d1['actual_skipped_windows']>0
    assert d1['m1_snapshot_read'] and not d1['window_reuse_quality_certified']
    assert rt.cache.version==2 and rt.cache.context==ctx
    assert d1['total_windows']==d1['executed_windows']+d1['reused_windows']==36
    assert d1['planned_windows']==d1['executed_windows']==12
    assert d1['reused_windows']==d1['actual_skipped_windows']==24
    assert sum(c['reused_windows'] for c in d1['per_camera'])==24
    for stage in d1['per_block']:
        assert stage['planned_window_ids']==stage['executed_window_ids']
        assert all(stage['selection_consistency'].values())
        reused=set(range(stage['total_windows']))-set(stage['executed_window_ids'])
        assert all(rt.cache.sources[stage['block']][j]==source.sources[stage['block']][j] for j in reused)
    assert not hasattr(rt,'costs')


@torch.no_grad()
def test_formal_and_d3_share_cache_namespace_and_plan(eva,context,tmp_path):
    rt,manager=runtime_and_manager(eva,tmp_path)
    image=torch.randint(0,256,(2,3,40,40),dtype=torch.uint8)
    step(rt,manager,image,context,0)
    ctx=dict(context,frame_id='1',timestamp=100.1)
    frame=frame_for(image,ctx);result=manager.precheck(frame)
    ctx['state_epoch']=result.diagnostics['state_epoch']
    source=copy.copy(rt.cache)
    formal,da=rt.forward_from_m1(image.float()/255,manager,ctx)
    saved=rt.cache;rt.cache=copy.copy(source)
    offline,db=rt.evaluate_offline_from_m1(image.float()/255,manager,ctx,variant='D')
    assert torch.equal(formal[0],offline[0])
    assert da['stages']==db['stages'] and da['cache_version']==db['cache_version']==2
    assert rt.cache.signature==saved.signature


@torch.no_grad()
def test_mandatory_closure_with_different_window_sizes(eva,context,tmp_path):
    from .test_wsgrr import step as synthetic_step,region
    from projects.mmdet3d_plugin.models.backbones.eva_vit import VisionRotaryEmbeddingFast
    eva.blocks[1].window_size=6
    eva.blocks[1].attn.rope=VisionRotaryEmbeddingFast(dim=4,pt_seq_len=4,ft_seq_len=6).eval()
    rt,_=runtime_and_manager(eva,tmp_path)
    image=torch.randn(2,3,40,40)
    synthetic_step(rt,image,context,0,[region(bbox_xyxy=[1.,1.,4.,4.])])
    _,diag=synthetic_step(rt,image,context,1,[region(bbox_xyxy=[18.,1.,22.,4.])])
    assert diag['region_refresh']['mandatory_windows']>0
    for stage in diag['stages'][:2]:
        assert all(stage['selection_consistency'].values())
        assert set(stage['mandatory_window_ids'])<=set(stage['executed_window_ids'])
        assert stage['planned_window_ids']==stage['executed_window_ids']


@torch.no_grad()
@pytest.mark.parametrize('fault',['full_refresh','source','scene','version','aborted','time','epoch'])
def test_forced_refresh_scene_reset_and_invalid_state(eva,context,tmp_path,fault):
    from .test_source_linked_admission import bundle_for
    rt,manager=runtime_and_manager(eva,tmp_path)
    images=torch.randint(0,256,(2,3,40,40),dtype=torch.uint8)
    _,_,ctx=step(rt,manager,images,context,0)
    ctx=dict(ctx,frame_id='next',timestamp=100.1)
    bundle=bundle_for(rt,ctx)
    if fault=='full_refresh':bundle.diagnostics['full_refresh_hint']=True
    if fault=='source':bundle.diagnostics['observed_source_frame_id']='stale'
    if fault=='scene':ctx['scene_id']=bundle.scene_id='new-scene'
    if fault=='version':ctx['source_version']=rt.cache.version+1
    if fault=='aborted':bundle.diagnostics['transaction_status']='aborted'
    if fault=='time':bundle.timestamp=float('nan')
    if fault=='epoch':bundle.state_epoch+=1
    if fault in ('full_refresh','scene'):
        out,diag=rt.forward(images.float()/255,execution_context=ctx,computation_state=bundle)
        assert diag['full_recompute'] and diag['actual_skipped_windows']==0
        assert torch.equal(out[0],eva(images.float()/255)[0])
        assert diag['cache_saved'] and diag['cache_version']==2
        assert all(s['computation_frame_id']=='next' for units in rt.cache.sources.values() for s in units)
        assert diag['full_refresh_reason']==('scene_change' if fault=='scene' else 'm1_full_refresh')
    else:
        with pytest.raises(ValueError):
            rt.forward(images.float()/255,execution_context=ctx,computation_state=bundle)
        assert rt.cache.version==1


@torch.no_grad()
@pytest.mark.parametrize('failure',['budget','nonfinite'])
def test_publication_errors_propagate_and_preserve_history(eva,context,tmp_path,failure):
    from worldstream.m2.state_cache import NonFiniteCacheError
    rt,manager=runtime_and_manager(eva,tmp_path)
    image=torch.randint(0,256,(2,3,40,40),dtype=torch.uint8)
    step(rt,manager,image,context,0)
    source=rt.cache.outputs
    if failure=='budget':rt.cache.budget_bytes=1
    else:
        original=rt.cache.commit
        def bad(images,outputs,*args,**kwargs):
            outputs=dict(outputs);outputs[0]=outputs[0]*float('nan')
            return original(images,outputs,*args,**kwargs)
        rt.cache.commit=bad
    with pytest.raises((CachePublicationError,NonFiniteCacheError)):
        step(rt,manager,image,context,1)
    assert rt.cache.outputs is source and rt.cache.version==1
    assert not rt.last_diagnostics['execution_completed']
