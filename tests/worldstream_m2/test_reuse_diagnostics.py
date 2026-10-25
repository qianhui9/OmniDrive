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
    runtime=IncrementalVisionRuntime(eva,dict(cost_table=str(path)))
    manager=WorldStateManager(M1Config.load('configs/worldstream_m1_final.yaml'))
    return runtime,manager


def step(runtime,manager,images,context,i,offline=True):
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
def test_offline_adaptive_initialization_and_live_m1_reuse(eva,context,tmp_path):
    rt,manager=runtime_and_manager(eva,tmp_path)
    image=torch.randint(0,256,(2,3,40,40),dtype=torch.uint8)
    first,d0,_=step(rt,manager,image,context,0)
    assert d0['execution_completed'] and d0['full_recompute']
    assert d0['cache_capture_decision']['allowed'] and d0['cache_capture_decision']['attempted']
    assert d0['cache_capture_decision']['published'] and rt.cache.version==1
    assert d0['cache_capture_decision']['reason']=='offline_candidate_capture'
    assert torch.equal(first[0],eva(image.float()/255)[0])
    source=copy.copy(rt.cache)
    image=image.clone();image[0,:,0,0]=0
    second,d1,ctx=step(rt,manager,image,context,1)
    assert d1['full_update_reason'] is None and d1['actual_skipped_windows']>0
    assert d1['m1_snapshot_read'] and not d1['window_reuse_quality_certified']
    assert rt.cache.version==2 and rt.cache.context==ctx
    for stage in d1['stages'][:2]:
        assert stage['executed_windows']==stage['planned_windows']<stage['total_windows']
        assert all(stage['selection_consistency'].values())
        assert set(stage['mandatory_window_ids'])<=set(stage['executed_window_ids'])
        assert sum(c['executed_windows'] for c in stage['per_camera'])==stage['executed_windows']
        assert stage['executed_windows']+stage['reused_windows']==stage['total_windows']
        assert not stage['cost_admission_applied'] and not stage['cost_record_found']
        for j,record in enumerate(rt.cache.sources[stage['block']]):
            if j in stage['executed_window_ids']:
                assert record['computation_frame_id']=='1' and record['computation_version']==2
            else:
                assert record==source.sources[stage['block']][j]
                assert torch.equal(rt.cache.outputs[stage['block']][j],source.outputs[stage['block']][j])
    # Same input except a local perturbation. Current v2 CPU operator tolerance;
    # real changed-frame quality is separately quantified, never auto-approved.
    assert torch.allclose(second[0],eva(image.float()/255)[0],atol=3e-6,rtol=3e-6)


@torch.no_grad()
def test_empty_production_costs_explain_persistent_cache_miss(eva,context,tmp_path):
    rt,manager=runtime_and_manager(eva,tmp_path)
    image=torch.randint(0,256,(2,3,40,40),dtype=torch.uint8)
    for i in range(2):
        output,diag,_=step(rt,manager,image,context,i,offline=False)
        assert diag['cost_table_status']=='empty_cost_table' and diag['cost_record_count']==0
        assert diag['execution_completed'] and diag['actual_skipped_windows']==0
        assert torch.equal(output[0],eva(image.float()/255)[0])
        decision=diag['cache_capture_decision']
        assert not decision['allowed'] and not decision['attempted'] and not decision['published']
        assert decision['rejection_reasons']==['empty_cost_table']
        assert diag['cache_publication_error'] is None
    assert diag['cache_miss_root_cause']=='empty_cost_table'
    assert rt.cache.images is None


@torch.no_grad()
def test_exact_count_missing_expands_actual_execution_not_original_plan(eva,context,tmp_path):
    from .test_source_linked_admission import cold,bundle_for
    rt,_=runtime_and_manager(eva,tmp_path)
    # Explicit capture is a unit fixture; no cost/quality approval is fabricated.
    from dataclasses import replace
    rt.config=replace(rt.config,cache_capture='always')
    image=torch.randn(2,3,40,40);cold(rt,image,context)
    x=block0_input(eva,image)
    records=[dict(key=cost_key(eva.blocks[0],x,0,model=eva,refresh_policy=rt.config.refresh_signature()),
        updated=1,block=0,numerical_verified=False,quality_certified=False,
        incremental_gpu_ms=.1,full_gpu_ms=1.,incremental_wall_ms=.1,full_wall_ms=1.)]
    candidate=tmp_path/'unapproved_unit_candidate.json'
    candidate.write_text(json.dumps(dict(schema_version=SCHEMA_VERSION,records=records,production_ready=False)))
    rt.costs=CostModel(str(candidate))
    ctx=dict(context,frame_id='1',timestamp=1.1)
    out,diag=rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    first,second=diag['stages'][:2]
    assert first['selector_planned_windows']==first['planned_windows']==6
    assert first['executed_windows']==18 and not first['cost_record_found']
    assert first['cost_key_matched'] and first['cost_rejection_reason']=='unprofiled_cost'
    assert not first['beneficial'] and first['cost_admission_applied']
    assert second['selector_planned_windows']==6 and second['planned_windows']==18
    assert second['executed_windows']==18 and all(second['selection_consistency'].values())
    assert diag['actual_skipped_windows']==0 and torch.equal(out[0],eva(image)[0])


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
        assert len(stage['mandatory_window_ids'])==sum(c['mandatory_windows'] for c in stage['per_camera'])


@torch.no_grad()
def test_cache_mismatch_cannot_hide_aborted_current_transaction(eva,context,tmp_path):
    from .test_source_linked_admission import bundle_for
    rt,manager=runtime_and_manager(eva,tmp_path)
    image=torch.randint(0,256,(2,3,40,40),dtype=torch.uint8)
    _,_,ctx=step(rt,manager,image,context,0)
    ctx=dict(ctx,scene_id='new-scene',frame_id='next',timestamp=100.1)
    bundle=bundle_for(rt,ctx);bundle.diagnostics['transaction_status']='aborted'
    _,diag=rt.evaluate_offline(image.float()/255,execution_context=ctx,computation_state=bundle)
    assert diag['actual_skipped_windows']==0 and diag['full_recompute_reason']=='scene_change'
    assert not diag['cache_saved']
    assert diag['cache_capture_decision']['source_validation_reason']=='m1_aborted_transaction'


@torch.no_grad()
@pytest.mark.parametrize('fault',['full_refresh','source','scene','version','aborted','time','epoch'])
def test_forced_or_invalid_sources_cannot_reuse_or_mask_reason(eva,context,tmp_path,fault):
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
    out,diag=rt.evaluate_offline(images.float()/255,execution_context=ctx,computation_state=bundle,variant='D')
    assert diag['full_recompute'] and diag['actual_skipped_windows']==0
    assert torch.equal(out[0],eva(images.float()/255)[0])
    assert diag['full_recompute_reason']
    if fault in ('source','aborted','time','epoch'):
        assert not diag['cache_capture_decision']['allowed']
        assert not diag['cache_saved'] and diag['cache_publication_error'] is None


@torch.no_grad()
@pytest.mark.parametrize('failure',['budget','nonfinite'])
def test_publication_failure_distinct_from_policy_refusal(eva,context,tmp_path,failure):
    rt,manager=runtime_and_manager(eva,tmp_path)
    image=torch.randint(0,256,(2,3,40,40),dtype=torch.uint8)
    if failure=='budget':rt.cache.budget_bytes=1
    else:
        original=rt.cache.commit
        def bad(images,outputs,*args,**kwargs):
            outputs=dict(outputs);outputs[0]=outputs[0]*float('nan')
            return original(images,outputs,*args,**kwargs)
        rt.cache.commit=bad
    _,d,_=step(rt,manager,image,context,0)
    decision=d['cache_capture_decision']
    assert decision['policy_allowed'] and not decision['allowed'] and decision['attempted'] and not decision['published']
    assert decision['reason']=='publication_failed' and decision['publication_failure_reason']
    assert d['cache_publication_error'] is not None and rt.cache.images is None
