"""Synthetic contract cases; natural-stream evidence lives in window_validate."""
import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from worldstream.m1.mask_codec import encode_mask
from worldstream.m1.state_types import ComputationStateBundle
from worldstream.m2 import IncrementalVisionRuntime, M2Config
from worldstream.m2.region_refresh import pooled_descriptors, cosine_drift, region_tokens
from worldstream.m2.dependency_planner import patch_updates, DependencyPlanner
from worldstream.m2.state_cache import StateCache
from .test_source_linked_admission import bundle_for


def region(**kw):
    row=dict(camera_id='front',track_id='real-track',bbox_xyxy=[1.,1.,7.,7.],
        class_id='car',state_signature=('object','car'),event_flags=[],
        visible=True,occluded=None,coordinate_space='preprocessed_image')
    row.update(kw);return row


def step(rt,image,context,index,regions=(),events=(),variant='D',edit=None):
    ctx=dict(context,frame_id=str(index),timestamp=1.+index*.01)
    b=bundle_for(rt,ctx) if rt.cache.images is not None else ComputationStateBundle(
        ctx['scene_id'],ctx['frame_id'],ctx['timestamp'],ctx['state_epoch'],ctx['execution_signature'],
        diagnostics=dict(full_refresh_hint=True))
    b.region_states=copy.deepcopy(list(regions));b.region_events=list(events)
    if edit:edit(b)
    return rt.evaluate_offline(image,variant=variant,execution_context=ctx,computation_state=b)


@pytest.fixture
def rt(eva):
    return IncrementalVisionRuntime(eva,dict(cache_capture='always',region_margin_px=0,ordinary_budget=0))


@torch.no_grad()
def test_stable_state_and_single_edge_pixel_are_not_mandatory(rt,context):
    image=torch.randn(2,3,40,40);step(rt,image,context,0,[region()])
    image[0,:,7,7]+=.1
    _,d=step(rt,image,context,1,[region(bbox_xyxy=[1.,1.,8.,7.])])
    assert d['region_refresh']['mandatory_windows']==0 and d['recomputed_windows']==0
    assert d['change_detection']=='wsgrr_v1'


@torch.no_grad()
@pytest.mark.parametrize('change',[dict(state_signature=('object','truck')),dict(event_flags=['state_jump']),dict(identity_link_qualified=False)])
def test_explicit_state_or_association_change_overrides_zero_budget(rt,context,change):
    image=torch.randn(2,3,40,40)
    step(rt,image,context,0,[region(identity_link_qualified=True)])
    _,d=step(rt,image,context,1,[region(**change)])
    assert 0<d['region_refresh']['mandatory_windows']<18
    assert d['region_refresh']['ordinary_windows']==0 and d['recomputed_windows']>0


@torch.no_grad()
def test_cross_window_move_covers_old_and_new_positions(rt,context):
    image=torch.randn(2,3,40,40);step(rt,image,context,0,[region()])
    _,d=step(rt,image,context,1,[region(bbox_xyxy=[18.,1.,24.,7.])])
    assert d['region_refresh']['mandatory_window_ids']==[0,1]
    assert d['stages'][0]['executed_window_ids']==d['stages'][1]['executed_window_ids']==[0,1]


@torch.no_grad()
def test_missing_is_not_disappearance_but_expiry_refreshes_old_region(rt,context):
    image=torch.randn(2,3,40,40);step(rt,image,context,0,[region()])
    _,missing=step(rt,image,context,1,[])
    assert missing['region_refresh']['mandatory_windows']==0 and len(rt.cache.region_states)==1
    _,expired=step(rt,image,context,2,[],[dict(track_id='real-track',event='state_expired')])
    assert expired['region_refresh']['mandatory_window_ids']==[0] and rt.cache.region_states==[]


@torch.no_grad()
def test_missing_track_retention_respects_real_m1_state_horizon(rt,context):
    image=torch.randn(2,3,40,40);step(rt,image,context,0,[region(valid_until=1.005)])
    _,d=step(rt,image,context,1,[])
    assert d['region_refresh']['mandatory_window_ids']==[0]
    assert rt.cache.region_states==[]


@torch.no_grad()
@pytest.mark.parametrize('event',['disappeared','reappeared','source_invalid'])
def test_explicit_lifecycle_events_refresh_affected_region(rt,context,event):
    image=torch.randn(2,3,40,40);step(rt,image,context,0,[region()])
    _,d=step(rt,image,context,1,[region(event_flags=[event])])
    assert d['region_refresh']['mandatory_window_ids']==[0]
    if event=='disappeared':assert rt.cache.region_states==[]


@torch.no_grad()
def test_occlusion_is_hypothesis_and_supported_reappearance_is_refresh(rt,context):
    image=torch.randn(2,3,40,40);step(rt,image,context,0,[region()])
    _,d=step(rt,image,context,1,[region(occluded=True,visible=None)])
    assert d['region_refresh']['mandatory_windows']==0
    _,d=step(rt,image,context,2,[region(occluded=None,visible=True)])
    assert d['region_refresh']['event_regions'][0]['events']==['reappeared']


@torch.no_grad()
def test_higher_drift_wins_camera_budget_and_descriptors_keep_computation_age(rt,context):
    rt.config=replace(rt.config,ordinary_budget=1,age_weight=0,uncertainty_weight=0,score_threshold=.01)
    image=torch.randn(2,3,40,40);step(rt,image,context,0)
    source=rt.cache.descriptors.clone()
    image[0,:,32:40,32:40]+=8
    _,d=step(rt,image,context,1)
    assert d['region_refresh']['ordinary_window_ids']==[8]
    assert torch.equal(rt.cache.descriptors[:8],source[:8])
    assert not torch.equal(rt.cache.descriptors[8],source[8])
    assert rt.cache.sources[0][0]['computation_frame_id']=='0'
    assert rt.cache.sources[0][8]['computation_frame_id']=='1'
    assert rt.cache.window_ages[0][0]==1 and rt.cache.window_ages[0][8]==0


@torch.no_grad()
def test_age_cap_forces_refresh_even_with_zero_budget(rt,context):
    rt.config=replace(rt.config,max_reuse_frames=2)
    image=torch.randn(2,3,40,40);step(rt,image,context,0)
    _,d=step(rt,image,context,1);assert d['recomputed_windows']==0
    _,d=step(rt,image,context,2)
    assert d['region_refresh']['mandatory_windows']==18 and d['recomputed_windows']==36
    assert all(t.eq(0).all() for t in rt.cache.window_ages.values())


@torch.no_grad()
def test_per_camera_budget_ties_and_event_overflow(rt,context):
    rt.config=replace(rt.config,ordinary_budget=2)
    image=torch.randn(2,3,40,40);step(rt,image,context,0)
    _,d=step(rt,image,context,1)
    assert d['region_refresh']['ordinary_window_ids']==[0,1,9,10]
    _,d=step(rt,image,context,2,[region(bbox_xyxy=[0,0,40,40],event_flags=['state_jump'])])
    assert d['region_refresh']['mandatory_windows']==9
    assert d['region_refresh']['ordinary_windows']==2


def test_padding_pooling_and_zero_descriptors():
    x=torch.ones(2,5,7,3)
    assert pooled_descriptors(x,4).eq(1).all()
    zero=torch.zeros(1,3)
    assert cosine_drift(zero,zero).item()==0
    assert cosine_drift(zero,torch.ones_like(zero)).item()==1


@torch.no_grad()
def test_local_unknown_has_priority_without_becoming_mandatory(rt,context):
    rt.config=replace(rt.config,ordinary_budget=1,age_weight=0)
    image=torch.randn(2,3,40,40);step(rt,image,context,0)
    def local(b):
        for cid,contract in b.refresh_contract.items():
            unknown=np.zeros((40,40),bool)
            if cid=='back':unknown[32:,32:]=True
            contract.update(unknown_mask=encode_mask(unknown),reuse_candidate_mask=encode_mask(~unknown))
    _,d=step(rt,image,context,1,edit=local)
    assert d['region_refresh']['mandatory_windows']==0
    assert d['region_refresh']['ordinary_window_ids']==[17]


@torch.no_grad()
def test_invalid_current_region_cannot_seed_a_cold_cache(rt,context):
    _,d=step(rt,torch.randn(2,3,40,40),context,0,[region(bbox_xyxy=[0,0,float('nan'),2])])
    assert d['full_update_reason']=='invalid_region_state' and not d['cache_saved']


@pytest.mark.parametrize('kernel,stride,padding,dilation',[(3,2,2,2),(5,3,2,1),(2,2,0,1)])
def test_box_mapping_matches_actual_convolution_with_multicamera_padding(kernel,stride,padding,dilation):
    conv=torch.nn.Conv2d(3,4,kernel,stride=stride,padding=padding,dilation=dilation)
    image_hw=(19,23);pixels=torch.zeros(2,*image_hw,dtype=torch.bool);pixels[1,5:11,7:15]=True
    expected=patch_updates(pixels,conv)
    actual=region_tokens([7,5,15,11],1,expected.shape,conv,image_hw)
    assert torch.equal(actual,expected) and not actual[0].any()


def test_selected_windows_close_across_different_window_sizes(eva):
    eva.blocks[1].window_size=6
    planner=DependencyPlanner(eva);selected=torch.zeros(18,dtype=torch.bool);selected[4]=True
    plans=planner.plan_windows(selected,(2,10,10),stop_after=3)
    assert plans[0].updated.sum()==1 and plans[1].updated.sum()==4
    assert plans[2].reason=='global_dependency' and plans[2].updated.tolist()==[True,False]


@torch.no_grad()
@pytest.mark.parametrize('mutation',['policy','region','ages','descriptor'])
def test_incompatible_policy_or_mutated_state_never_reuses(rt,context,mutation):
    image=torch.randn(2,3,40,40);step(rt,image,context,0,[region()])
    if mutation=='policy':rt.config=replace(rt.config,ordinary_budget=4)
    if mutation=='region':rt.cache.region_states[0]['class_id']='forged'
    if mutation=='ages':rt.cache.window_ages[0].add_(10)
    if mutation=='descriptor':rt.cache.descriptors.add_(1)
    out,d=step(rt,image,context,1)
    assert d['actual_skipped_windows']==0 and torch.equal(out[0],rt.model(image)[0])


def test_snapshot_failure_is_atomic_for_all_new_fields():
    cache=StateCache(100000);x=torch.ones(1,1,1,4);image=torch.ones(1,3,4,4)
    cache.commit(image,{0:x},dict(frame_id='0',camera_ids=['front']),'sig',block0_input=x,
        region_states=[region()],window_ages={0:torch.zeros(1,dtype=torch.long)},descriptors=x.reshape(1,4))
    previous=cache.__dict__.copy()
    class Broken:
        def __deepcopy__(self,memo):raise RuntimeError('injected region copy failure')
    with pytest.raises(RuntimeError,match='injected'):
        cache.commit(image,{0:x*2},dict(frame_id='1',camera_ids=['front']),'next',block0_input=x*2,
            region_states=[Broken()],window_ages={0:torch.ones(1,dtype=torch.long)},descriptors=x.reshape(1,4)*2)
    assert all(cache.__dict__[k] is v for k,v in previous.items())


@torch.no_grad()
def test_offline_candidate_uses_same_executor_and_cannot_enable_production(rt,context,monkeypatch):
    rt.config=replace(rt.config,ordinary_budget=2)
    image=torch.randn(2,3,40,40);step(rt,image,context,0)
    calls=[];execute=rt.executor.execute
    def observe(block,x,cache,ids):calls.append(len(ids));return execute(block,x,cache,ids)
    monkeypatch.setattr(rt.executor,'execute',observe)
    _,d=step(rt,image,context,1)
    assert calls==[4,4] and d['offline_experiment'] and not d['task_quality_certified']
    assert not rt.costs.production_ready and not rt.costs.records
    ctx=dict(context,frame_id='2',timestamp=1.02)
    _,d=rt.forward(image,execution_context=ctx,computation_state=bundle_for(rt,ctx))
    assert not d['offline_experiment'] and d['actual_skipped_windows']==0
    assert d['full_update_reason']=='execution_signature_mismatch'


@torch.no_grad()
def test_m1_exports_real_projection_and_qualified_fields(eva,context):
    from worldstream.m1 import WorldStateManager,M1Config,FullOutputs,PerceptionAnchor
    from worldstream.m1.geometry import box_geometry
    from .test_live_m1_snapshot import frame_for
    manager=WorldStateManager(M1Config(source_linked_state=True,temporal_visibility=True,local_support=True))
    image=torch.randint(0,255,(2,3,40,40),dtype=torch.uint8)
    f=frame_for(image,context);r=manager.precheck(f)
    manager.commit(f,FullOutputs([PerceptionAnchor('object','car',box_geometry(np.array([0,0,10]),np.array([4,4,4])))]),r)
    ctx=dict(context,frame_id='1',timestamp=1.1);f=frame_for(image,ctx);r=manager.precheck(f)
    ctx['state_epoch']=r.diagnostics['state_epoch'];snapshot=manager.export_m2_inputs(ctx)
    regions=snapshot.computation_state.region_states
    assert regions and all(row['track_id'] in manager.store.anchors for row in regions)
    assert all(row['velocity'] is None and not row['identity_link_qualified'] for row in regions)
    assert all(row['last_observed_time']==1. and row['state_signature']==('object','car') for row in regions)
    assert not any(row['event_flags'] for row in regions)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@torch.no_grad()
def test_cuda_wsgrr_runtime_preserves_shapes_sources_and_approximate_output(eva,context):
    rt=IncrementalVisionRuntime(eva.cuda(),dict(cache_capture='always',ordinary_budget=2))
    image=torch.randn(2,3,40,40,device='cuda');step(rt,image,context,0)
    image+=.02
    out,d=step(rt,image,context,1)
    assert [s['executed_windows'] for s in d['stages'][:2]]==[4,4]
    assert torch.isfinite(out[0]).all() and out[0].dtype==torch.float32
    assert out[0].stride()==rt.model(image)[0].stride()
    assert (out[0]-rt.model(image)[0]).abs().mean()>0
    assert rt.cache.descriptors.is_cuda and rt.cache.window_ages[0].is_cuda
    assert d['actual_skipped_windows']==28 and not d['window_reuse_quality_certified']


@torch.no_grad()
def test_no_m1_region_is_offline_only_and_retains_safety_expiry(rt,context):
    rt.config=replace(rt.config,ordinary_budget=1,max_reuse_frames=2)
    image=torch.randn(2,3,40,40)
    step(rt,image,context,0,[region()],variant='NoM1Region')
    _,d=step(rt,image,context,1,[region(event_flags=['state_jump'])],variant='NoM1Region')
    assert d['region_refresh']['mandatory_windows']==0 and d['region_refresh']['ordinary_windows']==2
    _,d=step(rt,image,context,2,[region(event_flags=['state_jump'])],variant='NoM1Region')
    assert d['region_refresh']['mandatory_windows']==16
    with pytest.raises(ValueError):M2Config(mode='NoM1Region')


@torch.no_grad()
def test_compact_m1_export_preserves_contract_and_owns_metadata(context):
    from worldstream.m1 import WorldStateManager,M1Config,FullOutputs
    from .test_live_m1_snapshot import frame_for
    manager=WorldStateManager(M1Config(source_linked_state=True,temporal_visibility=True,local_support=True))
    image=torch.randint(0,255,(2,3,40,40),dtype=torch.uint8)
    frame=frame_for(image,context);r=manager.precheck(frame);manager.commit(frame,FullOutputs([]),r)
    ctx=dict(context,frame_id='1',timestamp=1.1);r=manager.precheck(frame_for(image,ctx));ctx['state_epoch']=r.diagnostics['state_epoch']
    compact=manager.export_m2_inputs(ctx,computation_only=True);full=manager.export_m2_inputs(ctx)
    assert compact.computation_state.refresh_contract==full.computation_state.refresh_contract
    assert compact.computation_state.region_states==full.computation_state.region_states
    assert compact.diagnostics['snapshot_scope']=='m2_computation_only'
    compact.computation_state.refresh_contract.clear()
    assert manager.last_result.computation_state.refresh_contract
    with pytest.raises(RuntimeError):manager.export_m2_inputs(dict(ctx,frame_id='forged'),computation_only=True)


@torch.no_grad()
def test_compact_executor_only_projects_selected_windows_and_valid_mlp_tokens(eva,monkeypatch):
    from worldstream.m2.window_executor import WindowExecutor
    from worldstream.m2.packing import partition
    import torch.nn.functional as F
    block=eva.blocks[0];x=torch.randn(2,10,10,32)
    cache=partition(block(x),4)[0];seen=[];native=F.linear
    def observe(input,weight,bias=None):
        if weight is block.attn.q_proj.weight:seen.append(('qkv',tuple(input.shape)))
        if weight is block.mlp.w1.weight:seen.append(('mlp',tuple(input.shape)))
        return native(input,weight,bias)
    monkeypatch.setattr(F,'linear',observe)
    output,packed=WindowExecutor().execute(block,x,cache,torch.tensor([8,17]))
    assert ('qkv',(2,16,32)) in seen and ('mlp',(8,32)) in seen
    assert torch.allclose(output,block(x),atol=3e-6,rtol=3e-6)
    assert packed[[8,17],2:].eq(0).all()


@torch.no_grad()
def test_matched_offline_control_never_overrides_safety(rt,context):
    rt.config=replace(rt.config,max_reuse_frames=2)
    image=torch.randn(2,3,40,40)
    def run(i):
        ctx=dict(context,frame_id=str(i),timestamp=1+i*.01)
        b=bundle_for(rt,ctx) if i else ComputationStateBundle(ctx['scene_id'],ctx['frame_id'],ctx['timestamp'],ctx['state_epoch'],ctx['execution_signature'],diagnostics={'full_refresh_hint':True})
        return rt.evaluate_offline(image,variant='NoM1RegionMatched',execution_context=ctx,computation_state=b,window_targets=[2,4])[1]
    run(0);d=run(1)
    assert d['stages'][0]['executed_windows']==6
    d=run(2)
    assert d['region_refresh']['mandatory_windows']==12
    assert d['stages'][0]['executed_windows']>=12
