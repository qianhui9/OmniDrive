"""Image-local risk must survive failed world association without authorizing reuse."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from worldstream.m1.state_types import ComputationStateBundle
from worldstream.m2 import M2Config
from worldstream.m2.region_refresh import select_regions,pooled_descriptors
from worldstream.m2.dependency_planner import patch_updates
from worldstream.m2.packing import window_mask
from .test_wsgrr import step,region,rt


def risk(bundle,camera='front',box=(1,1,7,7),mandatory=True,scope='region'):
    return dict(camera_id=camera,source_track_id='historical-hypothesis',
        bbox_xyxy=list(box) if box is not None else None,scope=scope,mandatory=mandatory,
        reason='uncertainty_exceeds_search',frame_id=bundle.frame_id,timestamp=bundle.timestamp,
        coordinate_space='preprocessed_image',world_identity_authorized=False)


@torch.no_grad()
def test_geometry_invalid_local_box_overrides_zero_budget_without_identity(rt,context):
    x=torch.randn(2,3,40,40);step(rt,x,context,0)
    _,d=step(rt,x,context,1,edit=lambda b:b.uncertainty_regions.append(risk(b)))
    assert d['region_refresh']['mandatory_window_ids']==[0]
    assert d['region_refresh']['ordinary_windows']==0
    assert d['stages'][0]['executed_window_ids']==d['stages'][1]['executed_window_ids']==[0]
    assert d['region_refresh']['uncertainty_regions'][0]['world_identity_authorized'] is False
    assert rt.cache.region_states==[]
    assert rt.cache.window_ages[0][0]==0 and rt.cache.window_ages[0][1:].eq(1).all()
    assert rt.cache.sources[0][0]['computation_frame_id']=='1'
    assert rt.cache.sources[0][1]['computation_frame_id']=='0'


@torch.no_grad()
def test_local_observation_risk_survives_threshold_and_keeps_camera_budget(rt,context):
    rt.config=replace(rt.config,ordinary_budget=1,score_threshold=10.)
    x=torch.randn(2,3,40,40);step(rt,x,context,0)
    def edit(b):
        b.uncertainty_regions.extend([risk(b,box=(33,33,34,34),mandatory=False),
            risk(b,'back',(17,1,18,2),mandatory=False)])
    _,d=step(rt,x,context,1,edit=edit)
    assert d['region_refresh']['mandatory_windows']==0
    assert d['region_refresh']['ordinary_window_ids']==[8,10]
    assert d['region_refresh']['uncertainty'][8]==d['region_refresh']['uncertainty'][10]==1.


@torch.no_grad()
def test_unknown_extent_and_lost_projection_only_revoke_actual_scope(rt,context):
    x=torch.randn(2,3,40,40);step(rt,x,context,0,[region()])
    def edit(b):b.uncertainty_regions.append(risk(b,'back',None,scope='camera'))
    _,d=step(rt,x,context,1,events=[dict(track_id='real-track',camera_id='front',event='association_invalid')],edit=edit)
    assert d['region_refresh']['mandatory_window_ids']==[0,*range(9,18)]
    assert rt.cache.region_states==[]
    _,next_frame=step(rt,x,context,2)
    assert next_frame['region_refresh']['mandatory_windows']==0


@torch.no_grad()
def test_invalid_moved_geometry_refreshes_old_and_current_local_scopes(rt,context):
    x=torch.randn(2,3,40,40);step(rt,x,context,0,[region()])
    def edit(b):b.uncertainty_regions.append(risk(b,box=(33,33,39,39)))
    _,d=step(rt,x,context,1,events=[dict(track_id='real-track',camera_id='front',
        event='association_invalid',reason='current_geometry_invalid')],edit=edit)
    assert d['region_refresh']['mandatory_window_ids']==[0,8]
    assert d['stages'][0]['executed_window_ids']==[0,8]
    assert rt.cache.region_states==[]


@torch.no_grad()
@pytest.mark.parametrize('fault',['frame','coordinate','identity','nan'])
def test_uncertainty_context_cannot_bypass_source_validation(rt,context,fault):
    x=torch.randn(2,3,40,40);step(rt,x,context,0)
    def edit(b):
        row=risk(b)
        if fault=='frame':row['frame_id']='wrong'
        if fault=='coordinate':row['coordinate_space']='raw_image'
        if fault=='identity':row['world_identity_authorized']=True
        if fault=='nan':row['bbox_xyxy'][0]=float('nan')
        b.uncertainty_regions.append(row)
    with pytest.raises(ValueError,match='invalid_region_state'):
        step(rt,x,context,1,edit=edit)


@pytest.mark.parametrize('kernel,stride,padding,dilation',[(3,2,2,2),(5,3,2,1),(2,2,0,1)])
def test_uncertain_box_windows_equal_actual_conv_lattice(kernel,stride,padding,dilation):
    from types import SimpleNamespace
    conv=torch.nn.Conv2d(3,4,kernel,stride=stride,padding=padding,dilation=dilation)
    pixels=torch.zeros(2,19,23,dtype=torch.bool);pixels[1,5:11,7:15]=True
    tokens=patch_updates(pixels,conv);x=torch.zeros(*tokens.shape,4);size=4
    descriptors=pooled_descriptors(x,size)
    cache=SimpleNamespace(descriptors=descriptors,window_ages={0:torch.zeros(len(descriptors),dtype=torch.long)},region_states=[])
    bundle=ComputationStateBundle('scene','current',1.,0,'sig')
    bundle.uncertainty_regions=[risk(bundle,'back',(7,5,15,11))]
    selected,*_=select_regions(x,cache,bundle,dict(camera_ids=['front','back']),conv,size,
        M2Config(ordinary_budget=0,region_margin_px=0),np.zeros((2,19,23)),torch.zeros_like(tokens))
    assert torch.equal(selected,window_mask(tokens,size))


@torch.no_grad()
def test_m1_exports_image_scope_without_promoting_failed_geometry(context):
    from worldstream.m1 import WorldStateManager,M1Config,FullOutputs,PerceptionAnchor
    from worldstream.m1.geometry import box_geometry
    from worldstream.m1.computation_state import build_bundle
    from .test_live_m1_snapshot import frame_for
    manager=WorldStateManager(M1Config(source_linked_state=True,temporal_visibility=True,local_support=True))
    image=torch.randint(0,255,(2,3,40,40),dtype=torch.uint8)
    frame=frame_for(image,context);r=manager.precheck(frame)
    manager.commit(frame,FullOutputs([PerceptionAnchor('object','car',box_geometry(np.array([0,0,10]),np.array([4,4,4])))]),r)
    ctx=dict(context,frame_id='1',timestamp=1.1);frame=frame_for(image,ctx);r=manager.precheck(frame)
    aid=next(iter(manager.store.anchors));view=r.diagnostics['anchors'][aid]['cameras']['front']
    assert view['predicted_region'] is not None
    view.update(geometry_valid=False,valid=False,reason='uncertainty_exceeds_search')
    r.computation_state=build_bundle(manager,frame,r)
    assert not any(row['camera_id']=='front' for row in r.computation_state.region_states)
    rows=[row for row in r.computation_state.uncertainty_regions if row['camera_id']=='front']
    assert len(rows)==1 and rows[0]['mandatory'] and not rows[0]['world_identity_authorized']
    ctx['state_epoch']=r.diagnostics['state_epoch']
    exported=manager.export_m2_inputs(ctx,computation_only=True)
    with pytest.raises(TypeError,match='read-only'):
        exported.computation_state.uncertainty_regions[0]['bbox_xyxy'][0]=-100
    assert r.computation_state.uncertainty_regions[0]['bbox_xyxy'][0]>=0
    view.update(predicted_region=None,projected_points=np.array([[1.,2.],[np.nan,np.nan]]))
    unbounded=build_bundle(manager,frame,r)
    assert not any(row['camera_id']=='front' for row in unbounded.uncertainty_regions)
    assert dict(track_id=aid,camera_id='front',event='association_invalid',
        reason='current_projection_unavailable') in unbounded.region_events
    view['projected_points']=np.full((8,2),np.nan)
    invisible=build_bundle(manager,frame,r)
    assert not any(row['scope']=='camera' and row['camera_id']=='front' for row in invisible.uncertainty_regions)
    # Clipping can return an empty rectangle instead of None when a previously
    # visible object leaves the camera; it must revoke the old footprint too.
    for invalid_region in ([0.,0.,0.,0.],[45.,45.,50.,50.],[np.nan,0.,5.,5.]):
        view.update(predicted_region=invalid_region,geometry_valid=False)
        invalid=build_bundle(manager,frame,r)
        assert not any(row['camera_id']=='front' for row in invalid.uncertainty_regions)
        assert dict(track_id=aid,camera_id='front',event='association_invalid',
            reason='current_projection_unavailable') in invalid.region_events
