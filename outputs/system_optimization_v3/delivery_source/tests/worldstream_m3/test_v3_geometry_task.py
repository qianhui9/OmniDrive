from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from worldstream.m1 import M1Config
from worldstream.m1.geometry import Projection,intersect_area,region_area,box_geometry
from worldstream.m1.bidirectional_geometry import occlusion_table,verify_anchor_geometry
from worldstream.m1.state_types import Frame,FullOutputs,PerceptionAnchor
from worldstream.m3.world_state import VerifiedWorldState
from worldstream.m3.post_perception import post_verify,local_query,direct_query


def test_vectorized_occlusion_preserves_original_scalar_rule():
    rng=np.random.RandomState(8);cfg=M1Config();views={}
    for i in range(70):
        xy=rng.uniform(0,30,4);box=tuple([min(xy[0],xy[2]),min(xy[1],xy[3]),max(xy[0],xy[2]),max(xy[1],xy[3])]) if i%7 else None
        views[str(i)]={'c':Projection(box,np.empty((0,2)),float(rng.uniform(0,30)),1,True,'ok')}
    result=occlusion_table(views,['c'],cfg)
    for aid,v in views.items():
        p=v['c'];expected=p.region is not None and any(other!=aid and q['c'].region is not None and q['c'].depth<p.depth-cfg.occlusion_depth_margin_m and intersect_area(p.region,q['c'].region)>cfg.occlusion_overlap_fraction*region_area(p.region) for other,q in views.items())
        assert result[aid]['c']==expected


def state_frame():
    record=dict(anchor_id='a',anchor_type='object',semantic_class='car',center_ego_m=[5.,2.,0.],source_frame='old',source_timestamp=1.,verified_timestamp=1.5)
    state=VerifiedWorldState('s','now',1.5,3,'sig',(record,),dict(total_history_anchors=2))
    frame=Frame('s','now',1.5,np.eye(4),{},'sig')
    context=dict(scene_id='s',frame_id='now',timestamp=1.5,state_epoch=3,state_execution_signature='sig')
    return state,frame,context


def test_post_verification_preserves_history_and_rejects_ambiguous_prediction():
    state,frame,context=state_frame()
    prediction=PerceptionAnchor('object','car',box_geometry([5.1,2,0],[2,2,2]),source_score=.8)
    actual=post_verify(state,frame,FullOutputs([prediction]))
    assert len(actual.records)==1 and 'post_perception' not in state.records[0]
    assert actual.records[0]['source_frame']=='old' and actual.records[0]['source_timestamp']==1.
    assert not post_verify(state,frame,FullOutputs([prediction,prediction])).records
    assert not post_verify(state,replace(frame,frame_id='other'),FullOutputs([prediction])).records
    assert not local_query(actual,'How many cars are in front?',context)['admitted']
    assert local_query(actual,'What is the distance to the car in front?',context)['admitted']
    assert not direct_query(actual,'What is the distance to the car in front?',context)['admitted']
    assert not local_query(actual,'What is the distance to the car in front?',dict(context,timestamp=2.))['admitted']


def test_dynamic_motion_never_certified_by_static_epipolar_tracks():
    anchor=SimpleNamespace(anchor_type='object',semantic_class='car',velocity_reliable=False)
    assert verify_anchor_geometry(anchor,None,None,None,None,M1Config())['reason']=='dynamic_geometry_requires_independent_motion'


def test_depth_gate_requires_observable_metric_correspondence():
    from worldstream.m1.state_types import CameraFrame,AnchorObservation,WorldAnchor
    from worldstream.m1.geometry import project_anchor,project_points
    cfg=M1Config(temporal_visibility=True,bidirectional_geometry=True,uncertainty_px=0.)
    k=np.array([[100.,0,64],[0,100.,64],[0,0,1.]])
    previous=CameraFrame('c',np.zeros((128,128),np.uint8),k,np.eye(4),1.)
    matrix=np.eye(4);matrix[0,3]=-1.
    current=replace(previous,world_to_camera=matrix,timestamp=1.5)
    anchor=WorldAnchor('a','s','object','traffic_cone',box_geometry([0,0,10],[2,2,2]),1.)
    old=project_anchor(anchor,previous,cfg);p=project_anchor(anchor,current,cfg)
    anchor.observation_support['c']=AnchorObservation('c',old.region,np.ones((16,16)),1.,1.)
    points=np.array([[-.5,-.5,10],[-.5,.5,10],[.5,-.5,10],[.5,.5,10]])
    a=project_points(points,k,previous.world_to_camera)[0];b=project_points(points,k,current.world_to_camera)[0]
    ev=SimpleNamespace(source_timestamp=1.,source_points=a,current_points=b,correspondence_valid=np.ones(4,bool),geometry_valid=np.ones(4,bool),supported=np.ones(4,bool),fb_error_px=np.zeros(4),patch_residual=np.zeros(4))
    assert verify_anchor_geometry(anchor,previous,current,ev,p,cfg)['passed']
    # Zero-baseline stationary observations cannot localize an erroneous depth.
    current=replace(current,world_to_camera=np.eye(4));ev.current_points=a
    assert not verify_anchor_geometry(anchor,previous,current,ev,old,cfg)['passed']
