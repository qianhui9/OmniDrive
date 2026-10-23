import numpy as np
import pytest
from tools.worldstream_m2.task_diagnostics import (check_coordinates,line_distances,
    gated_assignment,diagnose_map,clean_line)


def line(count=11,offset=(0,0,0)):
    t=np.linspace(0,10,count)
    return np.c_[t,t*0,t*0]+offset


def test_wrong_coordinate_frames_units_or_origins_are_rejected():
    contract=dict(frame='ego',units='m',dimensions=3)
    for changed in (dict(frame='world',units='m',dimensions=3),
                    dict(frame='ego',units='cm',dimensions=3),dict(frame='ego',units='m',dimensions=2)):
        with pytest.raises(ValueError,match='coordinate'):check_coordinates(contract,changed)


def test_reversed_point_order_is_same_geometry():
    r=line_distances(line(),line()[::-1])
    assert r['reverse_m']<1e-12 and r['symmetric_m']<1e-12 and r['forward_m']>0


def test_different_sampling_density_and_point_count_are_invariant():
    a=line(3);b=np.c_[np.array([0,.01,.02,.03,1,2,10]),np.zeros(7),np.zeros(7)]
    r=line_distances(a,b)
    assert r['valid'] and r['symmetric_m']<1e-12 and r['forward_m']<1e-12
    assert r['symmetric_m']==line_distances(b,a)['symmetric_m']


@pytest.mark.parametrize('value',[np.nan,np.inf])
def test_invalid_points_are_not_silently_dropped(value):
    a=line();a[4,1]=value
    assert not line_distances(a,line())['valid']
    mask=np.ones(11,bool);mask[4]=False
    assert line_distances(a,line(),mask)['symmetric_m']==0


def test_zero_length_and_invalid_mask_are_reported():
    assert clean_line(np.ones((3,3)))[1]=='zero_length_line'
    assert clean_line(line(),np.zeros(2,bool))[1]=='invalid_point_mask'


def test_class_incompatible_and_remote_lines_remain_unmatched():
    matches,a,b=gated_assignment([[.01,50],[50,.02]],[[False,True],[True,False]],3)
    assert matches==[] and a==b==[0,1]


def test_real_prediction_shift_is_measured_symmetrically():
    a,b=line(),line(offset=(0,.6253,0))
    r=line_distances(a,b)
    assert r['symmetric_m']==pytest.approx(.6253)
    assert r['symmetric_m']==line_distances(b,a)['symmetric_m']


def test_official_map_audit_preserves_matcher_decisions_and_raw_geometry():
    from tools.worldstream_m2.official_map_audit import audit_frame
    import importlib
    official=importlib.import_module('openlanev2.centerline.evaluation.evaluate')
    gt=dict(lane_centerline=[dict(id='a',points=line()),dict(id='b',points=line(offset=(0,6,0)))],
        topology_lclc=np.array([[0,1],[0,0]]))
    predictions=np.stack([line(offset=(0,.5,0)),line(offset=(0,.5,0)),line(offset=(0,50,0))])
    scores=np.array([[.4],[.9],[.8]])
    result=audit_frame(('frame','scene',gt,predictions,scores))
    for row in result['thresholds']:
        assert row['tp']==1 and row['fp']==2 and row['fn']==1
        assert row['matches'][0]['prediction_index']==1
        assert row['matches'][0]['raw_frechet_m']==pytest.approx(.5)
        assert row['matches'][0]['official_relaxed_frechet_m']==pytest.approx(.5)
        assert row['unmatched_predictions'][0]['reason']=='nearest_gt_already_covered'
        assert row['unmatched_predictions'][1]['nearest_official_relaxed_frechet_m'] is None
        assert row['predicted_edges']==0 and row['ground_truth_edges']==1
    empty=dict(lane_centerline=[],topology_lclc=np.zeros((0,0)))
    row=audit_frame(('frame','scene',empty,predictions,scores))['thresholds'][0]
    assert row['tp']==0 and row['fp']==3 and row['fn']==0
    assert all(v['reason']=='no_ground_truth' for v in row['unmatched_predictions'])
    assert [v['threshold_m'] for v in result['thresholds']]==official.THRESHOLDS_FRECHET
    boundary=audit_frame(('frame','scene',gt,np.stack([line(offset=(0,1,0))]),np.ones((1,1))))
    assert [v['tp'] for v in boundary['thresholds']]==[0,1,1]


def test_gated_assignment_maximizes_valid_matches_before_cost():
    pairs,a,b=gated_assignment([[0,2.9],[2.9,20]],np.ones((2,2),bool),3)
    assert set(pairs)=={(0,1),(1,0)} and not a and not b


def test_legacy_extreme_is_retained_next_to_new_unmatched_result():
    a=np.stack([line(),line(offset=(0,50,0))]);b=np.stack([line(offset=(0,.1,0)),line(offset=(0,110,0))])
    r=diagnose_map((a,np.ones((2,1))),(b,np.ones((2,1))),'frame')
    assert max(row['legacy_cost_m'] for row in r['legacy'])==60
    assert len(r['symmetric_gated_matches'])==1
    assert r['unmatched_native'][0]['index']==1 and r['unmatched_candidate'][0]['index']==1


def test_map_classes_are_compared_before_gating():
    r=diagnose_map((np.stack([line()]),np.array([[.9,.1]])),
        (np.stack([line()]),np.array([[.1,.9]])),'frame')
    assert r['legacy'][0]['invalid_reason']=='class_incompatible'
    assert not r['symmetric_gated_matches']


def test_full_map_comparison_reports_ragged_and_invalid_lines():
    invalid=line(4);invalid[2,0]=np.nan
    r=diagnose_map(([line(3),invalid],np.ones((2,1))),
        ([line(7)],np.ones((1,1))),'frame')
    assert r['legacy']==[] and r['legacy_unavailable_reason']=='different_point_counts_or_shapes'
    assert len(r['symmetric_gated_matches'])==1
    assert r['unmatched_native'][0]['index']==1
    assert r['unmatched_native'][0]['reason']=='no_finite_geometry'


def test_parallel_official_map_evaluation_is_identical_to_original():
    import copy
    from tools.worldstream_m2.map_eval import evaluate_map
    from tools.worldstream_m123.benchmark.metrics import finite_metrics
    gt={};pred=dict(method='unit-test',authors=[],**{'e-mail':'','institution / company':'','country / region':'China'},results={})
    for index in range(2):
        key=('val','scene',str(index));points=line()+[-4,index,0]
        annotation=dict(lane_centerline=[dict(id=0,points=points)],traffic_element=[],
            topology_lclc=np.zeros((1,1)),topology_lcte=np.zeros((1,0)))
        gt[key]=dict(annotation=annotation)
        prediction=copy.deepcopy(annotation);prediction['lane_centerline'][0]['confidence']=.8
        pred['results'][key]=dict(predictions=prediction)
    original=finite_metrics(evaluate_map(copy.deepcopy(gt),copy.deepcopy(pred),workers=0))[0]
    parallel=finite_metrics(evaluate_map(copy.deepcopy(gt),copy.deepcopy(pred),workers=2))[0]
    assert parallel==original


def test_official_geometry_cache_binds_every_input_dimension():
    from tools.worldstream_m2.map_eval import geometry_key
    a,b=[line()],[line()];base=geometry_key('distance',a,b)
    assert base!=geometry_key('distance',a,[line(offset=(0,1,0))])
    assert base!=geometry_key('distance',a,b,relax=True)
    assert base!=geometry_key('distance',a,b,mask=np.ones((1,1),bool))


def test_separate_map_and_detection_extremes_use_their_own_frame_pose(monkeypatch,tmp_path):
    import builtins,io,json,pickle
    from tools.worldstream_m2.task_diagnostics import trace_runtime
    from worldstream.m1 import geometry
    camera='CAM_FRONT';rot=[1,0,0,0]
    metadata=dict(intrinsics=[np.eye(3).tolist()],shapes={camera:[100,100,3]})
    audits=[dict(precheck=dict(frame_id=frame,computation_state=dict(region_states=[]),
        diagnostics=dict(execution_context=metadata)),committed_state=dict(anchors=[])) for frame in ['map','det']]
    (tmp_path/'m1_state.jsonl').write_text('\n'.join(map(json.dumps,audits)))
    (tmp_path/'effective_config.json').write_text(json.dumps(dict(config=dict(class_names=['car']))))
    rows=[]
    for frame in ['map','det']:
        rt=dict(region_refresh=dict(ages=[1],drift=[0],scores=[.3]),cache_context=dict(camera_ids=[camera]),
            prefix_dependency_footprints={'0':[dict(input_region=[0,0,10,10])]},
            stages=[dict(executed_window_ids=[0] if frame=='map' else [])])
        rows.append(dict(frame_id=frame,m2=rt))
    (tmp_path/'per_frame_metrics.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    cam=dict(ego2global_rotation=rot,ego2global_translation=[0,0,0],sensor2ego_rotation=rot,sensor2ego_translation=[0,0,0])
    infos=[dict(token=frame,ego2global_rotation=rot,ego2global_translation=[x,0,0],cams={camera:cam}) for frame,x in [('map',10),('det',100)]]
    original=builtins.open
    def opened(path,*args,**kwargs):
        if str(path)=='data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl':return io.BytesIO(pickle.dumps(dict(infos=infos)))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',opened)
    captured=[]
    def projected(points,*args):
        captured.append(points.copy());return np.ones((len(points),2)),np.ones(len(points)),np.ones(len(points),bool)
    monkeypatch.setattr(geometry,'project_points',projected)
    summary=dict(worst_detection=dict(frame_id='det',native_center=[0,0,0],class_id=0),
        worst_map=dict(frame_id='map',native=dict(points=line(3).tolist())))
    result=trace_runtime(tmp_path,summary)
    assert result['detector_native_world_center']==[100,0,0]
    assert captured[0][0].tolist()==[10,0,0]
    assert result['map_frame_id']=='map'
    assert result['map_native_extreme_projections'][0]['windows'][0]['executed']


def test_official_detection_audit_uses_sdk_xy_threshold_and_no_gt_scope():
    import sys
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.utils import center_distance
    from nuscenes.eval.detection.data_classes import DetectionBox
    from tools.worldstream_m2.official_detection_audit import sdk_matches
    def box(x,score,z=0):
        return DetectionBox(sample_token='frame',translation=(x,0,z),size=(2,2,2),rotation=(1,0,0,0),
            velocity=(0,0),detection_name='car',detection_score=float(score),attribute_name='vehicle.parked')
    gt=EvalBoxes();gt.add_boxes('frame',[box(0,-1),box(10,-1)])
    pred=EvalBoxes();pred.add_boxes('frame',[box(.1,.9,z=100),box(12,.8)])
    before=sys.getprofile();result=sdk_matches(gt,pred,'car',center_distance,2)
    assert sys.getprofile() is before
    assert len(result['matches'])==len(result['unmatched_predictions'])==1
    assert result['matches'][0][2]==pytest.approx(.1)
    assert result['unmatched_predictions'][0].translation[0]==12
    assert not sdk_matches(gt,pred,'pedestrian',center_distance,2)['assessed']


def test_numeric_qa_leading_decimal_does_not_change_legacy_metric():
    from tools.worldstream_m2.followup_summary import numeric_diagnostic
    from tools.worldstream_m123.benchmark.metrics import answer_quality
    answer='The car is at (-.2, +2.05), distance 1e1 meters.'
    reference='The car is at (-0.2, +2.05), distance 10 meters.'
    old=answer_quality(answer,reference)
    new=numeric_diagnostic(answer,reference)
    assert old['numeric_mae']!=0
    assert new['answer_numbers']==[-.2,2.05,10]
    assert new['numeric_mae']==0
    assert numeric_diagnostic('two cars at (1, 2)','car at (1, 2, 3)')['numeric_mae'] is None
    assert not numeric_diagnostic('1e999','1')['valid']
