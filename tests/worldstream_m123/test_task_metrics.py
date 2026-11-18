import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.worldstream_m123.task_metrics import (
    parse_trajectory, performance, fact_summary, planning_summary, score_planning)
from tools.worldstream_m123.task_planning import footprint_pixels, drivable_violations, PlanningBackend
from tools.worldstream_m123.r0_task_evaluation import facts_questions, selected_scenes, render_report


def answer(points):
    return '[PT, '+', '.join('(%s, %s)' % tuple(p) for p in points)+']'


def info(mask=None):
    return dict(gt_planning=np.zeros((1,6,3)),
                gt_planning_mask=np.ones((1,6,2)) if mask is None else mask)


def safe_geometry(data, pred, gt):
    return dict(collision=[False]*len(pred), raw_collision=[False]*len(pred),
                road_boundary=[False]*len(pred), out_of_grid=[False]*len(pred), initially_offroad=False)


def metric():
    return SimpleNamespace(H=4.084, W=1.85, bev_start_position=np.array([-49.95,-49.95]),
                           bev_resolution=np.array([.1,.1]), bev_dimension=np.array([1000,1000]))


def test_parser_accepts_integer_signed_decimal_and_exponent():
    text='Trajectory: [PT, (1, -2), (+3.0, .5), (-.4, 1e-2), (0, 0), (2., +3), (4, -5)]'
    result=parse_trajectory(text)
    assert result.shape == (6,2)
    assert result[2] == pytest.approx([-.4,.01])


@pytest.mark.parametrize('text', [
    'I cannot plan', '[PT, (0,0)]', '[PT, (0,0), nan]',
    answer([[1e300,0]]*6).replace('1e+300','1e999'),
    answer([[0,0]]*6)+answer([[1,1]]*6),
    answer([[0,0]]*6).replace('(0, 0)', '(0, 0, 0)',1),
])
def test_parser_refuses_invalid_outputs(text):
    with pytest.raises(ValueError):
        parse_trajectory(text)


def test_performance_uses_total_wall_time_and_frame_not_question_denominator():
    result=performance([dict(e2e_ms=100,questions=['a','b']),dict(e2e_ms=300,questions=['c'])])
    assert result['mean_e2e_ms']==200
    assert result['p95_e2e_ms']==pytest.approx(290)
    assert result['FPS']==5 and result['QPS']==7.5
    assert performance([])['FPS'] is None
    with pytest.raises(ValueError):performance([dict(e2e_ms=0,questions=[])])


def test_fact_accuracy_keeps_unknown_and_fractional_scores_explicit():
    events=[dict(group='vehicle_category',quality=dict(known=True,score=1)),
            dict(group='other',quality=dict(known=True,score=.5)),
            dict(group='pedestrian_position',quality=dict(known=False,score=None,reason='ambiguous'))]
    result=fact_summary(events)
    assert result['overall']['scorable_fact_accuracy']==.75
    assert result['overall']['scored']==2 and result['overall']['unscored']==1
    assert result['groups']['pedestrian_position']['scorable_fact_accuracy'] is None
    assert result['groups']['other']['scorable_fact_accuracy_percent']==50


def test_horizon_ade_is_prefix_mean_not_endpoint_error():
    points=np.column_stack([np.arange(1,7),np.zeros(6)])
    result=score_planning(answer(points),info(),safe_geometry)
    for s,ade,fde in [(1,1.5,2),(2,2.5,4),(3,3.5,6)]:
        assert result['horizons'][str(s)+'s']['ADE_m']==ade
        assert result['horizons'][str(s)+'s']['FDE_m']==fde


def test_mask_requires_both_xy_coordinates_at_each_horizon():
    mask=np.ones((1,6,2));mask[0,2,1]=0
    calls=[]
    def backend(data,pred,gt):
        calls.append(len(gt));return safe_geometry(data,pred,gt)
    result=score_planning(answer([[0,0]]*6),info(mask),backend)
    assert calls==[2]
    assert result['horizons']['1s']['GT_valid']
    assert not result['horizons']['2s']['GT_valid']
    assert not result['horizons']['3s']['GT_valid']


def test_invalid_prediction_excluded_from_rates_but_counted_as_failure():
    good=score_planning(answer([[0,0]]*6),info(),safe_geometry)
    bad=score_planning('invalid',info(),lambda *a:pytest.fail('Must not call geometry'))
    result=planning_summary([good,bad])['horizons']['3s']
    assert result['GT_valid_frames']==2 and result['invalid_predictions']==1
    assert result['prediction_coverage']==.5 and result['collision_denominator']==1
    assert result['collision_rate_percent']==0
    assert planning_summary([bad])['horizons']['1s']['collision_rate_percent'] is None


def test_geometry_failures_have_separate_quality_denominators():
    def backend(*args):
        return dict(collision=[True]*6, raw_collision=[True]*6,road_boundary=[True]*6,
                    out_of_grid=[False,False,True,True,True,True],initially_offroad=True)
    result=planning_summary([score_planning(answer([[0,0]]*6),info(),backend)])['horizons']
    assert result['1s']['collision_rate_percent']==100
    assert result['2s']['collision_rate_percent'] is None
    assert result['1s']['road_boundary_rate_percent'] is None
    assert result['3s']['initially_offroad_excluded']==1


def test_rotated_ego_footprint_and_out_of_grid_are_not_silently_clipped():
    m=metric();straight,_=footprint_pixels((0,0),0,m);turned,_=footprint_pixels((0,0),np.pi/2,m)
    y,x=np.where(straight);yy,xx=np.where(turned)
    assert np.ptp(x)>np.ptp(y) and np.ptp(yy)>np.ptp(xx)
    result=drivable_violations(np.array([[0.,0.],[60.,0.]]),np.ones((1000,1000)),m)
    assert result['road_boundary']==[False,True] and result['out_of_grid']==[False,True]
    assert not result['initially_offroad']


def test_footprint_checks_body_not_just_center_point():
    m=metric();road=np.ones((1000,1000));road[:,530:]=0
    # Current ego rectangle reaches beyond +3m even though its origin is on road.
    result=drivable_violations(np.array([[0.,0.]]),road,m)
    assert result['initially_offroad'] and result['road_boundary']==[True]


def test_real_existing_collision_kernel_is_wired_to_new_backend(monkeypatch):
    import torch
    from evaluation.planning_utils import PlanningMetric
    backend=PlanningBackend.__new__(PlanningBackend)
    m=PlanningMetric.__new__(PlanningMetric);m.step=6
    m.X_BOUND=m.Y_BOUND=[-50.,50.,.1];m.Z_BOUND=[-10.,10.,20.]
    m.dx,m.bx,_=m.gen_dx_bx(m.X_BOUND,m.Y_BOUND,m.Z_BOUND);m.dx=m.dx[:2];m.bx=m.bx[:2]
    m.H=4.084;m.W=1.85
    parameters=m.calculate_birds_eye_view_parameters(m.X_BOUND,m.Y_BOUND,m.Z_BOUND)
    m.bev_resolution,m.bev_start_position,m.bev_dimension=[p.numpy() for p in parameters]
    occupancy=np.zeros((7,1000,1000));occupancy[:,495:505,550:570]=1
    m.get_birds_eye_view_label=lambda *a,**k:occupancy
    m.get_drivable_area=lambda *a:np.ones((1000,1000))
    backend.metric=m
    data=dict(gt_boxes=np.zeros((1,7)),gt_velocity=np.zeros((1,2)),gt_fut_traj=np.zeros((1,12,2)),
              gt_fut_traj_mask=np.ones((1,12)),gt_fut_yaw=np.zeros((1,12)),gt_fut_idx=np.array([[8]]),
              ego2global_translation=[0,0,0],ego2global_rotation=[1,0,0,0])
    result=backend(data,np.array([[6.,0.]]*6),np.zeros((6,2)))
    assert all(result['raw_collision']) and all(result['collision'])
    assert not any(result['road_boundary'])


def test_question_selection_is_deterministic_and_never_includes_reference_answers(tmp_path):
    folder=tmp_path/'conv/val';folder.mkdir(parents=True)
    (folder/'x.json').write_text(json.dumps([dict(question='first?',answer='SECRET'),dict(question='second?',answer='SECRET')]))
    questions,groups=facts_questions('x',tmp_path,2)
    assert questions[-2:]==['first?','second?'] and groups[-2:]==['other','other']
    assert 'SECRET' not in repr(questions)


def test_scene_selection_rejects_dropped_middle_frames(tmp_path):
    frames=[dict(scene_token='s',token=str(i),timestamp=i,prev=str(i-1),next=str(i+1)) for i in range(3)]
    (tmp_path/'scene_selection.json').write_text(json.dumps(dict(selected=[dict(scene_id='s',tokens=['0','2'])])))
    with pytest.raises(ValueError,match='complete chronological'):
        selected_scenes(frames,'main-scenes',tmp_path,None)
    assert selected_scenes(frames,'full-val',tmp_path,None)[0]['tokens']==['0','1','2']


def test_report_exports_requested_fields_and_never_labels_unscored_zero(tmp_path):
    q=fact_summary([]);p=planning_summary([score_planning('bad',info(),safe_geometry)])
    render_report(tmp_path,dict(performance=performance([]),quality=q),p,dict(scope='unit-test',scenes=[]))
    assert 'N/A' in (tmp_path/'TASK_METRICS.md').read_text()
    assert 'L2_ADE_m' in (tmp_path/'planning_metrics.csv').read_text()


def test_check_only_cannot_enter_inference(monkeypatch):
    import sys
    from tools.worldstream_m123 import r0_task_evaluation as runner
    monkeypatch.setattr(sys,'argv',['r0_task_evaluation','--check-only'])
    monkeypatch.setattr(runner,'preflight',lambda args:({'status':'PREFLIGHT_PASS_NO_INFERENCE'},[],{}))
    monkeypatch.setattr(runner,'run_inference',lambda *args:pytest.fail('check-only entered model inference'))
    monkeypatch.setattr(runner,'launch_worker',lambda *args:pytest.fail('check-only spawned GPU worker'))
    monkeypatch.setattr(runner,'report',lambda *args:pytest.fail('check-only entered scoring'))
    runner.main()


def test_scoring_rejects_tampered_raw_receipts(tmp_path):
    from tools.worldstream_m123.r0_task_evaluation import report, sha
    raw=tmp_path/'facts_receipts.jsonl';raw.write_text('original\n');digest=sha(raw)
    (tmp_path/'inference_status.json').write_text(json.dumps(dict(status='COMPLETED',
        full_model_calls=2,receipt_hashes={raw.name:digest})))
    (tmp_path/'run_manifest.json').write_text(json.dumps(dict(input_hashes={},reference_hashes={},scenes=[])))
    raw.write_text('tampered\n')
    with pytest.raises(ValueError,match='Raw inference receipt changed'):
        report(SimpleNamespace(output=tmp_path))


def test_gpu_worker_uses_new_interpreter_and_explicit_manifest_directory(tmp_path):
    import sys
    from tools.worldstream_m123.r0_task_evaluation import worker_command
    command=worker_command(SimpleNamespace(output=tmp_path,runtime_check_only=True))
    assert command[0]==sys.executable
    assert '--_inference-worker' in command and '--runtime-check-only' in command
    assert command[command.index('--output')+1]==str(tmp_path.resolve())
    assert 'faulthandler' in command and '-u' in command


def test_native_worker_failure_is_logged_and_reported_to_parent(tmp_path,monkeypatch):
    import sys
    from tools.worldstream_m123 import r0_task_evaluation as runner
    runner.write(tmp_path/'runtime_stage.json',dict(stage='CUDA_INITIALIZATION'))
    monkeypatch.setattr(runner,'worker_command',lambda args:[sys.executable,'-u','-c',
        "print('diagnostic marker',flush=True);raise SystemExit(139)"])
    with pytest.raises(RuntimeError,match='139'):
        runner.launch_worker(SimpleNamespace(output=tmp_path))
    assert 'diagnostic marker' in (tmp_path/'runtime.log').read_text()
    failure=json.loads((tmp_path/'worker_failure.json').read_text())
    assert failure['returncode']==139 and failure['last_stage']['stage']=='CUDA_INITIALIZATION'


def test_successful_worker_streams_output_and_records_child_process(tmp_path,monkeypatch):
    import sys
    from tools.worldstream_m123 import r0_task_evaluation as runner
    monkeypatch.setattr(runner,'worker_command',lambda args:[sys.executable,'-u','-c',"print('worker passed')"])
    runner.launch_worker(SimpleNamespace(output=tmp_path))
    assert 'worker passed' in (tmp_path/'runtime.log').read_text()
    process=json.loads((tmp_path/'worker_process.json').read_text())
    assert process['pid']!=process['parent_pid'] and not {4,5}&set(process['cpu_affinity'])
