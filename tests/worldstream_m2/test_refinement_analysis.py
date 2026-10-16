import numpy as np
from tools.worldstream_m2.refinement_common import memory_alignment,map_alignment
from tools.worldstream_m2.refine_integration import planning_metric
from worldstream.m2.cost_model import CostModel


def test_joint_memory_matching_preserves_age_and_fields():
    points=np.array([[[0.,0,0],[1,0,0],[10,0,0],[11,0,0]]])
    pose=np.tile(np.eye(4),(1,4,1,1))
    baseline={'det':dict(memory_reference_point=points,memory_timestamp=np.array([[[0.],[0.],[-1.],[-1.]]]),
        memory_egopose=pose,memory_embedding=np.arange(8).reshape(1,4,2))}
    permutation=[1,0,3,2]
    actual={'det':{k:v[:,permutation].copy() for k,v in baseline['det'].items()}}
    result=memory_alignment(actual,baseline)['det']
    assert result['matched']==4 and result['max_matched_point_error_m']==0
    assert result['leaves']['memory_embedding']['max_abs']==0
    assert sum(g['reordered'] for g in result['groups'])==4


def test_map_matching_reports_real_coordinate_error():
    points=np.arange(12,dtype=float).reshape(2,2,3);scores=np.array([.8,.6])
    actual=[[points[::-1].copy(),scores[::-1].copy()]]
    baseline=[[points,scores]]
    assert map_alignment(actual,baseline)['max_matched_control_coordinate_abs_m']==0
    actual[0][0][0,0,0]+=.02
    assert map_alignment(actual,baseline)['max_matched_control_coordinate_abs_m']>.01


def test_planning_parser_accepts_explicit_plus_and_mask():
    text=[{'A':['[PT, '+', '.join(['(+0.50, -0.25)']*6)+']']}]
    info={'gt_planning':np.array([[[.5,-.25,0]]*6]),'gt_planning_mask':np.ones((1,6,2))}
    result=planning_metric(text,info)
    assert result['parsed'] and result['ade_m']==result['fde_m']==0


def test_downstream_failure_blocks_measured_fast_operator():
    costs=CostModel();costs.records=[dict(key='actual-test-key',updated=1,
        incremental_gpu_ms=.1,full_gpu_ms=1.,incremental_wall_ms=.1,full_wall_ms=1.,
        backbone_quality_certified=True,quality_certified=False,
        quality_rejection_reason='downstream_regression_target_failed')]
    assert costs.beneficial('actual-test-key',1)==(False,'downstream_regression_target_failed')
