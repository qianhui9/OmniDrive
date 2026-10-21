from dataclasses import replace
import numpy as np
import pytest

from worldstream.m1.fact_certifier import calibrate_scene_maxima, certify_relative_direction, DirectionCalibration
from worldstream.m1.task_facts import extract_task_facts
from test_task_facts import inputs


def evidence():
    anchor,frame,check=inputs()
    anchor.geometry_world.center=np.array([10.,5.,1.])
    candidate=next(f for f in extract_task_facts(anchor,frame,check) if f.fact_type=='relative_direction')
    support=dict(identity=dict(unique=True,bidirectional=True,source_anchor='a',current_target='b',
        source_frame='source',current_frame='current',source='foreground_owned_bidirectional_correspondence'),
        foreground=dict(source_owned=True,current_owned=True,occlusion_rejected=True,evidence_digest='observations',
        source_frame='source',current_frame='current',source='independent_instance_segmentation'))
    calibration=calibrate_scene_maxima({f'cal{i}':[[.5,.3]] for i in range(19)},'sig','frozen-cal')
    return candidate,frame,support,calibration


def certify(**changes):
    candidate,frame,support,calibration=evidence()
    args=dict(candidate=candidate,frame=frame,center_xy=[10.,5.],evidence=support,
              calibration=calibration,associations=(('a','b'),))
    args.update(changes)
    return certify_relative_direction(**args)


def test_actual_certification_branch_remains_separate_from_deployment():
    result=certify()
    assert result.certified and result.rejection_reason=='ok'
    assert result.fact_value==dict(longitudinal=1,lateral=1)
    assert not result.deployment_approved
    assert 'statistical' in result.coverage_scope


@pytest.mark.parametrize('part',['identity','foreground'])
def test_missing_identity_or_foreground_rejects(part):
    _,_,support,_=evidence();support.pop(part)
    assert not certify(evidence=support).certified


@pytest.mark.parametrize('center',[[10,0],[0,5],[.5,5],[10,-.3]])
def test_crossing_or_touching_zero_rejects(center):
    assert certify(center_xy=center).rejection_reason=='direction_interval_crosses_zero'


@pytest.mark.parametrize('update',[dict(scene_id='new'),dict(timestamp=102.),dict(timestamp=100.)])
def test_stale_scene_and_future_sources_reject(update):
    _,frame,_,_=evidence()
    assert not certify(frame=replace(frame,**update)).certified


def test_gt_cannot_be_online_fact_evidence():
    _,_,support,_=evidence();support['nested']={'source':'offline_gt_assignment'}
    assert certify(evidence=support).rejection_reason=='forbidden_gt_online_input'


def test_score_cannot_substitute_for_calibrated_metre_error():
    assert certify(calibration={'detection_score':1.,'radius_xy_m':[0.,0.]}).rejection_reason=='calibrated_coordinate_error_bound_missing'


@pytest.mark.parametrize('pairs',[(('a','b'),('a','c')),(('a','b'),('c','b'))])
def test_incompatible_matches_reject_without_call_order_dependence(pairs):
    assert certify(associations=pairs).rejection_reason=='incompatible_or_ambiguous_association'


def test_scene_conformal_small_n_is_unbounded_and_xy_joint_radius():
    small=calibrate_scene_maxima({'cal':[[.1,.8]]},'sig','id')
    assert small.radius_xy_m is None
    assert certify(calibration=small).rejection_reason=='insufficient_independent_calibration_scenes'
    proper=calibrate_scene_maxima({f'cal{i}':[[.1,.8]] for i in range(19)},'sig','id')
    assert proper.radius_xy_m==(.8,.8)


def test_source_version_and_calibration_leak_reject():
    candidate,_,_,calibration=evidence()
    assert not certify(candidate=replace(candidate,computation_source={'origin':'oracle'})).certified
    assert not certify(calibration=replace(calibration,calibration_scenes=('scene',))).certified
    assert not certify(calibration=replace(calibration,execution_signature='new')).certified
