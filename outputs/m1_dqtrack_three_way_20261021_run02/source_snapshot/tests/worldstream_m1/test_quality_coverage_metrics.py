"""Offline statistical contracts: frames/pixels never become independent logs."""
from tools.worldstream_m1.quality_coverage_metrics import identity_metrics, log_risk, region_metrics, upper95


def identity_row(scene='s', log='l', correct=True):
    return dict(scene=scene,log_token=log,correct=correct,candidate=True,certified=False,
        current_instance='instance',source_instance='instance',frame='current',source_frame='old',rejection_reasons=[])


def test_repeated_frames_and_scenes_in_one_log_do_not_increase_risk_support():
    rows=[identity_row(scene=str(i)) for i in range(100)]
    r=identity_metrics(rows,200)
    assert r['log_risk']['fully_scorable_accepted_logs']==1
    assert r['log_risk']['complete_log_event_upper95']>.9
    assert r['unique_physical_identity_recovery']==1
    assert r['proposal_coverage']==.5
    assert r['qualified_certifications']==0 and r['qualified_FAR'] is None


def test_incomplete_log_does_not_become_a_zero_error_qualified_unit():
    r=log_risk([identity_row(correct=True),identity_row(scene='other',correct=None)])
    assert r['accepted_logs']==1 and r['fully_scorable_accepted_logs']==0
    assert r['complete_log_event_upper95'] is None
    assert r['worst_case_log_event_upper95']==1


def test_one_wrong_entity_makes_the_whole_log_an_error():
    r=log_risk([identity_row(),identity_row(correct=False),identity_row(log='other')])
    assert r['error_logs']==1 and r['complete_error_logs']==1
    assert upper95(0,58)>.05 and upper95(0,59)<=.05
    assert upper95(0,0) is None


def region_row():
    return dict(scene='s',log_token='l',frame='f',pixels=100,stable_pixels=80,effective_pixels=0,
        changed_pixels=0,unknown_pixels=20,accepted_points=0,false_points=0,critical_points=5,
        missed_critical_points=0,scorable_points=8,events=[dict(recalled=True,invalidation_coverage=1.,reasons=['moving_object'])])


def test_force_refresh_effective_area_and_zero_acceptance_far_remain_distinct():
    r=region_metrics([region_row()])
    assert r['observation_stable_area']==.8 and r['effective_stable_area_ratio']==0
    assert r['sparse_reference_far'] is None and r['log_event_upper95'] is None
    assert r['cluster_bootstrap_undefined_zero_support']==1000
    assert not r['risk_gates_passed']


def test_region_unit_error_groups_all_scenes_from_a_log():
    a=region_row();a.update(accepted_points=100,false_points=0,effective_pixels=50)
    b=dict(a,scene='other',false_points=1)
    r=region_metrics([a,b])
    assert r['accepted_scene_support']==2 and r['accepted_log_support']==1 and r['error_logs']==1
    assert r['sparse_reference_far']==1/200
    assert r['independently_correct_stable_support']==199
