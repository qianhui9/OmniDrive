import pytest
from tools.system_optimization.streaming_replay import replay
from tools.system_optimization.task_dependency_audit import dependency


def test_fcfs_retains_every_request_and_counts_pending_at_horizon():
    requests=[dict(request_id=str(i),sensor_arrival_s=i*.5,request_arrival_s=i*.5,measured_service_s=2.) for i in range(4)]
    r=replay(requests)
    assert r['completed_at_last_arrival']==0
    assert r['pending_at_last_arrival']==4
    assert r['completed_after_drain']==4
    assert r['queued_not_started_at_last_arrival']==3
    assert [e['queue_wait_s'] for e in r['events']]==[0,1.5,3,4.5]
    assert [e['output_age_s'] for e in r['events']]==[2,3.5,5,6.5]
    assert r['deadline_miss_rate_after_drain']==1
    assert r['actual_queue_memory_growth_bytes'] is None


def test_idle_gaps_and_delayed_request_arrival():
    r=replay([dict(request_id='a',sensor_arrival_s=10.,request_arrival_s=10.1,measured_service_s=.1),
        dict(request_id='b',sensor_arrival_s=11.,request_arrival_s=11.2,measured_service_s=.1)])
    assert r['queue_wait_s']['maximum']==0
    assert r['completed_at_last_arrival']==1
    assert r['output_age_s']['maximum']==pytest.approx(.3)


@pytest.mark.parametrize('cost',[0,-1,float('nan'),float('inf')])
def test_invalid_measured_service_rejected(cost):
    with pytest.raises(ValueError):replay([dict(request_id='a',sensor_arrival_s=0.,request_arrival_s=0.,measured_service_s=cost)])


def test_keywords_never_grant_input_completeness_or_planning_substitution():
    for q in ('Please provide the planning trajectory for the ego car without reasons.',
        'How far is the nearest car?', 'What is the speed of the red car?',
        'Are there any pedestrians on the left?', 'How many lanes are ahead?'):
        r=dependency(q)
        assert not r['input_complete'] and not r['direct_query_eligible'] and not r['compact_input_eligible']
    assert dependency('Are there any pedestrians on the left?')['category']=='count_absence_existence'
