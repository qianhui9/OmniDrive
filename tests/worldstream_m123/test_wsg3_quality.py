from tools.worldstream_m123.world_guided_decode_v3_report import facts


def test_incidental_where_and_compound_relations_abstain():
    assert 'relative_direction' not in facts('What building is visible from where we are?',
        'A modern building is directly ahead.')
    assert 'relative_direction' not in facts('Where is the car?',
        'The car is on the left, but a bus is on the right.')


def test_stay_alert_is_same_reference_precaution_as_watch():
    question='What actions should the driver take?'
    assert facts(question,'Stay alert for pedestrians.')['reference_action_set']==('observe',)
    assert facts(question,'Watch for pedestrians.')['reference_action_set']==('observe',)
