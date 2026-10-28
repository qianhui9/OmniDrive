from tools.worldstream_m123.world_guided_decode_v2_quality import facts,compare_facts


def test_fact_grading_uses_independent_reference_and_detects_direction_error():
    q='Where is the pedestrian?'
    gt=facts(q,'The pedestrian is to the left and ahead of us.')
    wrong=facts(q,'The pedestrian is to the right and ahead of us.')
    assert not compare_facts(gt,wrong)[0]['correct']
    assert compare_facts(gt,facts(q,'Ahead on the left.'))[0]['correct']


def test_signal_count_and_distance_are_task_facts():
    assert not compare_facts(facts('What color is the traffic light?','It is red.'),
        facts('What color is the traffic light?','It is green.'))[0]['correct']
    assert facts('How many lanes are visible?','There are three lanes, one leading left.')['object_count']==3
    assert not compare_facts({'distance_m':10.},{'distance_m':30.})[0]['correct']


def test_ambiguous_reference_abstains_and_negative_class_is_not_positive():
    assert facts('Where is the vehicle?','Vehicles are on the left and right.')=={}
    assert facts('What category of car?','There are no cars.')=={}
    assert not compare_facts({'binary_answer':'yes'}, {})[0]['correct']
