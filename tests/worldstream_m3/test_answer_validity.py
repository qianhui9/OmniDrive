"""Causality, question scope, source age and independent-label boundaries."""
from dataclasses import replace
import pytest

from worldstream.m3.answer_memory import AnswerMemory
from worldstream.m3.answer_validity import evaluate_answer, validity_task
from .test_world_guided_decode import state
from tools.worldstream_m123.answer_validity_quality import annotation_grade, object_claims


QUESTION = 'Are there any cars?'
BINDING = ('model', 'tokenizer', 'generation')


def setup_memory(question=QUESTION, answer='Yes, there is a car.'):
    memory = AnswerMemory()
    previous = state()
    context = dict(scene_id='s', frame_id='f0', frame_index=0, timestamp=1.,
                   state_epoch=2, current_state_version=1, state_execution_signature='actual')
    memory.remember(question, (1, 3), (6, 7, 2), context, BINDING, previous, answer, track_validity=True)
    current = state('f1', 1.5, 2)
    context = dict(context, frame_id='f1', frame_index=1, timestamp=1.5, current_state_version=2)
    record = next(iter(memory.records.values()))
    return memory, record, current, context


def test_local_witness_accepts_partial_identity_and_ignores_unrelated_change():
    _, record, current, ctx = setup_memory()
    uncertain = replace(current.anchors[0], identity_qualified=False, association_unambiguous=False)
    unrelated = replace(uncertain, anchor_id='other', semantic_class='bus', center_ego=(-40., 20., 0.))
    current = replace(current, anchors=(uncertain, unrelated))
    decision = evaluate_answer(record, QUESTION, current, ctx, BINDING)
    assert decision.eligible and .85 < decision.score < .95
    assert decision.relevant_anchor_ids == ('car1',)
    assert 'partial_identity_continuity' in decision.unknown_dependencies
    assert decision.score_kind == 'UNCALIBRATED_RULE_SCORE'


@pytest.mark.parametrize('mutation', ['scene', 'epoch', 'execution', 'binding', 'future', 'context', 'uncommitted'])
def test_state_and_time_boundaries(mutation):
    _, record, current, ctx = setup_memory()
    binding = BINDING
    if mutation == 'scene':
        current = replace(current, scene_id='other'); ctx = dict(ctx, scene_id='other')
    elif mutation == 'epoch':
        current = replace(current, epoch=3); ctx = dict(ctx, state_epoch=3)
    elif mutation == 'execution':
        current = replace(current, execution_signature='other'); ctx = dict(ctx, state_execution_signature='other')
    elif mutation == 'binding':
        binding = ('other',)
    elif mutation == 'future':
        record = replace(record, timestamp=2.)
    elif mutation == 'context':
        ctx = dict(ctx, frame_id='wrong')
    else:
        current = None
    assert not evaluate_answer(record, QUESTION, current, ctx, binding).eligible


@pytest.mark.parametrize('q,a', [
    ('Are there any cars?', 'No, there are no cars.'),
    ('How many cars?', 'There are two cars.'),
    ('What is the traffic light color?', 'The traffic light is red.'),
    ('Is it safe to drive?', 'It is safe to drive.'),
    ('Describe the environment', 'There is a car and a construction site.'),
    ('What lane is the car in?', 'The car is in the left lane.'),
])
def test_no_global_fact_from_one_unchanged_object(q, a):
    _, record, current, ctx = setup_memory(q, a)
    assert not evaluate_answer(record, q, current, ctx, BINDING).eligible


def test_current_support_not_retained_prediction_and_no_GT_promotion():
    _, record, current, ctx = setup_memory()
    for change in [dict(observed=False), dict(source_frame='f0'), dict(cameras=()),
                   dict(evidence_origin='oracle'), dict(score=float('nan'))]:
        modified = replace(current, anchors=(replace(current.anchors[0], **change),))
        assert not evaluate_answer(record, QUESTION, modified, ctx, BINDING).eligible


def test_full_answer_depends_on_more_than_question_class():
    _, record, current, ctx = setup_memory(answer='Yes, there is a car and a truck in front.')
    assert not evaluate_answer(record, QUESTION, current, ctx, BINDING).eligible


def test_claimed_direction_must_still_hold():
    _, record, current, ctx = setup_memory(answer='Yes, there is a car in front.')
    moved = replace(current, anchors=(replace(current.anchors[0], center_ego=(-10., 1., 0.)),))
    assert not evaluate_answer(record, QUESTION, moved, ctx, BINDING).eligible


def test_motion_is_a_dependency_even_when_question_only_asks_existence():
    _, record, current, ctx = setup_memory(answer='Yes, there is a moving car.')
    assert not evaluate_answer(record, QUESTION, current, ctx, BINDING).eligible


def test_direct_and_near_selectors_are_not_generic_front_existence():
    assert validity_task('Are there any cars directly ahead?').signature != validity_task('Are there any cars ahead?').signature
    assert validity_task('Are there any cars near the vehicle?').signature != validity_task(QUESTION).signature
    q = 'Are there any cars directly ahead?'
    _, record, current, ctx = setup_memory(q)
    current = replace(current, anchors=(replace(current.anchors[0], center_ego=(10., 8., 0.)),))
    assert not evaluate_answer(record, q, current, ctx, BINDING).eligible


def test_cache_observation_preserves_original_source_age_and_independence():
    memory, record, current, ctx = setup_memory()
    memory.observe_validity(record, 'f1', True, .93)
    updated = next(iter(memory.records.values()))
    assert updated.frame_id == 'f0' and updated.timestamp == 1. and updated.frame_index == 0
    assert updated.original_generation == record.original_generation
    assert updated.world_state is record.world_state
    assert dict(updated.validity_statistics) == dict(checks=1, hits=1, last_score=.93)
    assert updated.cache_provenance == (('f1', .93),)
    assert not memory.validity_candidates(QUESTION, dict(ctx, timestamp=6., frame_index=10), BINDING)
    assert evaluate_answer(replace(record, source='answer_cache'), QUESTION, current, ctx, BINDING).eligible is False


def test_semantic_candidates_are_separate_from_no_opportunity():
    memory, _, _, ctx = setup_memory()
    assert memory.validity_candidates('Is there any car?', ctx, BINDING)
    assert not memory.validity_candidates('What is the traffic signal?', ctx, BINDING)
    assert not memory.validity_candidates('Are there any trucks?', ctx, BINDING)
    assert not memory.validity_candidates('Are there any parked cars?', ctx, BINDING)


def test_broad_task_signature_is_not_semantic_equivalence():
    memory, _, _, ctx = setup_memory('Describe the area', 'There is an urban area.')
    assert not memory.validity_candidates('What is the weather in this area?', ctx, BINDING)
    assert validity_task('What type of public transportation vehicle can be seen?').signature != validity_task(
        'What types of vehicles can be seen?').signature


def test_annotation_absence_is_UNKNOWN_but_positive_witness_can_disprove_no():
    objects = [dict(annotation_id='gt', semantic_class='car', center_ego=(10., 1., 0.), visible=True)]
    no = annotation_grade('No, there are no cars.', objects)
    assert no['scored_assertions'] == 1 and no['score'] == 0.
    assert annotation_grade('Yes, there is a car.', objects)['score'] == 1.
    assert annotation_grade('No, there are no cars.', [])['score'] is None
    assert annotation_grade('Yes, there is a truck.', objects)['score'] is None


@pytest.mark.parametrize('answer', ['There might be a car.', 'You should watch for cars.',
                                    'No pedestrians are crossing the street.',
                                    'No pedestrians are waiting to cross.'])
def test_annotation_behavior_and_hypothetical_are_unknown(answer):
    objects = [dict(annotation_id='gt', semantic_class='pedestrian', center_ego=(10., 1., 0.), visible=True),
               dict(annotation_id='gt2', semantic_class='car', center_ego=(10., 1., 0.), visible=True)]
    assert annotation_grade(answer, objects)['score'] is None


def test_direction_not_shared_between_contrasting_objects():
    claims = object_claims('Cars are on the right, but no pedestrians are visible.')
    assert claims[0]['directions'] == ('right',) and not claims[1]['directions']
    objects = [dict(annotation_id='gt', semantic_class='car', center_ego=(10., -1., 0.), visible=True)]
    assert annotation_grade('Cars are on the left.', objects)['score'] is None


def test_question_scope_applies_to_negative_subordinate_clause():
    objects = [dict(annotation_id='gt', semantic_class='pedestrian', center_ego=(10., 1., 0.), visible=True)]
    answer = 'Cars are on the right, but no pedestrians are visible.'
    grade = annotation_grade(answer, objects, 'Are there any pedestrians to the right-front?')
    assert all(c['correct'] is None for c in grade['checks'])


def test_validity_parser_resolves_ego_road_and_car_synonyms():
    from worldstream.m3.answer_validity import dependencies_for_answer
    assert validity_task('What kind of intersection is the vehicle approaching?').intent == 'road'
    dep = dependencies_for_answer(QUESTION, 'Yes, there is a white van in front.')
    assert dep.classes == ('car',) and dep.polarity == 'positive'
    assert 'appearance_not_in_M1' in dep.unknown
