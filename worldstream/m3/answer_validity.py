from dataclasses import asdict
import math
from .task_conditioner import question_facts
from time import perf_counter

def fact_region(center, profile):
    (x, y) = center[:2]
    if math.hypot(x, y) > profile.near_m:
        return False
    if profile.direct and abs(y) > 4.0:
        return False
    return all(({'front': x > 0, 'rear': x < 0, 'left': y > 0, 'right': y < 0}[d] for d in profile.directions))

def fact_directions(center):
    (x, y) = center[:2]
    return tuple((d for (d, good) in [('front', x > 1.0), ('rear', x < -1.0), ('left', y > 1.0), ('right', y < -1.0)] if good))

def render_facts(profile, cls, directions=(), motion=''):
    noun = cls.replace('_', ' ')
    article = 'an' if noun[:1] in 'aeiou' else 'a'
    place = 'directly ahead' if profile.direct else {('front',): 'in front', ('rear',): 'behind', ('left',): 'on the left', ('right',): 'on the right'}.get(profile.directions, 'nearby' if profile.near_m == 30 else 'within view')
    if profile.intent == 'category':
        return f'{article.capitalize()} {noun} is {place}.'
    if profile.intent == 'existence':
        return f'Yes, there is {article} {noun} {place}.'
    if profile.intent == 'position':
        location = '-'.join(directions) or 'alongside'
        return f'The nearest {noun} is to the {location}.'
    if profile.intent == 'motion':
        return f'The nearest {noun} is {motion}.'
    return ''

def propose_fact_answer(question, current, context, memory, binding, policy=None, previous=None, target_binding_enabled=False, temporal_risk_enabled=False, target_completeness_enabled=False):
    """One fast-answer decision before expansion. Never accepts GT/current LLM.

    Full reuse is admitted only for the closed assertion grammar below; richer
    target answers may seed history, but uncovered modifiers cannot be copied.
    Current facts can answer a structured question without historical identity.
    """
    tick = perf_counter()
    profile = question_facts(question)
    current_only = (policy or {}).get('main_path') == 'current_only'
    if current_only and (temporal_risk_enabled or target_completeness_enabled):
        raise ValueError('Current-only bypass excludes historical risk and target completeness')
    parse_ms = (perf_counter() - tick) * 1000
    out = dict(supported=False, admitted=False, answer_source='none', answer_text='', profile=asdict(profile), required_facts=list(profile.required), reason=profile.reason, facts=[], history_candidate_count=0, question_fact_extraction_wall_ms=parse_ms, answer_validity_check_wall_ms=0.0, current_fact_extraction_wall_ms=0.0, target_completeness_wall_ms=0.0, position_admission_wall_ms=0.0, main_path='current_only' if current_only else 'legacy_reproduction', fast_answer_construction_wall_ms=0.0, score_kind='DETECTOR_SUPPORT_NOT_ERROR_PROBABILITY')
    tick = perf_counter()

    def finish(reason):
        out['reason'] = reason
        out['answer_validity_check_wall_ms'] = max(0.0, (perf_counter() - tick) * 1000 - out['fast_answer_construction_wall_ms'])
        return out
    if profile.intent == 'unsupported':
        return finish(profile.reason)
    if current is None:
        return finish('missing_current_committed_state')
    try:
        current.validate(context)
    except ValueError:
        return finish('current_commit_context_mismatch')
    from .world_delta import VEHICLES
    extraction_tick = perf_counter()
    classes = set(profile.classes)
    if 'vehicle' in classes:
        classes = classes - {'vehicle'} | VEHICLES
    if 'object' in classes:
        classes = set()
    witnesses = [a for a in current.anchors if a.anchor_type == 'object' and a.observed and (a.evidence_origin == 'actual_perception') and (a.source_frame == current.frame_id) and (a.source_timestamp == current.timestamp) and a.cameras and (a.score is not None) and math.isfinite(a.score) and (a.score >= 0.35) and (not classes or a.semantic_class in classes) and all((math.isfinite(v) for v in a.center_ego)) and fact_region(a.center_ego, profile)]
    if not witnesses:
        out['current_fact_extraction_wall_ms'] = (perf_counter() - extraction_tick) * 1000
        return finish('no_positive_current_witness_UNKNOWN_not_absent')
    witnesses.sort(key=lambda a: (math.hypot(*a.center_ego[:2]), -a.score, a.anchor_id))
    witness = max(witnesses, key=lambda a: a.score) if profile.selector == 'any' else witnesses[0]
    out['current_fact_extraction_wall_ms'] = (perf_counter() - extraction_tick) * 1000
    complete_enabled = False
    if target_binding_enabled:
        from .target_binding import target_binding_risk
        out['target_binding'] = target_binding_risk(profile, witness, witnesses, current, previous, temporal_risk_enabled)
    if profile.selector == 'nearest' and any((a.semantic_class != witness.semantic_class and abs(math.hypot(*a.center_ego[:2]) - math.hypot(*witness.center_ego[:2])) < 2.0 for a in witnesses[1:])):
        return finish('nearest_target_class_ambiguous')
    motion = ''
    if profile.intent == 'motion':
        if not witness.velocity_ego:
            return finish('no_reliable_velocity_evidence')
        speed = math.hypot(*witness.velocity_ego[:2])
        if 0.5 < speed < 1.0:
            return finish('motion_transition_unknown')
        motion = 'stationary' if speed <= 0.5 else 'moving'
    directions = fact_directions(witness.center_ego)
    if profile.intent == 'position' and (abs(abs(witness.center_ego[0]) - 1.0) < 0.5 or abs(abs(witness.center_ego[1]) - 1.0) < 0.5):
        return finish('position_sector_boundary_unknown')
    position_ok = True
    if profile.intent == 'position':
        out['position_admission'] = dict(mode='basic', eligible=True, min_confidence=None, scope='current_detector_support_only_not_nearest_identity_certification')
    facts = dict(anchor_id=witness.anchor_id, semantic_class=witness.semantic_class, center_ego=witness.center_ego, directions=directions, motion=motion, velocity_ego=witness.velocity_ego, confidence=witness.score, cameras=witness.cameras, source_frame=witness.source_frame, source_timestamp=witness.source_timestamp, state_version=current.version, state_epoch=current.epoch, evidence_origin=witness.evidence_origin, identity_qualified=witness.identity_qualified, observation_kind=witness.observation_kind, identity_required=False, association_unambiguous=witness.association_unambiguous)
    prior = next((a for a in previous.anchors if a.anchor_id == witness.anchor_id), None) if not current_only and previous is not None and (previous.scene_id == current.scene_id) and (previous.epoch == current.epoch) and (previous.timestamp < current.timestamp) and (previous.execution_signature == current.execution_signature) else None
    out.update(supported=True, facts=[facts], support_score=witness.score, continuity=dict(same_anchor_observed=bool(prior and prior.observed), ego_displacement_m=math.dist(prior.center_ego[:2], witness.center_ego[:2]) if prior else None), current_commit=(current.epoch, current.version, current.frame_id), source_generation=None, source_age_s=None, original_model_binding=binding)
    build_tick = perf_counter()
    text = render_facts(profile, witness.semantic_class, directions, motion)
    reusable = None
    out['answer_source'] = 'fact_grounded_response'
    out['answer_text'] = text
    out['fast_answer_construction_wall_ms'] = (perf_counter() - build_tick) * 1000
    expected = float((policy or {}).get('expected_normal_m3_ms', 0.0))
    check_ms = (perf_counter() - tick) * 1000 + parse_ms
    out['estimated_net_gain_ms'] = expected - check_ms
    out['empirical_policy_id'] = (policy or {}).get('policy_id')
    risk_ok = not target_binding_enabled or out['target_binding']['total_risk'] <= float((policy or {}).get('max_binding_risk', 0.35))
    complete_ok = not complete_enabled or out['target_completeness']['evidence_sufficient']
    out['admitted'] = bool(policy and witness.score >= policy['min_confidence'] and (expected - check_ms > float((policy or {}).get('min_net_gain_ms', 0.0))) and risk_ok and complete_ok and position_ok)
    if not position_ok:
        return finish('position_admission_requires_target_aware_generation')
    if not complete_ok:
        return finish('question_target_evidence_insufficient')
    if not risk_ok:
        return finish('target_binding_risk_exceeds_frozen_policy')
    return finish('supported_and_admitted' if out['admitted'] else 'supported_shadow_or_below_policy')
