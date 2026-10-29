"""Causal, CPU-only answer-validity proposals over committed perception.

Scores are provisional rule scores, NOT calibrated probabilities or a promise
that an epsilon quality budget is satisfied. Annotation grading lives in tools/.
This module does not authorize runtime caching without offline qualification.
"""
from dataclasses import asdict, dataclass, replace
import math
import re

from .task_conditioner import answer_facts, question_dependencies
from .world_delta import relevant
from .task_conditioner import question_facts
from time import perf_counter


@dataclass(frozen=True)
class AnswerDependencies:
    task: object
    classes: tuple
    directions: tuple
    attributes: tuple
    polarity: str
    motion: str
    unknown: tuple
    anchor_ids: tuple = ()
    observation_confidence: tuple = ()


def validity_task(question):
    """Refine object existence without changing v3's token-draft classifier."""
    dep = question_dependencies(question)
    q = str(question).lower()
    if re.search(r'\b(?:type|kind|sort) of (?:intersection|junction|road|lane)\b', q):
        dep = replace(dep, intent='road', classes=(), attributes=('map_geometry', 'map_class', 'ego'))
    if (dep.classes and dep.intent in ('position', 'existence')
            and re.match(r'(?:are|is|do|does|can)\b', q)
            and re.search(r'\b(any|there|visible|see|present)\b', q)):
        dep = replace(dep, intent='existence', attributes=('class', 'positive_observation', 'geometry'))
    attrs = dep.attributes
    if re.search(r'\b(near|nearby|close|vicinity|immediate|around)\b', q):
        attrs = tuple(sorted(set(attrs) | {'near'}))
    # Specific selectors cannot silently become an arbitrary same-class object.
    selector = 'direct' if re.search(r'\b(directly|straight)\b', q) else 'any'
    qualifiers = tuple(word for word in ('public', 'emergency', 'nearest', 'closest', 'parked', 'moving', 'crosswalk', 'crossing')
                       if re.search(r'\b' + word + r'\b', q))
    return replace(dep, attributes=attrs,
                   signature=(dep.intent, dep.classes, dep.directions,
                              'near' if 'near' in attrs else 'all_views', qualifiers, selector))


def dependencies_for_answer(question, answer, world=None):
    dep = validity_task(question)
    text = str(answer).lower()
    # This answer-level parser recognizes ordinary car subclasses without
    # changing the existing token-draft parser or its retained measurements.
    normalized = re.sub(r'\b(sedans?|suvs?|vans?|taxis?)\b', 'car', text)
    facts = answer_facts(normalized)
    unknown = []
    if facts.uncertain:
        unknown.append('uncertain_answer')
    if not facts.classes:
        unknown.append('unbound_answer_objects')
    if len(facts.classes) > 1:
        unknown.append('multi_object_attribute_binding')
    if re.search(r'\b(white|black|red|green|blue|yellow|large|small|sedan|suv)\b', text):
        unknown.append('appearance_not_in_M1')
    if re.search(r'\b(crossing|crosswalks?|sidewalks?|lanes?|roads?|following|followed|oncoming|approaching)\b', text):
        unknown.append('road_or_behaviour_relation')
    if re.search(r'\b(\d+|one|two|three|four|five|several|many|few|only|all)\b', text):
        unknown.append('quantity_or_metric')
    if re.search(r'\b(should|safe|cautious|alert|construction|buildings?|weather|signs?|signals?|lights?)\b', text):
        unknown.append('non_object_answer_content')
    directions = tuple(sorted(set(dep.directions) | set(facts.directions)))
    task = replace(dep, classes=facts.classes or dep.classes, directions=directions)
    anchors = tuple(a for a in world.anchors if a.observed and relevant(a, task)) if world else ()
    return AnswerDependencies(dep, facts.classes, directions,
                              tuple(sorted(set(dep.attributes) | ({'velocity'} if facts.motion != 'unknown' else set()))),
                              facts.polarity, facts.motion, tuple(unknown),
                              tuple(a.anchor_id for a in anchors),
                              tuple((a.anchor_id, a.score) for a in anchors))


@dataclass(frozen=True)
class ValidityDecision:
    score: float
    eligible: bool
    reasons: tuple
    risk_components: tuple = ()
    relevant_anchor_ids: tuple = ()
    evidence_versions: tuple = ()
    age_s: object = None
    unknown_dependencies: tuple = ()
    score_kind: str = 'UNCALIBRATED_RULE_SCORE'

    def to_dict(self):
        return asdict(self)


# Fixed a priori candidates; empirical quality is checked separately on DEV.
QUALITY_CANDIDATES = ((.01, .95), (.02, .90), (.03, .85))


def evaluate_answer(record, question, current, context, binding):
    """Question-referenced risk, with no labels, current answer, or GPU work."""
    dep = validity_task(question)
    evidence = record.answer_dependencies or dependencies_for_answer(record.question, record.answer_text, record.world_state)
    previous = record.world_state
    versions = ((previous.epoch, previous.version) if previous else None,
                (current.epoch, current.version) if current else None)

    def reject(reason, unknown=()):
        return ValidityDecision(0., False, (reason,), evidence_versions=versions,
                                unknown_dependencies=tuple(unknown))

    if current is None or previous is None:
        return reject('missing_committed_state', ('committed_state',))
    try:
        current.validate(context)
    except ValueError:
        return reject('current_state_context_mismatch')
    if (record.scene_id != current.scene_id or previous.scene_id != current.scene_id
            or previous.epoch != current.epoch or record.binding != binding
            or previous.execution_signature != current.execution_signature):
        return reject('scene_epoch_or_execution_reset')
    age = current.timestamp - record.timestamp
    if (not math.isfinite(age) or not 0 < age <= 4.
            or not 0 < context['frame_index'] - record.frame_index <= 8
            or previous.timestamp != record.timestamp or previous.frame_id != record.frame_id
            or current.version <= previous.version or record.frame_id == current.frame_id):
        return reject('noncausal_or_expired_source')
    if evidence.task.signature != dep.signature:
        return reject('question_dependency_mismatch')
    if record.source != 'current_target_model_generation':
        return reject('not_an_original_target_generation')
    if dep.intent not in ('category', 'existence', 'position', 'motion'):
        return reject('M1_does_not_cover_task', dep.attributes)
    if dep.complete_coverage_required or evidence.polarity != 'positive':
        return reject('absence_or_global_coverage_unknown', ('complete_scene_coverage',))
    if not evidence.classes or 'vehicle' in evidence.classes:
        return reject('unbound_answer_class', evidence.unknown)
    if 'non_object_answer_content' in evidence.unknown:
        return reject('answer_contains_uncovered_fact_domains', evidence.unknown)

    unknown = list(evidence.unknown)
    used = []
    identity_risks = []
    geometry_risks = []
    confidence_risks = []
    freshness = []
    for cls in evidence.classes:
        task = replace(dep, classes=(cls,), directions=evidence.directions,
                       attributes=evidence.attributes)

        def supported(state):
            return [a for a in state.anchors if a.observed and a.evidence_origin == 'actual_perception'
                    and a.source_frame == state.frame_id and a.source_timestamp == state.timestamp
                    and a.cameras and a.score is not None and math.isfinite(a.score)
                    and a.score >= .35 and all(math.isfinite(v) for v in a.center_ego)
                    and relevant(a, task)
                    and (dep.signature[-1] != 'direct' or abs(a.center_ego[1]) <= 4.)]

        old, now = supported(previous), supported(current)
        if not old or not now:
            return reject('missing_answer_referenced_observation', (cls,))
        # Local witnesses only. Unrelated objects cannot invalidate an answer.
        pairs = [(a, b) for a in old for b in now]
        a, b = min(pairs, key=lambda ab: (ab[0].anchor_id != ab[1].anchor_id,
                                          math.dist(ab[0].center_ego[:2], ab[1].center_ego[:2])))
        identity_risks.append(0. if a.anchor_id == b.anchor_id and b.identity_qualified and b.association_unambiguous
                              else .45 if a.anchor_id == b.anchor_id else .8)
        if identity_risks[-1]:
            unknown.append('partial_identity_continuity')
        geometry_risks.append(min(1., math.dist(a.center_ego[:2], b.center_ego[:2]) / 10.)
                              if 'geometry' in evidence.attributes else 0.)
        confidence_risks.append(1. - min(a.score, b.score))
        freshness.append(current.timestamp - b.source_timestamp)
        if evidence.motion != 'unknown' or dep.intent == 'motion':
            if not b.velocity_ego or not a.velocity_ego or evidence.motion == 'unknown':
                return reject('unobserved_answer_motion', ('motion',))
            speed = math.hypot(*b.velocity_ego[:2])
            if (evidence.motion == 'stationary' and speed > .5) or (evidence.motion == 'moving' and speed < 1.):
                return reject('answer_motion_changed')
        used.append(b.anchor_id)
    components = dict(semantic=0., geometry=max(geometry_risks), identity=max(identity_risks),
                      staleness=min(1., age / 4. + max(freshness)),
                      unknown=min(1., .25 * len(set(evidence.unknown)) + max(confidence_risks) * .25))
    risk = sum(components[k] * w for k, w in
               (('semantic', .35), ('geometry', .20), ('identity', .10), ('staleness', .15), ('unknown', .20)))
    return ValidityDecision(max(0., 1. - risk), True, ('local_answer_facts_supported',),
                            tuple(components.items()), tuple(sorted(set(used))), versions, age,
                            tuple(sorted(set(unknown))))


def fact_region(center, profile):
    x,y=center[:2]
    if math.hypot(x,y)>profile.near_m:return False
    if profile.direct and abs(y)>4.:return False
    return all({'front':x>0,'rear':x<0,'left':y>0,'right':y<0}[d] for d in profile.directions)


def fact_directions(center):
    x,y=center[:2]
    return tuple(d for d,good in [('front',x>1.),('rear',x< -1.),('left',y>1.),('right',y< -1.)] if good)


def render_facts(profile, cls, directions=(), motion=''):
    noun=cls.replace('_',' ')
    article='an' if noun[:1] in 'aeiou' else 'a'
    place='directly ahead' if profile.direct else {
        ('front',):'in front',('rear',):'behind',('left',):'on the left',('right',):'on the right'
    }.get(profile.directions,'nearby' if profile.near_m==30 else 'within view')
    if profile.intent=='category':return f'{article.capitalize()} {noun} is {place}.'
    if profile.intent=='existence':return f'Yes, there is {article} {noun} {place}.'
    if profile.intent=='position':
        location='-'.join(directions) or 'alongside'
        return f'The nearest {noun} is to the {location}.'
    if profile.intent=='motion':return f'The nearest {noun} is {motion}.'
    return ''


def propose_fact_answer(question,current,context,memory,binding,policy=None,previous=None):
    """One fast-answer decision before expansion. Never accepts GT/current LLM.

    Full reuse is admitted only for the closed assertion grammar below; richer
    target answers may seed history, but uncovered modifiers cannot be copied.
    Current facts can answer a structured question without historical identity.
    """
    tick=perf_counter();profile=question_facts(question)
    parse_ms=(perf_counter()-tick)*1000
    out=dict(supported=False,admitted=False,answer_source='none',answer_text='',
             profile=asdict(profile),required_facts=list(profile.required),
             reason=profile.reason,facts=[],history_candidate_count=0,
             question_fact_extraction_wall_ms=parse_ms,answer_validity_check_wall_ms=0.,
             fast_answer_construction_wall_ms=0.,score_kind='DETECTOR_SUPPORT_NOT_ERROR_PROBABILITY')
    tick=perf_counter()
    def finish(reason):
        out['reason']=reason
        out['answer_validity_check_wall_ms']=max(0.,(perf_counter()-tick)*1000-out['fast_answer_construction_wall_ms'])
        return out
    if profile.intent=='unsupported':return finish(profile.reason)
    if current is None:return finish('missing_current_committed_state')
    try:current.validate(context)
    except ValueError:return finish('current_commit_context_mismatch')
    from .world_delta import VEHICLES
    classes=set(profile.classes)
    if 'vehicle' in classes:classes=(classes-{'vehicle'})|VEHICLES
    if 'object' in classes:classes=set()
    witnesses=[a for a in current.anchors if a.anchor_type=='object' and a.observed
               and a.evidence_origin=='actual_perception' and a.source_frame==current.frame_id
               and a.source_timestamp==current.timestamp and a.cameras
               and a.score is not None and math.isfinite(a.score) and a.score>=.35
               and (not classes or a.semantic_class in classes)
               and all(math.isfinite(v) for v in a.center_ego) and fact_region(a.center_ego,profile)]
    if not witnesses:return finish('no_positive_current_witness_UNKNOWN_not_absent')
    witnesses.sort(key=lambda a:(math.hypot(*a.center_ego[:2]),-a.score,a.anchor_id))
    witness=max(witnesses,key=lambda a:a.score) if profile.selector=='any' else witnesses[0]
    if (profile.selector=='nearest' and any(a.semantic_class!=witness.semantic_class
            and abs(math.hypot(*a.center_ego[:2])-math.hypot(*witness.center_ego[:2]))<2. for a in witnesses[1:])):
        return finish('nearest_target_class_ambiguous')
    motion=''
    if profile.intent=='motion':
        if not witness.velocity_ego:return finish('no_reliable_velocity_evidence')
        speed=math.hypot(*witness.velocity_ego[:2])
        if .5<speed<1.:return finish('motion_transition_unknown')
        motion='stationary' if speed<=.5 else 'moving'
    directions=fact_directions(witness.center_ego)
    if profile.intent=='position' and (abs(abs(witness.center_ego[0])-1.)<.5 or abs(abs(witness.center_ego[1])-1.)<.5):
        return finish('position_sector_boundary_unknown')
    facts=dict(anchor_id=witness.anchor_id,semantic_class=witness.semantic_class,
               center_ego=witness.center_ego,directions=directions,motion=motion,
               velocity_ego=witness.velocity_ego,confidence=witness.score,
               cameras=witness.cameras,source_frame=witness.source_frame,
               source_timestamp=witness.source_timestamp,state_version=current.version,
               state_epoch=current.epoch,evidence_origin=witness.evidence_origin,
               identity_qualified=witness.identity_qualified,
               identity_required=False,association_unambiguous=witness.association_unambiguous)
    prior=next((a for a in previous.anchors if a.anchor_id==witness.anchor_id),None) if (
        previous is not None and previous.scene_id==current.scene_id and previous.epoch==current.epoch
        and previous.timestamp<current.timestamp and previous.execution_signature==current.execution_signature) else None
    out.update(supported=True,facts=[facts],support_score=witness.score,
               continuity=dict(same_anchor_observed=bool(prior and prior.observed),
                 ego_displacement_m=math.dist(prior.center_ego[:2],witness.center_ego[:2]) if prior else None),
               current_commit=(current.epoch,current.version,current.frame_id),
               source_generation=None,source_age_s=None,original_model_binding=binding)
    build_tick=perf_counter();text=render_facts(profile,witness.semantic_class,directions,motion)
    normalize=lambda s:' '.join(str(s).lower().strip().rstrip('.').split())
    alternatives={normalize(text)}
    if profile.intent=='existence':
        alternatives.add(normalize(text.replace('Yes, there is','There is')))
    if profile.direct:
        alternatives|={t.replace('directly ahead','directly in front of us') for t in tuple(alternatives)}
    history=[r for r in memory.records.values() if memory.valid(r,context,binding)
             and r.source=='current_target_model_generation'
             and question_facts(r.question).signature==profile.signature]
    out['history_candidate_count']=len(history)
    reusable=next((r for r in reversed(history) if normalize(r.answer_text) in alternatives),None)
    if reusable is not None:
        text=reusable.answer_text
        out.update(answer_source='full_answer_reuse',source_generation=reusable.original_generation,
                   source_age_s=current.timestamp-reusable.timestamp,source_prompt_ids=reusable.prompt_ids,
                   source_world_version=reusable.world_version,source_model_binding=reusable.binding)
    else:out['answer_source']='fact_grounded_response'
    out['answer_text']=text
    out['fast_answer_construction_wall_ms']=(perf_counter()-build_tick)*1000
    expected=float((policy or {}).get('expected_normal_m3_ms',0.))
    check_ms=(perf_counter()-tick)*1000+parse_ms
    out['estimated_net_gain_ms']=expected-check_ms
    out['empirical_policy_id']=(policy or {}).get('policy_id')
    out['admitted']=bool(policy and witness.score>=policy['min_confidence'] and expected>check_ms)
    return finish('supported_and_admitted' if out['admitted'] else 'supported_shadow_or_below_policy')
