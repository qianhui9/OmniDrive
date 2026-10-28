"""Question-aware differences over real committed M1 observations, no GT."""
from dataclasses import asdict,dataclass
import math

VEHICLES={'car','truck','bus','trailer','construction_vehicle','motorcycle','bicycle'}


def relevant(anchor,dependencies):
    if dependencies.intent=='road':return anchor.anchor_type=='map'
    if anchor.anchor_type!='object':return False
    classes=set(dependencies.classes)
    if 'vehicle' in classes:classes=classes-{'vehicle'}|VEHICLES
    if classes and anchor.semantic_class not in classes:return False
    x,y,_=anchor.center_ego
    if 'front' in dependencies.directions and x<=0:return False
    if 'rear' in dependencies.directions and x>=0:return False
    if 'left' in dependencies.directions and y<=0:return False
    if 'right' in dependencies.directions and y>=0:return False
    if 'near' in dependencies.attributes and math.hypot(x,y)>30:return False
    return True


@dataclass(frozen=True)
class QuestionAwareWorldDelta:
    scene_id: str
    previous_frame: object
    current_frame: object
    question_signature: tuple
    task_profile: str
    relevant_anchor_ids: tuple
    matched_anchor_ids: tuple
    changed_attributes: tuple
    unknown_dependencies: tuple
    evidence_versions: tuple
    draft_risk: float
    suggested_draft_length: int
    decision_reason: str

    def to_dict(self):return asdict(self)


def question_world_delta(previous,current,dependencies):
    def result(risk,length,reason,relevant_ids=(),matched=(),changed=(),unknown=()):
        return QuestionAwareWorldDelta(current.scene_id if current else '',
            previous.frame_id if previous else None,current.frame_id if current else None,
            dependencies.signature,dependencies.intent,tuple(relevant_ids),tuple(matched),tuple(changed),tuple(unknown),
            ((previous.epoch,previous.version) if previous else None,
             (current.epoch,current.version) if current else None),risk,length,reason)
    if current is None:return result(1.,0,'UNKNOWN_current_committed_state',unknown=('current_observation',))
    if previous is None:return result(1.,0,'UNKNOWN_history_state',unknown=('previous_observation',))
    if current.scene_id!=previous.scene_id or current.epoch!=previous.epoch:
        return result(1.,0,'scene_reset',changed=('scene_reset',))
    if (not 0<current.timestamp-previous.timestamp<=4. or current.version<=previous.version
            or current.execution_signature!=previous.execution_signature):
        return result(1.,0,'UNKNOWN_state_sequence',unknown=('state_sequence',))
    old={a.anchor_id:a for a in previous.anchors if relevant(a,dependencies) and a.observed}
    now={a.anchor_id:a for a in current.anchors if relevant(a,dependencies) and a.observed}
    matched=tuple(sorted(set(old)&set(now)))
    changed=[];unknown=[]
    observed_all={a.anchor_id:a for a in current.anchors if a.observed}
    if set(now)-set(old):changed.append('new_observed_anchors')
    if set(old)-set(observed_all):
        changed.append('lost_current_observation_support');unknown.append('departure_or_occlusion')
    for aid in set(old)&set(observed_all):
        if old[aid].semantic_class!=observed_all[aid].semantic_class:changed.append('class')
        if aid not in now:changed.append('left_question_region')
    for aid in matched:
        a,b=old[aid],now[aid]
        if a.semantic_class!=b.semantic_class:changed.append('class')
        if not (b.identity_qualified and b.association_unambiguous):unknown.append('identity_continuity')
        if 'geometry' in dependencies.attributes:
            if any((x>0)!=(y>0) for x,y in zip(a.center_ego[:2],b.center_ego[:2])):changed.append('relative_direction')
            if math.dist(a.center_ego[:2],b.center_ego[:2])>2.:changed.append('relative_position')
        if 'velocity' in dependencies.attributes:
            if not a.velocity_ego or not b.velocity_ego:unknown.append('motion')
            elif math.dist(a.velocity_ego,b.velocity_ego)>1.:changed.append('motion')
    ids=tuple(sorted(now))
    common=dict(relevant_ids=ids,matched=matched,changed=tuple(sorted(set(changed))))
    if dependencies.complete_coverage_required:
        return result(1.,0,'UNKNOWN_incomplete_scene_dependency',unknown=('complete_scene_coverage',),**common)
    if dependencies.intent=='road':
        return result(1.,0,'UNKNOWN_map_semantic_coverage',unknown=('map_semantic_coverage',),**common)
    # Positive existential evidence needs no negative/full-scene inference.
    # It does NOT certify identity or absence. The scheduler also checks the
    # history's positive polarity and compatible question intent before proposing.
    if dependencies.intent=='existence' and now and old:
        if any(a.score is not None and a.score>=.35 for a in now.values()):
            return result(.35,8,'positive_observations_both_frames',unknown=tuple(sorted(set(unknown))),**common)
    if not now:return result(1.,0,'UNKNOWN_no_relevant_current_observation',
        unknown=tuple(sorted(set(unknown+['current_relevant_evidence']))),**common)
    if dependencies.intent=='category' and old and (unknown or changed or not matched):
        old_classes={a.semantic_class for a in old.values() if a.score is not None and a.score>=.35}
        now_classes={a.semantic_class for a in now.values() if a.score is not None and a.score>=.35}
        if old_classes&now_classes:
            return result(.65,4,'partial_class_observations_prefix_only',
                unknown=tuple(sorted(set(unknown+['uncertified_identity_or_scene_coverage']))),**common)
    if not matched:return result(.8,0,'association_not_continuous',unknown=('identity_continuity',),**common)
    if unknown:return result(.7,0,'UNKNOWN_question_dependency',unknown=tuple(sorted(set(unknown))),**common)
    if changed:return result(.55,4,'changed_relevant_attributes',**common)
    return result(.1,8,'qualified_relevant_state_continues',**common)
