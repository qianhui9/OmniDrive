"""Offline instance evaluation. Never imported by the inference runtime.

Use the existing v4 2m match radius / .5m ambiguity margin. Assignment is
one-to-one over *all* current anchors, without using predicted class as a
matching constraint. A wrong class must remain measurable as a wrong class.
Center projection and nuScenes visibility are proxies, not image adjudication.
"""
import math
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from worldstream.m3.answer_validity import fact_region, fact_directions
from worldstream.m3.task_conditioner import question_facts
from worldstream.m3.world_delta import VEHICLES
from .selective_answer_reuse_report import grade

UNKNOWN_REASONS = ('no_visible_GT_target', 'multiple_equally_plausible_GT_targets',
    'visibility_ambiguous', 'detector_GT_matching_ambiguous',
    'question_semantics_ambiguous', 'missing_annotation_support')


def match_instances(anchors, objects, frame_id, timestamp):
    """Conservative global assignment; duplicates cannot share a GT receipt."""
    anchors=[a for a in anchors if a['anchor_type']=='object' and a['observed']
        and a['source_frame']==frame_id and a['source_timestamp']==timestamp
        and a['evidence_origin']=='actual_perception'
        and len(a['center_ego'])==3 and all(math.isfinite(x) for x in a['center_ego'])]
    result={a['anchor_id']:dict(known=False, reason='detector_GT_matching_ambiguous',
        annotation_id=None, instance_id=None) for a in anchors}
    if not anchors or not objects:return result
    cost=np.array([[math.dist(a['center_ego'],g['center_ego']) for g in objects] for a in anchors])
    # Private dummy columns allow leaving detections unmatched; no forced far
    # pairing can steal a valid match. No class or future-frame information.
    rows,cols=linear_sum_assignment(np.c_[cost,np.full((len(anchors),len(anchors)),2.)])
    for i,j in zip(rows,cols):
        if j>=len(objects) or cost[i,j]>=2.:continue
        if j!=int(np.argmin(cost[i])):continue
        alternatives=sorted(cost[i])
        if len(alternatives)>1 and alternatives[1]-alternatives[0]<=.5:continue
        # Competing duplicate detections stay UNKNOWN rather than counting both.
        others=[cost[k,j] for k in range(len(anchors)) if k!=i]
        if others and min(others)<=cost[i,j]+.5:continue
        a,g=anchors[i],objects[j]
        result[a['anchor_id']]=dict(known=True,reason=None,annotation_id=g['annotation_id'],
            instance_id=g['instance_id'],semantic_class=g['semantic_class'],
            class_correct=a['semantic_class']==g['semantic_class'],
            GT_center_ego=g['center_ego'],
            center_distance_m=float(cost[i,j]),geometry_within_2m=True,
            visible=g['visible'],cameras=g['cameras'],source_frame=frame_id,
            source_timestamp=timestamp,
            correct=a['semantic_class']==g['semantic_class'])
    assert len([r['annotation_id'] for r in result.values() if r['known']])==len(set(
        r['annotation_id'] for r in result.values() if r['known']))
    return result


def question_target(question,objects):
    """Identity and answer labels have separate eligibility and denominators."""
    p=question_facts(question)
    def unknown(reason):return dict(known=False,reason=reason,answer_known=False)
    if p.intent=='unsupported' or p.classes==('object',):return unknown('question_semantics_ambiguous')
    if objects is None:return unknown('missing_annotation_support')
    classes=set(p.classes)
    if 'vehicle' in classes:classes=(classes-{'vehicle'})|VEHICLES
    region=[g for g in objects if g['semantic_class'] in classes and fact_region(g['center_ego'],p)]
    visible=sorted([g for g in region if g['visible']],key=lambda g:math.hypot(*g['center_ego'][:2]))
    if not visible:return unknown('visibility_ambiguous' if any(g['cameras'] for g in region) else 'no_visible_GT_target')
    # Existence asks for ANY positive witness, not a particular nearest identity.
    if p.selector=='any':return dict(known=True,reason=None,answer_known=True,
        acceptable_annotation_ids=[g['annotation_id'] for g in visible],selected_annotation=None,
        directions=(),identity_semantics='positive_witness_set')
    first=visible[0];distance=math.hypot(*first['center_ego'][:2])
    ambiguous=[g for g in visible[1:] if math.hypot(*g['center_ego'][:2])-distance<2.]
    occluded=[g for g in region if not g['visible'] and g['cameras']
        and math.hypot(*g['center_ego'][:2])<distance+2.]
    if occluded:return unknown('visibility_ambiguous')
    boundary=p.intent=='position' and any(abs(abs(v)-1.)<.5 for v in first['center_ego'][:2])
    if boundary:return unknown('question_semantics_ambiguous')
    if ambiguous:
        # A common answer sector/category can be labelled without inventing a
        # uniquely identified target. Preserve the strict 2m identity margin.
        same=(all(fact_directions(g['center_ego'])==fact_directions(first['center_ego']) for g in ambiguous)
            if p.intent=='position' else all(g['semantic_class']==first['semantic_class'] for g in ambiguous)
            if p.intent=='category' else False)
        return dict(known=False,reason='multiple_equally_plausible_GT_targets',answer_known=same,
            acceptable_annotation_ids=[g['annotation_id'] for g in [first]+ambiguous],
            directions=fact_directions(first['center_ego']),semantic_class=first['semantic_class'])
    return dict(known=True,reason=None,answer_known=True,
        acceptable_annotation_ids=[first['annotation_id']],selected_annotation=first['annotation_id'],
        selected_instance=first['instance_id'],center_ego=first['center_ego'],
        directions=fact_directions(first['center_ego']),semantic_class=first['semantic_class'])


def evaluate(question,answer,objects,target,token):
    """Reuse the existing answer parser, never count UNKNOWN as correct."""
    if not target['answer_known']:
        return dict(known=False,score=None,severe_error=None,reason=target['reason'])
    candidates=[g for g in objects if g['annotation_id'] in target['acceptable_annotation_ids']]
    # Common-sector ambiguity is explicitly a task-answer label, never identity.
    # Removing equivalent witnesses avoids the old evaluator's blanket veto.
    if not target['known']:candidates=candidates[:1]
    return grade(question,answer,token,candidates)


def request_identity(proposal,target,matches):
    facts=(proposal or {}).get('facts',[])
    if not facts:return dict(object=dict(known=False,reason='missing_annotation_support'),
        target=dict(known=False,reason=target.get('reason') or 'missing_annotation_support'))
    physical=matches.get(facts[0]['anchor_id'],dict(known=False,reason='detector_GT_matching_ambiguous'))
    identity=dict(known=False,reason=target.get('reason') or physical.get('reason'))
    if physical['known'] and target['known']:
        identity=dict(known=True,reason=None,correct=bool(physical['correct'] and physical['visible']
            and physical['annotation_id'] in target['acceptable_annotation_ids']),
            matched_annotation=physical['annotation_id'],acceptable_annotation_ids=target['acceptable_annotation_ids'])
    return dict(object=physical,target=identity)


def coverage(records,key):
    known=[r for r in records if r[key]['known']]
    reasons=Counter(r[key].get('reason') or 'missing_annotation_support' for r in records if not r[key]['known'])
    return dict(requests=len(records),scored=len(known),UNKNOWN=len(records)-len(known),
        score=sum(bool(r[key].get('correct')) for r in known)/len(known) if known else None,
        unknown_reasons={k:reasons[k] for k in UNKNOWN_REASONS},
        scored_scenes=len({r['scene_id'] for r in known}),
        unique_scored_frame_questions=len({(r['frame_id'],r['question']) for r in known}))
