"""Offline-only current-frame labels for WS-AVC and active v3 requests.

Camera-visible nuScenes annotations establish positive object witnesses, not
exhaustive visible absence. Current QA fact recall and annotation assertion
checks have separate denominators. Neither metric is full driving accuracy.
"""
from pathlib import Path
import re

import numpy as np

from worldstream.m3.answer_validity import validity_task
from .world_guided_decode_v2_quality import qa_pairs, compare_facts
from .world_guided_decode_v3_report import facts


CLASS_PATTERNS = (
    ('pedestrian', r'pedestrians?|people|persons?'),
    ('car', r'cars?|sedans?|suvs?|vans?|taxis?'),
    ('truck', r'trucks?'), ('bus', r'buses|bus'),
    ('motorcycle', r'motorcycles?'), ('bicycle', r'bicycles?|cyclists?|bikes?'),
    ('vehicle', r'vehicles?'))
VEHICLES = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'motorcycle', 'bicycle'}


def original_qa_grade(frame_id, question, answer):
    refs = qa_pairs(frame_id)
    reference = refs.get(question)
    if not isinstance(reference, str) or not reference.strip():
        return dict(status='NOT_SCORED', reason='no_exact_current_frame_QA', score=None, facts=[])
    expected = facts(question, reference)
    if not expected:
        return dict(status='NOT_SCORED', reason='current_reference_facts_unknown', score=None,
                    reference=reference, facts=[])
    checks = compare_facts(expected, facts(question, answer))
    return dict(status='SCORED_TASK_FACTS', reference=reference,
                score=sum(x['correct'] for x in checks) / len(checks), facts=checks,
                source='exact_current_frame_original_QA', metric='limited_task_fact_recall')


def object_claims(answer):
    """Only explicit assertions; no cross-object direction binding."""
    result = []
    for text in re.split(r'[.!?;]|\b(?:but|however|while)\b', str(answer).lower()):
        if not text.strip():
            continue
        found = [(cls, m) for cls, pattern in CLASS_PATTERNS
                 for m in [re.search(r'\b(?:' + pattern + r')\b', text)] if m]
        if not found:
            continue
        neg = bool(re.search(r"\b(no|none|not|without|aren't|isn't)\b", text))
        ambiguous = bool(re.search(r'\b(may|might|could|perhaps|possibly|should|if|expect|potential)\b', text))
        behaviour = bool(re.search(r'\b(crossing|waiting to cross|crosswalk|oncoming|following)\b', text))
        directions = tuple(d for d, p in [('front', 'front|ahead'), ('rear', 'rear|behind'),
                                          ('left', 'left'), ('right', 'right')]
                           if re.search(r'\b(?:' + p + r')\b', text))
        for cls, _ in found:
            result.append(dict(semantic_class=cls, negative=neg,
                               directions=directions if len(found) == 1 else (),
                               near=bool(re.search(r'\b(near|nearby|vicinity|close|immediate)\b', text)),
                               direct=bool(re.search(r'\bdirectly\b', text)),
                               unknown=ambiguous or behaviour or (len(found) > 1 and bool(directions)),
                               clause=text.strip()))
    return result


def in_region(center, claim):
    x, y = center[:2]
    if claim['near'] and np.hypot(x, y) > 30.:
        return False
    if claim['direct'] and abs(y) > 4.:
        return False
    return all({'front': x > 0, 'rear': x < 0, 'left': y > 0, 'right': y < 0}[d]
               for d in claim['directions'])


def annotation_grade(answer, objects, question=''):
    checks = []
    task = validity_task(question)
    for claim in object_claims(answer):
        # A subordinate "but no pedestrians" remains scoped to the actual
        # question. It must not silently become an all-six-camera absence.
        if not claim['directions'] and task.directions:
            claim = dict(claim, directions=task.directions)
        if 'near' in task.attributes:
            claim = dict(claim, near=True)
        if task.signature[-1] == 'direct':
            claim = dict(claim, direct=True)
        classes = VEHICLES if claim['semantic_class'] == 'vehicle' else {claim['semantic_class']}
        witnesses = [o for o in objects if o['semantic_class'] in classes and o['visible']
                     and in_region(o['center_ego'], claim)]
        # Absence of annotations does not prove absence of a visible object.
        correct = (not claim['negative']) if witnesses and not claim['unknown'] else None
        checks.append(dict(claim=claim, correct=correct,
                           reason='visible_current_annotation_witness' if correct is not None
                           else 'behaviour_scope_unknown' if claim['unknown'] else 'no_positive_witness_UNKNOWN',
                           witness_ids=[o['annotation_id'] for o in witnesses]))
    scored = [c for c in checks if c['correct'] is not None]
    return dict(status='SCORED_PARTIAL_ASSERTIONS' if scored else 'NOT_SCORED',
                score=sum(c['correct'] for c in scored) / len(scored) if scored else None,
                checks=checks, scored_assertions=len(scored), unknown_assertions=len(checks) - len(scored),
                metric='positive_witness_partial_assertion_consistency', full_answer_validated=False,
                proximity_protocol='near <=30m; directly |ego_y|<=4m: operational scope, not human-validated',
                conflict_status='AUTOMATED_ANNOTATION_FINDING_REQUIRES_IMAGE_REVIEW')


def load_current_annotations(infos_path, frame_ids, data_root):
    """Stream authoritative reverse links; use each camera's exposure pose.

    No online module imports this reader. No future trajectories are loaded.
    """
    import pickle
    from pyquaternion import Quaternion
    from tools.worldstream_m1.io_utils import iter_json_array

    def transform(rotation, translation):
        t = np.eye(4)
        t[:3, :3] = Quaternion(rotation).rotation_matrix
        t[:3, 3] = translation
        return t

    with Path(infos_path).open('rb') as handle:
        infos = {i['token']: i for i in pickle.load(handle)['infos'] if i['token'] in frame_ids}
    if set(frame_ids) - set(infos):
        raise ValueError('Missing current-frame metadata: ' + repr(sorted(set(frame_ids) - set(infos))))
    table = Path(data_root) / 'v1.0-trainval'
    categories = {r['token']: r['name'] for r in iter_json_array(table / 'category.json')}
    instances = {r['token']: categories[r['category_token']] for r in iter_json_array(table / 'instance.json')}
    objects = {t: [] for t in frame_ids}
    for ann in iter_json_array(table / 'sample_annotation.json'):
        token = ann['sample_token']
        if token not in infos:
            continue
        name = instances[ann['instance_token']]
        cls = None
        if name.startswith('human.pedestrian.'):
            cls = 'pedestrian'
        elif name == 'vehicle.construction':
            cls = 'construction_vehicle'
        elif name == 'movable_object.barrier':
            cls = 'barrier'
        elif name == 'movable_object.trafficcone':
            cls = 'traffic_cone'
        else:
            for c in sorted(VEHICLES, key=len, reverse=True):
                if name == 'vehicle.' + c or name.startswith('vehicle.' + c + '.'):
                    cls = c
                    break
        if cls is None:
            continue
        info = infos[token]
        world = np.r_[ann['translation'], 1.]
        ego = np.linalg.inv(transform(info['ego2global_rotation'], info['ego2global_translation'])) @ world
        cameras = []
        for cid, cam in info['cams'].items():
            c2w = transform(cam['ego2global_rotation'], cam['ego2global_translation']) @ transform(
                cam['sensor2ego_rotation'], cam['sensor2ego_translation'])
            point = np.linalg.inv(c2w) @ world
            projection = np.array(cam['cam_intrinsic']) @ point[:3]
            if point[2] > 0 and 0 <= projection[0] / point[2] < 1600 and 0 <= projection[1] / point[2] < 900:
                cameras.append(cid)
        visible = bool(cameras and int(ann.get('visibility_token') or 0) >= 3
                       and ann['num_lidar_pts'] + ann['num_radar_pts'] > 0)
        objects[token].append(dict(annotation_id=ann['token'], instance_id=ann['instance_token'],
                                   semantic_class=cls, center_ego=ego[:3].tolist(), visible=visible,
                                   cameras=cameras, source_frame=token,
                                   source='current_sample_annotation_reverse_link'))
    return objects


def coverage_summary(events):
    """Keep the complete request denominator, including UNKNOWN outcomes."""
    groups = []
    for key in sorted({(e['condition'], e['workload'], e['execution']) for e in events}):
        part = [e for e in events if (e['condition'], e['workload'], e['execution']) == key]
        qa = [e['qa'] for e in part if e['qa']['score'] is not None]
        assertions = [c for e in part for c in e['annotations']['checks'] if c['correct'] is not None]
        groups.append(dict(condition=key[0], workload=key[1], execution=key[2], requests=len(part),
                           exact_current_QA_scored=len(qa), QA_coverage=len(qa) / len(part),
                           mean_task_fact_recall=sum(x['score'] for x in qa) / len(qa) if qa else None,
                           annotation_scored_requests=sum(e['annotations']['score'] is not None for e in part),
                           annotation_scored_assertions=len(assertions),
                           annotation_false_assertions=sum(not c['correct'] for c in assertions),
                           annotation_partial_consistency=sum(c['correct'] for c in assertions) / len(assertions) if assertions else None,
                           independent_frames=len({e['frame_id'] for e in part}),
                           full_driving_quality_qualified=False))
    return groups
