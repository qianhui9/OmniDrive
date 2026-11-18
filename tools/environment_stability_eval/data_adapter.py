"""Authoritative offline GT and unchanged factual-QA query/evaluator adapters.

No prediction, image difference, or online cache is used to construct GT.
"""
from collections import Counter
import math
from pathlib import Path
import pickle

from .common import ROOT, read


def load_infos(config):
    with (ROOT / config['infos']).open('rb') as handle:
        infos = pickle.load(handle)['infos']
    by = {i['token']: i for i in infos}
    if len(by) != len(infos):
        raise ValueError('Duplicate frame tokens in info pickle')
    return infos, by


def validate_frames(scenes, by, check_images=True):
    seen = set()
    for scene in scenes:
        frames = [by[t] for t in scene['tokens']]
        if not frames or len({i['timestamp'] for i in frames}) != len(frames):
            raise ValueError('Empty sequence or duplicated timestamp')
        for info in frames:
            token = info['token']
            if token in seen or info['scene_token'] != scene['scene_id']:
                raise ValueError('Duplicate frame or mismatched sequence: ' + token)
            seen.add(token)
            if not isinstance(info['timestamp'], (int, float)) or not math.isfinite(info['timestamp']):
                raise ValueError('Invalid timestamp')
            if len(info['cams']) != 6:
                raise ValueError('Expected exactly six cameras: ' + token)
            for cam in info['cams'].values():
                for key in ('cam_intrinsic', 'sensor2ego_rotation', 'sensor2ego_translation',
                            'ego2global_rotation', 'ego2global_translation', 'timestamp'):
                    if key not in cam:
                        raise ValueError('Missing camera geometry: ' + key)
                if check_images and not (ROOT / cam['data_path']).is_file():
                    raise FileNotFoundError(cam['data_path'])
        for a, b in zip(frames, frames[1:]):
            if a['timestamp'] >= b['timestamp'] or a.get('next') != b['token'] or b.get('prev') != a['token']:
                raise ValueError('Non-contiguous or reordered sequence: ' + scene['scene_id'])


def select_scenes(config, infos):
    from tools.worldstream_m123.r0_task_evaluation import selected_scenes
    selected = selected_scenes(infos, config['scope'], ROOT / Path(config['main_manifest']).parent, None)
    selected_ids = {s['scene_id'] for s in selected}
    # Main 18 scenes form the evaluation set; the other val scenes calibrate GT only.
    remaining = [s for s in selected_scenes(infos, 'full-val', ROOT, None)
                 if s['scene_id'] not in selected_ids]
    if config['calibration'] == 'remaining-val-scenes' and not remaining:
        raise ValueError('No disjoint calibration scenes; select calibration=evaluation-GT explicitly')
    calibration = remaining if config['calibration'] == 'remaining-val-scenes' else selected
    return selected, calibration


def load_objects(config, tokens):
    from tools.worldstream_m123.answer_validity_quality import load_current_annotations
    from tools.worldstream_m1.io_utils import iter_json_array
    table = ROOT / config['data_root'] / 'v1.0-trainval'
    # Missing sample rows must never turn an unavailable frame into an empty scene.
    samples = {s['token'] for s in iter_json_array(table / 'sample.json') if s['token'] in tokens}
    if set(tokens) - samples:
        raise ValueError('Missing authoritative sample rows: ' + repr(sorted(set(tokens) - samples)))
    objects = load_current_annotations(ROOT / config['infos'], set(tokens), ROOT / config['data_root'])
    attrs = {a['token']: a['name'] for a in iter_json_array(table / 'attribute.json')}
    indexed = {g['annotation_id']: g for values in objects.values() for g in values}
    # Extend only the offline annotations with authoritative attributes, never the model.
    for ann in iter_json_array(table / 'sample_annotation.json'):
        if ann['token'] in indexed:
            indexed[ann['token']]['attributes'] = sorted(attrs[t] for t in ann.get('attribute_tokens', []))
    for values in objects.values():
        if len({g['instance_id'] for g in values}) != len(values):
            raise ValueError('Duplicate instance in a single GT sample')
    return objects


def query_gt(question, token, objects):
    """Represent ONLY facts required by the existing condition-blind evaluator.

    Target identity changes alone are not answer changes. Negative visual
    existence is UNKNOWN under the main benchmark's positive-witness policy.
    """
    from worldstream.m3.task_conditioner import question_facts
    from tools.worldstream_m123.fact_bypass_identity import question_target
    from tools.worldstream_m123.world_guided_decode_v3_report import facts
    from tools.worldstream_m123.target_complete_report import qa_references
    from tools.worldstream_m123.selective_answer_reuse_report import label
    profile = question_facts(question)
    if token not in objects:
        return dict(known=False, state=None, reason='missing_current_annotation')
    if profile.intent == 'unsupported':
        reference = qa_references(token, 'val').get(question)
        state = facts(question, reference) if isinstance(reference, str) and reference.strip() else {}
        # JSON representation normalizes tuples and sets to stable lists.
        import json
        state = json.loads(json.dumps(state))
        return dict(known=bool(state), state=state or None, reference=reference,
                    source='exact_current_frame_original_QA', fact_type='original_QA_limited_facts',
                    reason=None if state else 'no_reliable_reference_facts')
    target = question_target(question, objects[token])
    if not target.get('answer_known'):
        return dict(known=False, state=None, reason=target.get('reason'))
    candidates = [g for g in objects[token] if g['annotation_id'] in target['acceptable_annotation_ids']]
    if not target['known']:
        candidates = candidates[:1]
    reference = label(question, candidates)
    if not reference['known']:
        return dict(known=False, state=None, reason=reference.get('reason'))
    if profile.intent == 'category':
        state = dict(semantic_class=reference['semantic_class'])
        kind = 'category'
    elif profile.intent == 'position':
        state = dict(directions=sorted(reference['directions']))
        kind = 'relative_position'
    elif profile.intent == 'existence':
        state = dict(present=True, acceptable_classes=reference['available_classes'])
        kind = 'existence'
    elif profile.intent == 'motion':
        state = dict(motion=reference['motion'])
        kind = 'attribute'
    else:
        return dict(known=False, state=None, reason='unsupported_GT_state')
    return dict(known=True, state=state, reference=reference, fact_type=kind,
                source='current_sample_annotation_reverse_link', reason=None)


def build_query_records(config, scenes, by, objects):
    from tools.worldstream_m123.r0_task_evaluation import facts_questions
    history, result, question_ids = {}, [], {}
    for scene in scenes:
        for token in scene['tokens']:
            questions, groups = facts_questions(token, ROOT / config['data_root'], config['other_questions'])
            if len(questions) != len(set(questions)):
                raise ValueError('Duplicate query in frame: ' + token)
            for qi, (question, group) in enumerate(zip(questions, groups)):
                qid = question_ids.setdefault(question, f'q{len(question_ids):05d}')
                gt = query_gt(question, token, objects)
                key = (scene['scene_id'], question)
                previous = history.get(key)
                comparable = bool(gt['known'] and previous and
                                  set(gt['state']) == set(previous['ground_truth']['state']))
                label_value = ('Stable' if gt['state'] == previous['ground_truth']['state'] else 'Changed') if comparable else 'Unclassified'
                row = dict(scene_id=scene['scene_id'], frame_id=token, timestamp=by[token]['timestamp'] / 1e6,
                           query_id=qid, query_index=qi, question=question, group=group,
                           ground_truth=gt, fact_status=label_value,
                           previous_frame_id=previous['frame_id'] if comparable else None,
                           previous_timestamp=previous['timestamp'] if comparable else None,
                           previous_ground_truth=previous['ground_truth'] if comparable else None,
                           change_type=gt.get('fact_type') if label_value == 'Changed' else None)
                result.append(row)
                if gt['known']:
                    history[key] = row
    return result


def coverage_report(query_rows, objects):
    known = [r for r in query_rows if r['ground_truth']['known']]
    return dict(frames=len(objects), annotated_objects=sum(map(len, objects.values())),
                visible_objects=sum(g['visible'] for values in objects.values() for g in values),
                objects_with_attribute_GT=sum(bool(g.get('attributes')) for values in objects.values() for g in values),
                queries=len(query_rows), known_queries=len(known),
                query_GT_coverage=len(known) / len(query_rows) if query_rows else None,
                known_by_group=dict(Counter(r['group'] for r in known)),
                queries_by_group=dict(Counter(r['group'] for r in query_rows)),
                status_counts=dict(Counter(r['fact_status'] for r in query_rows)),
                unknown_reasons=dict(Counter(r['ground_truth'].get('reason') for r in query_rows if not r['ground_truth']['known'])),
                supported_environment_facts=['annotation-set presence', 'visible ego-relative direction sector',
                    'persistent-instance class', 'nonempty official attributes', 'visible object-pair direction sector'],
                limitations=['Annotation-set entry/exit is not proof of physical creation/destruction or visual absence.',
                    'Missing attributes are unknown, never an empty known property.',
                    'Negative existence remains UNKNOWN under the reused main QA evaluator.',
                    'Original QA supports only the existing finite fact parser; partial scores are retained.',
                    'Object-pair relations support environment grouping, not unrestricted relation-QA scoring.'])
