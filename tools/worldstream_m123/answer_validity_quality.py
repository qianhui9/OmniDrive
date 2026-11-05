"""R0-required utilities; historical alternative runners have been retired."""

from pathlib import Path

import numpy as np

from .world_guided_decode_v2_quality import qa_pairs, compare_facts

from .world_guided_decode_v3_report import facts

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
