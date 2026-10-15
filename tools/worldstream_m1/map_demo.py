"""Annotation-assisted real OpenLane map continuity demo; not live Map Head inference."""
import argparse
import pickle
from pathlib import Path
import numpy as np
from worldstream.m1 import FullOutputs, Geometry, PerceptionAnchor, WorldStateManager, WorldAnchor
from worldstream.m1.geometry import transform_points, project_anchor
from .dataset import select_infos, frame_from_info
from .io_utils import write_json
from .visualize import draw_frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--infos', default='data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    p.add_argument('--lanes', default='data/nuscenes/data_dict_sample.pkl')
    p.add_argument('--data-root', default='data/nuscenes')
    p.add_argument('--frames', type=int, default=6)
    p.add_argument('--config', default=None)
    p.add_argument('--output', default='outputs/module1/map_annotation_demo')
    args = p.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    infos = select_infos(args.infos, limit=args.frames)
    with open(args.lanes, 'rb') as handle: lanes = pickle.load(handle)
    manager = WorldStateManager(args.config)
    rows = []
    for index, info in enumerate(infos):
        frame = frame_from_info(info, args.data_root)
        result = manager.precheck(frame)
        # This annotation access occurs after precheck and is explicitly labelled.
        annotations = lanes[info['lane_info']]['annotation']['lane_centerline']
        anchors = []
        frame.metadata['current_map_polylines'] = {cid: [] for cid in frame.cameras}
        for line in annotations:
            points = transform_points(line['points'], frame.ego_to_world)
            if len(points) < 2 or not np.isfinite(points).all(): continue
            anchors.append(PerceptionAnchor('map', 'lane_annotation_polyline', Geometry(points, points.mean(0))))
            a = WorldAnchor('current', frame.scene_id, 'map', 'lane_annotation_polyline', anchors[-1].geometry_world, frame.timestamp)
            for cid, camera in frame.cameras.items():
                projected = project_anchor(a, camera, manager.config)
                if projected.valid: frame.metadata['current_map_polylines'][cid].append(projected.uv)
        manager.commit(frame, FullOutputs(anchors, source='OpenLane_annotation_assisted_map_commit'), result)
        row = {'frame_id': frame.frame_id, 'timestamp': frame.timestamp, 'current_lines': len(anchors),
               'matched_historical_lines': len(manager.last_commit_diagnostics['association_pairs']),
               'valid': len(result.valid_anchor_ids), 'invalid': len(result.invalid_anchor_ids),
               'timings': manager.last_commit_diagnostics}
        rows.append(row)
        draw_frame(frame, result, root/'visualizations'/('{:03d}.jpg'.format(index)))
    write_json(root/'results.json', {'source': 'Real images/calibration and OpenLane annotations in commit only',
        'no_model_map_outputs_available': True, 'no_map_ID_ground_truth_claim': True, 'frames': rows})
    print(root/'results.json')


if __name__ == '__main__': main()
