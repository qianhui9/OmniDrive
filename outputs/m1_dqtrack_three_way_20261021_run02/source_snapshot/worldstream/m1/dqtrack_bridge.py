"""Fail-closed, commit-only nuScenes tracking bridge. No model dependencies.

The sidecar embedded as m1_manifest is mandatory: it establishes frame identity,
world coordinates and size order explicitly rather than inferring them from IDs.
"""
import json
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import numpy as np
from .association import associate
from .state_types import WorldAnchor
from .geometry import box_geometry


@lru_cache(maxsize=2)
def _load_results(path, mtime_ns, size):
    # Called exclusively from commit. At most two immutable JSON snapshots retained.
    with open(path) as handle:
        return json.load(handle)


def align(frame, predictions, config):
    start = perf_counter()
    diagnostics = dict(status='unavailable', available_tracks=0, aligned=0,
                       rejected=[], ambiguous=0, model_config=None)
    hints = {}
    try:
        path = Path(config.dqtrack_results_path).resolve()
        stat = path.stat()
        data = _load_results(str(path), stat.st_mtime_ns, stat.st_size)
        manifest = data['m1_manifest']
        if (manifest['coordinate_system'] != 'world' or manifest['size_order'] != 'wlh'
                or manifest['timestamp_unit'] != 'seconds'
                or not isinstance(manifest['run_id'], str) or not manifest['run_id']
                or not isinstance(manifest['model_config'], str) or not manifest['model_config']):
            raise ValueError('unverified_coordinate_or_model_contract')
        diagnostics['model_config'] = manifest['model_config']
        mapping = manifest['frames'][frame.frame_id]
        if (mapping['scene_id'] != frame.scene_id or
                not np.isfinite(mapping['timestamp']) or
                abs(mapping['timestamp'] - frame.timestamp) > 1e-6):
            raise ValueError('frame_scene_time_mismatch')
        token = mapping['sample_token']
        if not isinstance(token, str) or not token:
            raise ValueError('invalid_sample_token')
        # Ambiguous aliases of the same sample are rejected.
        if sum(m['sample_token'] == token for m in manifest['frames'].values()) != 1:
            raise ValueError('nonunique_frame_mapping')
        rows = data['results'][token]
        if not isinstance(rows, list):
            raise ValueError('invalid_results_rows')
        tracks = []
        identities = []
        ids = [r.get('tracking_id') for r in rows if isinstance(r, dict)]
        for index, row in enumerate(rows):
            try:
                if not isinstance(row, dict):
                    raise ValueError('invalid_track_row')
                tid = row['tracking_id']
                if (row['sample_token'] != token or not isinstance(tid, str) or not tid
                        or ids.count(tid) != 1 or not isinstance(row['tracking_name'], str)):
                    raise ValueError('invalid_or_duplicate_identity')
                center = np.asarray(row['translation'], float)
                size = np.asarray(row['size'], float)
                rotation = np.asarray(row['rotation'], float)
                score = float(row['tracking_score'])
                if (center.shape != (3,) or size.shape != (3,) or rotation.shape != (4,)
                        or not np.isfinite(np.r_[center, size, rotation, score]).all()
                        or np.any(size <= 0) or abs(np.linalg.norm(rotation)-1) > 1e-3
                        or not 0 <= score <= 1):
                    raise ValueError('invalid_geometry_or_score')
                if 'velocity' in row:
                    velocity = np.asarray(row['velocity'], float)
                    if velocity.shape != (2,) or not np.isfinite(velocity).all():
                        raise ValueError('invalid_velocity')
                # M1 uses ego box dims (length,width,height); nuScenes size is w,l,h.
                w, x, y, z = rotation / np.linalg.norm(rotation)
                pose = np.eye(4)
                pose[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
                pose[:3, 3] = center
                g = box_geometry(np.zeros(3), size[[1, 0, 2]], to_world=pose)
                tracks.append(WorldAnchor(str(index), frame.scene_id, 'object',
                                          row['tracking_name'], g, frame.timestamp))
                identities.append(dict(namespace=[manifest['run_id'], frame.scene_id],
                    tracking_id=tid, sample_token=token, frame_id=frame.frame_id,
                    timestamp=frame.timestamp, tracking_score=score,
                    rotation_wxyz=rotation.tolist(), certified=False))
            except (KeyError, ValueError, TypeError, OverflowError) as exc:
                diagnostics['rejected'].append(dict(index=index, reason=str(exc)))
        diagnostics['available_tracks'] = len(tracks)
        pairs, costs = associate(tracks, predictions, frame.timestamp, config)
        finite = np.isfinite(costs)
        for i, j in pairs:
            # Every competing legal edge rejects, even if Hungarian prefers one.
            if finite[i].sum() != 1 or finite[:, j].sum() != 1:
                diagnostics['ambiguous'] += 1
                continue
            hints[j] = dict(identities[i], status='aligned')
        matched_tracks = {i for i,j in pairs if j in hints}
        diagnostics['unaligned_records'] = [dict(tracking_id=identities[i]['tracking_id'],
            reason='competing_legal_candidates' if finite[i].any() else 'no_legal_class_distance_size_edge')
            for i in range(len(tracks)) if i not in matched_tracks]
        diagnostics['unaligned_tracks'] = len(tracks)-len(hints)
        diagnostics['alignments'] = hints
        diagnostics.update(status='ok', aligned=len(hints))
    except (OSError, KeyError, ValueError, TypeError, OverflowError, AttributeError, IndexError) as exc:
        diagnostics['rejected'].append(dict(reason=str(exc)))
    diagnostics['elapsed_ms'] = (perf_counter()-start)*1000
    return hints, diagnostics
