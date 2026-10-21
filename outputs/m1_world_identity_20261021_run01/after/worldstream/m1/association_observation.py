"""Commit-only RGB disambiguation on the original legal geometry graph.

Missing descriptors are unknown. Only measured mismatches remove edges; a
component changes only when both endpoints have a complete unique bijection.
This is an unqualified association hypothesis, never a physical certificate.
"""
import numpy as np
from .geometry import project_anchor
from .state_types import WorldAnchor


def refine(anchors, predictions, costs, accepted, frame, last_timestamp, config, validator):
    legal = np.isfinite(costs)
    compatible = legal.copy()
    measured = {}
    descriptors = {}
    for i, j in zip(*np.nonzero(legal)):
        if legal[i].sum()==1 and legal[:, j].sum()==1:
            continue
        old = anchors[i]
        if (old.anchor_type != 'object' or old.scene_id != frame.scene_id
                or old.source_timestamp != last_timestamp
                or not 0 < frame.timestamp-old.source_timestamp <= config.max_state_age_s
                or old.source_execution_signature != frame.execution_signature):
            continue
        residuals = []
        for cid, support in old.observation_support.items():
            cam = frame.cameras.get(cid)
            if (cam is None or not cam.timing_reliable or support.timestamp != old.source_timestamp
                    or abs(cam.timestamp-frame.timestamp) > config.max_camera_skew_s):
                continue
            key = (j, cid)
            if key not in descriptors:
                pred = predictions[j]
                target = WorldAnchor('', frame.scene_id, pred.anchor_type, pred.semantic_class,
                                     pred.geometry_world, frame.timestamp,
                                     geometry_uncertainty=pred.geometry_uncertainty)
                projection = project_anchor(target, cam, config)
                descriptors[key] = validator.descriptor(cam.image, projection.region) if projection.valid else None
            desc = descriptors[key]
            if desc is not None:
                residuals.append(validator.residual(support.descriptor, desc))
        if residuals:
            measured[int(i), int(j)] = residuals
            compatible[i, j] = all(np.isfinite(v) and v <= config.residual_threshold for v in residuals)
    replacements = set()
    replaced_rows = set()
    replaced_cols = set()
    seen = set()
    for start in range(len(anchors)):
        if start in seen or not legal[start].any():
            continue
        rows, cols = {start}, set()
        while True:
            new_cols = set(np.flatnonzero(legal[list(rows)].any(axis=0)))
            new_rows = set(np.flatnonzero(legal[:, list(new_cols)].any(axis=1)))
            if new_rows == rows and new_cols == cols:
                break
            rows, cols = new_rows, new_cols
        seen |= rows
        if len(rows) < 2 or len(rows) != len(cols):
            continue
        rs, cs = sorted(rows), sorted(cols)
        sub = compatible[np.ix_(rs, cs)]
        if not (np.all(sub.sum(axis=0) == 1) and np.all(sub.sum(axis=1) == 1)):
            continue
        pairs = {(rs[a], cs[b]) for a, b in zip(*np.nonzero(sub))}
        # Every accepted edge needs actual positive RGB, never unknown evidence.
        if not all(edge in measured for edge in pairs):
            continue
        if pairs == {edge for edge in accepted if edge[0] in rows}:
            continue
        replacements |= pairs
        replaced_rows |= rows
        replaced_cols |= cols
    result = [(i, j) for i, j in accepted if i not in replaced_rows and j not in replaced_cols]
    result.extend(sorted(replacements))
    return result, replacements, dict(measured_edges=len(measured),
        mismatching_edges=int((legal & ~compatible).sum()),
        changed_pairs=[list(e) for e in sorted(replacements)], certified=False)
