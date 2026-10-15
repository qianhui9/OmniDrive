"""Class/geometry gates followed by one distance cost; no weighted scores."""
import numpy as np
from .geometry import predict_geometry, sample_polyline


def legal_assignment(costs):
    """Maximum-cardinality legal assignment, then minimum distance; deterministic fallback."""
    costs = np.asarray(costs, dtype=float)
    if costs.size == 0:
        return []
    finite = np.isfinite(costs)
    if not finite.any():
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        # Dominates any sum of legitimate costs, so forbidden edges cannot steal a legal match.
        penalty = (max(costs.shape) + 1) * (1 + costs[finite].max())
        rows, cols = linear_sum_assignment(np.where(finite, costs, penalty))
        return [(int(i), int(j)) for i, j in zip(rows, cols) if finite[i, j]]
    except ImportError:
        edges = sorted((costs[i, j], i, j) for i, j in zip(*np.where(finite)))
        rows, cols, result = set(), set(), []
        for _, i, j in edges:
            if i not in rows and j not in cols:
                result.append((i, j)); rows.add(i); cols.add(j)
        return result


def associate(anchors, predictions, timestamp, config):
    """Same legal gates as scalar matching, with curves sampled once per input."""
    costs = np.full((len(anchors), len(predictions)), np.inf)
    groups = {}
    curves = {}
    for j, new in enumerate(predictions):
        groups.setdefault((new.anchor_type, new.semantic_class), []).append(j)
        if new.anchor_type != "object":
            try:
                curves[j] = sample_polyline(new.geometry_world.points)
            except ValueError:
                pass
    for i, old in enumerate(anchors):
        if timestamp - old.source_timestamp > config.max_state_age_s:
            continue
        try:
            g = predict_geometry(old, timestamp)
        except ValueError:
            continue
        indices = groups.get((old.anchor_type, old.semantic_class), [])
        if old.anchor_type == "object":
            if g.size is None:
                continue
            indices = [j for j in indices if predictions[j].geometry_world.size is not None]
            if not indices:
                continue
            sizes = np.asarray([predictions[j].geometry_world.size for j in indices])
            centers = np.asarray([predictions[j].geometry_world.center for j in indices])
            ratios = np.maximum(g.size / sizes, sizes / g.size)
            distances = np.linalg.norm(g.center - centers, axis=1)
            legal = (~np.any(ratios > config.box_size_ratio, axis=1)
                     & (distances <= config.association_gate_m))
            costs[i, np.asarray(indices)[legal]] = distances[legal]
        else:
            try:
                a = sample_polyline(g.points)
            except ValueError:
                continue
            da = a[-1] - a[0]
            for j in indices:
                if j not in curves:
                    continue
                b = curves[j]
                db = b[-1] - b[0]
                cosine = abs(np.dot(da, db)) / max(np.linalg.norm(da) * np.linalg.norm(db), 1e-8)
                if cosine < config.map_direction_cos:
                    continue
                distance = min(np.linalg.norm(a - b, axis=1).mean(),
                               np.linalg.norm(a - b[::-1], axis=1).mean())
                if distance <= config.map_gate_m:
                    costs[i, j] = distance
    return legal_assignment(costs), costs
