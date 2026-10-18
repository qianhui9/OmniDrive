"""Class/geometry gates followed by one distance cost; no weighted scores."""
import numpy as np
from .geometry import predict_geometry, sample_polyline


def legal_assignment(costs):
    """Maximum-cardinality legal assignment, then minimum distance.

    SciPy is required. A greedy fallback silently changes the cardinality objective.
    """
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 2 or np.isnan(costs).any() or (costs < 0).any():
        raise ValueError('Expected a nonnegative 2D distance matrix with +inf forbidden edges')
    if costs.size == 0:
        return []
    finite = np.isfinite(costs)
    if not finite.any():
        return []
    from scipy.optimize import linear_sum_assignment
    # Normalize to avoid overflow; forbidden cost dominates the sum of legal costs.
    scale = max(1., float(costs[finite].max()))
    rows, cols = linear_sum_assignment(np.where(finite, costs / scale, max(costs.shape) + 1.))
    return [(int(i), int(j)) for i, j in zip(rows, cols) if finite[i, j]]


def association_ambiguities(pairs, costs, margin=0.):
    """Reject competing edges in either direction, without inventing identity proof.

    This is a rejection gate, not a probability or a calibrated error guarantee.
    An assignment forced away from either endpoint's best edge also rejects.
    """
    result = {}
    for i, j in pairs:
        row = np.delete(costs[i], j)
        col = np.delete(costs[:, j], i)
        alternative = min(float(row.min()) if row.size else float('inf'),
                          float(col.min()) if col.size else float('inf'))
        gap = alternative - costs[i, j]
        result[(i, j)] = dict(ambiguous=bool(gap <= margin + 1e-12),
                              margin_m=float(gap), threshold_m=margin)
    return result


def assignment_edge_alternatives(pairs, costs):
    """OFFLINE diagnostic: remove each chosen edge and re-solve the full problem.

    Losing cardinality is distinct from a distance margin. Local competing
    nearest neighbours can be globally forced; no online gate is changed here.
    """
    costs=np.asarray(costs,float);base=len(pairs)
    total=float(sum(costs[i,j] for i,j in pairs));result={}
    for i,j in pairs:
        counterfactual=costs.copy();counterfactual[i,j]=np.inf
        alternative=legal_assignment(counterfactual)
        cardinality=len(alternative)
        result[i,j]=dict(alternative_cardinality=cardinality,cardinality_lost=base-cardinality,
            global_cost_margin_m=(float(sum(costs[a,b] for a,b in alternative))-total) if cardinality==base else None,
            alternative_pairs=alternative)
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
