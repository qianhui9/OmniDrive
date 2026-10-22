"""Online, same-sensor temporal visibility evidence from the last committed image."""
from time import perf_counter
import cv2
import numpy as np
from .state_types import OnlineVisibilityEvidence
from .geometry_support import calibrated_correspondence, measured_patch_support

def gray_image(camera):
    image=camera.image
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) if image.ndim==3 else image.copy()


def build_evidence(previous, current, config):
    start = perf_counter()
    gray = gray_image(current)
    (h, w) = gray.shape
    points = np.empty((0, 2), np.float32)
    evidence = OnlineVisibilityEvidence(current.camera_id, previous.timestamp if previous else float('nan'), current.timestamp, points, points.copy(), np.empty(0), np.empty(0, bool), np.empty(0, bool), np.empty(0, bool), np.empty(0), np.zeros((h, w), bool), np.zeros((h, w), bool))
    d = evidence.diagnostics
    d.update(reason='no_history', temporal_correspondence_ms=0.0, visibility_verification_ms=0.0, geometry_support_ms=0.0)
    if previous is None:
        return evidence
    dt = current.timestamp - previous.timestamp
    if not np.isfinite(dt) or not 0 < dt <= config.max_state_age_s or previous.image.shape[:2] != (h, w) or (not previous.timing_reliable) or (not current.timing_reliable):
        d['reason'] = 'incompatible_history'
        return evidence
    old_gray = gray_image(previous)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=config.flow_max_points, qualityLevel=0.01, minDistance=config.flow_min_distance_px, blockSize=5)
    if p0 is None:
        p0 = np.empty((0, 1, 2), np.float32)
    if not len(p0):
        d['reason'] = 'insufficient_texture'
        return evidence
    (p1, s1, error) = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if p1 is None:
        d['reason'] = 'forward_failed'
        return evidence
    p1 = np.where(np.isfinite(p1), p1, p0)
    (back, s0, _) = cv2.calcOpticalFlowPyrLK(gray, old_gray, p1, None, winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if back is None:
        d['reason'] = 'backward_failed'
        return evidence
    (source, target) = (p0[:, 0], p1[:, 0])
    fb = np.linalg.norm(back[:, 0] - source, axis=1)
    good = s1[:, 0].astype(bool) & s0[:, 0].astype(bool) & (fb <= config.flow_fb_max_px)
    good &= np.isfinite(target).all(1) & (error[:, 0] <= config.flow_lk_max_error)
    good &= (target[:, 0] >= 0) & (target[:, 0] < w) & (target[:, 1] >= 0) & (target[:, 1] < h)
    cells = np.rint(target / 2).astype(np.int64)
    (_, inverse, counts) = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    collision = counts[inverse] > 1
    good &= ~collision
    d['temporal_correspondence_ms'] = (perf_counter() - start) * 1000
    t = perf_counter()
    geom = calibrated_correspondence(previous, current, source, target, config.flow_epipolar_max_px)
    d['visibility_verification_ms'] = (perf_counter() - t) * 1000
    t = perf_counter()
    (stable, changed, supported, residual) = measured_patch_support(old_gray, gray, source, target, good & geom, config)
    d['geometry_support_ms'] = (perf_counter() - t) * 1000
    evidence.source_points = source
    evidence.current_points = target
    evidence.fb_error_px = fb
    evidence.correspondence_valid = good
    evidence.geometry_valid = geom
    evidence.supported = supported
    evidence.patch_residual = residual
    evidence.stable_mask = stable
    evidence.changed_mask = changed
    if config.observation_diagnostics:
        from .state_types import json_value
        m = evidence.motion_measurements or {}
        d['point_measurements'] = json_value(dict(source_points=source, current_points=target, fb_error_px=fb, correspondence_valid=good, geometry_valid=geom, patch_supported=supported, patch_residual=residual, **{k: v for (k, v) in m.items() if k != 'source_points'}))
    d.update(reason='measured', points=len(source), bidirectional_points=int(good.sum()), geometric_points=int((good & geom).sum()), supported_points=int(supported.sum()), collision_count=int(collision.sum()), source='same_RGB_not_independent_sensor')
    return evidence

def anchor_visibility(evidence, old, projection, config, anchor=None, previous=None, current=None):
    """Certify the observed crop support, not foreground ownership or full box surface."""
    if evidence is None or old is None or abs(evidence.source_timestamp - old.timestamp) > 1e-05:
        return (False, 'temporal_source_missing', {})
    (p, q) = (evidence.source_points, evidence.current_points)
    (r, s) = (old.region, projection.region)
    inside = (p[:, 0] >= r[0]) & (p[:, 0] < r[2]) & (p[:, 1] >= r[1]) & (p[:, 1] < r[3])
    size = np.maximum(np.array(r[2:]) - r[:2], 1)
    expected = (p - np.array(r[:2])) / size * (np.array(s[2:]) - s[:2]) + np.array(s[:2])
    prediction_kind = 'legacy_box_affine'
    prediction_valid = np.ones(len(p), bool)
    residual = np.linalg.norm(q - expected, axis=1)
    bounded = prediction_valid & (residual <= projection.search_radius_px)
    in_current = (q[:, 0] >= s[0]) & (q[:, 0] < s[2]) & (q[:, 1] >= s[1]) & (q[:, 1] < s[3])
    accepted = inside & evidence.supported & bounded & in_current
    total = int(inside.sum())
    count = int(accepted.sum())
    quadrants = np.minimum(((p[accepted] - r[:2]) / size * 2).astype(int), 1)
    coverage = len(np.unique(quadrants, axis=0)) / 4 if count else 0.0
    fraction = count / total if total else 0.0
    d = dict(selected_points=total, supported_points=count, support_fraction=fraction, quadrant_fraction=coverage, source=evidence.source, foreground_ownership_certified=False, prediction_kind=prediction_kind, metric_geometry_certified=False, prediction_valid_points=int((inside & prediction_valid).sum()), prediction_residual_px=residual[inside].tolist())
    ok = count >= config.support_min_points and fraction >= config.support_point_fraction and (coverage >= config.support_quadrant_fraction)
    return (bool(ok), 'ok' if ok else 'temporal_support_insufficient', d)
