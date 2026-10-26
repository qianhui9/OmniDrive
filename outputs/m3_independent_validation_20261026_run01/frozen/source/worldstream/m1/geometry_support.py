"""Calibrated static-scene correspondence and bounded, measured pixel support.

No box fills, global homographies, semantic background labels or token claims.
"""
import cv2
import numpy as np
from .geometry import valid_transform

def calibrated_correspondence(previous, current, source, target, tolerance):
    n = len(source)
    rejected = np.zeros(n, bool)
    if not n or not all((valid_transform(c.world_to_camera) and c.timing_reliable for c in (previous, current))):
        return rejected
    try:
        inv0 = np.linalg.inv(previous.intrinsic)
        inv1 = np.linalg.inv(current.intrinsic)
        if not np.isfinite(inv0).all() or not np.isfinite(inv1).all():
            return rejected
        relative = current.world_to_camera @ np.linalg.inv(previous.world_to_camera)
        (r, t) = (relative[:3, :3], relative[:3, 3])
        x = np.c_[source, np.ones(n)] @ inv0.T
        y = np.c_[target, np.ones(n)] @ inv1.T
        if np.linalg.norm(t) < 0.02:
            rotated = x @ r.T
            uv = rotated @ current.intrinsic.T
            return (rotated[:, 2] > 0) & (np.linalg.norm(uv[:, :2] / uv[:, 2:] - target, axis=1) <= tolerance)
        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        f = inv1.T @ tx @ r @ inv0
        a = np.c_[source, np.ones(n)]
        b = np.c_[target, np.ones(n)]
        (la, lb) = (a @ f.T, b @ f)
        residual = np.abs(np.sum(b * la, axis=1))
        distance = np.maximum(residual / np.maximum(np.linalg.norm(la[:, :2], axis=1), 1e-10), residual / np.maximum(np.linalg.norm(lb[:, :2], axis=1), 1e-10))
        xyz = cv2.triangulatePoints(np.c_[np.eye(3), np.zeros(3)], relative[:3], x[:, :2].T.copy(), y[:, :2].T.copy())
        with np.errstate(invalid='ignore', divide='ignore'):
            xyz = (xyz[:3] / xyz[3:]).T
        z0 = xyz[:, 2]
        z1 = (xyz @ r.T + t)[:, 2]
        return (distance <= tolerance) & np.isfinite(xyz).all(1) & (z0 > 0.5) & (z1 > 0.5) & (z0 < 200) & (z1 < 200)
    except (ValueError, np.linalg.LinAlgError, cv2.error):
        return rejected

def measured_patch_support(old_gray, gray, source, target, eligible, config):
    """Vectorized subpixel patches; mark only pixels whose actual samples agree.

    Patch-wide support and texture are required before marking individual pixels.
    Residual failure alone is not semantic change: only a geometrically matched,
    bidirectional patch with a majority of consistent pixels can expose changed pixels.
    """
    (h, w) = gray.shape
    (stable, changed) = (np.zeros((h, w), bool), np.zeros((h, w), bool))
    residual = np.full(len(source), np.inf, np.float32)
    supported = np.zeros(len(source), bool)
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return (stable, changed, supported, residual)
    rad = config.support_patch_radius_px
    (yy, xx) = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    offsets = np.stack([xx.ravel(), yy.ravel()], -1)
    target_grid = np.rint(target[indices]).astype(np.float32)[:, None, :] + offsets
    source_grid = target_grid + (source[indices] - target[indices])[:, None, :]
    boundary = ((target_grid[..., 0] >= 1) & (target_grid[..., 0] < w - 1) & (target_grid[..., 1] >= 1) & (target_grid[..., 1] < h - 1) & (source_grid[..., 0] >= 1) & (source_grid[..., 0] < w - 1) & (source_grid[..., 1] >= 1) & (source_grid[..., 1] < h - 1)).all(1)

    def sample(im, grid):
        return cv2.remap(im, grid[..., 0].astype(np.float32), grid[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    def gray_measurement(gray, derivative=None):
        image=gray.astype(np.float32)/255.
        return image if derivative is None else cv2.Sobel(image,cv2.CV_32F,int(derivative==0),int(derivative==1),ksize=3)/8
    a = sample(gray_measurement(old_gray), source_grid)
    b = sample(gray_measurement(gray), target_grid)
    offset = np.clip(np.median(b - a, axis=1, keepdims=True), -0.03, 0.03)
    diff = np.abs(b - a - offset)
    gradient_diff = np.zeros_like(diff)
    for axis in (0, 1):
        g0 = gray_measurement(old_gray, axis)
        g1 = gray_measurement(gray, axis)
        gradient_diff += np.abs(sample(g0, source_grid) - sample(g1, target_grid))
    agree = (diff <= config.support_residual_max) & (gradient_diff <= config.support_gradient_max)
    texture = (a.std(1) >= config.min_texture_std) & (b.std(1) >= config.min_texture_std)
    qualified = boundary & texture & (agree.mean(1) >= 0.9)
    residual[indices] = diff.mean(1)
    supported[indices] = qualified
    locations = target_grid[qualified].astype(int)
    good = agree[qualified]
    if len(locations):
        stable[locations[..., 1][good], locations[..., 0][good]] = True
        bad = ~good & (diff[qualified] > config.support_residual_max * 2)
        changed[locations[..., 1][bad], locations[..., 0][bad]] = True
    stable &= ~changed
    return (stable, changed, supported, residual)
