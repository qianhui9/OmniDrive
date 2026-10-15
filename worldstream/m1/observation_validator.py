"""CPU image evidence only. No GT, detector outputs or EVA features enter this API."""
from dataclasses import dataclass
import cv2
import numpy as np
from .geometry import region_area, region_slices, clip_region


@dataclass
class Validation:
    valid: bool
    residual: float
    region: tuple
    observation_consistent: bool
    visibility_and_coverage_valid: bool
    reason: str


class ObservationValidator:
    def __init__(self, config):
        self.config = config

    def descriptor(self, image, region):
        if region_area(region) < self.config.min_region_area_px:
            return None
        ys, xs = region_slices(region, image.shape)
        patch = image[ys, xs]
        if patch.size == 0:
            return None
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY) if patch.ndim == 3 else patch
        gray = cv2.resize(gray.astype(np.float32) / 255.,
                          (self.config.descriptor_size, self.config.descriptor_size),
                          interpolation=cv2.INTER_AREA)
        std = float(gray.std())
        if std < self.config.min_texture_std:
            return None  # uniform sky/road is insufficient evidence
        if self.config.descriptor == "gradient":
            dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            desc = np.stack([dx, dy])
            return desc / max(float(np.sqrt(np.mean(desc ** 2))), 0.01)
        return (gray - gray.mean()) / max(std, 0.01)

    @staticmethod
    def residual(a, b):
        if a is None or b is None or a.shape != b.shape:
            return float("inf")
        return float(np.mean(np.abs(a - b)))

    def validate(self, old, camera, projection, occluded=False):
        region = projection.region
        if region is None:
            return Validation(False, float("inf"), (), False, False, projection.reason)
        if old is None:
            return Validation(False, float("inf"), region, False, False, "no_historical_observation")
        if not projection.valid:
            return Validation(False, float("inf"), region, False, False, projection.reason)
        best, best_region = float("inf"), region
        radius = projection.search_radius_px
        offsets = sorted(set(range(-radius, radius + 1, self.config.search_step_px)) | {0})
        for dy in offsets:
            for dx in offsets:
                r = (region[0] + dx, region[1] + dy, region[2] + dx, region[3] + dy)
                if region_area(clip_region(r, camera.image.shape)) < region_area(r) * self.config.min_coverage:
                    continue
                value = self.residual(old.descriptor, self.descriptor(camera.image, r))
                if value < best:
                    best, best_region = value, clip_region(r, camera.image.shape)
        consistent = best <= self.config.residual_threshold
        visible = not occluded
        visibility_reason = "projected_occlusion" if occluded else "ok"
        if self.config.require_visibility_evidence:
            mask = camera.visibility_mask
            ys, xs = region_slices(best_region, camera.image.shape)
            if mask is None or mask.shape != camera.image.shape[:2] or mask[ys, xs].size == 0:
                visible = False; visibility_reason = "visibility_unknown"
            elif np.asarray(mask[ys, xs], dtype=bool).mean() < self.config.min_coverage:
                visible = False; visibility_reason = "insufficient_visible_support"
        valid = consistent and visible
        reason = "ok" if valid else ("observation_mismatch" if not consistent else visibility_reason)
        return Validation(valid, best, best_region, consistent, visible, reason)
