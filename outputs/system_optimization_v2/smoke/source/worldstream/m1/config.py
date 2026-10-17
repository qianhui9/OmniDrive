"""All defaults are provisional; distances are metres and times are seconds."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class M1Config:
    enabled: bool = True
    execution_mode: str = "observe_only"
    storage_device: str = "cpu"
    max_anchors: int = 512
    max_state_age_s: float = 1.0
    max_frame_gap_s: float = 1.0
    association_gate_m: float = 2.0
    map_gate_m: float = 1.0
    map_direction_cos: float = 0.8
    box_size_ratio: float = 2.0
    residual_threshold: float = 0.18
    descriptor: str = "normalized_gray"
    descriptor_size: int = 16
    min_texture_std: float = 0.02
    min_coverage: float = 0.9
    min_region_area_px: int = 64
    search_radius_px: int = 4
    search_step_px: int = 2
    uncertainty_px: float = 2.0
    velocity_uncertainty_mps: float = 2.0
    max_camera_skew_s: float = 0.01
    require_visibility_evidence: bool = True
    trust_detection_velocity: bool = False
    min_detection_score: float = 0.3
    min_map_score: float = 0.3
    occlusion_overlap_fraction: float = 0.1
    occlusion_depth_margin_m: float = 0.5
    novelty_threshold: float = 0.18
    novelty_cell_px: int = 32
    full_refresh_interval: int = 10
    capture_tensors: bool = False
    capture_backbone: bool = False
    max_tensor_bytes: int = 16 * 1024 * 1024
    model_id: str = "unspecified"
    calibration_path: Optional[str] = None
    logging: bool = True
    log_path: Optional[str] = None
    visualization: bool = True
    profiling: bool = True
    temporal_visibility: bool = False  # opt-in refinement; false reproduces round one
    local_support: bool = False
    flow_max_points: int = 1600
    flow_min_distance_px: int = 5
    flow_fb_max_px: float = 0.75
    flow_lk_max_error: float = 12.0  # OpenCV mean absolute intensity error, 0..255
    flow_epipolar_max_px: float = 1.0
    support_patch_radius_px: int = 3
    support_residual_max: float = 0.04  # grayscale intensity, 0..1
    support_gradient_max: float = 0.08
    support_min_points: int = 4
    support_point_fraction: float = 0.8
    support_quadrant_fraction: float = 0.75
    dense_motion_guard: bool = False
    dense_local_support: bool = False
    dense_guard_radius_px: int = 4
    temporal_workers: int = 1
    certification_min_source_score: float = 0.0
    certification_min_history: int = 1
    certification_max_innovation_m: float = 0.0  # 0 disables this additional gate

    def __post_init__(self):
        if not isinstance(self.dense_guard_radius_px, int) or self.dense_guard_radius_px < 1:
            raise ValueError("Dense support needs a positive integer boundary guard")
        if self.dense_local_support and not (self.temporal_visibility and self.local_support):
            raise ValueError("Dense support requires temporal visibility and local support")
        if self.execution_mode != "observe_only":
            raise ValueError("M1 supports observe_only; GPU incremental execution belongs to M2")
        if not 1 <= self.temporal_workers <= 6 or self.certification_min_history < 1:
            raise ValueError("Invalid temporal worker/history count")
        if not 0 <= self.certification_min_source_score <= 1 or not 0 <= self.certification_max_innovation_m < float('inf'):
            raise ValueError("Invalid source quality gate")
        for key in ("max_anchors", "descriptor_size", "search_step_px", "novelty_cell_px",
                    "full_refresh_interval", "max_tensor_bytes", "min_region_area_px"):
            if getattr(self, key) <= 0:
                raise ValueError(key + " must be positive")
        for key in ("flow_max_points", "flow_min_distance_px", "support_patch_radius_px", "support_min_points"):
            if getattr(self, key) <= 0:
                raise ValueError(key + " must be positive")
        for key in ("flow_fb_max_px", "flow_lk_max_error", "flow_epipolar_max_px",
                    "support_residual_max", "support_gradient_max"):
            import math
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(key + " must be finite and positive")
        if not all(0 < x <= 1 for x in (self.support_point_fraction, self.support_quadrant_fraction)):
            raise ValueError("support fractions must be in (0, 1]")
        for key in ("max_state_age_s", "max_frame_gap_s", "association_gate_m", "map_gate_m",
                    "box_size_ratio", "residual_threshold", "min_texture_std", "uncertainty_px",
                    "velocity_uncertainty_mps", "max_camera_skew_s", "novelty_threshold"):
            import math
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(key + " must be finite and nonnegative")
        if not 0 < self.min_coverage <= 1 or not 0 <= self.map_direction_cos <= 1:
            raise ValueError("coverage/direction thresholds out of range")
        if not all(0 <= x <= 1 for x in (self.min_detection_score, self.min_map_score,
                                        self.occlusion_overlap_fraction)):
            raise ValueError("score/occlusion fractions out of range")
        if self.box_size_ratio < 1 or self.search_radius_px < 0:
            raise ValueError("invalid size ratio/search radius")
        if self.descriptor not in ("normalized_gray", "gradient"):
            raise ValueError("unsupported descriptor")

    @classmethod
    def load(cls, source=None):
        if isinstance(source, cls):
            return source
        if isinstance(source, (str, Path)):
            import yaml
            with open(source) as handle:
                source = yaml.safe_load(handle)
        values = dict(source or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError("Unknown M1 settings: " + str(sorted(unknown)))
        if values.get("calibration_path"):
            import json
            with open(values["calibration_path"]) as handle:
                calibration = json.load(handle)
            if calibration.get("split") != "calibration" or calibration.get("threshold") is None:
                raise ValueError("Expected a supported calibration-only artifact")
            for name in ("descriptor", "descriptor_size"):
                if calibration.get(name) != values.get(name, getattr(cls, name)):
                    raise ValueError("Calibration descriptor mismatch")
            values["residual_threshold"] = calibration["threshold"]
        return cls(**values)

    def to_dict(self):
        return asdict(self)
