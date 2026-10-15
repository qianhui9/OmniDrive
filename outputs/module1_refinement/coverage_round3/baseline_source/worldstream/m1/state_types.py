"""Explicit coordinate, observation and computation contracts (no model imports)."""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

Region = Tuple[float, float, float, float]  # half-open xyxy, image pixels


@dataclass
class OnlineVisibilityEvidence:
    """Same-RGB temporal evidence, NOT an independent visibility sensor or object mask.

    Arrays are camera pixel coordinates; supported points have bidirectional,
    calibrated-geometry and local photometric evidence. Failed tracks are unknown.
    """
    camera_id: str
    source_timestamp: float
    timestamp: float
    source_points: np.ndarray
    current_points: np.ndarray
    fb_error_px: np.ndarray
    geometry_valid: np.ndarray
    correspondence_valid: np.ndarray
    supported: np.ndarray
    patch_residual: np.ndarray
    stable_mask: np.ndarray
    changed_mask: np.ndarray
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    source: str = "same_rgb_bidirectional_lk_calibrated_geometry_local_patch"
    transformer_dependency_certified: bool = False


@dataclass
class CameraFrame:
    camera_id: str
    image: np.ndarray  # RGB uint8, unpadded raw or preprocessed image
    intrinsic: np.ndarray  # 3x3, in coordinate_space
    world_to_camera: Optional[np.ndarray]
    timestamp: float
    coordinate_space: str = "raw_image"
    raw_to_image: Optional[np.ndarray] = None
    timing_reliable: bool = True
    visibility_mask: Optional[np.ndarray] = None  # independently supported foreground visibility
    geometry_support_mask: Optional[np.ndarray] = None  # reliable dense surface support, not box interiors


@dataclass
class Frame:
    scene_id: str
    frame_id: str
    timestamp: float  # absolute seconds, never a Head's memory_timestamp
    ego_to_world: Optional[np.ndarray]
    cameras: Dict[str, CameraFrame]
    execution_signature: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Geometry:
    points: np.ndarray  # world points: 8 box corners, or map polyline/control points
    center: np.ndarray
    size: Optional[np.ndarray] = None  # box dimensions along source local axes


@dataclass
class AnchorObservation:
    camera_id: str
    region: Region
    descriptor: np.ndarray
    timestamp: float
    coverage: float


@dataclass
class WorldAnchor:
    anchor_id: str
    scene_id: str
    anchor_type: str
    semantic_class: str
    geometry_world: Geometry
    source_timestamp: float
    last_verified_timestamp: Optional[float] = None
    velocity_world: Optional[np.ndarray] = None
    velocity_reliable: bool = False
    geometry_uncertainty: float = 0.0  # metres
    observation_support: Dict[str, AnchorObservation] = field(default_factory=dict)
    associated_feature_ids: List[str] = field(default_factory=list)
    source_score: Optional[float] = None  # prediction confidence, NOT localization probability
    history_observations: int = 1
    last_innovation_m: Optional[float] = None


@dataclass
class PerceptionAnchor:
    """Only accepted by commit after actual perception, or explicit offline replay."""
    anchor_type: str
    semantic_class: str
    geometry_world: Geometry
    velocity_world: Optional[np.ndarray] = None
    velocity_reliable: bool = False
    geometry_uncertainty: float = 0.0
    source_score: Optional[float] = None


@dataclass
class ComputedFeature:
    feature_id: str
    feature_kind: str
    shape: Tuple[int, ...]
    dtype: str
    device: str
    source_layer: str
    tensor: Any = None
    source_camera_ids: List[str] = field(default_factory=list)
    source_region: Dict[str, List[Region]] = field(default_factory=dict)
    dependency_scope: str = "unknown/conservative"
    source_anchor_ids: List[str] = field(default_factory=list)


@dataclass
class FeatureRecord(ComputedFeature):
    computation_version: int = 0
    computation_timestamp: float = 0.0
    last_verified_timestamp: Optional[float] = None
    execution_signature: str = ""


@dataclass
class ComputationalProvenance:
    feature_id: str
    anchor_ids: List[str]
    camera_ids: List[str]
    image_regions: Dict[str, List[Region]]
    computation_unit: str
    dependency_scope: str
    upstream_versions: Dict[str, int] = field(default_factory=dict)
    downstream_dependencies: str = "unknown/conservative"
    coordinate_space: Dict[str, str] = field(default_factory=dict)


@dataclass
class FullOutputs:
    anchors: List[PerceptionAnchor]
    features: List[ComputedFeature] = field(default_factory=list)
    source: str = "actual_perception"


@dataclass
class M1Result:
    scene_id: str
    frame_id: str
    timestamp: float
    valid_anchor_ids: List[str] = field(default_factory=list)
    invalid_anchor_ids: List[str] = field(default_factory=list)
    valid_feature_candidates: List[dict] = field(default_factory=list)
    invalidation_regions: Dict[str, dict] = field(default_factory=dict)
    provenance: Dict[str, dict] = field(default_factory=dict)
    feature_versions: Dict[str, dict] = field(default_factory=dict)
    full_refresh_hint: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return json_value(asdict(self))


def json_value(value):
    """Strict JSON: unavailable/infinite residuals become null, never NaN."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value
