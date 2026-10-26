"""The sole M2 contract: source-linked window Auto with maintained M1 state."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class M2Config:
    enabled: bool = True
    mode: str = 'auto'
    reuse_boundary: str = 'window'
    state_interface: str = 'source_linked'
    m1_service: str = 'maintain'
    memory_budget_mb: int = 512
    ordinary_budget: int = 3
    max_reuse_frames: int = 8
    region_margin_px: int = 8
    movement_threshold_px: float = 16.0
    drift_weight: float = 1.0
    age_weight: float = 0.25
    uncertainty_weight: float = 0.25
    score_threshold: float = 0.05

    def refresh_signature(self):
        return ('wsgrr_image_uncertainty_v2', self.ordinary_budget, self.max_reuse_frames,
                self.region_margin_px, self.movement_threshold_px, self.drift_weight,
                self.age_weight, self.uncertainty_weight, self.score_threshold)

    def __post_init__(self):
        if (self.mode!='auto' or self.reuse_boundary!='window' or self.state_interface!='source_linked' or self.m1_service!='maintain'):
            raise ValueError('M2 supports only source-linked window Auto with maintained M1 state')
        if type(self.memory_budget_mb) is not int or self.memory_budget_mb<0:
            raise ValueError('Invalid cache budget')
        for name, minimum in (('ordinary_budget', 0), ('max_reuse_frames', 1), ('region_margin_px', 0)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError('Invalid '+name)
        for name in ('movement_threshold_px','drift_weight','age_weight','uncertainty_weight','score_threshold'):
            value = getattr(self, name)
            if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid '+name)
