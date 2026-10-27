"""The sole M2 contract: source-linked windows and temporal anchor residuals."""
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
    global_anchor_interval: int = 2
    global_drift_threshold: float = 0.7721027880907059
    global_risk_weights: tuple = (1., 0., 0., 0.)
    global_risk_scales: tuple = (1., 1., 1., 1.)
    global_risk_threshold: float = None
    global_tail_fraction: float = .10

    @property
    def risk_threshold(self):
        # Preserve the previous strictly-greater RMS boundary with the same
        # single >= score comparison. No alternate selector or executor.
        return (math.nextafter(float(self.global_drift_threshold), math.inf)
                if self.global_risk_threshold is None else self.global_risk_threshold)

    def global_signature(self):
        return ('wsgrr_world_task_anchor_v2', self.global_anchor_interval, self.risk_threshold,
                tuple(self.global_risk_weights),tuple(self.global_risk_scales),self.global_tail_fraction)

    def refresh_signature(self):
        return ('wsgrr_image_uncertainty_v2', self.ordinary_budget, self.max_reuse_frames,
                self.region_margin_px, self.movement_threshold_px, self.drift_weight,
                self.age_weight, self.uncertainty_weight, self.score_threshold)

    def __post_init__(self):
        for key in ('global_risk_weights','global_risk_scales'):
            values=getattr(self,key)
            if len(values)!=4 or any(type(v) not in (float,int) or not math.isfinite(v) or (v<0 if key.endswith('weights') else v<=0) for v in values):
                raise ValueError('Invalid '+key)
        if not any(self.global_risk_weights):raise ValueError('Empty risk weights')
        if not 0<self.global_tail_fraction<=1:raise ValueError('Invalid tail fraction')
        if self.global_risk_threshold is not None and (not math.isfinite(self.global_risk_threshold) or self.global_risk_threshold<0):raise ValueError('Invalid risk threshold')
        if (self.mode!='auto' or self.reuse_boundary!='window' or self.state_interface!='source_linked' or self.m1_service!='maintain'):
            raise ValueError('M2 supports only source-linked window Auto with maintained M1 state')
        if type(self.memory_budget_mb) is not int or self.memory_budget_mb<0:
            raise ValueError('Invalid cache budget')
        for name, minimum in (('ordinary_budget', 0), ('max_reuse_frames', 1), ('region_margin_px', 0), ('global_anchor_interval', 2)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError('Invalid '+name)
        for name in ('movement_threshold_px','drift_weight','age_weight','uncertainty_weight','score_threshold','global_drift_threshold'):
            value = getattr(self, name)
            if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid '+name)
