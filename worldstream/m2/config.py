"""Opt-in, inference-only incremental EVA configuration."""
from dataclasses import dataclass


@dataclass(frozen=True)
class M2Config:
    enabled: bool = True
    mode: str = 'auto'  # exact forces legal sparse execution for experiments
    memory_budget_mb: int = 512
    cost_table: str = ''
    cache_capture: str = 'adaptive'  # always / adaptive / never
    cheap_gate: bool = False
    m1_service: str = 'maintain'  # maintain / reset_on_reject / output_only
    reuse_boundary: str = 'window'  # window / camera
    camera_cost_table: str = ''

    def __post_init__(self):
        if self.mode not in ('auto', 'exact'):
            raise ValueError('M2 mode must be auto or exact')
        if self.memory_budget_mb < 0:
            raise ValueError('Negative cache budget')
        if self.cache_capture not in ('adaptive', 'always', 'never'):
            raise ValueError('Invalid cache capture policy')
        if self.m1_service not in ('maintain','reset_on_reject','output_only'):
            raise ValueError('Invalid M1 service contract')
        if self.m1_service != 'maintain' and not self.cheap_gate:
            raise ValueError('On-demand M1 service requires the rejection gate')
        if self.reuse_boundary not in ('window','camera'):
            raise ValueError('Unknown visual reuse boundary')
