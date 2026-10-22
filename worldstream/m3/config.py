"""Trusted historical augmentation and exact prefix prefill, one executor."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class M3Config:
    mode: str = 'auto'
    memory_budget_mb: float = 512
    min_prefix_tokens: int = 32
    cost_table: str = ''
    quality_certificate: str = ''
    state_quality_certificate: str = ''
    model_version: str = ''
    state_context: str = 'augment'

    def __post_init__(self):
        if self.mode!='auto' or self.state_context!='augment':
            raise ValueError('M3 supports only trusted state augmentation and exact prefix Auto')
        if not math.isfinite(self.memory_budget_mb) or self.memory_budget_mb<0 or type(self.min_prefix_tokens) is not int or self.min_prefix_tokens<1:
            raise ValueError('Invalid prefix cache budget')
