"""Opt-in, inference-only incremental EVA configuration."""
from dataclasses import dataclass


@dataclass(frozen=True)
class M2Config:
    enabled: bool = True
    mode: str = 'auto'  # exact forces legal sparse execution for experiments
    memory_budget_mb: int = 512
    cost_table: str = ''
    cache_capture: str = 'adaptive'  # always / adaptive / never

    def __post_init__(self):
        if self.mode not in ('auto', 'exact'):
            raise ValueError('M2 mode must be auto or exact')
        if self.memory_budget_mb < 0:
            raise ValueError('Negative cache budget')
        if self.cache_capture not in ('adaptive', 'always', 'never'):
            raise ValueError('Invalid cache capture policy')
