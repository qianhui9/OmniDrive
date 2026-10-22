"""The sole M2 contract: source-linked window Auto with maintained M1 state."""
from dataclasses import dataclass

@dataclass(frozen=True)
class M2Config:
    enabled: bool = True
    mode: str = 'auto'
    reuse_boundary: str = 'window'
    state_interface: str = 'source_linked'
    m1_service: str = 'maintain'
    memory_budget_mb: int = 512
    cost_table: str = ''
    cache_capture: str = 'adaptive'

    def __post_init__(self):
        if (self.mode!='auto' or self.reuse_boundary!='window' or self.state_interface!='source_linked' or self.m1_service!='maintain'):
            raise ValueError('M2 supports only source-linked window Auto with maintained M1 state')
        if type(self.memory_budget_mb) is not int or self.memory_budget_mb<0:
            raise ValueError('Invalid cache budget')
        if self.cache_capture not in ('adaptive','always','never'):
            raise ValueError('Invalid capture policy')
