"""M3 is opt-in; approximate policies require an offline quality certificate."""
from dataclasses import dataclass


@dataclass(frozen=True)
class M3Config:
    mode: str = 'auto'
    memory_budget_mb: float = 512
    min_prefix_tokens: int = 32
    selector: str = 'task'
    keep_ratio: float = .5
    random_seed: int = 0
    offline_experiment: bool = False
    cost_table: str = ''
    quality_certificate: str = ''
    cross_frame: bool = False
    model_version: str = ''  # deployment's verified checkpoint/content identifier

    def __post_init__(self):
        if self.mode not in ('full', 'auto', 'exact', 'compact', 'combined'):
            raise ValueError('Unknown M3 mode')
        if self.selector not in ('full', 'task', 'topk', 'random'):
            raise ValueError('Unknown selector')
        if not 0 < self.keep_ratio <= 1 or self.memory_budget_mb < 0 or self.min_prefix_tokens < 1:
            raise ValueError('Invalid M3 budget, ratio or prefix length')
