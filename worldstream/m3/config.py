"""One quality-bounded WSDR policy; exact_prefix is a benchmark restriction."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class M3Config:
    mode: str = 'auto'
    memory_budget_mb: float = 512
    min_prefix_tokens: int = 32
    model_version: str = ''
    state_context: str = 'augment'
    reuse_strategy: str = 'wsdr_kv'
    reuse_policy: str = ''
    quality_budget: float = .02
    block_size: int = 32
    prefix_adapter: str = ''
    stable_anchor_max_frames: int = 8
    semantic_qualification: str = ''

    def __post_init__(self):
        if self.mode != 'auto' or self.state_context != 'augment':
            raise ValueError('M3 supports one automatic prefill/decode path')
        if self.reuse_strategy not in ('exact_prefix', 'wsdr_kv'):
            raise ValueError('Unknown M3 reuse strategy')
        if self.prefix_adapter and self.reuse_strategy != 'exact_prefix':
            raise ValueError('WRCPS requires exact_prefix; native approximate policy is inapplicable')
        if type(self.stable_anchor_max_frames) is not int or self.stable_anchor_max_frames < 1:
            raise ValueError('Invalid stable anchor horizon')
        if (not math.isfinite(self.memory_budget_mb) or self.memory_budget_mb < 0
                or type(self.min_prefix_tokens) is not int or self.min_prefix_tokens < 1
                or self.block_size not in (16, 32, 64)
                or not math.isfinite(self.quality_budget) or not 0 <= self.quality_budget <= 1):
            raise ValueError('Invalid cache or quality budget')
