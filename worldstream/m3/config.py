"""One quality-bounded WSDR policy; exact_prefix is a benchmark restriction."""
from dataclasses import dataclass

@dataclass(frozen=True)
class M3Config:
    mode: str = 'auto'
    memory_budget_mb: float = 512
    min_prefix_tokens: int = 32
    model_version: str = ''
    state_context: str = 'augment'
    reuse_strategy: str = 'exact_prefix'
    reuse_policy: str = ''
    quality_budget: float = 0.02
    block_size: int = 32
    prefix_adapter: str = ''
    stable_anchor_max_frames: int = 8
    semantic_qualification: str = ''
    prefix_layout: str = 'legacy_fixed'
    decode_strategy: str = 'world_guided'
    draft_length: int = 8
    experimental_speculative_sampling: bool = True
    target_aware_exact_only: bool = True
    selective_answer_mode: str = 'online'
    selective_answer_policy: str = 'outputs/m123_final_quality_validation_run01/reference_policy.json'
    question_rng_enabled: bool = True
    target_binding_enabled: bool = True
    target_completeness_enabled: bool = False
    temporal_risk_enabled: bool = False
    current_detection_support: bool = True

    def __post_init__(self):
        from dataclasses import asdict
        from worldstream.r0 import m3_settings
        expected = m3_settings(self.selective_answer_policy)
        actual = asdict(self)
        expected['model_version'] = self.model_version
        if actual != expected:
            bad = [k for k in expected if actual[k] != expected[k]]
            raise ValueError('Only frozen R0 M3 settings are supported: ' + str(bad))
