"""M3 is opt-in; approximate policies require an offline quality certificate."""
from dataclasses import dataclass


@dataclass(frozen=True)
class M3Config:
    task_evidence_enabled: bool = False
    shadow_evaluation: bool = True
    experimental_direct_query: bool = False
    direct_query_prefix_fallback: bool = True
    direct_task_certificate: str = ''
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
    single_question_fastpath: bool = True
    borrow_prefix: bool = False
    batched_decode: bool = False
    batch_workspace_mb: float = 6144
    state_context: str = 'none'  # none / only / augment; offline approximation
    post_perception: bool = False
    common_prefix_seed: bool = False
    adaptive_batch: bool = False  # offline length grouping; each shape is uncertified
    max_batch_size: int = 5
    batch_cost_table: str = ''
    # Append extensions: retain the established positional constructor contract.
    decode_cuda_graph: bool = False
    identical_prefill_only: bool = False  # conservative research guard, Auto still gated
    experimental_rng_isolated_microbatch: bool = False
    sampling_microbatch_size: int = 2
    sampling_microbatch_seed: int = 0

    def __post_init__(self):
        if self.sampling_microbatch_size not in (2,4):
            raise ValueError('Sampling microbatch research supports fixed groups of 2/4')
        if self.experimental_rng_isolated_microbatch and (not self.offline_experiment or
                self.batched_decode or self.experimental_direct_query or self.decode_cuda_graph or
                self.state_context!='none' or self.mode!='full'):
            raise ValueError('Isolated sampling requires explicit offline full-input mode with other shortcuts disabled')
        if self.experimental_direct_query and (not self.offline_experiment or not self.task_evidence_enabled):
            raise ValueError('Direct task execution requires an explicit offline evidence experiment')
        if self.task_evidence_enabled and self.state_context != 'none':
            raise ValueError('Direct contracts preserve original visual prompts on fallback')
        if self.post_perception and (not self.offline_experiment or self.state_context=='none'):
            raise ValueError('Post-perception task candidates require an offline state experiment')
        if self.mode not in ('full', 'auto', 'exact', 'compact', 'combined'):
            raise ValueError('Unknown M3 mode')
        if self.selector not in ('full', 'task', 'topk', 'random'):
            raise ValueError('Unknown selector')
        if not 0 < self.keep_ratio <= 1 or self.memory_budget_mb < 0 or self.min_prefix_tokens < 1:
            raise ValueError('Invalid M3 budget, ratio or prefix length')
        if self.state_context not in ('none','only','augment') or self.batch_workspace_mb < 0:
            raise ValueError('Invalid state context or batch workspace')
        if (self.batched_decode or (self.common_prefix_seed and self.mode!='auto') or self.state_context!='none') and not self.offline_experiment:
            raise ValueError('Uncertified batch/state strategies require offline_experiment=True')
        if self.batched_decode and (self.mode!='exact' or self.state_context!='none'):
            raise ValueError('Batch experiment requires exact mode and unchanged full carriers')
        if self.common_prefix_seed and (self.mode not in ('exact','auto') or self.state_context!='none' or self.batched_decode):
            raise ValueError('Independent prefix seeding requires sequential exact mode and full carriers')
        if self.max_batch_size < 1 or (self.adaptive_batch and not self.batched_decode):
            raise ValueError('Adaptive grouping requires the offline batched executor')
