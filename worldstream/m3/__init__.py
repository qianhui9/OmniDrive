"""State-conditioned LLM execution (approximate selection and exact KV isolated)."""
from .config import M3Config
from .runtime import M3Runtime
from .visual_state import CarrierLayout, ExpandedInput, expand_input

__all__ = ['M3Config', 'M3Runtime', 'CarrierLayout', 'ExpandedInput', 'expand_input']

from .task_evidence import TaskEvidenceContract, TaskEvidenceDecision, evaluate_task_evidence
