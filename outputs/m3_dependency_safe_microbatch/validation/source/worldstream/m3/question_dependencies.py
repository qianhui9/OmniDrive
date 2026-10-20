"""Conservative readiness for the inspected native single-turn QA interface.

No result or visual fact is inferred from text. A ready group needs completed
shared heads, immutable carriers, no state-writing questions, and prompt binding.
Unknown conversation wrappers and answer references fail closed to serial.
"""
from dataclasses import dataclass
import re
from .task_evidence import native_prompt_matches


@dataclass(frozen=True)
class QuestionDependency:
    request_id: int
    dependencies: tuple = ()
    independent: bool = False
    reason: str = 'uninspected_prompt'


def analyze_questions(tokenizer, ids, questions, *, inputs_ready, shared_carriers,
                      state_writes=False, ordered_execution=False):
    if len(ids) != len(questions):
        raise ValueError('Question/input order mismatch')
    ready = inputs_ready and shared_carriers and not state_writes and not ordered_execution
    rows=[]
    for index,(tokens,question) in enumerate(zip(ids,questions)):
        reference=bool(re.search(r'\b(previous|preceding|earlier|above|your|last)\s+(answer|response|reply)|'
            r'\bas you (said|mentioned)|\bcontinue (the|your)|\bbased on (that|your answer)',question,re.I))
        bound=native_prompt_matches(tokenizer,tokens,question)
        independent=bool(ready and bound and not reference)
        rows.append(QuestionDependency(index,tuple(range(index)) if not independent else (),independent,
            'native_bound_single_turn_shared_readonly_input' if independent else
            'answer_reference' if reference else 'unbound_prompt' if not bound else 'unready_or_ordered_or_state_writing'))
    return rows


def ready_groups(rows, maximum):
    """Do not cross a serial barrier; restore original submission order."""
    if maximum < 1:raise ValueError('Positive batch bound required')
    groups=[];pending=[]
    for row in rows:
        if row.independent:
            pending.append(row.request_id)
            if len(pending)==maximum:groups.append(pending);pending=[]
        else:
            if pending:groups.append(pending);pending=[]
            groups.append([row.request_id])
    if pending:groups.append(pending)
    return groups
