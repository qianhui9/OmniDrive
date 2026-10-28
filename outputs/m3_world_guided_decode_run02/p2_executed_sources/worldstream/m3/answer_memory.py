"""Bounded causal answer-token proposals. No tensors, KV, or annotation answers."""
from collections import OrderedDict
from dataclasses import dataclass
from .task_conditioner import question_dependencies
from .world_delta import question_world_delta
from time import perf_counter


@dataclass(frozen=True)
class AnswerRecord:
    question: str
    prompt_ids: tuple
    tokens: tuple
    scene_id: str
    frame_id: str
    frame_index: int
    timestamp: float
    world_version: object
    binding: tuple
    source: str = 'current_target_model_generation'
    dependencies: object = None
    world_state: object = None
    polarity: str = 'unknown'


@dataclass(frozen=True)
class DraftSelection:
    record: object = None
    length: int = 0
    match_type: str = 'none'
    exact_available: bool = False
    semantic_available: bool = False
    delta: object = None
    reason: str = 'no_history'
    world_diff_ms: float = 0.


class AnswerMemory:
    def __init__(self, max_entries=128, max_age_frames=8, max_age_s=4.):
        self.max_entries = max_entries
        self.max_age_frames = max_age_frames
        self.max_age_s = max_age_s
        self.records = OrderedDict()
        self.acceptance = {}

    def reset(self):
        self.records.clear()
        self.acceptance.clear()

    def valid(self,record,context,binding):
        return (record.scene_id==context['scene_id'] and record.binding==binding
            and 0<context['frame_index']-record.frame_index<=self.max_age_frames
            and 0<context['timestamp']-record.timestamp<=self.max_age_s
            and record.frame_id!=context['frame_id'])

    def select(self, prompt_ids, context, binding):
        key = (context['scene_id'], tuple(prompt_ids))
        record = self.records.get(key)
        if record is None:
            return None
        if not self.valid(record,context,binding):
            return None
        return record

    def retrieve(self,question,prompt_ids,context,binding,current_world=None,guided=False):
        deps=question_dependencies(question)
        exact=self.select(prompt_ids,context,binding)
        if not guided:
            return DraftSelection(exact,8 if exact else 0,'exact' if exact else 'none',
                exact is not None,False,None,'exact_history' if exact else 'no_exact_history')
        def compatible(other):
            if other is None:return False
            if other.signature==deps.signature:return True
            # Positive existential drafts may cross near/all-view wording only
            # when current AND source observations satisfy the CURRENT scope.
            return (other.intent==deps.intent=='existence' and other.classes==deps.classes
                    and other.directions==deps.directions)
        candidates=[r for r in reversed(list(self.records.values())) if self.valid(r,context,binding)
            and compatible(r.dependencies)]
        if exact is not None:candidates=[exact]+[r for r in candidates if r is not exact]
        diff_ms=0.
        for record in candidates:
            tick=perf_counter()
            delta=question_world_delta(record.world_state,current_world,deps)
            diff_ms+=(perf_counter()-tick)*1000
            if not delta.suggested_draft_length:continue
            if deps.intent=='existence' and record.polarity!='positive':continue
            key=(record.dependencies.signature,deps.signature)
            outcome=self.acceptance.get(key)
            if outcome and outcome[0]==0 and context['frame_index']-outcome[1]<=2:continue
            length=delta.suggested_draft_length
            return DraftSelection(record,length,'exact' if record is exact else 'semantic',
                exact is not None,bool(candidates),delta,delta.decision_reason,diff_ms)
        tick=perf_counter()
        delta=question_world_delta(candidates[0].world_state if candidates else None,current_world,deps)
        diff_ms+=(perf_counter()-tick)*1000
        return DraftSelection(None,0,'none',exact is not None,bool(candidates),delta,
            delta.decision_reason if candidates else 'no_compatible_semantic_history',diff_ms)

    def observe_acceptance(self,selection,diagnostics,frame_index):
        if selection.record is None:return
        logs=diagnostics.get('verification_log',[])
        if logs:
            signature=selection.record.dependencies.signature
            self.acceptance[(signature,signature)]=(logs[0]['accepted'],frame_index)

    def remember(self, question, prompt_ids, tokens, context, binding,world_state=None,answer_text=''):
        ids = tuple(prompt_ids)
        key = (context['scene_id'], ids)
        self.records[key] = AnswerRecord(str(question), ids, tuple(tokens), context['scene_id'],
            context['frame_id'], context['frame_index'], context['timestamp'],
            context.get('current_state_version',context.get('state_version')), binding,
            dependencies=question_dependencies(question),world_state=world_state,
            polarity='positive' if answer_text.lstrip().lower().startswith('yes') else
                     'negative' if answer_text.lstrip().lower().startswith('no') else 'unknown')
        self.records.move_to_end(key)
        while len(self.records) > self.max_entries:
            self.records.popitem(last=False)
