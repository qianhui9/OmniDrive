"""Bounded causal answer-token proposals. No tensors, KV, or annotation answers."""
from collections import OrderedDict
from dataclasses import dataclass, replace
from .task_conditioner import question_dependencies, answer_facts
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
    facts: object = None
    answer_text: str = ''
    answer_dependencies: object = None
    original_generation: tuple = ()
    generation_cost_ms: object = None
    validity_statistics: tuple = ()
    cache_provenance: tuple = ()

@dataclass(frozen=True)
class DraftCandidate:
    tokens: tuple
    kind: str
    record: object
    delta: object
    block_length: int = 8
    expected_accepted: float = 2.0
    forward_cost_ratio: float = 1.05
    fact_token_start: int = -1
    text: str = ''
    history_statistics: object = None
    selection_cost_steps: float = 0.003

    def metadata(self):
        return dict(kind=self.kind, source_frame=self.record.frame_id, source_question=self.record.question, tokens=list(self.tokens), text=self.text, block_length=self.block_length, expected_accepted=self.expected_accepted, forward_cost_ratio=self.forward_cost_ratio, fact_token_start=self.fact_token_start, history_statistics=self.history_statistics, selection_cost_steps=self.selection_cost_steps, world_delta=self.delta.to_dict() if self.delta else None)

@dataclass(frozen=True)
class DraftSelection:
    record: object = None
    length: int = 0
    match_type: str = 'none'
    exact_available: bool = False
    semantic_available: bool = False
    delta: object = None
    reason: str = 'no_history'
    world_diff_ms: float = 0.0
    candidates: tuple = ()
    construction_ms: float = 0.0
    tokenizer_ms: float = 0.0

class AnswerMemory:

    def __init__(self, max_entries=128, max_age_frames=8, max_age_s=4.0):
        self.max_entries = max_entries
        self.max_age_frames = max_age_frames
        self.max_age_s = max_age_s
        self.records = OrderedDict()
        self.acceptance = {}
        self.performance = OrderedDict()

    def reset(self):
        self.records.clear()
        self.acceptance.clear()
        self.performance.clear()

    def valid(self, record, context, binding):
        return record.scene_id == context['scene_id'] and record.binding == binding and (0 < context['frame_index'] - record.frame_index <= self.max_age_frames) and (0 < context['timestamp'] - record.timestamp <= self.max_age_s) and (record.frame_id != context['frame_id'])

    def select(self, prompt_ids, context, binding):
        key = (context['scene_id'], tuple(prompt_ids))
        record = self.records.get(key)
        if record is None:
            return None
        if not self.valid(record, context, binding):
            return None
        return record

    def retrieve(self, question, prompt_ids, context, binding, current_world=None, guided=False):
        deps = question_dependencies(question)
        exact = self.select(prompt_ids, context, binding)
        return DraftSelection(exact, 8 if exact else 0, 'exact' if exact else 'none', exact is not None, False, None, 'exact_history' if exact else 'no_exact_history')

    def construct(self, selection, question, prompt_ids, context, binding, current_world, tokenizer):
        """At most four deterministic proposals; no forward, GT or current answer.

        M1 controls which historical *facts* may seed composition. It does not
        certify these strings as current answers. Every candidate still enters
        the one current-target verifier, after prefill-aware selection.
        """
        tick = perf_counter()
        deps = question_dependencies(question)
        out = []
        token_ms = 0.0
        diff_ms = selection.world_diff_ms

        def add(tokens, kind, record, delta, text='', fact_start=-1):
            tokens = tuple(tokens)
            if len(tokens) < 2 or any((c.tokens == tokens for c in out)) or len(out) >= 4:
                return
            stat = self.performance.get((record.prompt_ids if kind == 'exact' else deps.signature, kind))
            if stat and stat['unproductive_streak'] >= 2:
                return
            expected = stat['accepted_per_verification'] if stat else 6.0 if kind == 'exact' else 2.0
            out.append(DraftCandidate(tokens, kind, record, delta, 8, min(8.0, max(1.0, expected)), stat['forward_cost_ratio'] if stat else 1.05, fact_start, text, dict(stat) if stat else None, stat['selection_cost_steps'] if stat else 0.003))
        if selection.record is not None:
            kind = 'exact' if selection.exact_available and selection.match_type == 'exact' else 'semantic_raw'
            add(selection.record.tokens, kind, selection.record, selection.delta, selection.record.answer_text)
            if kind == 'exact':
                return replace(selection, candidates=tuple(out), construction_ms=(perf_counter() - tick) * 1000)
        return replace(selection, candidates=tuple(out), world_diff_ms=diff_ms, semantic_available=selection.semantic_available or any((c.kind != 'exact' for c in out)), construction_ms=(perf_counter() - tick) * 1000 - (diff_ms - selection.world_diff_ms) - token_ms, tokenizer_ms=token_ms)

    def observe_acceptance(self, selection, diagnostics, frame_index):
        if selection.record is None:
            return
        logs = diagnostics.get('verification_log', [])
        if logs:
            signature = selection.record.dependencies.signature
            target_signature = selection.delta.question_signature if selection.delta else signature
            self.acceptance[signature, target_signature] = (logs[0]['accepted'], frame_index)
            if not diagnostics.get('target_aware_enabled'):
                return
            kind = diagnostics.get('selected_candidate_kind', selection.match_type)
            key = (selection.record.prompt_ids if kind == 'exact' else target_signature, kind)
            count = len(logs)
            accepted = sum((l['accepted'] for l in logs))
            tokens = len(diagnostics.get('output_token_ids', []))
            calls = diagnostics.get('decode_forward_calls', 0)
            events = diagnostics.get('decode_forward_cuda_events', [])
            single = [e['cuda_ms'] for e in events if e['input_tokens'] == 1]
            blocks = [e['cuda_ms'] for e in events if e['input_tokens'] > 1]
            step = sum(single) / len(single) if single else diagnostics.get('decode_forward_cuda_ms', 0.0) / max(1, calls)
            block_ms = sum(blocks) / len(blocks) if blocks else step
            saved = tokens - 1 - calls
            old = self.performance.get(key, {})
            blend = lambda name, value: 0.5 * old[name] + 0.5 * value if name in old else value
            selection_ms = diagnostics.get('draft_selection_wall_ms', 0.0) + diagnostics.get('target_selection_cuda_span_ms', 0.0)
            self.performance[key] = dict(requests=old.get('requests', 0) + 1, recent_accepted=accepted, accepted_per_verification=blend('accepted_per_verification', accepted / count), acceptance_rate=accepted / max(1, sum((len(l['proposed']) for l in logs))), first_rejection=logs[0]['rejection_position'], mean_verification_ms=blend('mean_verification_ms', block_ms), forward_cost_ratio=blend('forward_cost_ratio', block_ms / step if step > 0 else 1.05), selection_cost_steps=blend('selection_cost_steps', selection_ms / step if step > 0 else 0.003), estimated_saved_forwards=saved, estimated_net_ms=saved * step - selection_ms, unproductive_streak=old.get('unproductive_streak', 0) + 1 if accepted < 2 or saved <= 0 else 0)
            self.performance.move_to_end(key)
            while len(self.performance) > self.max_entries:
                self.performance.popitem(last=False)

    def remember(self, question, prompt_ids, tokens, context, binding, world_state=None, answer_text='', extract_facts=True, track_validity=False, generation_cost_ms=None):
        ids = tuple(prompt_ids)
        key = (context['scene_id'], ids)
        facts = answer_facts(answer_text) if extract_facts else None
        answer_dependencies = None
        self.records[key] = AnswerRecord(str(question), ids, tuple(tokens), context['scene_id'], context['frame_id'], context['frame_index'], context['timestamp'], context.get('current_state_version', context.get('state_version')), binding, dependencies=question_dependencies(question), world_state=world_state, polarity=facts.polarity if facts else 'unknown', facts=facts, answer_text=str(answer_text), answer_dependencies=answer_dependencies, original_generation=(context['scene_id'], context['frame_id'], context['timestamp']), generation_cost_ms=generation_cost_ms)
        self.records.move_to_end(key)
        while len(self.records) > self.max_entries:
            self.records.popitem(last=False)
