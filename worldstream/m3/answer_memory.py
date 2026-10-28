"""Bounded causal answer-token proposals. No tensors, KV, or annotation answers."""
from collections import OrderedDict
from dataclasses import dataclass


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


class AnswerMemory:
    def __init__(self, max_entries=128, max_age_frames=8, max_age_s=4.):
        self.max_entries = max_entries
        self.max_age_frames = max_age_frames
        self.max_age_s = max_age_s
        self.records = OrderedDict()

    def reset(self):
        self.records.clear()

    def select(self, prompt_ids, context, binding):
        key = (context['scene_id'], tuple(prompt_ids))
        record = self.records.get(key)
        if record is None:
            return None
        age = context['frame_index']-record.frame_index
        dt = context['timestamp']-record.timestamp
        if (record.binding != binding or not 0 < age <= self.max_age_frames
                or not 0 < dt <= self.max_age_s or record.frame_id == context['frame_id']):
            return None
        return record

    def remember(self, question, prompt_ids, tokens, context, binding):
        ids = tuple(prompt_ids)
        key = (context['scene_id'], ids)
        self.records[key] = AnswerRecord(str(question), ids, tuple(tokens), context['scene_id'],
            context['frame_id'], context['frame_index'], context['timestamp'],
            context.get('state_version'), binding)
        self.records.move_to_end(key)
        while len(self.records) > self.max_entries:
            self.records.popitem(last=False)
