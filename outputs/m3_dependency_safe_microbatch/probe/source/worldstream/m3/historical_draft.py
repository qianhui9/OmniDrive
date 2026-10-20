"""Bounded, immutable answer proposals. No visual facts or target logits live here.

The caller commits only after native parsing/submission. Matching requires the
entire text prompt (including camera markers), exact question and all protocol
identities. Changed images are intentionally not a matching requirement: every
proposed token still needs verification against the current full visual prompt.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class DraftContract:
    scene: str
    cameras: tuple
    task: str
    question: str
    prompt_signature: str
    model_signature: str
    tokenizer_signature: str
    generation_signature: str

    def valid(self):
        return (all(isinstance(v, str) and bool(v) for v in (
            self.scene, self.task, self.question, self.prompt_signature,
            self.model_signature, self.tokenizer_signature, self.generation_signature))
            and len(self.cameras) == 6 and len(set(self.cameras)) == 6)


@dataclass(frozen=True)
class HistoricalAnswer:
    contract: DraftContract
    frame: str
    timestamp: float
    token_ids: tuple
    stop_reason: str
    cache_version: int = 1


class HistoricalDraftCache:
    def __init__(self, max_entries=64, max_tokens=20480):
        if max_entries < 1 or max_tokens < 1:
            raise ValueError('Positive history budgets required')
        self.max_entries, self.max_tokens = max_entries, max_tokens
        self.entries = OrderedDict()
        self.scope = None
        self.frame = None
        self.timestamp = None

    def reset(self):
        self.entries.clear()
        self.scope = self.frame = self.timestamp = None

    @property
    def nbytes(self):
        # Serialized payload bytes, not Python allocator/RSS accounting.
        return sum(len(json.dumps(dict(contract=vars(a.contract), frame=a.frame,
            timestamp=a.timestamp, tokens=a.token_ids, stop=a.stop_reason,
            version=a.cache_version)).encode()) for a in self.entries.values())

    def begin_frame(self, contract, frame, timestamp):
        if not contract.valid() or not frame or not math.isfinite(timestamp):
            self.reset()
            return False
        scope = (contract.scene, contract.cameras, contract.model_signature,
                 contract.tokenizer_signature, contract.generation_signature)
        if (scope != self.scope or self.timestamp is not None and
                (timestamp < self.timestamp or timestamp == self.timestamp and frame != self.frame)):
            self.reset()
        self.scope, self.frame, self.timestamp = scope, frame, timestamp
        return True

    def lookup(self, contract):
        answer = self.entries.get(contract)
        if (answer is None or not contract.valid() or self.scope != (
                contract.scene, contract.cameras, contract.model_signature,
                contract.tokenizer_signature, contract.generation_signature)):
            return None
        return answer

    def commit(self, contract, frame, timestamp, token_ids, stop_reason, *,
               completed=False, parsed=False, legally_submitted=False):
        tokens = tuple(token_ids)
        if (not completed or not parsed or not legally_submitted or not tokens
                or any(type(t) is not int or t < 0 for t in tokens)
                or stop_reason not in ('eos', 'max_new_tokens')
                or len(tokens) > self.max_tokens or not contract.valid()
                or frame != self.frame or timestamp != self.timestamp
                or self.scope != (contract.scene, contract.cameras, contract.model_signature,
                                  contract.tokenizer_signature, contract.generation_signature)):
            return False
        self.entries.pop(contract, None)
        self.entries[contract] = HistoricalAnswer(contract, frame, timestamp, tokens, stop_reason)
        while (len(self.entries) > self.max_entries or
               sum(len(a.token_ids) for a in self.entries.values()) > self.max_tokens):
            self.entries.popitem(last=False)
        return True
