"""Benchmark-only equal-token HF generation, retaining every real forward.

This module is never imported by serving. The same logits processor is applied
to native Llava.generate and the existing M3 PrefillExecutor.generate. It changes
only next-token selection, not embeddings, attention, positions, KV or forwards.
"""
from functools import wraps
from time import perf_counter
import json
from pathlib import Path
import torch
from transformers import LogitsProcessor,LogitsProcessorList


class FixedSequenceProcessor(LogitsProcessor):
    def __init__(self,tokens):
        if not tokens or any(type(t) is not int or t<0 for t in tokens):
            raise ValueError('Expected nonempty nonnegative token IDs')
        self.tokens=tuple(tokens);self.calls=0;self.events=[];self.enqueue_ms=0.

    def __call__(self,input_ids,scores):
        tick=perf_counter()
        if input_ids.shape[0]!=1 or self.calls>=len(self.tokens):
            raise ValueError('Fixed-token step/batch mismatch')
        token=self.tokens[self.calls]
        if token>=scores.shape[-1]:raise ValueError('Reference token outside model vocabulary')
        pair=None
        if scores.is_cuda:
            pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));pair[0].record()
        scores.fill_(-float('inf'));scores[:,token]=0.
        if pair:pair[1].record();self.events.append(pair)
        self.calls+=1
        self.enqueue_ms+=(perf_counter()-tick)*1000
        return scores


def forced_generation(tokens):
    processor=FixedSequenceProcessor(tokens)
    return processor,dict(logits_processor=LogitsProcessorList([processor]),
        # This pinned HF version resolves None back to the model EOS. An empty
        # EOS set disables early EOS while max_new_tokens fixes the exact work.
        max_new_tokens=len(tokens),eos_token_id=[],forced_eos_token_id=None)


class FixedTokenBenchmark:
    def __init__(self,model,reference):
        self.reference_path=str(reference)
        self.reference=json.loads(Path(reference).read_text())
        self.current=[];self.index=0;self.processors=[];self.decode_inputs=[];self.forward_calls=0
        self._wrap_generate(model.lm_head,'generate')
        if model.m3_runtime is not None:self._wrap_generate(model.m3_runtime.executor,'generate')
        from worldstream.m3.visual_state import base_model
        base=base_model(model.lm_head);original=base.forward
        @wraps(original)
        def observe(*args,**kwargs):
            self.forward_calls+=1
            if kwargs.get('inputs_embeds') is None and kwargs.get('past_key_values') is not None:
                self.decode_inputs.append(kwargs['input_ids'].detach())
            return original(*args,**kwargs)
        observe._worldstream_read_only_observer=True
        base.forward=observe

    def _wrap_generate(self,obj,name):
        original=getattr(obj,name)
        @wraps(original)
        def generate(*args,**kwargs):
            if self.index>=len(self.current):raise ValueError('Unexpected extra generation request')
            request=self.current[self.index];self.index+=1
            processor,overrides=forced_generation(request['tokens'])
            if 'logits_processor' in kwargs:raise ValueError('Unexpected existing custom logits processor')
            kwargs.update(overrides);self.processors.append(processor)
            return original(*args,**kwargs)
        setattr(obj,name,generate)

    def prepare(self,frame_id):
        self.current=self.reference['frames'][frame_id]
        self.index=0;self.processors=[];self.decode_inputs=[];self.forward_calls=0

    def collect(self,answers):
        if self.index!=len(self.current) or len(answers)!=len(self.current):
            raise ValueError('Fixed reference request count differs')
        expected=[token for r in self.current for token in r['tokens'][:-1]]
        actual=torch.cat([t.reshape(-1) for t in self.decode_inputs]).cpu().tolist() if self.decode_inputs else []
        if actual!=expected:raise ValueError('Actual model decode inputs differ from reference history')
        if [a['Q'] for a in answers]!=[r['question'] for r in self.current]:raise ValueError('Fixed reference questions differ')
        if [p.calls for p in self.processors]!=[len(r['tokens']) for r in self.current]:raise ValueError('Generation stopped before all fixed steps')
        return dict(reference=self.reference_path,source=self.reference['source'],
            generated_steps=sum(p.calls for p in self.processors),actual_decode_forward_calls=len(self.decode_inputs),
            actual_total_llm_forward_calls=self.forward_calls,decode_input_history_equal=True,
            selector_enqueue_ms=sum(p.enqueue_ms for p in self.processors),
            selector_cuda_span_ms=sum(a.elapsed_time(b) for p in self.processors for a,b in p.events),
            scope='Actual HF generation with common forced selection; prefill emits first token, remaining T-1 tokens invoke real incremental forwards. Selection overhead is included in E2E.')

    def validate_output(self,question_index,question,output):
        reference=self.current[question_index]
        if reference['question']!=question or output[0][1:]!=reference['tokens']:
            raise ValueError('Returned token sequence differs from the fixed reference')
