"""Actual tiny Llama execution through the sole R0 runtime."""
import torch

from worldstream.m3 import M3Runtime


class Tokenizer:
    special_tokens_map = {}

    def get_vocab(self):
        return {'unit': 3}

    def decode(self,tokens,skip_special_tokens=True):
        return ' '.join(map(str,tokens))


def test_r0_full_current_and_exact_draft_preserve_native_tokens(llm,carriers,context):
    runtime=M3Runtime()
    tokenizer=Tokenizer()
    ids=torch.tensor([1,-200,4])
    kw=dict(max_new_tokens=15,do_sample=False,eos_token_id=None)
    for i in range(3):
        current=carriers+i*.001
        ctx=dict(context,frame_id='f'+str(i),timestamp=1.+i*.5)
        answer,diagnostic=runtime.generate(llm,[ids],current,context=ctx,
            questions=['Describe the complete scene.'],tokenizer=tokenizer,**kw)
        expected=llm.generate(inputs=ids[None],images=current,**kw)
        assert torch.equal(answer[0],expected)
        row=diagnostic['requests'][0]
        assert row['prefill_completed'] and row['decode_completed']
        assert row['reused_tokens']==0 and not row['persistent_prefill_kv']
        assert not row['prefix_planner_executed']
        assert row['llm_forward_calls']>0
        if i:
            assert row['draft_exact_available']
            assert row['draft_semantic_available'] is False


def test_r0_tokenizer_replacement_invalidates_exact_history(llm,carriers,context):
    runtime=M3Runtime()
    tokenizers=[Tokenizer(),Tokenizer()]
    ids=torch.tensor([1,-200,4])
    for i,tokenizer in enumerate(tokenizers):
        ctx=dict(context,frame_id='f'+str(i),timestamp=1.+i*.5)
        _,diagnostic=runtime.generate(llm,[ids],carriers,context=ctx,
            questions=['Describe the complete scene.'],tokenizer=tokenizer,
            max_new_tokens=6,do_sample=False,eos_token_id=None)
        row=diagnostic['requests'][0]
        assert not row['draft_exact_available']
        assert row['tokenizer_metadata_cache_hit'] is False
    records=list(runtime.answer_memory.records.values())
    assert len(records)==1
    assert records[0].frame_id=='f1'
