import torch
from worldstream.m3.native_decode import zero_mask_decode
from worldstream.m3.visual_state import deployment_key,execution_signature,base_model


@torch.no_grad()
def test_zero_mask_decode_preserves_complete_sampling_ids_rng_and_backend_scope(llm,carriers):
    ids=torch.tensor([[1,3,-200,4,5]])
    kw=dict(inputs=ids,images=carriers,do_sample=True,temperature=.1,top_p=.75,max_new_tokens=7,use_cache=True)
    torch.manual_seed(91);reference=llm.generate(**kw);rng=torch.get_rng_state()
    key=deployment_key(llm);signature=execution_signature(llm)
    with zero_mask_decode(llm) as stats:
        assert key!=deployment_key(llm) and signature!=execution_signature(llm)
        torch.manual_seed(91);actual=llm.generate(**kw)
        assert torch.equal(reference,actual) and torch.equal(rng,torch.get_rng_state())
        assert stats['skipped_zero_masks']>0
    assert deployment_key(llm)==key and execution_signature(llm)==signature


@torch.no_grad()
def test_padded_decode_keeps_native_additive_mask_and_logits(llm):
    model=base_model(llm);ids=torch.tensor([[1,3,4]])
    seed=model(input_ids=ids,use_cache=True)
    mask=torch.tensor([[1,0,1,1]])
    kw=dict(input_ids=torch.tensor([[5]]),past_key_values=seed.past_key_values,attention_mask=mask,use_cache=True)
    expected=model(**kw)
    with zero_mask_decode(llm) as stats:
        actual=model(**kw)
        assert stats['skipped_zero_masks']==0 and stats['full_masks']==1
    assert torch.equal(expected.logits,actual.logits)
