"""Opt-in native LLaMA decode probe: omit only provably zero additive masks.

No Q/K/V/LoRA, KV layout, sampling or generation loop is replaced. The all-ones
check is real work (including its GPU sync) and must be charged to benchmarks.
"""
from contextlib import contextmanager
from functools import wraps
import torch
from .visual_state import base_model


@contextmanager
def zero_mask_decode(lm_head):
    import transformers
    from transformers.models.llama.modeling_llama import LlamaModel
    base=base_model(lm_head);decoder=base.model
    if not transformers.__version__.startswith('4.31.') or not isinstance(decoder,LlamaModel):
        raise ValueError('Only inspected native Transformers 4.31 LLaMA is supported')
    if any(m.training for m in base.modules()):raise ValueError('Decode probe requires eval mode')
    attr='_prepare_decoder_attention_mask';owned=attr in vars(decoder);original=getattr(decoder,attr)
    policy='_worldstream_decode_policy';previous={id(m):(policy in vars(m),getattr(m,policy,None)) for m in (lm_head,base)}
    stats=dict(skipped_zero_masks=0,full_masks=0,all_ones_host_checks=0)
    @wraps(original)
    def prepare(attention_mask,input_shape,inputs_embeds,past_key_values_length):
        if (tuple(input_shape)==(1,1) and past_key_values_length>0 and attention_mask is not None
                and tuple(attention_mask.shape)==(1,past_key_values_length+1)):
            stats['all_ones_host_checks']+=1
            if bool((attention_mask==1).all()):
                stats['skipped_zero_masks']+=1
                return None
        stats['full_masks']+=1
        return original(attention_mask,input_shape,inputs_embeds,past_key_values_length)
    try:
        setattr(decoder,attr,prepare)
        for module in (lm_head,base):setattr(module,policy,'native_zero_mask_v1')
        yield stats
    finally:
        if owned:setattr(decoder,attr,original)
        else:delattr(decoder,attr)
        for module in {id(m):m for m in (lm_head,base)}.values():
            existed,value=previous[id(module)]
            if existed:setattr(module,policy,value)
            else:delattr(module,policy)
