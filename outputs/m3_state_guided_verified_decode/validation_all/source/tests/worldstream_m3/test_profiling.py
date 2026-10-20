"""Profiling must preserve the signature inspected by PEFT/HF generate."""
import inspect
import torch
from peft import LoraConfig,get_peft_model
from tools.worldstream_m3.experiment import Timer


def test_forward_instrumentation_preserves_peft_generation(llm,carriers,monkeypatch):
    class CPUEvent:
        def __init__(self,**kwargs):pass
        def record(self):pass
        def elapsed_time(self,other):return 1.
    monkeypatch.setattr(torch.cuda,'Event',CPUEvent)
    model=get_peft_model(llm,LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    ids=torch.tensor([[1,-200,3,4]])
    torch.manual_seed(41)
    expected=model.generate(inputs=ids,images=carriers,do_sample=True,temperature=.7,max_new_tokens=3)
    signature=inspect.signature(llm.forward)
    timer=Timer();timer.wrap(llm,'llm',dynamic=True)
    timer.wrap_method(llm,'prepare_inputs_labels_for_multimodal','multimodal_preparation')
    assert inspect.signature(llm.forward)==signature
    torch.manual_seed(41)
    actual=model.generate(inputs=ids,images=carriers,do_sample=True,temperature=.7,max_new_tokens=3)
    assert torch.equal(actual,expected)
    assert timer.collect()['prefill']['calls']==1
