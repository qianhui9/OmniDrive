import copy
import pytest
import torch
from worldstream.m3.sampling_microbatch import SamplingMicrobatchExecutor
from worldstream.m3.question_dependencies import QuestionDependency
from worldstream.m3.visual_state import base_model


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@torch.no_grad()
def test_cuda_independent_generators_survive_grouping_without_global_rng_pollution(llm,carriers,monkeypatch):
    llm=llm.cuda();carriers=carriers.cuda()
    monkeypatch.setattr('worldstream.m3.sampling_microbatch.analyze_questions',
        lambda t,ids,qs,**kw:[QuestionDependency(i,independent=True) for i in range(len(ids))])
    config=copy.deepcopy(base_model(llm).generation_config);config.max_new_tokens=13
    def probs(history,logits):
        p=torch.zeros_like(logits);p[:,3]=.4;p[:,4]=.6;return p
    monkeypatch.setattr('worldstream.m3.sampling_microbatch.sampling_tools',lambda b,g:(config,probs))
    ids=[torch.tensor([1,-200,3+i],device='cuda') for i in range(4)]
    before=torch.cuda.get_rng_state().clone();answers=[];states=[]
    for size in (2,4):
        out,diag=SamplingMicrobatchExecutor(size,231).try_generate(llm,ids,carriers,['q']*4,
            dict(scene_id='s',frame_id='f',camera_ids=list('abcdef')),object())
        answers.append([t.cpu().tolist() for t in out]);states.append([r['final_rng_sha256'] for r in diag['requests']])
    assert answers[0]==answers[1] and states[0]==states[1]
    assert torch.equal(before,torch.cuda.get_rng_state())
