from dataclasses import replace
import pytest
import torch
from transformers import LogitsProcessor,LogitsProcessorList
from worldstream.m3.answer_memory import AnswerMemory,DraftCandidate
from worldstream.m3.task_conditioner import answer_facts,question_dependencies
from worldstream.m3 import expand_input
from .test_speculative_decode import executor
from .test_world_guided_decode import state


@pytest.mark.parametrize('text',[
    'Yes, there is a car.','There is a white car ahead.',
    'Several vehicles are visible.','A truck can be seen on the left.'])
def test_positive_facts_without_yes_prefix(text):
    f=answer_facts(text)
    assert f.polarity=='positive' and f.classes and not f.uncertain


@pytest.mark.parametrize('text',[
    'There are no cars.','A car might be visible.','Is there a truck?',
    'There is a car?','It is not possible to see any vehicles.','Perhaps a car ahead.'])
def test_mentions_not_promoted_to_positive_facts(text):
    f=answer_facts(text)
    assert f.polarity!='positive'


def test_exact_changed_geometry_preserves_complete_draft():
    m=AnswerMemory();ctx=dict(scene_id='s',frame_id='f0',frame_index=0,timestamp=1.)
    old=state();now=state('f1',1.5,2)
    now=replace(now,anchors=(replace(now.anchors[0],center_ego=(-3.,-2.,0.),identity_qualified=False),))
    m.remember('Where is the car?',[1,4],range(12),ctx,('binding',),old,'There is a car ahead.')
    current=dict(ctx,frame_id='f1',timestamp=1.5,frame_index=1)
    s=m.retrieve('Where is the car?',[1,4],current,('binding',),now,True)
    s=m.construct(s,'Where is the car?',[1,4],current,('binding',),now,None)
    assert len(s.candidates)==1 and s.candidates[0].tokens==tuple(range(12))
    assert s.candidates[0].block_length==8 and s.delta.unknown_dependencies


def test_clear_path_is_full_scene_safety_not_vehicle_existence():
    d=question_dependencies('Is there a clear path for the vehicle to proceed?')
    assert d.intent=='safety' and d.complete_coverage_required


def test_road_question_scope_and_negation():
    assert question_dependencies('What type of road is the vehicle currently on?').intent=='road'
    for question in ('Are there no cars?', 'Is there any road without pedestrians?'):
        assert question_dependencies(question).complete_coverage_required


def candidate(tokens,kind='question_conditioned',expected=8.,cost=1.05):
    return DraftCandidate(tuple(tokens),kind,None,None,8,expected,cost)


@pytest.mark.parametrize('wrong_middle',[False,True])
def test_real_target_selects_without_extra_forward(llm,carriers,wrong_middle):
    ex=expand_input(llm,torch.tensor([[1,-200,4]]),carriers)
    kw=dict(max_new_tokens=15,do_sample=False,eos_token_id=None,capture=False)
    native,_=executor().generate(llm,ex,**kw)
    correct=native[0,1:].tolist()
    if wrong_middle:correct[4]=(correct[4]+1)%64
    wrong=[(native[0,1].item()+1)%64]*8
    result,d=executor().generate(llm,ex,draft_candidates=[candidate(wrong),candidate(correct)],**kw)
    assert torch.equal(result,native)
    assert d['selected_candidate_index']==1 and d['draft_accepted_tokens']>=4
    assert d['target_selection_extra_forwards']==d['target_selection_rng_draws']==0
    if wrong_middle:assert d['verification_log'][0]['rejection_position']==4


def test_sampling_filter_preserves_rng_when_no_verification(llm,carriers):
    ex=expand_input(llm,torch.tensor([[1,-200,4]]),carriers)
    kw=dict(max_new_tokens=12,do_sample=True,eos_token_id=None,capture=False,temperature=.1,top_p=.75)
    torch.manual_seed(73);native,_=executor().generate(llm,ex,**kw);rng=torch.get_rng_state()
    torch.manual_seed(73)
    result,d=executor().generate(llm,ex,draft_candidates=[candidate([5,6],cost=100.)],experimental_sampling=True,**kw)
    assert torch.equal(native,result) and torch.equal(rng,torch.get_rng_state())
    assert d['draft_status']=='target_filtered_native' and not d['verification_log']
    assert d['target_selection_rng_draws']==0


def test_sampling_selection_uses_processed_probability_not_argmax(llm,carriers):
    class Force(LogitsProcessor):
        def __call__(self,ids,scores):
            scores[:]=-float('inf');scores[:,7]=0.;return scores
    ex=expand_input(llm,torch.tensor([[1,-200,4]]),carriers)
    result,d=executor().generate(llm,ex,draft_candidates=[candidate([8]*8),candidate([7]*8)],
        experimental_sampling=True,capture=False,max_new_tokens=12,do_sample=True,
        temperature=.1,top_p=.75,eos_token_id=None,logits_processor=LogitsProcessorList([Force()]))
    assert result[0,1:].tolist()==[7]*12
    assert d['target_selection_rankings'][0]['first_token_probability']==0.
    assert d['target_selection_rankings'][1]['first_token_probability']==1.
    assert d['selected_candidate_index']==1 and d['draft_accepted_tokens']==8


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA target-aware path')
def test_cuda_selector_span_and_rng(llm,carriers):
    llm=llm.cuda();carriers=carriers.cuda()
    ex=expand_input(llm,torch.tensor([[1,-200,4]],device='cuda'),carriers)
    kw=dict(max_new_tokens=12,do_sample=True,eos_token_id=None,capture=False,profile=True,temperature=.1,top_p=.75)
    torch.cuda.manual_seed(73);native,_=executor().generate(llm,ex,**kw);rng=torch.cuda.get_rng_state()
    torch.cuda.manual_seed(73)
    result,d=executor().generate(llm,ex,draft_candidates=[candidate([5,6],cost=100.)],experimental_sampling=True,**kw)
    assert torch.equal(native,result) and torch.equal(rng,torch.cuda.get_rng_state())
    assert d['target_selection_cuda_span_ms']>0 and d['draft_status']=='target_filtered_native'
