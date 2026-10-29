from dataclasses import replace
import json
import subprocess
import sys
import pytest
import torch
from worldstream.m3.request_rng import request_seed,question_rng
from worldstream.m3.target_binding import target_binding_risk
from worldstream.m3.task_conditioner import question_facts
from worldstream.m3.answer_validity import propose_fact_answer
from worldstream.m3.answer_memory import AnswerMemory
from worldstream.m3.config import M3Config
from .test_world_guided_decode import state
from .test_selective_answer_reuse import context,policy,BINDING,Q


def test_question_rng_restores_outer_and_ignores_prior_question_draws():
    ctx=dict(scene_id='s',frame_id='f',repeat=1,experiment_seed=17,rng_audit=True)
    torch.manual_seed(1);outer=torch.get_rng_state().clone()
    with question_rng(ctx,2,'cpu',True) as d:a=torch.rand(40)
    assert torch.equal(outer,torch.get_rng_state()) and d['outer_before']==d['outer_after']
    torch.rand(101)
    with question_rng(ctx,2,'cpu',True):b=torch.rand(40)
    assert torch.equal(a,b)
    with question_rng(ctx,1,'cpu',True):c=torch.rand(40)
    assert not torch.equal(a,c)


def test_question_seed_is_cross_process_stable_and_condition_independent():
    ctx=dict(scene_id='s',frame_id='f',repeat=2,experiment_seed=19)
    code='from worldstream.m3.request_rng import request_seed;print(request_seed('+repr(ctx)+',2))'
    assert int(subprocess.check_output([sys.executable,'-c',code]))==request_seed(ctx,2)
    assert request_seed(dict(ctx,condition='B'),2)==request_seed(dict(ctx,condition='C'),2)


def test_question_rng_exception_restores_state():
    outer=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with question_rng(dict(scene_id='s',frame_id='f'),0,'cpu',True):
            torch.rand(10);raise RuntimeError('test')
    assert torch.equal(outer,torch.get_rng_state())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
def test_cuda_request_sampling_is_isolated():
    ctx=dict(scene_id='s',frame_id='f');outer=torch.cuda.get_rng_state().clone()
    with question_rng(ctx,1,'cuda',True):a=torch.multinomial(torch.ones(50,device='cuda'),20,replacement=True)
    assert torch.equal(outer,torch.cuda.get_rng_state())
    torch.rand(30,device='cuda')
    with question_rng(ctx,1,'cuda',True):b=torch.multinomial(torch.ones(50,device='cuda'),20,replacement=True)
    assert torch.equal(a,b)


def worlds():
    prev=state();car=replace(prev.anchors[0],center_ego=(8.,0.,0.),velocity_ego=())
    prev=replace(prev,anchors=(car,))
    now=state('f1',1.5,2);truck=replace(now.anchors[0],anchor_id='truck',semantic_class='truck',
        center_ego=(20.,0.,0.),velocity_ego=())
    return prev,replace(now,anchors=(truck,))


def propose(now,prev,temporal=True):
    return propose_fact_answer(Q,now,context(now.frame_id,now.timestamp,now.version),AnswerMemory(),
        BINDING,dict(policy(),max_binding_risk=.35),prev,True,temporal)


def test_history_competitor_veto_never_becomes_current_fact():
    prev,now=worlds();current=propose(now,prev,False);temporal=propose(now,prev)
    assert current['admitted'] and not temporal['admitted']
    assert temporal['facts'][0]['semantic_class']=='truck'
    comp=temporal['target_binding']['temporal_competitors'][0]
    assert comp['semantic_class']=='car' and not comp['current_fact']


def test_ego_translation_changes_historical_competition():
    prev,now=worlds()
    pose=list(now.ego_to_world);pose[3]=30.
    now=replace(now,ego_to_world=tuple(pose))
    assert not propose(now,prev)['target_binding']['temporal_competitors']


@pytest.mark.parametrize('change',[dict(scene_id='other'),dict(epoch=3),dict(execution_signature='other'),dict(timestamp=-10.)])
def test_illegal_history_does_not_cross_boundary(change):
    prev,now=worlds()
    assert not propose(now,replace(prev,**change))['target_binding']['temporal_competitors']


def test_reobserved_qualified_target_resolves_competition():
    prev,now=worlds();car=replace(prev.anchors[0],source_frame=now.frame_id,source_timestamp=now.timestamp)
    now=replace(now,anchors=now.anchors+(car,))
    p=propose(now,prev)
    assert p['admitted'] and p['facts'][0]['semantic_class']=='car'


def test_unrelated_history_cannot_veto_positive_existence():
    prev,now=worlds();now=replace(now,anchors=(replace(now.anchors[0],semantic_class='car'),))
    p=propose_fact_answer('Is there a car in front?',now,context('f1',1.5,2),AnswerMemory(),
        BINDING,policy(),prev,True,True)
    assert p['admitted'] and not p['target_binding']['temporal_competitors']


def test_current_competition_and_sector_boundaries():
    prev,now=worlds();a=replace(now.anchors[0],center_ego=(12.,0.,0.))
    b=replace(a,anchor_id='car',semantic_class='car',center_ego=(13.9,0.,0.))
    risk=target_binding_risk(question_facts(Q),a,[a,b],replace(now,anchors=(a,b)),None)
    assert risk['current_competitors'] and risk['current_risk']>.35
    a=replace(a,center_ego=(12.,3.7,0.))
    assert target_binding_risk(question_facts(Q),a,[a],now,None)['boundary']


def test_temporal_switch_requires_current_binding_and_defaults_off():
    cfg=M3Config()
    assert not cfg.question_rng_enabled and not cfg.target_binding_enabled and not cfg.temporal_risk_enabled
    with pytest.raises(ValueError):M3Config(temporal_risk_enabled=True)


def test_camera_timestamp_support_remains_current_only():
    prev,now=worlds();stale=replace(now.anchors[0],source_timestamp=now.timestamp-.001)
    p=propose(replace(now,anchors=(stale,)),prev)
    assert not p['supported'] and not p['admitted']


def test_reliable_motion_and_ego_rotation_use_one_current_coordinate_system():
    prev,now=worlds();car=replace(prev.anchors[0],center_ego=(8.,0.,0.),velocity_ego=(-30.,0.,0.))
    p=propose(now,replace(prev,anchors=(car,)))
    assert not p['target_binding']['temporal_competitors']
    pose=(0.,-1.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.)
    p=propose(replace(now,ego_to_world=pose),prev)
    assert p['target_binding']['temporal_competitors'][0]['predicted_center_current_ego']==[0.,-8.,0.]


def test_same_category_history_cannot_change_category_fact():
    prev,now=worlds();prev=replace(prev,anchors=(replace(prev.anchors[0],semantic_class='truck'),))
    assert propose(now,prev)['admitted']


def test_runtime_prior_fast_answer_cannot_shift_later_sampling(llm,carriers,tmp_path):
    from worldstream.m3 import M3Runtime
    from .test_selective_answer_reuse import Tokenizer
    p=tmp_path/'policy.json';p.write_text(json.dumps(policy()))
    answers=[];diags=[]
    for mode in ('off','online'):
        rt=M3Runtime(dict(reuse_strategy='exact_prefix',decode_strategy='world_guided',target_aware_exact_only=True,
            experimental_speculative_sampling=True,
            question_rng_enabled=True,selective_answer_mode=mode,selective_answer_policy=str(p) if mode=='online' else ''))
        a,d=rt.generate(llm,[torch.tensor([1,-200,4]),torch.tensor([1,-200,5])],carriers,
            context=dict(context(),experiment_seed=92,repeat=3,rng_audit=True),questions=[Q,'Describe the road'],
            tokenizer=Tokenizer(),committed_world_state=state(),do_sample=True,temperature=.7,top_p=.75,
            max_new_tokens=5)
        answers.append(a);diags.append(d)
    assert diags[1]['requests'][0]['llm_skipped']
    assert torch.equal(answers[0][1],answers[1][1])
    assert diags[0]['requests'][1]['question_rng']['sampling_before']==diags[1]['requests'][1]['question_rng']['sampling_before']
