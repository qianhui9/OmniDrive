from dataclasses import replace
import json

import pytest
import torch

from worldstream.m3.answer_memory import AnswerMemory
from worldstream.m3.answer_validity import propose_fact_answer
from .test_world_guided_decode import state
from .test_selective_answer_reuse import context,BINDING,Q,Tokenizer
from tools.worldstream_m123.fact_bypass_identity import (
    match_instances,question_target,request_identity,evaluate)

POSITION='Where is the nearest pedestrian relative to the vehicle?'


def policy(**kw):
    return dict(status='FROZEN_RESEARCH_POLICY',main_path='current_only',position_mode='restricted',
        min_confidence=.45,position_min_confidence=.65,expected_normal_m3_ms=1000.,
        max_binding_risk=.75,calibration_sha256='unit-test-only',**kw)


def propose(q,w,p=None,memory=None,**kw):
    return propose_fact_answer(q,w,context(),memory or AnswerMemory(),BINDING,p or policy(),
        target_binding_enabled=True,**kw)


def pedestrian(score=.7):
    w=state()
    return replace(w,anchors=(replace(w.anchors[0],semantic_class='pedestrian',center_ego=(20.,-5.,0.),score=score),))


def test_current_only_does_not_read_history_but_legacy_still_can():
    class ForbiddenMemory:
        @property
        def records(self):raise AssertionError('Current-only read answer history')
    p=propose(Q,state(),memory=ForbiddenMemory())
    assert p['admitted'] and p['answer_source']=='fact_grounded_response'
    assert p['history_candidate_count']==0


def test_restriction_preserves_safe_position_without_changing_other_tasks():
    assert propose(POSITION,pedestrian())['admitted']
    rejected=propose(POSITION,pedestrian(.6))
    assert rejected['supported'] and not rejected['admitted']
    assert rejected['position_admission_wall_ms']>=0
    assert propose('Are there pedestrians on the right?',pedestrian(.6))['admitted']
    w=state();w=replace(w,anchors=(replace(w.anchors[0],score=.6),))
    assert propose(Q,w)['admitted']


@pytest.mark.parametrize('flag',['temporal_risk_enabled','target_completeness_enabled'])
def test_current_only_refuses_historical_risk_modules(flag):
    with pytest.raises(ValueError,match='Current-only'):
        propose(Q,state(),**{flag:True})


@pytest.mark.parametrize('floor',[None,float('nan'),.2,1.1])
def test_position_restricted_requires_valid_dev_floor(floor):
    p=policy();p['position_min_confidence']=floor
    with pytest.raises(ValueError,match='DEV confidence floor'):
        propose(POSITION,pedestrian(),p)


def test_position_fallback_executes_same_target_aware_generation(llm,carriers,tmp_path):
    from worldstream.m3 import M3Runtime
    p=policy();p['position_mode']='fallback';path=tmp_path/'policy.json';path.write_text(json.dumps(p))
    common=dict(reuse_strategy='exact_prefix',decode_strategy='world_guided',
        target_aware_exact_only=True,question_rng_enabled=True,target_binding_enabled=True)
    b=M3Runtime(common);c=M3Runtime(dict(common,selective_answer_mode='online',selective_answer_policy=str(path)))
    kwargs=dict(context=context(),questions=[POSITION],tokenizer=Tokenizer(),committed_world_state=pedestrian(),
        do_sample=False,max_new_tokens=3)
    ids=[torch.tensor([1,-200,4])]
    ba,bd=b.generate(llm,ids,carriers,**kwargs);ca,cd=c.generate(llm,ids,carriers,**kwargs)
    assert torch.equal(ba[0],ca[0])
    br,cr=bd['requests'][0],cd['requests'][0]
    assert not cr['llm_skipped'] and cr['prefill_completed'] and cr['decode_completed']
    assert br['llm_forward_calls']==cr['llm_forward_calls']>0


def test_current_only_runtime_hit_skips_input_expansion_and_all_forwards(llm,carriers,tmp_path,monkeypatch):
    from worldstream.m3 import M3Runtime
    import worldstream.m3.runtime as module
    path=tmp_path/'policy.json';path.write_text(json.dumps(policy()))
    runtime=M3Runtime(dict(reuse_strategy='exact_prefix',decode_strategy='world_guided',
        target_aware_exact_only=True,question_rng_enabled=True,target_binding_enabled=True,
        selective_answer_mode='online',selective_answer_policy=str(path)))
    def forbidden(*a,**kw):raise AssertionError('Fact hit entered language model')
    monkeypatch.setattr(module,'expand_input',forbidden)
    monkeypatch.setattr(llm,'forward',forbidden)
    monkeypatch.setattr(runtime.executor,'generate',forbidden)
    _,diag=runtime.generate(llm,[torch.tensor([1,-200,4])],carriers,context=context(),questions=[Q],
        tokenizer=Tokenizer(),committed_world_state=state(),do_sample=False,max_new_tokens=3)
    row=diag['requests'][0]
    assert row['llm_skipped'] and row['llm_forward_calls']==0
    assert row['layout_preparation_ms']==0 and not row['prefill_completed'] and not row['decode_completed']


@pytest.mark.parametrize('change',[dict(main_path='unknown'),dict(position_mode='invalid'),
    dict(position_min_confidence=float('nan')),dict(target_completeness_enabled=True),dict(temporal_risk_enabled=True)])
def test_runtime_rejects_inconsistent_current_policy_before_processing(change,tmp_path):
    from worldstream.m3 import M3Runtime
    p=policy();p.update(change);path=tmp_path/'bad.json';path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        M3Runtime(dict(reuse_strategy='exact_prefix',decode_strategy='world_guided',
            target_aware_exact_only=True,selective_answer_mode='online',selective_answer_policy=str(path)))


def gt(a='g',cls='pedestrian',center=(20.,-5.,0.),visible=True):
    return dict(annotation_id=a,instance_id='i'+a,semantic_class=cls,center_ego=center,
        visible=visible,cameras=['front'],source_frame='f0')


def anchors(w):
    from dataclasses import asdict
    return [asdict(a) for a in w.anchors]


def test_instance_match_does_not_condition_on_class_and_detects_wrong_target():
    w=pedestrian();a=anchors(w)
    m=match_instances(a,[gt(cls='truck')],'f0',1.)
    assert m[a[0]['anchor_id']]['known'] and not m[a[0]['anchor_id']]['class_correct']
    gs=[gt(),gt('closer',center=(10.,4.,0.))]
    m=match_instances(a,gs,'f0',1.)
    r=request_identity(propose(POSITION,w),question_target(POSITION,gs),m)
    assert r['object']['correct'] and r['target']['known'] and not r['target']['correct']


def test_duplicate_detections_never_count_twice():
    w=pedestrian();a=anchors(w);a.append(dict(a[0],anchor_id='duplicate',center_ego=(20.05,-5.,0.)))
    m=match_instances(a,[gt()],'f0',1.)
    assert all(not r['known'] for r in m.values())


def test_assignment_cannot_promote_a_second_choice_to_certain_identity():
    w=pedestrian();a=anchors(w)
    a[0]['center_ego']=(20.1,-5.,0.)
    a.append(dict(a[0],anchor_id='second',center_ego=(19.2,-5.,0.)))
    m=match_instances(a,[gt(),gt('other',center=(21.,-5.,0.))],'f0',1.)
    assert all(not r['known'] for r in m.values())


def test_noncurrent_anchor_cannot_receive_an_instance_label():
    a=anchors(pedestrian());a[0]['source_frame']='future'
    assert not match_instances(a,[gt()],'f0',1.)


def test_common_sector_answer_label_does_not_create_identity_label():
    gs=[gt(),gt('second',center=(21.,-5.,0.))]
    target=question_target(POSITION,gs)
    assert not target['known'] and target['answer_known']
    assert target['reason']=='multiple_equally_plausible_GT_targets'
    assert evaluate(POSITION,'The nearest pedestrian is to the front-right.',gs,target,'f0')['score']==1
    gs[1]['center_ego']=(20.,5.,0.)
    assert not question_target(POSITION,gs)['answer_known']


def test_visibility_ambiguity_and_missing_annotations_are_unknown():
    assert question_target(POSITION,None)['reason']=='missing_annotation_support'
    assert question_target(POSITION,[])['reason']=='no_visible_GT_target'
    gs=[gt(),gt('occluded',center=(10.,4.,0.),visible=False)]
    assert question_target(POSITION,gs)['reason']=='visibility_ambiguous'
    assert not question_target(POSITION,gs)['answer_known']


def test_positive_existence_uses_witness_set_not_nearest_identity():
    q='Are there pedestrians on the right?'
    gs=[gt(),gt('second',center=(21.,-5.,0.))]
    t=question_target(q,gs)
    assert t['known'] and len(t['acceptable_annotation_ids'])==2
    assert question_target('What color is the pedestrian?',gs)['reason']=='question_semantics_ambiguous'
