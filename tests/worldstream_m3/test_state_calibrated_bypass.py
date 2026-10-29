from dataclasses import replace
import pytest
from worldstream.m3.task_conditioner import question_facts,_question_facts_cached
from worldstream.m3.answer_validity import propose_fact_answer
from worldstream.m3.answer_memory import AnswerMemory
from worldstream.m3.config import M3Config
from .test_selective_answer_reuse import context,policy,BINDING,Q
from .test_world_guided_decode import state


def test_task_cache_is_bounded_immutable_and_preserves_semantics():
    _question_facts_cached.cache_clear()
    first=question_facts(Q);assert question_facts(Q) is first
    assert _question_facts_cached.cache_info().hits==1
    for i in range(600):question_facts('Describe weather '+str(i))
    assert _question_facts_cached.cache_info().currsize<=512
    assert question_facts('What color is the traffic light?').intent=='unsupported'
    with pytest.raises(Exception):first.intent='position'


def test_projection_supported_fact_still_requires_current_source_and_confidence():
    w=state();a=replace(w.anchors[0],observation_kind='current_detector_center_projection_not_reuse_certificate')
    p=propose_fact_answer(Q,replace(w,anchors=(a,)),context(),AnswerMemory(),BINDING,policy(),None,True,False)
    assert p['admitted'] and p['facts'][0]['observation_kind']==a.observation_kind
    for kwargs in (dict(score=.2),dict(source_frame='old'),dict(cameras=()),dict(evidence_origin='oracle')):
        p=propose_fact_answer(Q,replace(w,anchors=(replace(a,**kwargs),)),context(),AnswerMemory(),BINDING,policy(),None,True,False)
        assert not p['admitted']


def test_unsupported_task_exits_without_binding_or_memory_scan(monkeypatch):
    import worldstream.m3.target_binding as target
    def fail(*args,**kwargs):raise AssertionError('Unsupported task ran target/history analysis')
    monkeypatch.setattr(target,'target_binding_risk',fail)
    class NoMemory:
        @property
        def records(self):return fail()
    p=propose_fact_answer('Describe the weather and road conditions',state(),context(),NoMemory(),BINDING,
                          policy(),state(),True,True)
    assert not p['admitted'] and not p['supported']


def test_new_observation_readout_defaults_off():
    assert not M3Config().current_detection_support


def test_missing_pose_is_explicit_temporal_uncertainty_not_current_position():
    from worldstream.m3.target_binding import target_binding_risk
    current=state('f1',1.5,2);previous=replace(state(),ego_to_world=())
    a=current.anchors[0]
    r=target_binding_risk(question_facts(Q),a,[a],current,previous,True)
    assert r['temporal_status']=='invalid_pose_uncertainty'
    assert not r['temporal_competitors'] and r['temporal_risk']>0


def test_continuity_never_credits_unqualified_history_or_supplies_facts():
    from worldstream.m3.target_binding import target_binding_risk
    current=state('f1',1.5,2);previous=state()
    a=replace(current.anchors[0],identity_qualified=False)
    r=target_binding_risk(question_facts(Q),a,[a],current,previous,True)
    assert r['continuity']['matched'] and not r['continuity']['identity_qualified']
    assert r['continuity']['admission_credit']==0.
