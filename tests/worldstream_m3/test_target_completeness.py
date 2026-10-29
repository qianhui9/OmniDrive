from dataclasses import replace
import pytest
from worldstream.m3.world_state import CurrentCompetitionEvidence
from worldstream.m3.config import M3Config
from worldstream.m3.answer_validity import propose_fact_answer
from worldstream.m3.answer_memory import AnswerMemory
from .test_world_guided_decode import state
from .test_selective_answer_reuse import context,BINDING,Q

POSITION='Where is the nearest pedestrian relative to the vehicle?'


def policy(**kw):
    return dict(min_confidence=.45,expected_normal_m3_ms=1000.,max_binding_risk=.75,
        target_completeness_enabled=True,position_min_confidence=.55,max_target_risk=.5,
        center_nearest_min_cameras=2,**kw)


def propose(q,w,p=None):
    return propose_fact_answer(q,w,context(),AnswerMemory(),BINDING,p or policy(),None,True,False,True)


def candidate(cls='pedestrian',center=(12.,4.,0.),score=.3,**kw):
    return CurrentCompetitionEvidence('decoded:0',cls,center,score,('front',),'f0',1.,**kw)


def pedestrian():
    w=state();a=replace(w.anchors[0],semantic_class='pedestrian',center_ego=(20.,-5.,0.),score=.8)
    return replace(w,anchors=(a,))


def test_low_confidence_competitor_vetoes_nearest_but_never_supplies_facts():
    w=replace(pedestrian(),competition_evidence=(candidate(score=.45),))
    p=propose(POSITION,w)
    assert p['supported'] and not p['admitted']
    assert p['facts'][0]['center_ego']==w.anchors[0].center_ego
    assert not p['target_completeness']['competitors'][0]['can_support_answer']
    p=propose(POSITION,replace(w,anchors=()))
    assert not p['supported'] and not p['answer_text']


def test_safe_position_remains_available_not_unconditionally_disabled():
    p=propose(POSITION,pedestrian())
    assert p['admitted'] and 'front-right' in p['answer_text']


def test_same_answer_and_unrelated_competitors_do_not_veto():
    w=replace(pedestrian(),competition_evidence=(candidate(center=(12.,-4.,0.),score=.99),candidate(cls='truck',score=.99)))
    p=propose(POSITION,w)
    assert p['admitted'] and not p['target_completeness']['competitors']


def test_positive_existence_does_not_need_nearest_completeness_or_absence():
    w=state();w=replace(w,competition_evidence=(candidate(cls='car',score=.99),))
    p=propose('Is there a car in front?',w)
    assert p['admitted'] and p['target_completeness']['risk']==0.
    p=propose('Is there a car in front?',replace(w,anchors=()))
    assert not p['supported'] and 'UNKNOWN_not_absent' in p['reason']


@pytest.mark.parametrize('change',[dict(source_frame='future'),dict(source_timestamp=2.),
    dict(evidence_origin='oracle'),dict(cameras=()),dict(answer_witness_eligible=True)])
def test_noncurrent_or_nonperception_competition_is_ignored(change):
    w=replace(pedestrian(),competition_evidence=(replace(candidate(score=.99),**change),))
    assert propose(POSITION,w)['admitted']


def test_center_projection_is_task_conditioned_and_car_fix_remains_possible():
    w=state();a=replace(w.anchors[0],observation_kind='current_detector_center_projection_not_reuse_certificate')
    one=replace(w,anchors=(a,));two=replace(w,anchors=(replace(a,cameras=('front','front_right')),))
    assert not propose(Q,one)['admitted']
    assert propose(Q,two)['admitted']
    assert propose('Is there a car in front?',one)['admitted']


def test_basic_ablation_preserves_v3_admission_on_same_evidence():
    w=replace(pedestrian(),competition_evidence=(candidate(score=.99),))
    p=policy();p['target_completeness_enabled']=False
    out=propose(POSITION,w,p)
    assert out['admitted'] and 'target_completeness' not in out


def test_unsupported_tasks_never_run_completeness(monkeypatch):
    import worldstream.m3.target_binding as module
    def fail(*args,**kwargs):raise AssertionError('Unsupported task scanned target evidence')
    monkeypatch.setattr(module,'question_target_completeness',fail)
    out=propose('Describe the traffic lights and road',state())
    assert not out['supported'] and out['target_completeness_wall_ms']==0.


def test_production_completeness_flag_defaults_off():
    assert not M3Config().target_completeness_enabled
