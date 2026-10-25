"""History integrity and task quality are independent of exact prefix matching."""
import copy
import pytest
import torch
from worldstream.m3 import M3Runtime
from worldstream.m3.world_state import VerifiedWorldState,query_world_state


def fixture_state():
    record=dict(anchor_id='old:car',anchor_type='object',semantic_class='car',
        center_ego_m=[10.,0.,0.],velocity_ego_mps=None,source_timestamp=.8,
        verified_timestamp=1.,source_frame='previous',source_state_epoch=2,
        source_execution_signature='bound_m1',evidence_origin='actual_perception')
    return VerifiedWorldState('s','f',1.,2,'bound_m1',(record,),{},state_version=3)


def state_context(context):
    return dict(context,state_epoch=2,state_version=3,state_execution_signature='bound_m1')


@pytest.mark.parametrize('field,value',[('scene_id','other'),('frame_id','other'),('timestamp',2.),
    ('state_epoch',4),('state_version',4),('state_execution_signature','other')])
def test_context_binding_rejects_stale_history(context,field,value):
    ctx=state_context(context);ctx[field]=value
    assert not query_world_state(fixture_state(),'where is the car in front',ctx)['admitted']


def test_payload_mutation_cannot_reuse_frozen_receipt(context):
    state=fixture_state();state.records[0]['center_ego_m'][0]=1000.
    result=query_world_state(state,'where is the car in front',state_context(context))
    assert not result['admitted'] and result['reason']=='historical_snapshot_modified'


@pytest.mark.parametrize('question',['how many cars are there','is it safe','are there any cars'])
def test_partial_history_cannot_answer_count_absence_or_safety(context,question):
    assert not query_world_state(fixture_state(),question,state_context(context))['admitted']


def test_unqualified_history_keeps_original_ids_and_full_visual(llm,carriers,context,monkeypatch):
    runtime=M3Runtime();ids=torch.tensor([1,-200,4]);captured=[]
    original=runtime.executor.generate
    def observe(model,expanded,*a,**kw):
        captured.append(expanded)
        return original(model,expanded,*a,**kw)
    monkeypatch.setattr(runtime.executor,'generate',observe)
    answers,d=runtime.generate(llm,[ids],carriers,context=state_context(context),
        questions=['where is the car in front'],verified_state=fixture_state(),max_new_tokens=2)
    row=d['requests'][0]
    assert d['runtime_entered'] and d['execution_completed']
    assert not row['historical_text_consumed']
    assert row['full_prefill'] and row['prefill_completed'] and row['decode_completed']
    assert row['state_retrieval']['reason']=='historical_context_quality_unverified'
    assert not row['state_retrieval']['admitted'] and row['current_visual_complete']
    assert captured[0].token_ids[0,0]==1 and captured[0].token_ids[0,-1]==4
    assert torch.equal(captured[0].embeddings[:,1:10],carriers)
    assert row['input_tokens']==11 and row['reused_tokens']==0


def test_state_augmentation_preserves_every_current_carrier(llm,carriers,context,monkeypatch):
    runtime=M3Runtime();captured=[]
    monkeypatch.setattr(runtime.scheduler,'state_allowed',lambda *a:(True,'unit_fixture'))
    original=runtime.executor.generate
    def observe(model,expanded,*a,**kw):
        captured.append(expanded)
        return original(model,expanded,*a,**kw)
    monkeypatch.setattr(runtime.executor,'generate',observe)
    class Tokenizer:
        def encode(self,text,**kw):return [7,8]
    _,d=runtime.generate(llm,[torch.tensor([1,-200,4])],carriers,context=state_context(context),
        questions=['where is the car in front'],verified_state=fixture_state(),tokenizer=Tokenizer(),max_new_tokens=2)
    assert d['requests'][0]['state_retrieval']['admitted']
    assert d['runtime_entered'] and d['execution_completed']
    assert d['requests'][0]['historical_text_consumed']
    assert d['requests'][0]['full_prefill'] and d['requests'][0]['decode_completed']
    assert torch.equal(captured[0].embeddings[:,1:10],carriers)
    assert captured[0].token_ids[0,-3:].tolist()==[7,8,4]
