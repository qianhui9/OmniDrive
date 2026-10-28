from dataclasses import replace
import pytest
import torch
from worldstream.m3.task_conditioner import question_dependencies
from worldstream.m3.world_state import CommittedAnchor,CommittedWorldState
from worldstream.m3.world_delta import question_world_delta
from worldstream.m3.answer_memory import AnswerMemory


def state(frame='f0',timestamp=1.,version=1,anchors=None,scene='s',epoch=2):
    a=CommittedAnchor('car1','object','car',(10.,1.,0.),(0.,0.,0.),True,('front',),
        True,True,frame,timestamp,str(version),'actual_perception',.9)
    return CommittedWorldState(scene,frame,timestamp,epoch,version,'actual',tuple([a] if anchors is None else anchors),
        (1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.))


@pytest.mark.parametrize('question,intent',[
    ('Are there any cars nearby?','existence'),('What category is the truck?','category'),
    ('How many vehicles are there?','count'),('Where is the pedestrian?','position'),
    ('How far is the car?','distance'),('Which direction is the bus moving?','motion'),
    ('How many lanes does the road have?','road'),('What color is the traffic light?','signal'),
    ('Is it safe to turn?','safety'),('Describe the weather conditions','environment'),('hello','unknown')])
def test_dependencies(question,intent):
    assert question_dependencies(question).intent==intent


def test_question_aware_changes_and_missing_evidence():
    previous=state();current=state('f1',1.5,2)
    q=question_dependencies('Where is the car?')
    assert question_world_delta(previous,current,q).suggested_draft_length==8
    moved=replace(current,anchors=(replace(current.anchors[0],center_ego=(-3.,-2.,0.)),))
    diff=question_world_delta(previous,moved,q)
    assert 'relative_direction' in diff.changed_attributes and diff.suggested_draft_length==4
    missing=replace(current,anchors=(replace(current.anchors[0],observed=False),))
    diff=question_world_delta(previous,missing,q)
    assert diff.suggested_draft_length==0 and 'departure_or_occlusion' in diff.unknown_dependencies
    assert question_world_delta(previous,replace(current,scene_id='other'),q).decision_reason=='scene_reset'
    uncertain=replace(current,anchors=(replace(current.anchors[0],identity_qualified=False),))
    assert question_world_delta(previous,uncertain,q).suggested_draft_length==0
    assert question_world_delta(previous,uncertain,question_dependencies('Are there any cars?')).suggested_draft_length==8


@pytest.mark.parametrize('q',['How many cars?','Is it safe to drive?','What is the traffic light color?',
                              'Describe the environment','What lane are we in?'])
def test_partial_world_never_becomes_complete_scene(q):
    diff=question_world_delta(state(),state('f1',1.5,2),question_dependencies(q))
    assert diff.suggested_draft_length==0 and diff.unknown_dependencies


def test_partial_category_is_short_proposal_not_stability_certificate():
    previous=state();current=state('f1',1.5,2)
    current=replace(current,anchors=(replace(current.anchors[0],identity_qualified=False),))
    delta=question_world_delta(previous,current,question_dependencies('What type of car is visible?'))
    assert delta.suggested_draft_length==4 and delta.draft_risk==.65
    assert 'identity_continuity' in delta.unknown_dependencies
    assert delta.decision_reason=='partial_class_observations_prefix_only'


def test_scene_description_is_not_ego_vehicle_classification():
    for q in ('What type of establishment is to the left of the vehicle?',
              'What type of business is the vehicle associated with?','Describe the area behind the vehicle'):
        assert question_dependencies(q).intent=='environment'


def test_semantic_retrieval_requires_world_and_positive_answer():
    memory=AnswerMemory();binding=('model','tok','config')
    ctx=dict(scene_id='s',frame_id='f0',frame_index=0,timestamp=1.,current_state_version=1)
    memory.remember('Are there any cars nearby?',[1,3],[7,8,9,10,11,12,2],ctx,binding,state(),'Yes, there are cars')
    now=dict(ctx,frame_id='f1',frame_index=1,timestamp=1.5,current_state_version=2)
    selection=memory.retrieve('Is there any car in the immediate vicinity?',[1,4],now,binding,state('f1',1.5,2),True)
    assert selection.match_type=='semantic' and selection.length==8 and not selection.exact_available
    assert selection.delta.decision_reason=='positive_observations_both_frames'
    assert memory.retrieve('Is there any car in the immediate vicinity?',[1,4],now,binding,None,True).record is None
    assert memory.retrieve('How many cars are nearby?',[1,4],now,binding,state('f1',1.5,2),True).record is None
    memory.observe_acceptance(selection,dict(verification_log=[dict(accepted=0)]),1)
    assert memory.retrieve('Is there any car in the immediate vicinity?',[1,4],now,binding,state('f1',1.5,2),True).record is None


def test_fast_runtime_no_legacy_prefix_planner_or_cache(llm,carriers,context):
    from worldstream.m3 import M3Runtime
    runtime=M3Runtime(dict(reuse_strategy='exact_prefix',decode_strategy='history_draft'))
    def forbidden(*args,**kw):raise AssertionError('legacy prefill reuse entered')
    runtime.planner.plan=forbidden;runtime.cache.put=forbidden
    ids=[torch.tensor([1,-200,4])]
    for i in range(2):
        _,d=runtime.generate(llm,ids,carriers,context=dict(context,frame_id=str(i),timestamp=1.+i*.5),
            questions=['Any cars?'],max_new_tokens=10,do_sample=False)
        assert not d['prefix_planner_executed'] and runtime.cache.entry is None
        assert d['requests'][0]['model_metadata_cache_hit']==bool(i)
    with torch.no_grad():llm.model.embed_tokens.weight.add_(.01)
    _,d=runtime.generate(llm,ids,carriers,context=dict(context,frame_id='2',timestamp=2.),max_new_tokens=3,do_sample=False)
    assert not d['requests'][0]['model_metadata_cache_hit'] and d['requests'][0]['draft_status']=='none'
