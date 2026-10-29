from dataclasses import replace
import json
import pytest
import torch

from worldstream.m3.answer_memory import AnswerMemory
from worldstream.m3.answer_validity import propose_fact_answer
from worldstream.m3.task_conditioner import question_facts
from .test_world_guided_decode import state


Q='What type of vehicle is directly ahead?'
BINDING=('model','tokenizer','generation')


def context(frame='f0',timestamp=1.,version=1,index=0):
    return dict(scene_id='s',frame_id=frame,timestamp=timestamp,current_state_version=version,
                state_epoch=2,state_execution_signature='actual',frame_index=index,
                camera_ids=['front'],model_version='m')


def policy():
    return dict(status='FROZEN_RESEARCH_POLICY',min_confidence=.65,expected_normal_m3_ms=1000.,
                calibration_sha256='unit-test-only',policy_id='unit')


@pytest.mark.parametrize('q,intent',[
    (Q,'category'),('Is there a car in front?','existence'),
    ('Are there pedestrians on the left?','existence'),
    ('Where is the nearest pedestrian relative to the vehicle?','position'),
    ('Is the nearest car moving or stationary?','motion'),
    ('What color is the car ahead?','unsupported'),('How many cars are there?','unsupported'),
    ('Is it safe to turn?','unsupported'),('What types of vehicles are around?','unsupported'),
    ('Is there a car and a pedestrian in front?','unsupported'),
    ('What is the traffic light color?','unsupported'),('Describe the road ahead','unsupported'),
])
def test_question_required_fact_domain(q,intent):
    assert question_facts(q).intent==intent


def test_current_class_and_geometry_do_not_require_identity_or_full_history():
    world=state();world=replace(world,anchors=(replace(world.anchors[0],identity_qualified=False),))
    p=propose_fact_answer(Q,world,context(),AnswerMemory(),BINDING,policy())
    assert p['admitted'] and p['answer_source']=='fact_grounded_response'
    assert p['answer_text']=='A car is directly ahead.' and not p['source_generation']
    assert p['facts'][0]['evidence_origin']=='actual_perception'
    assert p['facts'][0]['identity_required'] is False


def test_no_observation_is_UNKNOWN_never_negative_answer():
    world=replace(state(),anchors=())
    p=propose_fact_answer('Is there a car in front?',world,context(),AnswerMemory(),BINDING,policy())
    assert not p['admitted'] and not p['supported'] and not p['answer_text']
    assert 'UNKNOWN_not_absent' in p['reason']


def test_rich_historical_answer_is_not_copied_but_core_question_can_be_answered():
    mem=AnswerMemory();mem.remember(Q,[1,3],[5,2],context(),BINDING,state(),'A white sedan is directly ahead.')
    p=propose_fact_answer(Q,state('f1',1.5,2),context('f1',1.5,2,1),mem,BINDING,policy())
    assert p['admitted'] and p['answer_source']=='fact_grounded_response'
    assert p['answer_text']=='A car is directly ahead.' and p['history_candidate_count']==1


def test_full_reuse_retains_original_generation_and_does_not_refresh_source():
    mem=AnswerMemory();mem.remember(Q,[1,3],[5,2],context(),BINDING,state(),'A car is directly ahead.')
    for i in (1,2,3):
        world=state('f'+str(i),1.+i*.5,i+1)
        p=propose_fact_answer(Q,world,context(world.frame_id,world.timestamp,world.version,i),mem,BINDING,policy())
        assert p['answer_source']=='full_answer_reuse' and p['source_generation']==('s','f0',1.)
        r=next(iter(mem.records.values()));mem.observe_validity(r,world.frame_id,True,p['support_score'])
        assert next(iter(mem.records.values())).frame_id=='f0'
    p=propose_fact_answer(Q,state('f10',6.,11),context('f10',6.,11,10),mem,BINDING,policy())
    assert p['answer_source']=='fact_grounded_response' and p['history_candidate_count']==0


@pytest.mark.parametrize('change',[
    dict(evidence_origin='oracle'),dict(source_frame='other'),dict(observed=False),dict(cameras=()),dict(score=.1),
])
def test_real_current_camera_supported_perception_only(change):
    world=state();world=replace(world,anchors=(replace(world.anchors[0],**change),))
    assert not propose_fact_answer(Q,world,context(),AnswerMemory(),BINDING,policy())['supported']


def test_nearest_target_is_chosen_before_confidence_filtering():
    world=state();a=replace(world.anchors[0],center_ego=(5.,0.,0.),score=.4)
    b=replace(a,anchor_id='b',semantic_class='truck',center_ego=(20.,0.,0.),score=.99)
    p=propose_fact_answer(Q,replace(world,anchors=(a,b)),context(),AnswerMemory(),BINDING,policy())
    assert p['supported'] and not p['admitted'] and p['facts'][0]['semantic_class']=='car'


def test_motion_uses_existing_reliable_velocity_and_abstains_when_missing():
    q='Is the nearest car moving or stationary?';world=state()
    assert propose_fact_answer(q,world,context(),AnswerMemory(),BINDING,policy())['answer_text'].endswith('stationary.')
    world=replace(world,anchors=(replace(world.anchors[0],velocity_ego=()),))
    assert not propose_fact_answer(q,world,context(),AnswerMemory(),BINDING,policy())['supported']


class Tokenizer:
    special_tokens_map={}
    def get_vocab(self):return {'a':3,'car':4}
    def encode(self,text,add_special_tokens=False):return [3,4,5]
    def decode(self,ids,skip_special_tokens=True):return 'A car is directly ahead.'


def test_runtime_online_fast_answer_has_zero_LLM_and_expansion(llm,carriers,tmp_path,monkeypatch):
    from worldstream.m3 import M3Runtime
    import worldstream.m3.runtime as module
    p=tmp_path/'policy.json';p.write_text(json.dumps(policy()))
    rt=M3Runtime(dict(reuse_strategy='exact_prefix',decode_strategy='world_guided',
        target_aware_exact_only=True,selective_answer_mode='online',selective_answer_policy=str(p)))
    def forbidden(*a,**k):raise AssertionError('Fast response executed language expansion/forward')
    monkeypatch.setattr(module,'expand_input',forbidden)
    monkeypatch.setattr(llm,'forward',forbidden)
    monkeypatch.setattr(rt.executor,'generate',forbidden)
    answers,d=rt.generate(llm,[torch.tensor([1,-200,4])],carriers,context=context(),questions=[Q],
        tokenizer=Tokenizer(),committed_world_state=state(),do_sample=False,max_new_tokens=10)
    r=d['requests'][0]
    assert r['llm_forward_calls']==0 and r['llm_skipped']
    assert answers[0].tolist()==[[1,3,4,5,2]] and not rt.answer_memory.records
    assert not r['prefill_completed'] and not r['decode_completed'] and r['layout_preparation_ms']==0.


def test_runtime_shadow_generates_and_miss_uses_same_target_aware_executor(llm,carriers):
    from worldstream.m3 import M3Runtime
    rt=M3Runtime(dict(reuse_strategy='exact_prefix',decode_strategy='world_guided',
        target_aware_exact_only=True,selective_answer_mode='shadow'))
    for i,q in enumerate([Q,'What color is the traffic light?']):
        world=state('f'+str(i),1.+i*.5,i+1)
        _,d=rt.generate(llm,[torch.tensor([1,-200,4])],carriers,
            context=context(world.frame_id,world.timestamp,world.version,i),questions=[q],
            tokenizer=Tokenizer(),committed_world_state=world,do_sample=False,max_new_tokens=3)
        r=d['requests'][0]
        assert not r['llm_skipped'] and r['llm_forward_calls']>0 and r['prefill_completed']
        assert r['fast_answer']['supported']==(i==0)


def test_quality_requires_all_question_facts_and_keeps_missing_GT_unknown():
    from tools.worldstream_m123.selective_answer_reuse_report import grade
    obj=dict(annotation_id='a',source_frame='f',semantic_class='pedestrian',center_ego=[10.,4.,0.],visible=True)
    q='Where is the nearest pedestrian relative to the vehicle?'
    assert grade(q,'The pedestrian is to the front-left.','f',[obj])['score']==1
    assert grade(q,'The pedestrian is in front.','f',[obj])['score']==0
    assert grade(q,'The pedestrian is on the right.','f',[obj])['severe_error']
    assert grade('Is there a car in front?','No cars.','f',[])['score'] is None


@pytest.mark.parametrize('location',['boston-seaport','singapore-queenstown'])
def test_introduced_question_tokens_equal_native_loader_city_context(location):
    from pathlib import Path
    from types import SimpleNamespace
    from transformers import AutoTokenizer
    from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import LoadAnnoatationVQATest
    from tools.worldstream_m123.selective_answer_reuse import encode_questions
    path=Path('ckpts/pretrain_qformer')
    if not (path/'tokenizer.model').exists():pytest.skip('Existing local tokenizer required; no model/download')
    tokenizer=AutoTokenizer.from_pretrained(str(path),model_max_length=2048,padding_side='right',use_fast=False,local_files_only=True)
    tokenizer.pad_token=tokenizer.unk_token
    loader=object.__new__(LoadAnnoatationVQATest);loader.tokenizer=tokenizer
    loader.preprocess_vqa=lambda results:[[dict(**{'from':'human'},value=Q),dict(**{'from':'gpt'},value='')]]
    native=loader(dict(location=location.split('-')[0]))['input_ids'][0]
    model=SimpleNamespace(tokenizer=tokenizer,lm_head=torch.nn.Linear(1,1))
    encoded=encode_questions(model,[Q],location)[0]
    assert torch.equal(encoded,native)


@pytest.mark.parametrize('category,expected',[
    ('vehicle.construction','construction_vehicle'),('movable_object.barrier','barrier'),
    ('movable_object.trafficcone','traffic_cone')])
def test_annotation_reader_uses_actual_nuscenes_category_names(tmp_path,category,expected):
    import pickle
    from tools.worldstream_m123.answer_validity_quality import load_current_annotations
    tables=tmp_path/'v1.0-trainval';tables.mkdir()
    for name,rows in {
        'category':[dict(token='c',name=category)],
        'instance':[dict(token='i',category_token='c')],
        'sample_annotation':[dict(token='a',sample_token='f',instance_token='i',translation=[1.,1.,10.],
                                  visibility_token='4',num_lidar_pts=2,num_radar_pts=0)],
    }.items():(tables/(name+'.json')).write_text(json.dumps(rows))
    cam=dict(ego2global_rotation=[1,0,0,0],ego2global_translation=[0,0,0],
             sensor2ego_rotation=[1,0,0,0],sensor2ego_translation=[0,0,0],
             cam_intrinsic=[[100,0,800],[0,100,450],[0,0,1]])
    info=dict(token='f',ego2global_rotation=[1,0,0,0],ego2global_translation=[0,0,0],cams={'front':cam})
    path=tmp_path/'infos.pkl';path.write_bytes(pickle.dumps(dict(infos=[info])))
    result=load_current_annotations(path,{'f'},tmp_path)['f']
    assert len(result)==1 and result[0]['semantic_class']==expected and result[0]['visible']


def test_formal_completion_requires_explicit_mode_and_current_fast_receipt():
    from types import SimpleNamespace
    from worldstream.m3.runtime import language_request_completed
    world=state();p=propose_fact_answer(Q,world,context(),AnswerMemory(),BINDING,policy())
    rt=SimpleNamespace(config=SimpleNamespace(selective_answer_mode='online'),fast_answer_policy=policy())
    row=dict(executor='QuestionReferencedFactReuse',fast_answer=p,llm_skipped=True,llm_forward_calls=0,
             layout_preparation_ms=0.,prefill_completed=False,decode_completed=False)
    assert language_request_completed(rt,row,world)
    assert not language_request_completed(rt,row,state('f1',1.5,2))
    assert not language_request_completed(rt,dict(row,llm_forward_calls=1),world)
    rt.config.selective_answer_mode='off'
    assert not language_request_completed(rt,row,world)
    assert language_request_completed(rt,dict(executor='PrefillExecutor',prefill_completed=True,decode_completed=True))
