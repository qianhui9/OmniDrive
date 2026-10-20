"""Controlled contract fixtures are functionality tests, not deployment evidence."""
import copy
import json
from types import SimpleNamespace
import pytest
import torch
from worldstream.m3 import M3Runtime
from worldstream.m3.config import M3Config
from worldstream.m3.task_evidence import evaluate_task_evidence, population_task

QUESTION='What is the distance to observed object obj-1 in meters?'


@pytest.fixture
def evidence():
    context=dict(scene_id='s',frame_id='f',timestamp=1.,state_execution_signature='sig',source_version='v1',camera_ids=['front'])
    record=dict(referent_id='obj-1',scene_id='s',frame_id='f',timestamp=1.,execution_signature='sig',source_frame='f',source_version='v1',
        current_fields=dict(center_ego_m=[3.,4.,2.]),current_observation_available=True,contradiction_free=True,units='m',
        identity_evidence_origin='independent_current_observation',quality_certificate='controlled-fixture')
    certificate=dict(certificate_id='controlled-fixture',independent_quality_pass=True,evidence_origin='independent_task_validation',online_ground_truth_used=False,
        protocol_sha256='fixture-protocol',evidence_sha256='fixture-evidence',heldout_samples=1,calibration_scenes=['calibration'],heldout_scenes=['heldout'],
        execution_signature='sig',template='distance_observed_id',answer_format='distance_m_2dp',units='m',identity_evidence_independent=True)
    return context,record,certificate


def test_complete_metric_is_candidate_until_independent_certificate(evidence):
    c,r,cert=evidence
    d=evaluate_task_evidence(QUESTION,c,[r])
    assert d.evidence_complete and not d.quality_qualified and d.candidate=='5.00 m'
    assert evaluate_task_evidence(QUESTION,c,[r],cert).quality_qualified


@pytest.mark.parametrize('question',['Is it safe to drive?','How many cars are there?','Is there any car?','What is the distance to the car in front?',
    'Please provide the planning trajectory for the ego car without reasons.','What is the distance to observed object obj-1 in feet?',
    'What is the distance to observed object obj-1 in meters? and drive left'])
def test_unrestricted_semantics_never_admit(evidence,question):
    c,r,cert=evidence;assert not evaluate_task_evidence(question,c,[r],cert).evidence_complete


@pytest.mark.parametrize('changes,reason',[
    ({'current_observation_available':False},'no_current_observation'),
    ({'source_frame':'old'},'stale_source_or_signature'),({'source_version':'v0'},'stale_source_or_signature'),
    ({'timestamp':.9},'stale_source_or_signature'),({'execution_signature':'changed'},'stale_source_or_signature'),
    ({'current_fields':{}},'missing_or_nonfinite_fields'),({'current_fields':{'center_ego_m':[float('nan'),1,2]}},'missing_or_nonfinite_fields'),
    ({'contradiction_free':False},'current_history_contradiction'),({'units':'cm'},'unsupported_units'),
    ({'identity_evidence_origin':'current_head_history_agreement'},'no_independent_referent_identity')])
def test_evidence_hard_rejections(evidence,changes,reason):
    c,r,cert=evidence;r.update(changes);d=evaluate_task_evidence(QUESTION,c,[r],cert)
    assert not d.evidence_complete and d.reason==reason


def test_ambiguous_and_missing_referent(evidence):
    c,r,cert=evidence
    for rows in ([],[r,r]):assert not evaluate_task_evidence(QUESTION,c,rows,cert).evidence_complete


@pytest.mark.parametrize('changes',[{'online_ground_truth_used':True},{'evidence_origin':'actual_perception'},{'identity_evidence_independent':False},
    {'heldout_samples':0},{'calibration_scenes':['heldout']},{'units':'cm'},{'execution_signature':'other'}, {'answer_format':'free_text'}, {'evidence_sha256':''}])
def test_certificate_scope_is_not_a_boolean(evidence,changes):
    c,r,cert=evidence;cert.update(changes);assert not evaluate_task_evidence(QUESTION,c,[r],cert).quality_qualified


def test_position_schema_and_units(evidence):
    c,r,cert=evidence;cert.update(template='relative_position_observed_id',answer_format='ego_xyz_m_json_2dp')
    d=evaluate_task_evidence('What is the relative position of observed object obj-1 in meters?',c,[r],cert)
    assert d.quality_qualified and json.loads(d.candidate)==dict(x_m=3.,y_m=4.,z_m=2.)


def test_direct_config_requires_opt_in():
    assert M3Config().shadow_evaluation and not M3Config().task_evidence_enabled
    with pytest.raises(ValueError):M3Config(experimental_direct_query=True)


def configured(tmp_path,cert,**kw):
    path=tmp_path/'controlled_certificate.json';path.write_text(json.dumps(cert))
    return M3Runtime(dict(mode='exact',offline_experiment=True,min_prefix_tokens=1,task_evidence_enabled=True,
        direct_task_certificate=str(path),experimental_direct_query=True,**kw))


def test_shadow_executes_original_and_credits_zero(evidence,tmp_path,llm,carriers):
    c,r,cert=evidence;runtime=configured(tmp_path,cert,shadow_evaluation=True);calls=[]
    h=llm.register_forward_hook(lambda *a:calls.append(True))
    ids=torch.tensor([1,-200,3]);actual,d=runtime.generate(llm,[ids],carriers,context=c,questions=[QUESTION],task_evidence=[r],max_new_tokens=2,
        tokenizer=SimpleNamespace(encode=lambda *a,**k:[5,6],decode=lambda *a,**k:"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: \nYou are driving in boston. "+QUESTION+" ASSISTANT:"))
    h.remove();assert calls and d['actual_direct_requests']==0 and d['requests'][0]['candidate']=='5.00 m'
    assert d['requests'][0]['actual_avoided_decode_requests']==0


def test_direct_skips_all_multimodal_and_transformer_calls(evidence,tmp_path,llm,carriers,monkeypatch):
    c,r,cert=evidence;runtime=configured(tmp_path,cert,shadow_evaluation=False)
    def forbidden(*a,**k):raise AssertionError('Direct route invoked original model')
    monkeypatch.setattr(llm,'generate',forbidden);monkeypatch.setattr(llm,'forward',forbidden)
    monkeypatch.setattr(llm,'prepare_inputs_labels_for_multimodal',forbidden)
    out,d=runtime.generate(llm,[torch.tensor([1,-200,3])],carriers,context=c,questions=[QUESTION],task_evidence=[r],
        tokenizer=SimpleNamespace(encode=lambda *a,**k:[5,6],decode=lambda *a,**k:"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: \nYou are driving in boston. "+QUESTION+" ASSISTANT:"))
    assert out[0].tolist()==[[5,6]] and d['actual_direct_requests']==1 and runtime.cache.entry is None


def test_rejected_direct_can_still_use_exact_prefix(llm,carriers,context):
    runtime=M3Runtime(dict(mode='exact',offline_experiment=True,min_prefix_tokens=1,task_evidence_enabled=True,
        shadow_evaluation=False,experimental_direct_query=True,direct_query_prefix_fallback=True))
    ids=torch.tensor([1,-200,3]);out,d=runtime.generate(llm,[ids,ids],carriers,context=context,questions=['Is it safe?']*2,max_new_tokens=2)
    assert d['actual_direct_requests']==0 and d['actual_visual_requests']==2
    assert d['visual_fallback']['requests'][1]['reused_tokens']>0
    assert torch.equal(out[0],out[1])


def test_shadow_never_consumes_historical_head_truth(llm,carriers,context):
    runtime=M3Runtime(dict(task_evidence_enabled=True))
    state=SimpleNamespace(records=({'anchor_id':'obj-1','center_ego_m':[3,4,2]},))
    _,d=runtime.generate(llm,[torch.tensor([1,-200,3])],carriers,context=context,questions=[QUESTION],verified_state=state,max_new_tokens=2)
    assert not d['requests'][0]['evidence_complete']


def test_population_labels_do_not_authorize():
    assert population_task('Please provide the planning trajectory for the ego car without reasons.')=='planning'
    assert population_task('What is the distance of the car?')=='local_metric'
    assert not evaluate_task_evidence('What is the distance of the car?',{}).evidence_complete


def test_interrupted_generation_clears_transaction_state(llm,carriers,context,monkeypatch):
    runtime=M3Runtime(dict(mode='exact',offline_experiment=True,min_prefix_tokens=1))
    ids=torch.tensor([1,-200,3]);runtime.generate(llm,[ids,ids],carriers,context=context,max_new_tokens=2)
    assert runtime.cache.entry is not None
    def oom(*a,**k):raise torch.cuda.OutOfMemoryError('injected controlled test')
    monkeypatch.setattr(llm,'forward',oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):runtime.generate(llm,[ids,ids],carriers,context=context,max_new_tokens=2)
    assert runtime.cache.entry is None and runtime._scope is None


def test_direct_rich_generation_schema_falls_back(evidence,tmp_path,llm,carriers):
    c,r,cert=evidence;runtime=configured(tmp_path,cert,shadow_evaluation=False)
    out,d=runtime.generate(llm,[torch.tensor([1,-200,3])],carriers,context=c,questions=[QUESTION],task_evidence=[r],
        tokenizer=SimpleNamespace(encode=lambda *a,**k:[5,6],decode=lambda *a,**k:"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: \nYou are driving in boston. "+QUESTION+" ASSISTANT:"),return_dict_in_generate=True,max_new_tokens=2)
    assert d['actual_direct_requests']==0 and hasattr(out[0],'sequences')


def test_malformed_current_evidence_is_unknown(evidence):
    c,r,cert=evidence;c['timestamp']=None;r['current_fields']=None
    assert not evaluate_task_evidence(QUESTION,c,[r],cert).evidence_complete


def test_prefill_decode_classification_preserves_one_token_suffix():
    from tools.worldstream_m3.task_serving_profile import stage
    assert stage((),dict(inputs_embeds=torch.zeros(1,1,2),past_key_values=('cached',)))=='prefill'
    assert stage((),dict(input_ids=torch.ones(1,1),past_key_values=('cached',)))=='decode'


def test_prompt_metadata_cannot_replace_planning_query():
    from worldstream.m3.task_evidence import native_prompt_matches
    tokenizer=SimpleNamespace(decode=lambda *a,**k:'USER: Please provide the planning trajectory. ASSISTANT:')
    assert not native_prompt_matches(tokenizer,torch.tensor([1,-200,3]),QUESTION)


def test_direct_respects_generation_token_budget(evidence,tmp_path,llm,carriers):
    c,r,cert=evidence;runtime=configured(tmp_path,cert,shadow_evaluation=False)
    tokenizer=SimpleNamespace(encode=lambda *a,**k:[5,6],decode=lambda *a,**k:"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: \nYou are driving in boston. "+QUESTION+" ASSISTANT:")
    _,d=runtime.generate(llm,[torch.tensor([1,-200,3])],carriers,context=c,questions=[QUESTION],task_evidence=[r],tokenizer=tokenizer,max_new_tokens=1)
    assert d['actual_direct_requests']==0 and d['requests'][0]['fallback_reason']=='direct_answer_exceeds_generation_budget'
