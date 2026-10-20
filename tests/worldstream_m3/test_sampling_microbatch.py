from dataclasses import replace
import copy
import pytest
import torch
from worldstream.m3 import M3Runtime
from worldstream.m3.config import M3Config
from worldstream.m3.sampling_microbatch import SamplingMicrobatchExecutor,request_seed,sampling_tools
from worldstream.m3.question_dependencies import QuestionDependency
from worldstream.m3.visual_state import base_model


def inputs():return [torch.tensor([1,3,-200,4+i]) for i in range(3)]
def context():return dict(scene_id='s',frame_id='f',camera_ids=list('abcdef'))


@pytest.fixture
def independent(monkeypatch):
    monkeypatch.setattr('worldstream.m3.sampling_microbatch.analyze_questions',
        lambda t,ids,q,**kw:[QuestionDependency(i,independent=True) for i in range(len(ids))])


def test_default_off_and_requires_explicit_offline_full_mode():
    assert not M3Config().experimental_rng_isolated_microbatch
    for change in ({},{'offline_experiment':True},{'offline_experiment':True,'mode':'full','batched_decode':True}):
        with pytest.raises(ValueError):M3Config(experimental_rng_isolated_microbatch=True,**change)
    assert M3Config(experimental_rng_isolated_microbatch=True,offline_experiment=True,mode='full')


@torch.no_grad()
def test_request_eos_compaction_order_positions_and_rng_independence(llm,carriers,independent,monkeypatch):
    config=copy.deepcopy(base_model(llm).generation_config);config.max_new_tokens=8
    initial=[0]
    def probs(history,logits):
        if history.shape[1]==1:
            token=[2,3,4][initial[0]];initial[0]+=1
        else:token=2 if int(history[0,-1])==3 or history.shape[1]>=4 else 4
        p=torch.zeros_like(logits);p[:,token]=1;return p
    monkeypatch.setattr('worldstream.m3.sampling_microbatch.sampling_tools',lambda b,g:(config,probs))
    rng=torch.get_rng_state().clone()
    out,diag=SamplingMicrobatchExecutor(4,123).try_generate(llm,inputs(),carriers,['q']*3,context(),object())
    assert [a.tolist()[0] for a in out]==[[1,2],[1,3,2],[1,4,4,4,2]]
    assert [s['batch'] for s in diag['steps']]==[2,1,1]
    assert [r['request_id'] for r in diag['requests']]==[0,1,2]
    assert all(r['stop_reason']=='eos' for r in diag['requests'])
    assert torch.equal(rng,torch.get_rng_state())
    assert not hasattr(base_model(llm),'_worldstream_sampling_microbatch_owner')
    assert diag['persistent_prefix_cache_accessed'] is False


@torch.no_grad()
def test_private_generators_do_not_depend_on_group_size_or_scene_order(llm,carriers,independent,monkeypatch):
    config=copy.deepcopy(base_model(llm).generation_config);config.max_new_tokens=12
    def probs(ids,logits):
        p=torch.zeros_like(logits);p[:,3]=.3;p[:,4]=.7;return p
    monkeypatch.setattr('worldstream.m3.sampling_microbatch.sampling_tools',lambda b,g:(config,probs))
    results=[]
    for size in (2,4):
        out,diag=SamplingMicrobatchExecutor(size,11).try_generate(llm,inputs(),carriers,['q']*3,context(),object())
        results.append(([t.tolist() for t in out],[r['final_rng_sha256'] for r in diag['requests']]))
    assert results[0]==results[1]
    assert request_seed(11,context(),0)!=request_seed(11,dict(context(),scene_id='other'),0)
    assert request_seed(11,context(),0)!=request_seed(11,context(),1)


@torch.no_grad()
@pytest.mark.parametrize('exception',[RuntimeError('injected'),torch.cuda.OutOfMemoryError('injected oom')])
def test_failed_attempt_native_fallback_restores_rng_and_no_partial_answers(llm,carriers,independent,exception):
    generation=dict(do_sample=True,temperature=.1,top_p=.75,num_beams=1,max_new_tokens=4,use_cache=True)
    ids=inputs()
    torch.manual_seed(113)
    reference=[llm.generate(inputs=i[None],images=carriers,**generation) for i in ids]
    reference_rng=torch.get_rng_state().clone()
    model=base_model(llm);original=model.forward;count=[0]
    from functools import wraps
    @wraps(original)
    def fail(*a,**kw):
        if kw.get('input_ids') is not None and kw['input_ids'].shape[0]>1 and count[0]==0:
            count[0]+=1;raise exception
        return original(*a,**kw)
    model.forward=fail
    runtime=M3Runtime(dict(mode='full',offline_experiment=True,experimental_rng_isolated_microbatch=True))
    try:
        torch.manual_seed(113)
        out,diag=runtime.generate(llm,ids,carriers,context=context(),questions=['q']*3,tokenizer=object(),**generation)
    finally:model.forward=original
    assert count[0]==1
    assert diag['reason']=='sampling_microbatch_native_fallback'
    assert all(torch.equal(a,b) for a,b in zip(out,reference))
    assert torch.equal(reference_rng,torch.get_rng_state())
    assert runtime.cache.entry is None
    assert not hasattr(model,'_worldstream_sampling_microbatch_owner')


@torch.no_grad()
def test_unsupported_config_missing_source_and_one_question_fall_back(llm,carriers,independent):
    executor=SamplingMicrobatchExecutor()
    kw=dict(questions=['q']*3,context=context(),tokenizer=object())
    out,diag=executor.try_generate(llm,inputs(),carriers,do_sample=False,**kw)
    assert out is None and diag['fallback_reason']=='unsupported_generation_contract'
    out,diag=executor.try_generate(llm,inputs(),carriers,do_sample=True,stopping_criteria=object(),**kw)
    assert out is None
    out,diag=executor.try_generate(llm,inputs(),carriers,do_sample=True,max_new_tokens=4,max_time=1.,**kw)
    assert out is None and diag['fallback_reason']=='unsupported_generation_contract'
    out,diag=executor.try_generate(llm,inputs(),carriers,questions=['q']*3,context={},tokenizer=object())
    assert out is None and diag['fallback_reason']=='missing_six_camera_source_contract'


@torch.no_grad()
def test_pack_padding_positions_match_individual_native_forward(llm):
    from worldstream.m3.microbatch_kv import pack_independent_kv
    base=base_model(llm)
    states=[base(input_ids=torch.tensor([[1,3]]),use_cache=True).past_key_values,
            base(input_ids=torch.tensor([[1,3,4,5]]),use_cache=True).past_key_values]
    kv,mask,pos=pack_independent_kv(states)
    outputs=base(input_ids=torch.tensor([[6],[7]]),past_key_values=kv,
        attention_mask=torch.cat((mask,torch.ones(2,1,dtype=torch.long)),1),position_ids=pos,use_cache=True)
    for i,token in enumerate((6,7)):
        n=states[i][0][0].shape[-2]
        native=base(input_ids=torch.tensor([[token]]),past_key_values=states[i],attention_mask=torch.ones(1,n+1,dtype=torch.long),
            position_ids=torch.tensor([[n]]),use_cache=True)
        # Structural test in FP32 tiny Llama, not the real FP16 certificate.
        assert torch.allclose(outputs.logits[i],native.logits[0],atol=1e-6,rtol=1e-5)


@torch.no_grad()
def test_scene_frame_and_model_changes_always_recompute_request_prefill(llm,carriers,independent,monkeypatch):
    config=copy.deepcopy(base_model(llm).generation_config);config.max_new_tokens=4
    def eos(ids,logits):
        p=torch.zeros_like(logits);p[:,2]=1;return p
    monkeypatch.setattr('worldstream.m3.sampling_microbatch.sampling_tools',lambda b,g:(config,eos))
    executor=SamplingMicrobatchExecutor(2,11);base=base_model(llm);prefills=[]
    h=base.register_forward_pre_hook(lambda m,args:prefills.append(True))
    try:
        a,da=executor.try_generate(llm,inputs(),carriers,['q']*3,context(),object())
        with torch.no_grad():base.get_input_embeddings().weight.add_(.001)
        b,db=executor.try_generate(llm,inputs(),carriers,['q']*3,dict(context(),scene_id='new',frame_id='next'),object())
    finally:h.remove()
    assert len(prefills)==6
    assert all(q['source'][0]=='new' and q['source'][1]=='next' for q in db['requests'])
    assert [q['seed'] for q in da['requests']]!=[q['seed'] for q in db['requests']]
    assert not any('past_key' in name for name in vars(executor))
