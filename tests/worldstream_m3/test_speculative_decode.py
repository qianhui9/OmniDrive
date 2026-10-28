"""Real tiny Llama forwards: no mocked target, KV or timing."""
import copy
import pytest
import torch
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
from worldstream.m3 import expand_input
from worldstream.m3.prefill_executor import PrefillExecutor, deterministic_draft_sample
from worldstream.m3.prefix_kv_cache import PrefixKVCache


def executor():
    return PrefillExecutor(PrefixKVCache(1024**2))


@pytest.mark.parametrize('kind', ['correct','first','middle','wrong','eos','repeat','negative','overflow','empty'])
@pytest.mark.parametrize('k', [1,4,8])
def test_greedy_all_drafts(llm, carriers, kind, k):
    ex = expand_input(llm, torch.tensor([[1,3,-200,4]]), carriers)
    kw = dict(max_new_tokens=17,do_sample=False,eos_token_id=None)
    expected, _ = executor().generate(llm, ex, **kw)
    draft = expected[0,1:].tolist()
    if kind == 'first': draft[0] = (draft[0]+1)%64
    if kind == 'middle': draft[6] = (draft[6]+1)%64
    if kind == 'wrong': draft = [(v+1)%64 for v in draft]
    if kind == 'eos': draft[0] = 2
    if kind == 'repeat': draft = [7]*30
    if kind == 'negative': draft[2] = -1
    if kind == 'overflow': draft[2] = 64
    if kind == 'empty': draft = []
    actual, diag = executor().generate(llm, ex, draft_tokens=draft, draft_length=k, **kw)
    assert torch.equal(expected, actual)
    if kind == 'correct' and k>1:
        assert diag['decode_forward_calls'] < 16
        assert diag['draft_accepted_tokens'] > 0
    if kind in ('negative','overflow'):
        assert diag['draft_status'] == 'invalid_tokens_native'


@pytest.mark.parametrize('limit',[1,2,4,8,9])
def test_eos_length_stopping_scores_config(llm, carriers, limit):
    ex = expand_input(llm,torch.tensor([[1,-200,4]]),carriers)
    e = executor()
    full,_=e.generate(llm,ex,max_new_tokens=12,do_sample=False,eos_token_id=None)
    cfg = copy.deepcopy(llm.generation_config)
    cfg.do_sample=False; cfg.max_new_tokens=limit; cfg.eos_token_id=full[0,3].item()
    expected,_=executor().generate(llm,ex,generation_config=cfg,return_dict_in_generate=True,output_scores=True)
    actual,_=e.generate(llm,ex,draft_tokens=full[0,1:].tolist(),draft_length=8,
        generation_config=cfg,return_dict_in_generate=True,output_scores=True)
    assert torch.equal(expected.sequences,actual.sequences)
    for a,b in zip(expected.scores,actual.scores):
        torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)


def test_custom_processor_stopping_and_owned_cache(llm,carriers):
    ex=expand_input(llm,torch.tensor([[1,-200,4]]),carriers)
    class Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs): return input_ids.shape[-1]>=7
    kw=dict(max_new_tokens=20,do_sample=False,eos_token_id=None,repetition_penalty=1.2,
        no_repeat_ngram_size=2,stopping_criteria=StoppingCriteriaList([Stop()]))
    e=executor(); expected,_=e.generate(llm,ex,**kw)
    stored=tuple(tuple(t.clone() for t in layer) for layer in e.cache.entry.kv)
    actual,diag=e.generate(llm,ex,reuse_length=ex.length,draft_tokens=expected[0,1:].tolist(),**kw)
    assert torch.equal(actual,expected)
    assert all(torch.equal(a,b) for x,y in zip(stored,e.cache.entry.kv) for a,b in zip(x,y))
    assert diag['forward_prefill_lengths']==[]


@pytest.mark.parametrize('mask',[[0,1,1],[1,0,1],[1,1,0]])
def test_positions_after_rejection(llm,mask):
    ex=expand_input(llm,torch.tensor([[1,3,4]]),None,attention_mask=torch.tensor([mask]))
    expected,_=executor().generate(llm,ex,max_new_tokens=14,do_sample=False,eos_token_id=None)
    draft=expected[0,1:].tolist();draft[3]=(draft[3]+1)%64
    actual,diag=executor().generate(llm,ex,max_new_tokens=14,do_sample=False,
        eos_token_id=None,draft_tokens=draft,draft_length=8)
    assert torch.equal(actual,expected)
    assert diag['verification_log'][0]['rejection_position']==3


@pytest.mark.parametrize('probability',[0.,.01,.25,.9,1.])
def test_sampling_transition_law(probability):
    p=torch.tensor([[probability,(1-probability)*.3,(1-probability)*.7]],dtype=torch.float64)
    g=torch.Generator().manual_seed(410)
    counts=torch.zeros(3)
    accepted=0
    for _ in range(12000):
        token,ok=deterministic_draft_sample(p,0,generator=g)
        counts[token]+=1;accepted+=ok
        if not ok:assert token != 0
    assert torch.max(torch.abs(counts/12000-p[0]))<.018
    assert abs(accepted/12000-probability)<.018


def test_sampling_warpers_processors_and_eos(llm,carriers):
    ex=expand_input(llm,torch.tensor([[1,-200,4]]),carriers)
    kw=dict(max_new_tokens=12,do_sample=True,temperature=.1,top_p=.75,top_k=50,
        repetition_penalty=1.1,eos_token_id=None)
    # Exercise actual model distributions and every rejection path. Distribution
    # correctness is independently tested above; seed identity is not required.
    for seed in range(12):
        torch.manual_seed(seed)
        actual,diag=executor().generate(llm,ex,draft_tokens=[7,8,9,10]*4,
            experimental_sampling=True,**kw)
        assert actual.shape==(1,13)
        assert diag['verification_forward_calls']>=1
    with pytest.raises(ValueError,match='experimental_sampling'):
        executor().generate(llm,ex,draft_tokens=[7],**kw)


def test_empty_draft_same_rng_native(llm,carriers):
    ex=expand_input(llm,torch.tensor([[1,-200,4]]),carriers)
    kw=dict(max_new_tokens=12,do_sample=True,temperature=.1,top_p=.75)
    torch.manual_seed(4); a,_=executor().generate(llm,ex,**kw)
    torch.manual_seed(4); b,_=executor().generate(llm,ex,draft_tokens=[],**kw)
    assert torch.equal(a,b)


@pytest.mark.parametrize('change',['scene','age','model','tokenizer','generation','same_frame','time','world_epoch'])
def test_answer_memory_invalidation(change):
    from worldstream.m3.answer_memory import AnswerMemory
    memory=AnswerMemory()
    ctx=dict(scene_id='s',frame_id='f0',frame_index=0,timestamp=0.,state_version=1)
    binding=('model0','tok0','gen0','epoch0')
    memory.remember('question',[1,-200,3],[7,8,2],ctx,binding)
    now=dict(ctx,frame_id='f1',frame_index=1,timestamp=.5,state_version=2)
    assert memory.select([1,-200,3],now,binding).tokens==(7,8,2)
    assert memory.select([1,-200,4],now,binding) is None
    if change=='scene':now['scene_id']='other'
    if change=='age':now['frame_index']=10
    if change=='same_frame':now['frame_id']='f0'
    if change=='time':now['timestamp']=10.
    if change in ('model','tokenizer','generation','world_epoch'):
        i={'model':0,'tokenizer':1,'generation':2,'world_epoch':3}[change]
        binding=tuple('changed' if j==i else v for j,v in enumerate(binding))
    assert memory.select([1,-200,3],now,binding) is None


def test_runtime_history_multiquestion(llm,carriers,context):
    from worldstream.m3 import M3Runtime
    runtime=M3Runtime(dict(reuse_strategy='exact_prefix',decode_strategy='history_draft'))
    questions=[torch.tensor([1,-200,4]),torch.tensor([1,-200,5])]
    kw=dict(max_new_tokens=12,do_sample=False,eos_token_id=None)
    first,diag=runtime.generate(llm,questions,carriers,context=context,questions=['q1','q2'],**kw)
    assert all(r['draft_status']=='none' for r in diag['requests'])
    changed=carriers+.001
    now=dict(context,frame_id='f2',timestamp=1.5)
    actual,diag=runtime.generate(llm,questions,changed,context=now,questions=['q1','q2'],**kw)
    expected=[llm.generate(inputs=q[None],images=changed,**kw) for q in questions]
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    assert all(r['draft_status']=='target_verification' and r['reused_tokens']==0 for r in diag['requests'])
    assert all(r['draft_source_frame']==context['frame_id'] for r in diag['requests'])
    _,diag=runtime.generate(llm,questions,changed,context=dict(now,scene_id='other'),**kw)
    assert all(r['draft_status']=='none' for r in diag['requests'])


def test_generation_config_sampling_optin(llm,carriers):
    ex=expand_input(llm,torch.tensor([[1,-200,4]]),carriers)
    config=copy.deepcopy(llm.generation_config);config.do_sample=True;config.max_new_tokens=4
    with pytest.raises(ValueError,match='experimental_sampling'):
        executor().generate(llm,ex,draft_tokens=[7],generation_config=config)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real CUDA cache/forward lifecycle')
def test_cuda_forward_and_cache_lifecycle(llm,carriers):
    llm=llm.cuda();carriers=carriers.cuda()
    ex=expand_input(llm,torch.tensor([[1,-200,4]],device='cuda'),carriers)
    e=executor();kw=dict(max_new_tokens=17,do_sample=False,eos_token_id=None,profile=True)
    native,_=e.generate(llm,ex,**kw)
    stored=tuple(tuple(t.clone() for t in layer) for layer in e.cache.entry.kv)
    nbytes=e.cache.nbytes
    for bad in (None,0,5):
        draft=native[0,1:].tolist()
        if bad is not None:draft[bad]=(draft[bad]+1)%64
        actual,diag=e.generate(llm,ex,reuse_length=ex.length,draft_tokens=draft,**kw)
        assert torch.equal(native,actual)
        assert diag['decode_forward_cuda_ms']>0 and diag['verification_forward_cuda_ms']>0
        assert diag['verification_forward_calls']>=1
        assert e.cache.nbytes==nbytes
        assert all(torch.equal(a,b) for x,y in zip(stored,e.cache.entry.kv) for a,b in zip(x,y))
    e.cache.reset();assert e.cache.entry is None and e.cache.nbytes==0
