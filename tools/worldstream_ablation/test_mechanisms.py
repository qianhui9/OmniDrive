"""Mechanism checks use small real networks; GPU smoke is a separate command."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from .registry import VARIANTS, GROUPS, select
from .vision import ExperimentalVision, source_bundle, budget_selection
from .state import AlternativeState, flow_support
from .language import PromptKV
from .report import target_signature, grade_answer, summarize_events


def test_matrix_and_shared_full_reference():
    assert len(VARIANTS)==14
    assert len(select())==14
    assert all('worldstream' in names for names in GROUPS.values())
    assert VARIANTS['without_ogsm'].davp=='image_difference'
    assert VARIANTS['without_ogsm'].qcse=='candidate_only'


@pytest.mark.parametrize('policy',['periodic','feature_drift'])
def test_per_camera_budget_and_hard_age_with_adversarial_drift(policy):
    ages = [0]*9
    for frame in range(50):
        scores = ([10. if j<3 else 0. for j in range(9)] if policy=='feature_drift' else
                  [float(j//3==frame%3) for j in range(9)])
        ages = [a+1 for a in ages]
        selected = budget_selection(scores,ages,3,8)
        assert len(selected)==len(set(selected))==3
        for j in selected:
            ages[j]=0
        assert max(ages)<8


def expanded(embeddings,visual_start=3):
    from worldstream.m3.visual_state import ExpandedInput
    n = embeddings.shape[1]
    visual = torch.zeros((1,n),dtype=torch.bool)
    visual[:,visual_start:visual_start+2] = True
    return ExpandedInput(embeddings,torch.arange(n)[None],torch.arange(n)[None],torch.ones((1,n),dtype=torch.long),visual)


@torch.no_grad()
def test_prompt_kv_reuses_only_causal_text_and_decodes_current_visual_context():
    from transformers import LlamaConfig,LlamaForCausalLM
    from worldstream.m3.prefill_executor import PrefillExecutor
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=41,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,max_position_embeddings=128,
        bos_token_id=1,eos_token_id=None,pad_token_id=0)).eval()
    cache = PromptKV()
    x = torch.randn(1,9,16)
    generation = dict(max_new_tokens=4,do_sample=False,use_cache=True)
    _,cold = cache.generate(model,expanded(x),'query',**generation)
    assert cold['reused_tokens']==0
    y = x.clone()
    y[:,3:] = torch.randn_like(y[:,3:])
    cached,hit = cache.generate(model,expanded(y),'query',**generation)
    fresh,_ = PrefillExecutor().generate(model,expanded(y),**generation)
    assert hit['reused_tokens']==3
    assert hit['decode_forward_calls']==3
    assert hit['effective_prefill_tokens']==6
    assert torch.equal(cached,fresh)
    assert all(k.shape[-2]==3 for layer in cache.cache['query']['kv'] for k in layer)
    y[:,0] += 1
    _,invalid = cache.generate(model,expanded(y),'query',**generation)
    assert invalid['reused_tokens']==0


@pytest.fixture
def frame():
    from worldstream.m1.state_types import Frame,CameraFrame
    rng = np.random.RandomState(9)
    img = rng.randint(0,255,(40,40,3),dtype=np.uint8)
    K = np.array([[30.,0,20],[0,30.,20],[0,0,1]])
    cam = CameraFrame('front',img,K,np.eye(4),1.,'preprocessed_image',
        np.eye(3),True,raw_intrinsic=K.copy())
    return Frame('scene','0',1.,np.eye(4),{'front':cam},'test')


@pytest.mark.parametrize('policy',['tracking','geometry','flow'])
def test_state_baselines_keep_provenance_and_expire(frame,policy):
    from worldstream.m1 import M1Config
    from worldstream.m1.geometry import box_geometry
    from worldstream.m1.state_types import PerceptionAnchor,FullOutputs
    cfg = replace(M1Config(),min_region_area_px=1,min_coverage=.1)
    manager = AlternativeState(cfg,policy)
    result = manager.precheck_with_observation(frame)
    pred = PerceptionAnchor('object','car',box_geometry([2.,0.,10.],[2.,2.,2.]),source_score=.9)
    manager.commit(frame,FullOutputs([pred]),result)
    assert len(manager.records)==1
    assert manager.snapshot().anchors[0].source_frame=='0'
    frame2 = replace(frame,frame_id='1',timestamp=1.5,
        cameras={k:replace(c,timestamp=1.5) for k,c in frame.cameras.items()})
    result = manager.precheck_with_observation(frame2)
    manager.commit(frame2,FullOutputs([]),result)
    snapshot = manager.snapshot()
    assert len(snapshot.anchors)==1
    assert snapshot.anchors[0].source_frame=='0'
    assert snapshot.anchors[0].source_timestamp==1.
    assert snapshot.anchors[0].evidence_origin=='experimental_'+policy+'_validation'
    if policy=='flow':
        assert all('anchor' not in r for r in manager.records.values())
    frame3 = replace(frame2,frame_id='2',timestamp=3.)
    assert not manager.precheck_with_observation(frame3).valid_anchor_ids
    assert not manager.records


def test_bidirectional_flow_checks_local_appearance(frame):
    stable,_ = flow_support(frame.cameras['front'].image,frame.cameras['front'].image)
    assert stable[5:-5,5:-5].mean()>.9
    changed = frame.cameras['front'].image.copy()
    changed[10:30,10:30] = 0
    supported,_ = flow_support(frame.cameras['front'].image,changed)
    assert supported[15:25,15:25].mean()<.3


@torch.no_grad()
@pytest.mark.parametrize('policy',['periodic','feature_drift','image_difference','region_mapping','full'])
def test_real_window_executor_runs_multiple_frames(policy):
    from projects.mmdet3d_plugin.models.backbones.eva_vit import EVAViT
    from worldstream.m2 import M2Config
    torch.manual_seed(19)
    eva = EVAViT(img_size=40,patch_size=4,embed_dim=32,depth=4,num_heads=4,
        window_size=4,window_block_indexes=[0,1,3],pt_hw_seq_len=4,pretrain_img_size=40,
        flash_attn=False,with_cp=False).eval()
    rt = ExperimentalVision(eva,M2Config(),policy)
    x = torch.randn(2,3,40,40)
    context = dict(scene_id='scene',frame_id='0',timestamp=1.,camera_ids=['front','back'],
        state_epoch=1,execution_signature='test',coordinate_space='preprocessed_image')
    class Manager:
        def export_m2_inputs(self,expected_context=None,computation_only=False):
            return SimpleNamespace(computation_state=source_bundle(rt,x,expected_context,
                masks=np.zeros((2,40,40),bool),horizon=10.))
    manager = Manager()
    for i in range(10):
        context.update(frame_id=str(i),timestamp=1.+i*.1)
        output,diag = rt.forward_from_m1(x,manager,context)
        assert output[0].shape==(2,32,10,10)
        assert diag['execution_completed']
        if policy in ('periodic','feature_drift') and i:
            assert all(c['executed_windows']==3 for c in diag['stages'][0]['per_camera'])
        assert max(a.max().item() for a in rt.cache.window_ages.values())<8


def test_absence_and_unknown_have_explicit_denominators():
    from tools.worldstream_m123.selective_answer_reuse import QUESTIONS
    q = QUESTIONS[1]
    assert target_signature(q,[])==('existence','absent')
    assert grade_answer(q,'No, there is no vehicle directly ahead.','t',{'t':[]})['score']==1
    assert grade_answer(q,'Yes, there is a car directly ahead.','t',{'t':[]})['score']==0
    assert grade_answer(q,'I cannot tell if there is a vehicle.','t',{'t':[]})['score']==0
    category = QUESTIONS[0]
    assert grade_answer(category,'No pedestrians, a car is directly ahead.','t',{'t':[]})['score']==0
    assert grade_answer(category,'No vehicle is ahead, but a car is present.','t',{'t':[]})['score']==0
    assert grade_answer(category,'None.','t',{'t':[]})['score']==1
    assert grade_answer(q,'There is no car in front, but there is a truck.','t',{'t':[]})['score']==1
    summary = summarize_events([])
    assert summary['Factual_QA_Accuracy_percent'] is None
    assert summary['Stale_Answer_Rate_percent'] is None


@torch.no_grad()
def test_meter_keeps_native_forward_and_generation_functions_separate(monkeypatch):
    from transformers import LlamaConfig,LlamaForCausalLM
    from projects.mmdet3d_plugin.models.backbones.eva_vit import EVAViT
    from .instrumentation import Meter
    # CPU mechanism test: replace only CUDA clock functions, not model forwards.
    monkeypatch.setattr(Meter,'start',lambda self,label:None)
    monkeypatch.setattr(Meter,'stop',lambda self,label,start:None)
    llama = LlamaForCausalLM(LlamaConfig(vocab_size=41,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,bos_token_id=1,eos_token_id=None,pad_token_id=0)).eval()
    eva = EVAViT(img_size=40,patch_size=4,embed_dim=32,depth=4,num_heads=4,
        window_size=4,window_block_indexes=[0,1,3],pt_hw_seq_len=4,pretrain_img_size=40,
        flash_attn=False,with_cp=False).eval()
    model = SimpleNamespace(lm_head=llama,img_backbone=eva,m2_runtime=None,m3_runtime=None)
    before = llama.forward
    meter = Meter(model)
    answer = llama.generate(torch.tensor([[1,3,4]]),max_new_tokens=3,do_sample=False)
    eva(torch.randn(2,3,40,40))
    assert answer.shape==(1,6)
    assert meter.macs['llm']>0 and meter.macs['visual']>0
    assert len(meter.forward_lengths)==3
    meter.close()
    assert llama.forward==before


@torch.no_grad()
def test_executed_visual_macs_match_torch_profiler_for_partial_windows(monkeypatch):
    from transformers import LlamaConfig,LlamaForCausalLM
    from projects.mmdet3d_plugin.models.backbones.eva_vit import EVAViT
    from worldstream.m2 import M2Config
    from .instrumentation import Meter
    monkeypatch.setattr(Meter,'start',lambda self,label:None)
    monkeypatch.setattr(Meter,'stop',lambda self,label,start:None)
    eva = EVAViT(img_size=40,patch_size=4,embed_dim=32,depth=4,num_heads=4,
        window_size=4,window_block_indexes=[0,1,3],pt_hw_seq_len=4,pretrain_img_size=40,
        flash_attn=False,with_cp=False).eval()
    llama = LlamaForCausalLM(LlamaConfig(vocab_size=41,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2)).eval()
    runtime = ExperimentalVision(eva,M2Config(),'periodic')
    model = SimpleNamespace(img_backbone=eva,lm_head=llama,m2_runtime=runtime,m3_runtime=None)
    meter = Meter(model)
    x = torch.randn(2,3,40,40)
    context = dict(scene_id='scene',frame_id='0',timestamp=1.,camera_ids=['front','back'],
        state_epoch=1,execution_signature='test',coordinate_space='preprocessed_image')
    runtime.forward_from_m1(x,None,context)
    meter.clear()
    context.update(frame_id='1',timestamp=1.1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=True,with_flops=True) as profile:
        _,diag = runtime.forward_from_m1(x,None,context)
    assert diag['actual_skipped_windows']>0
    # Independent dispatcher records of the operations that really ran; exclude
    # elementwise operations consistently with the declared dense-MAC metric.
    measured_flops = sum(e.flops for e in profile.key_averages()
        if e.key in ('aten::mm','aten::addmm','aten::bmm','aten::conv2d'))
    assert measured_flops>0
    assert meter.macs['visual']*2==measured_flops
    meter.close()


@torch.no_grad()
@pytest.mark.parametrize('policy',['exact_cache','semantic_cache','prompt_kv','candidate_only','alternative_state'])
def test_language_routes_use_real_target_model_and_preserve_rng_indices(policy):
    from transformers import LlamaConfig,LlamaForCausalLM
    from worldstream.m3.runtime import M3Runtime
    from .language import LanguagePolicy
    torch.manual_seed(23)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=41,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=128,
        bos_token_id=1,eos_token_id=None,pad_token_id=0)).eval()
    class Tokenizer:
        special_tokens_map = {}
        def encode(self,text,add_special_tokens=False):
            return [3+ord(c)%38 for c in text[:10]]
        def decode(self,tokens,skip_special_tokens=True):
            return 'The car is ahead.'
        def get_vocab(self):
            return {str(i):i for i in range(41)}
    tokenizer = Tokenizer()
    manager = SimpleNamespace(snapshot=lambda:None)
    runtime = LanguagePolicy(M3Runtime(),policy,manager)
    context = dict(scene_id='scene',frame_id='0',timestamp=1.,camera_ids=['front'],
        weights_sha256='test',state_epoch=1,current_state_version=1,experiment_seed=7)
    for i in range(3):
        context.update(frame_id=str(i),timestamp=1.+i*.5,current_state_version=i+1)
        answers,diag = runtime.generate(lm_head=model,input_ids_list=[torch.tensor([1,3,4,5,6])],
            vision_carriers=None,context=context,questions=['What vehicle is ahead?'],tokenizer=tokenizer,
            max_new_tokens=4,do_sample=False,use_cache=True)
        assert len(answers)==1 and answers[0].shape==(1,5)
        assert diag['execution_completed']
        if policy in ('exact_cache','semantic_cache') and i:
            assert diag['requests'][0]['llm_skipped']
        if policy=='prompt_kv' and i:
            assert diag['requests'][0]['reused_tokens']==4
            assert diag['requests'][0]['decode_forward_calls']==3
        if policy=='candidate_only':
            assert all(not q['llm_skipped'] for q in diag['requests'])
