from dataclasses import replace
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from worldstream.m3 import M3Runtime,M3Config,expand_input
from worldstream.m3.batched_executor import BatchedExecutor
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from worldstream.m3.prefill_executor import PrefillExecutor
from worldstream.m3.visual_state import execution_signature
from worldstream.m3.world_state import freeze_verified_state,query_world_state,insert_state_context


def questions():
    return [torch.tensor([1,3,-200,4,5]),torch.tensor([1,3,-200,6]),torch.tensor([1,3,-200,7,8,9,10])]


def test_batched_variable_suffix_positions_and_eos(llm,carriers):
    ids=questions();xs=[expand_input(llm,x,carriers) for x in ids]
    # First sequence ends immediately, others continue; result order and
    # per-request EOS lengths must match native independent generation.
    eos=int(llm(inputs_embeds=xs[0].embeddings).logits[0,-1].argmax())
    kw=dict(do_sample=False,max_new_tokens=9,eos_token_id=eos,pad_token_id=0)
    expected=[llm.generate(inputs=x[None],images=carriers,**kw) for x in ids]
    cache=PrefixKVCache(1024**2)
    actual,diag=BatchedExecutor(cache,workspace_bytes=32*1024**2).try_generate(llm,xs,**kw)
    assert diag['admitted'] and diag['suffix_padding_tokens']>0
    assert diag['effective_prefill_tokens']==sum(x.length for x in xs)-diag['actual_reused_tokens']
    assert actual[0].shape[1]==2
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    saved=copy.deepcopy(cache.entry.kv)
    PrefillExecutor(cache).generate(llm,xs[1],cache.match(xs[1]),borrow_prefix=True,max_new_tokens=4)
    assert all(torch.equal(a,b) for x,y in zip(saved,cache.entry.kv) for a,b in zip(x,y))


def test_batch_sampling_rng_and_budget_fall_back(llm,carriers,context):
    runtime=M3Runtime(dict(mode='exact',offline_experiment=True,batched_decode=True,min_prefix_tokens=2))
    kw=dict(do_sample=True,temperature=.7,top_p=.8,max_new_tokens=4)
    torch.manual_seed(77);expected=[llm.generate(inputs=x[None],images=carriers,**kw) for x in questions()]
    torch.manual_seed(77);actual,diag=runtime.generate(llm,questions(),carriers,context=context,**kw)
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    assert diag['batch_fallback']['reason']=='sampling_requires_sequential_rng_isolation'
    xs=[expand_input(llm,x,carriers) for x in questions()]
    actual,diag=BatchedExecutor(PrefixKVCache(1024**2),0).try_generate(llm,xs,do_sample=False,max_new_tokens=320)
    assert actual is None and diag['reason']=='batch_workspace_budget'


def test_readonly_native_cat_does_not_mutate_cache(llm,carriers):
    cache=PrefixKVCache(1024**2);executor=PrefillExecutor(cache)
    xs=[expand_input(llm,x,carriers) for x in questions()]
    executor.generate(llm,xs[0],max_new_tokens=2)
    saved=copy.deepcopy(cache.entry.kv)
    a,diag=executor.generate(llm,xs[2],cache.match(xs[2]),borrow_prefix=True,max_new_tokens=6)
    b,_=executor.generate(llm,xs[2],cache.match(xs[2]),borrow_prefix=False,max_new_tokens=6)
    assert torch.equal(a,b) and diag['branch_workspace_bytes']==0
    assert diag['prefix_storage']=='borrowed_readonly_native_cat'
    assert all(torch.equal(a,b) for x,y in zip(saved,cache.entry.kv) for a,b in zip(x,y))
    original=llm.model.layers[0].self_attn.forward
    llm.model.layers[0].self_attn.forward=lambda *a,**kw:original(*a,**kw)
    _,diag=executor.generate(llm,xs[2],cache.match(xs[2]),borrow_prefix=True,max_new_tokens=2)
    assert diag['prefix_storage']=='owned_clone' and diag['branch_workspace_bytes']>0


def test_single_fastpath_no_cache_signature_or_expansion(llm,carriers,context,monkeypatch):
    import worldstream.m3.runtime as module
    def forbidden(*a,**kw):raise AssertionError('No M3 signature or preparation for single question')
    monkeypatch.setattr(module,'execution_signature',forbidden)
    monkeypatch.setattr(module,'expand_input',forbidden)
    runtime=M3Runtime(dict(mode='exact',offline_experiment=True))
    _,diag=runtime.generate(llm,questions()[:1],carriers,context=context,max_new_tokens=3)
    assert diag['reason']=='single_question_no_shared_prefix' and runtime.cache.entry is None


def test_numerical_settings_are_cache_dependencies(llm):
    signature=execution_signature(llm)
    flag=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=not flag
        assert execution_signature(llm)!=signature
    finally:torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=flag


def historical_state():
    from worldstream.m1.state_types import WorldAnchor,Geometry,Frame,M1Result,AnchorObservation
    anchor=WorldAnchor('a','s','object','car',Geometry(np.array([[8.,2.,0.],[10.,2.,0.]]),np.array([9.,2.,0.])),1.,2.)
    anchor.history_observations=2
    anchor.observation_support={'front':AnchorObservation('front',(0,0,4,4),np.ones(3),1.,1.)}
    result=M1Result('s','f',2.,valid_anchor_ids=['a'],diagnostics=dict(execution_signature='weights',
        anchors={'a':dict(valid=True,cameras={'front':dict(valid=True)})}))
    store=SimpleNamespace(scene_id='s',generation=1,anchors={'a':anchor},features={})
    manager=SimpleNamespace(store=store,last_result=result,config=SimpleNamespace(max_state_age_s=1.),_pending=('s','f',2.,'weights'))
    frame=Frame('s','f',2.,np.eye(4),{},'weights')
    return manager,frame


def test_world_state_is_historical_precommit_and_unknown_stays_unknown():
    manager,frame=historical_state();snapshot=freeze_verified_state(manager,frame)
    record=snapshot.records[0]
    assert record['source_frame'] is None and record['velocity_ego_mps'] is None and record['feature_versions']==[]
    manager.store.anchors['a'].geometry_world.center[0]=100
    assert record['center_ego_m'][0]==9
    context=dict(scene_id='s',frame_id='f',timestamp=2.,state_epoch=1,state_execution_signature='weights')
    assert query_world_state(snapshot,'Where is the car on the left?',context)['admitted']
    assert not query_world_state(snapshot,'How many cars are there?',context)['admitted']
    assert not query_world_state(snapshot,'Describe the speed of the car',context)['admitted']
    assert not query_world_state(snapshot,'Describe the environment',context)['admitted']
    assert not query_world_state(snapshot,'Describe the car on the left',dict(context,frame_id='old'))['admitted']
    manager.store.anchors['a'].source_timestamp=frame.timestamp
    assert not freeze_verified_state(manager,frame).records
    manager._pending=None
    assert freeze_verified_state(manager,frame).diagnostics['reason']=='stale_or_missing_precheck'


def test_context_insertion_preserves_original_ids_and_auto_gate():
    ids=torch.tensor([1,3,-200,4,5])
    tokenizer=SimpleNamespace(encode=lambda s,add_special_tokens:[6,7])
    assert insert_state_context(ids,'history',tokenizer).tolist()==[1,3,-200,6,7,4,5]
    with pytest.raises(ValueError):M3Config(state_context='only')
    with pytest.raises(ValueError):M3Config(batched_decode=True)
    with pytest.raises(ValueError):M3Config(mode='combined',batched_decode=True,offline_experiment=True)


def test_state_context_paths_change_actual_input_only_with_verified_history(llm,carriers,context):
    manager,frame=historical_state();state=freeze_verified_state(manager,frame)
    ids=[questions()[0]];ctx=dict(scene_id='s',frame_id='f',timestamp=2.,camera_ids=['front'],
        state_epoch=1,state_execution_signature='weights')
    tokenizer=SimpleNamespace(encode=lambda s,add_special_tokens:[11,12])
    for mode in ('only','augment'):
        runtime=M3Runtime(dict(mode='compact',state_context=mode,offline_experiment=True))
        _,diag=runtime.generate(llm,ids,carriers,context=ctx,questions=['Describe the car on the left'],
            verified_state=state,tokenizer=tokenizer,max_new_tokens=2)
        row=diag['requests'][0]
        assert row['state_retrieval']['admitted']
        assert row['visual_tokens']==(0 if mode=='only' else 9)
        assert row['input_tokens']==(6 if mode=='only' else 15)
        _,fallback=runtime.generate(llm,ids,carriers,context=ctx,questions=['Describe the car on the left'],
            verified_state=None,tokenizer=tokenizer,max_new_tokens=2)
        assert fallback['requests'][0]['mode']=='full' and runtime.cache.entry is None


def test_precision_probe_restores_lora_and_flags_on_failure(llm):
    from peft import get_peft_model,LoraConfig
    from worldstream.m3.numerics import precision_probe
    model=get_peft_model(llm,LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    versions=[(n,p._version,p.dtype) for n,p in model.named_parameters()]
    original={n:m.forward for n,m in model.named_modules() if '.lora_A.' in n or '.lora_B.' in n}
    with pytest.raises(RuntimeError):
        with precision_probe(model,'fp32_lora'):
            model(input_ids=torch.tensor([[1,3,4]]))
            raise RuntimeError('intentional unwinding')
    assert versions==[(n,p._version,p.dtype) for n,p in model.named_parameters()]
    assert all(m.forward==original[n] for n,m in model.named_modules() if n in original)


def test_independent_common_prefix_seed_and_suffix_only_generation(llm,carriers,context):
    runtime=M3Runtime(dict(mode='exact',offline_experiment=True,common_prefix_seed=True,
        borrow_prefix=True,min_prefix_tokens=2))
    kw=dict(do_sample=False,max_new_tokens=7)
    expected=[llm.generate(inputs=x[None],images=carriers,**kw) for x in questions()]
    actual,diag=runtime.generate(llm,questions(),carriers,context=context,**kw)
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    assert diag['shared_seed']['cache_stored']
    assert diag['shared_seed']['actual_saved_prefix_tokens']==22
    assert all(r['reused_tokens']==11 for r in diag['requests'])
    assert runtime.cache.entry.expanded.length==11
    assert diag['requests'][0]['forward_prefill_lengths']==[2]


def test_common_auto_requires_both_independent_shape_and_generation_evidence(llm,carriers,context):
    from worldstream.m3.scheduler import Scheduler
    cfg=M3Config(mode='auto',common_prefix_seed=True,min_prefix_tokens=2)
    scheduler=Scheduler(cfg);xs=[expand_input(llm,x,carriers) for x in questions()]
    ctx=dict(model_version='verified_weights',execution_key='backend')
    row=dict(strategy='independent_common_prefix',model_version='verified_weights',execution_key='backend',
        generation_key='greedy_320',input_tokens=[x.length for x in xs],prefix_tokens=11,
        protocol_sha256='protocol',evidence_sha256='evidence',frozen_targets_pass=True,heldout_samples=3,
        calibration_scenes=['s1'],heldout_scenes=['s2'],includes_cold_capture_and_maintenance=True,
        baseline_ms=10,optimized_including_overhead_ms=5,cache_bytes=4096)
    scheduler.costs={'approved':[row]};scheduler.quality={'approved':[]}
    assert not scheduler.common_seed_allowed(xs,11,ctx,'greedy_320')[0]
    scheduler.quality={'approved':[row]}
    assert scheduler.common_seed_allowed(xs,11,ctx,'greedy_320')[0]
    assert not scheduler.common_seed_allowed(xs,11,ctx,'sampling_320')[0]
    assert not scheduler.common_seed_allowed(xs,10,ctx,'greedy_320')[0]
    scheduler.quality={'approved':[dict(row,heldout_scenes=['s1'])]}
    assert not scheduler.common_seed_allowed(xs,11,ctx,'greedy_320')[0]


def test_common_auto_executes_only_certified_plan_and_rejects_other_generation(llm,carriers,context,tmp_path):
    import hashlib,json
    from worldstream.m3.visual_state import deployment_key
    ids=questions();xs=[expand_input(llm,x,carriers) for x in ids]
    generation=dict(do_sample=False,max_new_tokens=3)
    key=hashlib.sha256(json.dumps(dict(defaults=llm.generation_config.to_dict(),overrides=generation),
        sort_keys=True,allow_nan=False).encode()).hexdigest()
    row=dict(strategy='independent_common_prefix',model_version='v',execution_key=deployment_key(llm),
        generation_key=key,input_tokens=[x.length for x in xs],prefix_tokens=11,protocol_sha256='test_protocol',evidence_sha256='test_evidence',
        frozen_targets_pass=True,heldout_samples=3,calibration_scenes=['cal'],heldout_scenes=['held'],
        includes_cold_capture_and_maintenance=True,baseline_ms=100,optimized_including_overhead_ms=1,cache_bytes=32768)
    path=tmp_path/'test_only_evidence.json';path.write_text(json.dumps(dict(approved=[row])))
    runtime=M3Runtime(dict(mode='auto',common_prefix_seed=True,borrow_prefix=True,model_version='v',min_prefix_tokens=2,
        cost_table=str(path),quality_certificate=str(path)))
    _,diag=runtime.generate(llm,ids,carriers,context=context,**generation)
    assert diag['shared_seed']['cache_stored'] and all(r['reused_tokens']==11 for r in diag['requests'])
    _,diag=runtime.generate(llm,ids,carriers,context=dict(context,frame_id='next',timestamp=2.),do_sample=True,max_new_tokens=3)
    assert diag['mode']=='full' and 'no_independent_common_prefix' in diag['reason']


def test_frozen_original_questions_replace_only_cpu_prompt_fields():
    from mmcv.parallel import DataContainer
    from tools.worldstream_m3.refinement import freeze_sample_questions
    images=torch.randn(2,3,4,4)
    sample=dict(input_ids=[DataContainer([torch.tensor([1,2])])],
        img_metas=[DataContainer(dict(vlm_labels=DataContainer(['random template']),scene_token='s'),cpu_only=True)],img=[images])
    original=dict(ids=[torch.tensor([1,-200,8]),torch.tensor([1,-200,9,10])],questions=['original A','original B'])
    freeze_sample_questions(sample,original)
    assert sample['img'][0] is images and sample['img_metas'][0].data['scene_token']=='s'
    assert sample['img_metas'][0].data['vlm_labels'].data==original['questions']
    assert all(torch.equal(x,y) for x,y in zip(sample['input_ids'][0].data,original['ids']))
    sample['input_ids'][0].data[0][0]=11
    assert original['ids'][0][0]==1


def test_deployment_key_binds_actual_forward_chain_and_lora_state(llm):
    from functools import wraps
    from peft import get_peft_model,LoraConfig
    from worldstream.m3.visual_state import deployment_key
    model=get_peft_model(llm,LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    original=deployment_key(model)
    with model.disable_adapter():assert deployment_key(model)!=original
    assert deployment_key(model)==original
    layer=model.get_base_model().model.layers[0].self_attn.q_proj.lora_A['default']
    native=layer.forward
    @wraps(native)
    def different_arithmetic(x):return native(x.float().to(layer.weight.dtype))
    layer.forward=different_arithmetic
    assert deployment_key(model)!=original
    layer.forward=native
    assert deployment_key(model)==original


def test_fp32_base_projection_scope_restores_and_leaves_decode_native(llm):
    from peft import get_peft_model,LoraConfig
    from worldstream.m3.projection_precision import fp32_qkv_prefill
    model=get_peft_model(llm,LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    base=model.get_base_model();layer=base.model.layers[0].self_attn.q_proj.base_layer
    original=layer.forward;versions=[p._version for p in base.parameters()];calls=[]
    def native(*a,**kw):calls.append(a[0].shape[-2]);return original(*a,**kw)
    layer.forward=native
    with pytest.raises(RuntimeError):
        with fp32_qkv_prefill(base):
            layer(torch.randn(1,3,32));layer(torch.randn(1,1,32))
            assert calls==[1]
            raise RuntimeError('unwind')
    assert layer.forward is native and versions==[p._version for p in base.parameters()]
