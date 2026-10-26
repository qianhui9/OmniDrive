"""Group-level WSDR contracts; all test evidence stays in pytest temporary paths."""
import copy
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from worldstream.m3 import CarrierLayout,M3Runtime,expand_input
from worldstream.m3.reuse_planner import ReusePlanner,WSDR_VERSION,GROUPS,organize_carriers
from worldstream.m3.prefix_matcher import exact_lcp
from worldstream.m3.config import M3Config
from worldstream.m3.scheduler import Scheduler
from worldstream.m3.world_state import query_world_state
from .test_trusted_history import fixture_state,state_context


def expanded(llm,carriers,layout=None,order=None,signature='same'):
    return expand_input(llm,torch.tensor([1,3,-200,4,5]),carriers,carrier_layout=layout or CarrierLayout(3,1,5),
        carrier_order=order,scope=('s',('front',),2),signature=(signature,))


def plan(llm,carriers,context,question='where is the car',layout=None,previous=None):
    layout=layout or CarrierLayout(3,1,5)
    x=expanded(llm,carriers,layout)
    ctx=state_context(context)
    return x,ReusePlanner().plan(question,x,layout,previous,query_world_state(fixture_state(),question,ctx),ctx)


def test_deterministic_layout_and_no_invented_m2_mapping(llm,carriers,context):
    x,a=plan(llm,carriers,context)
    _,b=plan(llm,carriers,context)
    assert a==b and a.layout_permutation==GROUPS
    assert not a.m2_mapping_available and a.m2_mapping_reason=='m2_region_mapping_unavailable'
    assert a.fallback_reason=='native_group_order_unverified'
    assert a.m1_state_valid
    assert x.visual_mask.sum()==carriers.shape[1]


@pytest.mark.parametrize('question,profile,relevant',[
    ('where is the car','object_attributes',('object',)),
    ('which lane is on the road','map_relations',('map',)),
    ('what is the vehicle speed','object_ego_motion',('object','ego')),
    ('how many cars are on the road','full_visual',GROUPS),
    ('are there any cars','full_visual',GROUPS),
    ('is it safe to proceed','full_visual',GROUPS),
    ('unrecognized question','full_visual',GROUPS)])
def test_uses_existing_classifier_and_conservative_dependencies(llm,carriers,context,question,profile,relevant):
    _,p=plan(llm,carriers,context,question)
    assert p.task_profile==profile and p.relevant_groups==relevant


def test_stable_group_priority_is_exact_not_semantic(llm,carriers,context):
    old,_=plan(llm,carriers,context)
    changed=carriers.clone();changed[:,4:]+=.00001
    _,p=plan(llm,changed,context,'which road lane',previous=old)
    assert p.stable_groups==('object','ego') and p.changed_groups==('map',)
    assert p.mandatory_refresh_groups==('map',)
    assert p.candidate_group_order==('object','ego','map')
    _,same=plan(llm,carriers,context,'which road lane',previous=old)
    assert same.candidate_group_order==('map','object','ego')
    assert same.layout_permutation==GROUPS  # Native positional contract is fixed.


def test_declared_producer_contract_reorganizes_all_metadata_before_expansion(llm,carriers,context):
    layout=CarrierLayout(3,1,5,'group_permutation_v1')
    _,p=plan(llm,carriers,context,'which road lane',layout)
    assert p.layout_permutation==('map','object','ego')
    ordered=organize_carriers(carriers,layout,p)
    assert torch.equal(ordered,torch.cat((carriers[:,4:],carriers[:,:3],carriers[:,3:4]),1))
    x=expanded(llm,ordered,layout,p.layout_permutation)
    x.layout_signature=p.layout_signature
    assert x.token_ids.tolist()==[[1,3]+[-200]*9+[4,5]]
    assert x.position_ids.tolist()==[list(range(13))] and x.attention_mask.all()
    assert x.visual_group_spans==( ('map',2,7),('object',7,10),('ego',10,11) )
    assert torch.equal(x.embeddings[:,2:11],ordered) and x.visual_mask.sum()==9
    native=expanded(llm,carriers)
    native.layout_signature='original'
    assert exact_lcp(native,x)==0
    # A layout changes model behavior independently of KV. Compare full forwards.
    with torch.no_grad():
        a=llm(inputs_embeds=native.embeddings).logits
        b=llm(inputs_embeds=x.embeddings).logits
    assert not torch.equal(a,b) and (a-b).abs().max()>0


def test_high_similarity_never_skips_changed_or_noncontiguous_prefix(llm,carriers):
    previous=expanded(llm,carriers)
    changed=carriers.clone();changed[0,4,0]+=1e-7
    current=expanded(llm,changed)
    assert torch.nn.functional.cosine_similarity(carriers.flatten(),changed.flatten(),dim=0)>.999
    assert exact_lcp(previous,current)==6
    current.layout_signature='different_layout'
    assert exact_lcp(previous,current)==0


@torch.no_grad()
@pytest.mark.parametrize('fault',['scene','time','epoch','version','signature','expired','mutated'])
def test_invalid_history_forces_full_prefill(llm,carriers,context,fault):
    state=fixture_state();ctx=state_context(context)
    if fault=='scene':ctx['scene_id']='other'
    if fault=='time':ctx['timestamp']=.5
    if fault=='epoch':ctx['state_epoch']=4
    if fault=='version':ctx['state_version']=9
    if fault=='signature':ctx['state_execution_signature']='wrong'
    if fault=='expired':state=replace(state,diagnostics=dict(max_state_age_s=.01))
    if fault=='mutated':state.records[0]['center_ego_m'][0]=999.
    runtime=M3Runtime(dict(reuse_strategy='wsdr_kv',min_prefix_tokens=1))
    _,diag=runtime.generate(llm,[torch.tensor([1,-200,4])]*2,carriers,context=ctx,
        layout=CarrierLayout(3,1,5),verified_state=state,questions=['where is the car']*2,max_new_tokens=2)
    assert all(not r['m1_state_valid'] and r['reused_tokens']==0 and r['full_prefill'] for r in diag['requests'])


@torch.no_grad()
def test_scene_time_and_version_lifecycle_diagnostics(llm,carriers,context):
    runtime=M3Runtime(dict(reuse_strategy='wsdr_kv',min_prefix_tokens=1))
    ids=[torch.tensor([1,-200,4])];layout=CarrierLayout(3,1,5)
    def call(state,ctx):
        return runtime.generate(llm,ids,carriers,context=ctx,layout=layout,
            verified_state=state,questions=['where is the car'],max_new_tokens=2)[1]['requests'][0]
    state=fixture_state();ctx=state_context(context)
    call(state,ctx)
    row=call(state,ctx)
    assert row['matched_prefix_tokens']==11 and row['approved_prefix_tokens']==row['reused_tokens']==0
    assert row['prefix_rejection_reason']=='wsdr_quality_evidence_unavailable'
    assert row['effective_prefill_tokens']==11 and row['cache_branch_ms'] is None
    assert row['cache_invalidation_reason'] is None
    row=call(replace(state,state_version=2),dict(ctx,state_version=2))
    assert row['cache_invalidation_reason']=='state_version_rollback'
    row=call(replace(state,timestamp=.9),dict(ctx,timestamp=.9,state_version=2))
    assert row['cache_invalidation_reason']=='nonmonotonic_timestamp'
    row=call(state,dict(ctx,scene_id='other'))
    assert row['cache_invalidation_reason']=='scene_camera_or_epoch_changed'


@torch.no_grad()
def test_storage_metadata_and_m2_context_cannot_forge_mapping(llm,carriers,context):
    runtime=M3Runtime(dict(reuse_strategy='wsdr_kv',min_prefix_tokens=1))
    kwargs=dict(lm_head=llm,input_ids_list=[torch.tensor([1,-200,4])],vision_carriers=carriers,
        context=state_context(context),layout=CarrierLayout(3,1,5),verified_state=fixture_state(),
        questions=['where is the car'],max_new_tokens=2)
    runtime.generate(**kwargs)
    runtime.cache.entry.expanded.layout_signature='tampered'
    row=runtime.generate(**kwargs)[1]['requests'][0]
    assert row['cache_invalidation_reason']=='cache_storage_revision_changed'
    row=runtime.generate(**kwargs,m2_diagnostics=dict(execution_completed=True,cache_context=dict(frame_id='wrong')))[1]['requests'][0]
    assert not row['m2_mapping_available'] and row['layout_fallback_reason']=='stale_m2_execution_context'
    assert row['reused_tokens']==0


def test_strategy_configuration_is_explicit():
    assert M3Config().reuse_strategy=='exact_prefix'
    with pytest.raises(ValueError):M3Config(reuse_strategy='approximate')


def evidence(tmp_path,llm,carriers,context):
    x,p=plan(llm,carriers,context)
    x.layout_signature=p.layout_signature
    context=dict(p.diagnostics(),model_version='deployed',execution_key='backend',weights_sha256='a'*64,
                 generation_key='generation',effective_reuse_length=6)
    artifact=tmp_path/'unit_only_evidence.json'
    row=dict(context,strategy='wsdr_kv',input_tokens=x.length,prefix_tokens=6,reused_tokens=6,
        dtype=str(x.embeddings.dtype),evidence_path=str(artifact),protocol_path=str(artifact),
        tokens_equal=True,rng_equal=True,scores_close=True,approved_by='unit_fixture_only',task_quality_status='passed',
        independent_task_quality=True,task_metrics={'fixture_only':1.},heldout_samples=1,
        measurements=[dict(sample_id='unit',baseline_ms=20.,optimized_ms=.1)])
    keys=('strategy','strategy_version','model_version','execution_key','weights_sha256','task_type','task_profile',
          'layout_signature','input_tokens','prefix_tokens','reused_tokens','dtype','generation_key')
    artifact.write_text(json.dumps(dict(schema='wsdr_kv_evidence_v1',binding={k:row[k] for k in keys},
        numerical=dict(tokens_equal=True,rng_equal=True,scores_close=True),measurements=row['measurements'])))
    digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
    row.update(evidence_sha256=digest,protocol_sha256=digest)
    scheduler=Scheduler(M3Config(reuse_strategy='wsdr_kv',min_prefix_tokens=1))
    scheduler.quality={'approved':[copy.deepcopy(row)]};scheduler.costs={'approved':[copy.deepcopy(row)]}
    return scheduler,x,context,row


@pytest.mark.parametrize('field',['strategy','strategy_version','task_type','task_profile','layout_signature',
    'weights_sha256','input_tokens','prefix_tokens','reused_tokens','dtype','generation_key','evidence_sha256'])
def test_wsdr_certificates_are_strategy_layout_task_weight_and_count_bound(tmp_path,llm,carriers,context,field):
    scheduler,x,ctx,row=evidence(tmp_path,llm,carriers,context)
    assert scheduler.prefix_allowed(x,6,ctx,.1)[0]
    for table in (scheduler.quality,scheduler.costs):table['approved'][0][field]='mismatched'
    assert not scheduler.prefix_allowed(x,6,ctx,.1)[0]


def test_exact_approval_and_unhashed_or_unbeneficial_wsdr_evidence_rejected(tmp_path,llm,carriers,context):
    scheduler,x,ctx,row=evidence(tmp_path,llm,carriers,context)
    scheduler.quality['approved'][0]['strategy']='exact_prefill'
    assert scheduler.prefix_allowed(x,6,ctx,0)[1]=='wsdr_quality_evidence_unavailable'
    scheduler.quality['approved'][0]=copy.deepcopy(row)
    scheduler.costs['approved'][0]['measurements'][0]['optimized_ms']=30.
    assert scheduler.prefix_allowed(x,6,ctx,0)[1]=='wsdr_cost_evidence_unavailable'
    scheduler.costs['approved'][0]=copy.deepcopy(row)
    (tmp_path/'unit_only_evidence.json').write_text('modified')
    assert not scheduler.prefix_allowed(x,6,ctx,0)[0]


@torch.no_grad()
def test_wsdr_uses_only_verified_contiguous_prefix_in_existing_executor(llm,carriers,context,monkeypatch):
    runtime=M3Runtime(dict(reuse_strategy='wsdr_kv',min_prefix_tokens=1))
    # Isolate the tiny model executor from deployment approval, never write it.
    monkeypatch.setattr(runtime.scheduler,'prefix_allowed',lambda x,p,c,o:(p>0,'unit_executor_fixture'))
    ids=[torch.tensor([1,3,-200,4,5]),torch.tensor([1,3,-200,7,8])]
    outputs,diag=runtime.generate(llm,ids,carriers,context=state_context(context),layout=CarrierLayout(3,1,5),
        verified_state=fixture_state(),questions=['where is the car']*2,max_new_tokens=3,do_sample=False)
    second=diag['requests'][1]
    assert second['matched_prefix_tokens']==second['approved_prefix_tokens']==second['reused_tokens']==11
    assert second['forward_prefill_lengths']==[2] and second['effective_prefill_tokens']==2
    reference=llm.generate(inputs=ids[1][None],images=carriers,max_new_tokens=3,do_sample=False)
    assert torch.equal(outputs[1],reference)


@torch.no_grad()
def test_approved_match_and_actual_reuse_differ_for_last_token_logits(llm,carriers,context,monkeypatch):
    runtime=M3Runtime(dict(reuse_strategy='wsdr_kv',min_prefix_tokens=1))
    seen=[]
    def allow(x,p,c,o):
        seen.append(c['effective_reuse_length'])
        return p>0,'unit_executor_fixture'
    monkeypatch.setattr(runtime.scheduler,'prefix_allowed',allow)
    _,diag=runtime.generate(llm,[torch.tensor([1,3,-200,4,5]),torch.tensor([1,3,-200,4])],
        carriers,context=state_context(context),layout=CarrierLayout(3,1,5),verified_state=fixture_state(),
        questions=['where is the car']*2,max_new_tokens=2)
    row=diag['requests'][1]
    assert row['matched_prefix_tokens']==row['approved_prefix_tokens']==12
    assert row['reused_tokens']==11 and seen[-1]==11 and row['effective_prefill_tokens']==1


@torch.no_grad()
@pytest.mark.parametrize('strategy',['exact_prefix','wsdr_kv'])
def test_diagnostics_do_not_extend_old_kv_entry_lifetime(llm,carriers,context,monkeypatch,strategy):
    import gc
    import weakref
    runtime=M3Runtime(dict(reuse_strategy=strategy))
    original=runtime.cache.put;observed=[]
    def put(*args,**kwargs):
        old=weakref.ref(runtime.cache.entry) if runtime.cache.entry is not None else None
        value=original(*args,**kwargs)
        if old is not None:
            gc.collect();observed.append(old() is None)
        return value
    monkeypatch.setattr(runtime.cache,'put',put)
    runtime.generate(llm,[torch.tensor([1,-200,4]),torch.tensor([1,-200,5])],carriers,
        context=state_context(context),layout=CarrierLayout(3,1,5),verified_state=fixture_state(),
        questions=['where is the car']*2,max_new_tokens=2)
    assert observed==[True]
