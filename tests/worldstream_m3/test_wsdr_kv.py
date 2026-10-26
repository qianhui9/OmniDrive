"""Continuous quality-bounded KV; unit policies never leave temporary paths."""
from dataclasses import replace
import json
import pytest
import torch
from worldstream.m3 import M3Runtime, CarrierLayout, expand_input
from worldstream.m3.config import M3Config
from worldstream.m3.prefix_matcher import prefix_statistics, exact_lcp
from worldstream.m3.reuse_planner import ReuseDecision
from worldstream.m3.scheduler import Scheduler, POLICY_VERSION
from worldstream.m3.world_state import query_world_state
from .test_trusted_history import fixture_state, state_context


def bound(context, index):
    t=1.+index*.1
    state=fixture_state()
    record=dict(state.records[0],verified_timestamp=t,source_timestamp=t-.2)
    state=replace(state,frame_id='f'+str(index),timestamp=t,state_version=3+index,records=(record,))
    return state,dict(state_context(context),frame_id=state.frame_id,timestamp=t,state_version=state.state_version)


def trial(threshold=.2):
    runtime=M3Runtime(dict(min_prefix_tokens=1))
    runtime.planner.calibration_threshold=threshold
    return runtime


def call(runtime,llm,carriers,context,index,questions=None):
    state,ctx=bound(context,index)
    questions=questions or ['where is the car']
    ids=[torch.tensor([1,3,-200,4+i,5]) for i in range(len(questions))]
    return runtime.generate(llm,ids,carriers,context=ctx,layout=CarrierLayout(3,1,5),
        verified_state=state,questions=questions,max_new_tokens=3,do_sample=False)


@torch.no_grad()
def test_natural_order_and_same_frame_exact_reuse(llm,carriers,context):
    runtime=trial()
    answers,d=call(runtime,llm,carriers,context,0,['where is the car']*2)
    row=d['requests'][1]
    assert row['reuse_type']=='exact' and row['reused_tokens']==11
    assert row['forward_prefill_lengths']==[2] and row['effective_prefill_tokens']==2
    expected=llm.generate(inputs=torch.tensor([[1,3,-200,5,5]]),images=carriers,max_new_tokens=3,do_sample=False)
    assert torch.equal(answers[1],expected)
    assert row['layout_permutation']==('object','ego','map')
    assert not row['m2_mapping_available']


@torch.no_grad()
def test_cross_frame_approx_really_executes_only_suffix(llm,carriers,context):
    runtime=trial(.1)
    call(runtime,llm,carriers,context,0)
    changed=carriers+.001
    _,d=call(runtime,llm,changed,context,1)
    row=d['requests'][0]
    assert row['reuse_type']=='bounded_approx' and row['reused_tokens']>2
    assert row['matched_prefix_tokens']==2
    assert sum(row['forward_prefill_lengths'])==row['input_tokens']-row['reused_tokens']
    assert row['approx_reuse_depth']==1 and row['cache_age_frames']==1
    assert runtime.cache.entry.approx_reuse_depth==1
    assert runtime.cache.entry.root_frame_index==0
    assert runtime.cache.entry.prefix_lineage


@torch.no_grad()
def test_inherited_approx_cache_never_relabelled_as_exact(llm,carriers,context):
    runtime=trial(.1)
    call(runtime,llm,carriers,context,0)
    changed=carriers+.001
    call(runtime,llm,changed,context,1)
    _,d=call(runtime,llm,changed,context,1,['where is the car']*2)
    assert all(r['reuse_type']=='bounded_approx' for r in d['requests'])
    assert all(r['approx_reuse_depth']==1 for r in d['requests'])


@torch.no_grad()
def test_inherited_risk_is_reweighted_for_current_question(llm,carriers,context):
    runtime=trial(.5)
    call(runtime,llm,carriers,context,0)
    call(runtime,llm,carriers+.001,context,1)
    previous=runtime.cache.entry.quality_risk
    _,d=call(runtime,llm,carriers+.001,context,1,['is it safe'])
    assert d['requests'][0]['estimated_quality_risk']>=previous
    assert d['requests'][0]['risk_profile']=='full_visual'


@torch.no_grad()
def test_depth_bound_refreshes_instead_of_accumulating(llm,carriers,context):
    runtime=trial(.1)
    call(runtime,llm,carriers,context,0)
    call(runtime,llm,carriers+.001,context,1)
    _,d=call(runtime,llm,carriers+.002,context,2)
    assert d['requests'][0]['reuse_type']=='none'
    assert d['requests'][0]['full_prefill']
    assert runtime.cache.entry.approx_reuse_depth==0


@torch.no_grad()
def test_calibrated_visual_cap_limits_only_contiguous_cross_frame_prefix(llm,carriers,context):
    runtime=trial(.5);runtime.planner.calibration_max_visual_tokens=3
    call(runtime,llm,carriers,context,0)
    _,d=call(runtime,llm,carriers+.001,context,1)
    assert d['requests'][0]['reuse_type']=='bounded_approx'
    assert d['requests'][0]['reused_tokens']==5
    assert d['requests'][0]['forward_prefill_lengths']==[8]


@torch.no_grad()
def test_first_changed_block_stops_prefix_no_noncontiguous_kv(llm,carriers,context):
    runtime=trial(.01)
    call(runtime,llm,carriers,context,0)
    changed=carriers.clone();changed[:,0]+=10
    _,d=call(runtime,llm,changed,context,1)
    assert d['requests'][0]['reused_tokens']==2
    assert d['requests'][0]['reuse_type']=='exact'


@pytest.mark.parametrize('field',['position_ids','attention_mask','token_ids','visual_mask'])
def test_metadata_and_text_must_remain_exact(llm,carriers,field):
    a=expand_input(llm,torch.tensor([[1,3,-200,4]]),carriers,carrier_layout=CarrierLayout(3,1,5))
    b=a.prefix(a.length,clone=True)
    getattr(b,field)[0,3]=0 if field in ('attention_mask','visual_mask') else 30
    exact,legal,blocks=prefix_statistics(a,b)
    assert legal==3 and exact<=3
    assert all(b['end']>legal for b in blocks if b['start']<=3)


@torch.no_grad()
def test_same_frame_visual_change_is_never_approximate(llm,carriers,context):
    runtime=trial(10.)
    call(runtime,llm,carriers,context,0)
    _,d=call(runtime,llm,carriers+.001,context,0)
    assert d['requests'][0]['reused_tokens']==2
    assert d['requests'][0]['reuse_type']=='exact'


@torch.no_grad()
@pytest.mark.parametrize('fault',['scene','time','version','model'])
def test_lifecycle_incompatibility_cannot_use_old_kv(llm,carriers,context,fault):
    runtime=trial(10.)
    call(runtime,llm,carriers,context,1)
    state,ctx=bound(context,2)
    if fault=='scene':ctx['scene_id']='other'
    if fault=='time':ctx['timestamp']=.2
    if fault=='version':ctx['state_version']=0
    if fault=='model':
        with torch.no_grad():llm.model.embed_tokens.weight.add_(.01)
    _,d=runtime.generate(llm,[torch.tensor([1,3,-200,4,5])],carriers,context=ctx,
        layout=CarrierLayout(3,1,5),verified_state=state,questions=['where is the car'],max_new_tokens=2)
    assert d['requests'][0]['reused_tokens']==0
    assert d['requests'][0]['cache_invalidation_reason']


@torch.no_grad()
def test_unproved_approx_and_stale_proof_are_errors(llm,carriers,context):
    runtime=trial(.1)
    call(runtime,llm,carriers,context,0)
    state,ctx=bound(context,1);ctx['frame_index']=1
    x=expand_input(llm,torch.tensor([[1,3,-200,4,5]]),carriers+.001,
        scope=runtime.cache.entry.expanded.scope,signature=runtime.cache.entry.expanded.signature,
        source=('s','f1',1.1,0),carrier_layout=CarrierLayout(3,1,5))
    with pytest.raises(ValueError,match='evidence'):
        runtime.executor.generate(llm,x,decision=ReuseDecision(11,'bounded_approx'),max_new_tokens=2)
    decision,_=runtime.planner.plan('where is the car',x,runtime.cache,
        query_world_state(state,'where is the car',ctx),ctx)
    for broken in (replace(decision,approx_reuse_depth=0),replace(decision,cache_age_frames=0),
                   replace(decision,root_frame_index=99)):
        with pytest.raises(ValueError,match='evidence'):
            runtime.executor.generate(llm,x,decision=broken,max_new_tokens=2)
    x.position_ids[0,1]+=1
    with pytest.raises(ValueError,match='evidence'):
        runtime.executor.generate(llm,x,decision=decision,max_new_tokens=2)


def unit_scheduler(tmp_path):
    context=dict(model_version='unit',weights_sha256='unit',execution_key='unit',generation_key='unit',
        layout_signature='unit',dtype='unit',task_profile='full_visual',input_tokens=100)
    e=dict(quality_loss_upper=.01,calibration_samples=3,validation_samples=2,critical_regressions=0,
        risk_limit=.1,max_rms=.1,max_token_change=.2,
        cost=dict(net_saved_ms_per_token=.2,samples=3,min_input_tokens=90,max_input_tokens=110,max_reuse_tokens=90))
    policy=dict(version=POLICY_VERSION,binding={k:v for k,v in context.items() if k not in ('task_profile','input_tokens')},
        block_size=32,profiles={'full_visual':{'exact':e,'approx':e}},max_cache_age_frames=2,max_approx_depth=1)
    p=tmp_path/'unit.json';p.write_text(json.dumps(policy))
    return Scheduler(M3Config(reuse_policy=str(p))),context


def candidate(**kw):
    return dict(length=64,reuse_type='bounded_approx',risk=.05,rms=.04,max_token=.05,age=1,depth=1,**kw)


def test_unified_argmin_selects_quality_bounded_net_gain(tmp_path):
    scheduler,ctx=unit_scheduler(tmp_path)
    small=dict(candidate(),length=32)
    chosen,_=scheduler.choose([small,candidate()],ctx,1.)
    assert chosen['length']==64
    assert scheduler.choose([dict(candidate(),risk=.2)],ctx,1.)[0] is None
    assert scheduler.choose([dict(candidate(),depth=2)],ctx,1.)[0] is None
    assert scheduler.choose([dict(candidate(),age=3)],ctx,1.)[0] is None
    assert scheduler.choose([candidate()],ctx,100.)[0] is None
    scheduler.policy['profiles']['full_visual']['approx']['quality_loss_upper']=.03
    assert scheduler.choose([candidate()],ctx,0.)[0] is None


@pytest.mark.parametrize('key',['model_version','weights_sha256','execution_key','generation_key','layout_signature','dtype'])
def test_policy_bound_to_actual_execution(tmp_path,key):
    scheduler,ctx=unit_scheduler(tmp_path);ctx[key]='different'
    assert scheduler.choose([candidate()],ctx,0.)[1]=='reuse_policy_binding_mismatch'


@pytest.mark.parametrize('content',[None,'broken','[]','{}','{"version":"old"}',
    '{"version":"quality_bounded_wsdr_v1","profiles":{},"quality_budget":NaN}'])
def test_missing_policy_is_same_executor_full_prefill(tmp_path,content):
    p=tmp_path/'policy.json'
    if content is not None:p.write_text(content)
    scheduler=Scheduler(M3Config(reuse_policy=str(p)))
    assert scheduler.choose([],{},0.)[0] is None


def test_native_order_and_reference_strategy_only():
    assert M3Config().reuse_strategy=='wsdr_kv'
    with pytest.raises(ValueError):CarrierLayout(3,1,5,'group_permutation_v1')
    with pytest.raises(ValueError):M3Config(block_size=7)


@torch.no_grad()
@pytest.mark.parametrize('approximate',[False,True])
def test_no_old_cache_entry_reference_leak(llm,carriers,context,monkeypatch,approximate):
    import weakref,gc
    runtime=trial(.1) if approximate else M3Runtime();oldput=runtime.cache.put;observed=[]
    def put(*args,**kw):
        old=weakref.ref(runtime.cache.entry) if runtime.cache.entry else None
        value=oldput(*args,**kw)
        if old:gc.collect();observed.append(old() is None)
        return value
    monkeypatch.setattr(runtime.cache,'put',put)
    if approximate:
        call(runtime,llm,carriers,context,0)
        call(runtime,llm,carriers+.001,context,1)
    else:call(runtime,llm,carriers,context,0,['where is the car']*2)
    assert observed==[True]
