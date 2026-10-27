"""Evidence predicates, conservative task semantics and skipped unused RMS kernels."""
import pytest
import torch
from worldstream.m3.task_conditioner import task_profile
from worldstream.m3.prefix_matcher import prefix_statistics
from worldstream.m3 import expand_input,CarrierLayout,M3Runtime
from .test_wsdr_kv import unit_scheduler,candidate,call


@pytest.mark.parametrize('question',[
    'What type of area is the vehicle navigating?',
    'On the left-rear side, is there sufficient space for another vehicle to pass or park?',
    'What should the vehicle do in response to the traffic light?',
    'Is the vehicle currently in a position where it needs to change lanes or make a turn?',
    'How does the parking situation appear in the areas behind the vehicle?',
    'Are there any road markings that indicate specific lanes or directions?',
    'Can the driver make a safe lane change if needed?'])
def test_sensitive_questions_are_not_map_or_object_attributes(question):
    assert task_profile(question)[1]=='full_visual'


def test_static_map_and_genuine_object_tasks_remain_distinct():
    assert task_profile('Describe the road layout.')[1]=='map_relations'
    assert task_profile('What color is the car?')[1]=='object_attributes'
    assert task_profile('What is the speed of the car?')[1]=='object_ego_motion'


@pytest.mark.parametrize('fault,reason',[
    ('critical','critical_regressions'),('quality','quality_loss_exceeds_budget'),
    ('input','input_tokens_out_of_range'),('length','reuse_tokens_below_range'),
    ('risk','risk_limit'),('cost','nonpositive_net_saving')])
def test_explanation_uses_actual_first_predicate(tmp_path,fault,reason):
    scheduler,ctx=unit_scheduler(tmp_path);c=candidate();e=scheduler.policy['profiles']['full_visual']['approx'];overhead=0.
    if fault=='critical':e['critical_regressions']=1
    if fault=='quality':e['quality_loss_upper']=.025974025974025983
    if fault=='input':ctx['input_tokens']=120
    if fault=='length':e['cost']['min_reuse_tokens']=80
    if fault=='risk':c['risk']=.3
    if fault=='cost':overhead=100.
    assert scheduler.choose([c],ctx,overhead)[0] is None
    row=scheduler.explain([c],ctx,overhead)[0]
    assert row['first_rejection']==reason and row['candidate']==c


def test_unavailable_approximate_evidence_never_requires_rms(tmp_path):
    scheduler,_=unit_scheduler(tmp_path)
    assert scheduler.needs_approximate_statistics('full_visual')
    scheduler.policy['profiles']['full_visual']['approx']['critical_regressions']=1
    assert not scheduler.needs_approximate_statistics('full_visual')
    assert not scheduler.needs_approximate_statistics('unseen')


@torch.no_grad()
def test_omitting_block_statistics_preserves_exact_and_legal_prefix(llm,carriers):
    x=expand_input(llm,torch.tensor([[1,3,-200,4]]),carriers,carrier_layout=CarrierLayout(3,1,5))
    y=x.prefix(x.length,clone=True);y.embeddings[:,3]+=.001
    full=prefix_statistics(x,y,4)
    with torch.autograd.profiler.profile() as p:
        pruned=prefix_statistics(x,y,4,include_blocks=False)
    assert pruned[:2]==full[:2] and full[2] and pruned[2]==[]
    assert 'aten::sqrt' not in {e.name for e in p.function_events}


@torch.no_grad()
def test_runtime_omits_unusable_blocks_and_keeps_full_current_forward(llm,carriers,context,monkeypatch):
    import worldstream.m3.reuse_planner as module
    original=module.prefix_statistics;seen=[]
    def statistics(*args,**kwargs):
        seen.append(kwargs['include_blocks']);return original(*args,**kwargs)
    monkeypatch.setattr(module,'prefix_statistics',statistics)
    runtime=M3Runtime();call(runtime,llm,carriers,context,0)
    _,d=call(runtime,llm,carriers+.001,context,1)
    assert seen==[False] and d['requests'][0]['reused_tokens']==0
    assert d['requests'][0]['forward_prefill_lengths']==[13]


@torch.no_grad()
def test_benchmark_zero_control_keeps_same_executor_and_real_full_forward(llm,carriers,context):
    from tools.worldstream_m123.benchmark.m3_recovery import ZeroReuseControl
    from .test_wsdr_kv import trial
    runtime=trial(.1);call(runtime,llm,carriers,context,0)
    executor=runtime.executor;ZeroReuseControl(runtime)
    _,d=call(runtime,llm,carriers+.001,context,1)
    row=d['requests'][0]
    assert runtime.executor is executor
    assert row['benchmark_planned_reuse_length']>0 and row['reused_tokens']==0
    assert row['forward_prefill_lengths']==[13] and row['effective_prefill_tokens']==13
    assert runtime.cache.entry.approx_reuse_depth==0 and runtime.cache.entry.root_frame_index==1


def test_rejecting_approximate_rule_does_not_disable_qualified_exact(tmp_path):
    import copy
    scheduler,ctx=unit_scheduler(tmp_path)
    scheduler.policy['profiles']['full_visual']['approx']=copy.deepcopy(scheduler.policy['profiles']['full_visual']['approx'])
    scheduler.policy['profiles']['full_visual']['approx']['critical_regressions']=1
    exact=dict(candidate(),reuse_type='exact')
    chosen,_=scheduler.choose([candidate(),exact],ctx,1.)
    assert chosen['reuse_type']=='exact'
    assert not scheduler.needs_approximate_statistics('full_visual')
    assert scheduler.explain([candidate()],ctx,1.)[0]['first_rejection']=='critical_regressions'
