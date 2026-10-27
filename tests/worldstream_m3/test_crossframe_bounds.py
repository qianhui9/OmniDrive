"""Optimistic cost pruning preserves causal candidates and real GPU forwards."""
import pytest
import torch
from worldstream.m3 import CarrierLayout, expand_input
from worldstream.m3.prefix_matcher import prefix_statistics, visual_block_spans
from .test_wsdr_kv import unit_scheduler, trial, bound


@pytest.mark.parametrize('device',['cpu','cuda'])
@torch.no_grad()
def test_cap_preserves_full_causal_match_and_measured_blocks(llm,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    llm=llm.to(device)
    images=torch.randn(1,81,32,device=device)
    x=expand_input(llm,torch.tensor([[1,3,-200,4]],device=device),images,
        carrier_layout=CarrierLayout(48,1,32))
    y=x.prefix(x.length,clone=True)
    y.embeddings[:,2:]+=0.01
    full=prefix_statistics(x,y,16)
    capped=prefix_statistics(x,y,16,max_visual_tokens=32)
    assert capped[:2]==full[:2]
    assert capped[2]==full[2][:2]
    # Later unchanged map slots cannot bypass the first changed object slot.
    y.embeddings[:,51:]=x.embeddings[:,51:]
    assert prefix_statistics(x,y,16,max_visual_tokens=32)[0]==2
    # Text/position/mask/nonfinite failures beyond the cap remain visible.
    for name in ('token_ids','position_ids','attention_mask'):
        z=y.prefix(y.length,clone=True);getattr(z,name)[:,-1]=0
        assert prefix_statistics(x,z,16,max_visual_tokens=32)[:2]==prefix_statistics(x,z,16)[:2]
    y.embeddings[:,-1]=float('nan')
    assert prefix_statistics(x,y,16,max_visual_tokens=32)[:2]==prefix_statistics(x,y,16)[:2]


def test_visual_cap_does_not_invent_sub_block_endpoints():
    from types import SimpleNamespace
    x=SimpleNamespace(visual_group_spans=(('object',35,291),('ego',291,292),('map',292,548)))
    assert list(visual_block_spans(x,580,32,31))==[]
    assert list(visual_block_spans(x,580,32,128))[-1]==('object',131,163)
    assert [v[2] for v in visual_block_spans(x,580,32,257)][-2:]==[291,292]


@pytest.mark.parametrize('fault,expected',[
    ('length','reuse_tokens_below_range'),('input','input_tokens_out_of_range'),
    ('age','cache_age_limit'),('depth','approx_depth_limit'),
    ('quality','critical_regressions'),('binding','reuse_policy_binding_mismatch'),
    ('cost','nonpositive_net_saving')])
def test_optimistic_pruning_rejects_only_impossible_cost_quality_ranges(tmp_path,fault,expected):
    s,ctx=unit_scheduler(tmp_path)
    e=s.policy['profiles']['full_visual']['approx'];age=1;depth=1
    if fault=='length':e['cost']['min_reuse_tokens']=80
    if fault=='input':ctx['input_tokens']=120
    if fault=='age':age=3
    if fault=='depth':depth=2
    if fault=='quality':e['critical_regressions']=1
    if fault=='binding':ctx['execution_key']='different'
    if fault=='cost':e['cost']['fixed_overhead_ms']=10000
    assert s.approximate_statistics_rejection(ctx,[34,66],age,depth)==expected


def test_bounds_check_every_endpoint_not_only_the_largest(tmp_path):
    s,ctx=unit_scheduler(tmp_path)
    e=s.policy['profiles']['full_visual']['approx'];e['cost']['max_reuse_tokens']=65
    assert s.approximate_statistics_rejection(ctx,[34,66],1,1) is None
    assert s.approximate_statistics_rejection(ctx,[],1,1)=='no_visual_block_within_cap'


@pytest.mark.parametrize('device',['cpu','cuda'])
@torch.no_grad()
def test_optimized_bounded_reuse_has_actual_suffix_forward_and_lineage(llm,carriers,context,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    llm=llm.to(device);carriers=carriers.to(device)
    runtime=trial(.1);runtime.planner.calibration_max_visual_tokens=3
    def execute(images,index):
        state,ctx=bound(context,index)
        return runtime.generate(llm,[torch.tensor([1,3,-200,4,5],device=device)],images,
            context=ctx,layout=CarrierLayout(3,1,5),verified_state=state,
            questions=['where is the car'],max_new_tokens=3,do_sample=False)
    execute(carriers,0)
    _,d=execute(carriers+.001,1)
    row=d['requests'][0]
    assert row['reused_tokens']==5 and row['forward_prefill_lengths']==[8]
    assert row['computed_source'][1]=='f0' and row['suffix_computed_source'][1]=='f1'
    assert row['approx_reuse_depth']==1 and row['cache_age_frames']==1
    # Mutating the stored prefix still invalidates all cached computation.
    runtime.cache.entry.kv[0][0].zero_()
    _,d=execute(carriers+.002,2)
    assert d['requests'][0]['reused_tokens']==0 and d['requests'][0]['forward_prefill_lengths']==[13]
