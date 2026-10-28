"""Representation/cache contracts, not real-scene acceptance evidence."""
from dataclasses import replace
import pytest
import torch
from worldstream.m3 import M3Config,M3Runtime,CarrierLayout
from worldstream.m3.world_prefix_composer import SpatiallyFaithfulComposer,SlotPreservingCompression
from worldstream.m3.stable_world import StableAnchorManager
from worldstream.m3.visual_state import expand_input
from worldstream.m3.prefix_matcher import exact_lcp
from worldstream.m3.reuse_planner import ReusePlanner
from .test_wrcps import state,context


def test_no_core_is_native_storage_dtype_and_gradients():
    comp=SpatiallyFaithfulComposer(32,64,8)
    s=state(facts=());a,_=StableAnchorManager().update(s,context(s))
    current=torch.randn(1,513,32,dtype=torch.float64,requires_grad=True)
    output=comp(current,a)
    assert output is current and torch.equal(output,current)
    output.sum().backward();assert torch.equal(current.grad,torch.ones_like(current))
    assert all(p.grad is None for p in comp.parameters())


@pytest.mark.parametrize('k',[64,32])
def test_slot_identity_all_inputs_and_stable_cache(k):
    torch.manual_seed(17);comp=SpatiallyFaithfulComposer(32,k,8).eval()
    a,_=StableAnchorManager().update(state(),context(state()))
    current=torch.randn(1,513,32,requires_grad=True)
    result=comp(current,a);assert result.shape==current.shape
    count=256-k//2
    assert torch.equal(result[:,k+count:k+count+1],current[:,256:257])
    source,destination,retained=comp.objects.plan(current[:,:256])
    assert len(source)==k//2 and len(retained)==count
    # Slots not chosen as merge destinations remain exactly unchanged.
    untouched=~torch.isin(retained,destination)
    assert torch.equal(result[:,k:k+count][:,untouched],current[:,retained[untouched]])
    result[:,k:].square().sum().backward()
    assert (current.grad.abs().sum(-1)>0).all()
    with torch.no_grad():
        first=comp(current.detach(),a,use_anchor_cache=True)
        ptr=comp._cached_stable.data_ptr()
        second=comp(current.detach()+.1,a,use_anchor_cache=True)
        assert ptr==comp._cached_stable.data_ptr() and torch.equal(first[:,:k],second[:,:k])
        assert not torch.equal(first[:,k:],second[:,k:])


def test_no_core_runtime_embeddings_logits_and_tokens(llm):
    llm.config.max_position_embeddings=2048
    runtime=M3Runtime(M3Config(prefix_layout='sf_adaptive',reuse_strategy='exact_prefix'))
    runtime.planner=ReusePlanner(runtime.scheduler,calibration_threshold=0.)
    native=CarrierLayout(256,1,256);s=state(facts=());vision=torch.randn(1,513,32)
    ids=torch.tensor([1,7,8,-200,9]);ctx=dict(context(s),stable_state_version=s.state_version,camera_ids=['front'],profile=True)
    with torch.no_grad():
        result,diag=runtime.generate(llm,[ids],vision,context=ctx,stable_world_state=s,
            layout=native,do_sample=False,max_new_tokens=3)
        assert runtime.last_composed_carriers is vision and runtime.composer is None
        original=expand_input(llm,ids,vision,carrier_layout=native)
        identity=expand_input(llm,ids,runtime.last_composed_carriers,carrier_layout=native)
        for field in ('embeddings','position_ids','attention_mask','visual_mask'):
            assert torch.equal(getattr(original,field),getattr(identity,field))
        a=llm(inputs_embeds=original.embeddings,use_cache=False).logits
        b=llm(inputs_embeds=identity.embeddings,use_cache=False).logits
        assert torch.equal(a,b)
        from worldstream.m3.prefill_executor import PrefillExecutor
        from worldstream.m3.prefix_kv_cache import PrefixKVCache
        expected,_=PrefillExecutor(PrefixKVCache(0)).generate(llm,original,0,do_sample=False,max_new_tokens=3)
        assert torch.equal(result[0],expected)
        assert diag['requests'][0]['stable_prefix_tokens']==0
        assert diag['requests'][0]['visual_layout_mode']=='native_full_visual'
        # Same frame exact current vision is still eligible, including visual KV.
        _,repeat=runtime.generate(llm,[ids],vision,context=ctx,stable_world_state=s,
            layout=native,do_sample=False,max_new_tokens=3)
        assert repeat['requests'][0]['reused_tokens']==original.length


def test_layout_switch_and_anchor_revocation_are_causal(llm,tmp_path):
    llm.config.max_position_embeddings=2048
    comp=SpatiallyFaithfulComposer(32,64,8);path=tmp_path/'adapter.pt'
    torch.save(dict(layout_version=comp.layout.order_contract,spec=comp.spec,state_dict=comp.state_dict(),
        trained_steps=1,training_scene_ids=['synthetic'],base_weights_sha256='unit',data_manifest_sha256='unit'),path)
    runtime=M3Runtime(M3Config(prefix_layout='sf_adaptive',prefix_adapter=str(path),reuse_strategy='exact_prefix'))
    runtime.planner=ReusePlanner(runtime.scheduler,calibration_threshold=0.)
    current=torch.randn(1,513,32);ids=[torch.tensor([1]+[7]*34+[-200,9])]
    def run(s,x):
        return runtime.generate(llm,ids,x,context=dict(context(s),stable_state_version=s.state_version,camera_ids=['front'],weights_sha256='unit'),
            stable_world_state=s,layout=CarrierLayout(256,1,256),do_sample=False,max_new_tokens=2)
    _,cold=run(state(facts=()),current)
    _,valid=run(state('f2',1.5),current+.2)
    assert valid['requests'][0]['reused_tokens']==0
    _,kept=run(state('f3',2.),current+.3)
    d=kept['requests'][0];assert d['stable_anchor_hit'] and d['reused_tokens']==35+64
    _,lost=run(state('f4',2.5,facts=()),current+.4)
    assert lost['requests'][0]['reused_tokens']==0
    assert lost['requests'][0]['visual_layout_mode']=='native_full_visual'
    assert runtime.executor is not None


def test_directional_distillation_does_not_promote_teacher_conflict():
    from tools.worldstream_m123.wrcps_train import distillation_loss
    student=torch.randn(3,5,requires_grad=True)
    teacher=torch.tensor([[0.,8.,0.,0.,0.],[0.,0.,8.,0.,0.],[0.,0.,0.,8.,0.]])
    labels=torch.tensor([1,1,3]);mask=torch.tensor([True,True,False])
    loss,d=distillation_loss(student,teacher,labels,spatial_mask=mask,spatial_weight=.5)
    assert d['spatial_aligned_tokens']==1 and d['teacher_reference_direction_conflicts']==1
    loss.backward();assert student.grad.isfinite().all()
