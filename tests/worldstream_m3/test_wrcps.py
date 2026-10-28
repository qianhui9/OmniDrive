"""Synthetic contract tests; these are never research/quality acceptance data."""
from dataclasses import asdict, replace
import json
import pytest
import torch
from worldstream.m3 import M3Runtime, M3Config, CarrierLayout, WorldPrefixComposer, WorldPrefixLayout
from worldstream.m3.stable_world import StableWorldSnapshot, StableAnchorManager, digest
from worldstream.m3.visual_state import expand_input
from worldstream.m3.prefix_matcher import exact_lcp
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from worldstream.m3.prefill_executor import PrefillExecutor


def state(frame='f1', timestamp=1., facts=(('id1','object','car'),), **kwargs):
    return StableWorldSnapshot('s',frame,timestamp,1,1,'exec',facts,'{}',**kwargs)


def context(s):
    return dict(scene_id=s.scene_id,frame_id=s.frame_id,timestamp=s.timestamp,
        state_epoch=s.state_epoch,state_version=s.state_version,state_execution_signature=s.execution_signature)


def test_anchor_content_identity_excludes_per_frame_provenance():
    manager=StableAnchorManager(max_age_frames=3)
    a,hit=manager.update(state(),context(state()));assert not hit
    s=replace(state('f2',1.5),state_version=2)
    b,hit=manager.update(s,context(s))
    assert hit and b.anchor_id==a.anchor_id and b.anchor_age==1 and b.source_frame=='f1'
    s=replace(state('f3',2.,facts=(('id1','object','truck'),)),state_version=3)
    c,hit=manager.update(s,context(s))
    assert not hit and c.anchor_id!=a.anchor_id and c.update_reason=='stable_content_changed'


def test_invalid_state_never_keeps_anchor():
    manager=StableAnchorManager()
    a,_=manager.update(state(facts=()),context(state(facts=())))
    s=state('f2',1.5,facts=());b,hit=manager.update(s,context(s))
    assert not hit and not b.stable_state_valid and a.anchor_id!=b.anchor_id
    with pytest.raises(ValueError,match='context mismatch'):
        manager.update(s,dict(context(s),state_epoch=9))
    with pytest.raises(ValueError,match='Same-frame'):
        s=replace(s,state_version=3);manager.update(s,context(s))


@pytest.mark.parametrize('change,reason',[
    (dict(scene_id='new'),'scene_changed'),(dict(state_epoch=2),'state_epoch_changed'),
    (dict(execution_signature='new'),'state_execution_changed'),
    (dict(state_version=0),'state_version_rollback'),(dict(timestamp=.5),'time_discontinuity'),
    (dict(timestamp=10.),'time_discontinuity')])
def test_anchor_lifecycle_revokes_previous(change,reason):
    manager=StableAnchorManager();a,_=manager.update(state(),context(state()))
    s=replace(state('f2',1.5),**change);b,hit=manager.update(s,context(s))
    assert not hit and a.anchor_id!=b.anchor_id and b.update_reason==reason


def test_cached_stable_is_exact_and_all_current_tokens_have_gradients():
    torch.manual_seed(1)
    composer=WorldPrefixComposer(32,128,rank=8)
    manager=StableAnchorManager();a,_=manager.update(state(),context(state()))
    current=torch.randn(1,513,32,requires_grad=True)
    output=composer(current,a)
    assert output.shape==current.shape
    assert torch.equal(output[:,128+192:128+193],current[:,256:257])
    output[:,128:].square().sum().backward()
    assert (current.grad.abs().sum(-1)>0).all()
    assert all(p.grad is not None for p in composer.objects.parameters())
    composer.eval()
    with torch.no_grad():
        first=composer(current.detach(),a,use_anchor_cache=True)
        second=composer(current.detach()+.1,a,use_anchor_cache=True)
        assert torch.equal(first[:,:128],second[:,:128])
        assert not torch.equal(first[:,128:],second[:,128:])
        composer._cached_stable.add_(1)
        with pytest.raises(ValueError,match='modified'):composer(current.detach(),a,use_anchor_cache=True)


def test_explicit_layout_and_real_exact_suffix_forward(llm):
    llm.config.max_position_embeddings=2048
    torch.manual_seed(2)
    composer=WorldPrefixComposer(32,128,rank=8).eval()
    manager=StableAnchorManager();a,_=manager.update(state(),context(state()))
    ids=torch.tensor([[1,7,8,-200,9]])
    with torch.no_grad():
        native=torch.randn(1,513,32)
        x=composer(native,a,use_anchor_cache=True)
        y=composer(native+.2,a,use_anchor_cache=True)
        old=expand_input(llm,ids,x,scope=('s',a.anchor_id),signature=('adapter',),
            source=('s','f1'),carrier_layout=composer.layout)
        new=expand_input(llm,ids,y,scope=old.scope,signature=old.signature,
            source=('s','f2'),carrier_layout=composer.layout)
        assert old.visual_group_spans==(('stable_world',3,131),('dynamic_current',131,516))
        match=exact_lcp(old,new);assert match==131
        cache=PrefixKVCache(16*1024**2);executor=PrefillExecutor(cache)
        executor.generate(llm,old,0,max_new_tokens=3,do_sample=False)
        actual,diag=executor.generate(llm,new,match,max_new_tokens=3,do_sample=False)
        fresh,_=PrefillExecutor(PrefixKVCache(0)).generate(llm,new,0,max_new_tokens=3,do_sample=False)
        assert torch.equal(actual,fresh)
        assert diag['reused_tokens']==131 and diag['effective_prefill_tokens']==386
        assert diag['forward_prefill_lengths']==[386]
        with pytest.raises(ValueError,match='reorder'):
            expand_input(llm,ids,x,carrier_layout=composer.layout,carrier_order=('dynamic_current','stable_world'))
    with pytest.raises(ValueError):CarrierLayout(128,0,385,order_contract='wrcps')
    with pytest.raises(ValueError):WorldPrefixLayout(128,514)


def test_runtime_uses_same_executor_and_revokes_invalid_anchor(llm,tmp_path):
    from worldstream.m3.reuse_planner import ReusePlanner
    llm.config.max_position_embeddings=2048
    torch.manual_seed(3)
    composer=WorldPrefixComposer(32,128,rank=8)
    adapter=tmp_path/'adapter.pt'
    torch.save(dict(layout_version=composer.layout.order_contract,spec=composer.spec,
        state_dict=composer.state_dict(),trained_steps=1,training_scene_ids=['synthetic_test_only'],
        base_weights_sha256='synthetic_weights',data_manifest_sha256='synthetic_manifest'),adapter)
    runtime=M3Runtime(M3Config(prefix_adapter=str(adapter),reuse_strategy='exact_prefix'))
    runtime.planner=ReusePlanner(runtime.scheduler,calibration_threshold=0.)
    ids=[torch.tensor([1,7,8,-200,9])];current=torch.randn(1,513,32)
    def run(s,current):
        ctx=dict(context(s),camera_ids=['front'],weights_sha256='synthetic_weights',stable_state_version=s.state_version)
        return runtime.generate(llm,ids,current,context=ctx,stable_world_state=s,
            layout=CarrierLayout(256,1,256),do_sample=False,max_new_tokens=2)
    _,cold=run(state(),current)
    _,warm=run(state('f2',1.5),current+.2)
    d=warm['requests'][0]
    assert d['reuse_length']==d['reused_tokens']==131
    assert d['executor']=='PrefillExecutor' and d['stable_anchor_hit']
    _,invalid=run(state('f3',2.,facts=()),current+.3)
    assert invalid['requests'][0]['reused_tokens']==0
    assert not invalid['requests'][0]['stable_anchor_valid']
    runtime.reset();assert runtime.stable_anchors.anchor is None and runtime.composer._cached_stable is None


def test_adapter_and_legacy_policy_rejected(tmp_path):
    with pytest.raises(ValueError,match='exact_prefix'):M3Config(prefix_adapter='x')
    p=tmp_path/'untrained.pt';torch.save(dict(layout_version='wrong'),p)
    with pytest.raises(ValueError,match='trained'):WorldPrefixComposer.load_adapter(p,32,'cpu',torch.float32)
    from worldstream.m3.scheduler import Scheduler
    scheduler=Scheduler(M3Config(prefix_adapter='x',reuse_strategy='exact_prefix'))
    scheduler.policy=dict(binding=dict.fromkeys(('model_version','weights_sha256','execution_key','generation_key','layout_signature','dtype'),'same'),block_size=32)
    assert not scheduler.compatible(dict(scheduler.policy['binding'],adapter_sha256='new'))


def test_answer_distillation_backpropagates_only_student():
    from tools.worldstream_m123.wrcps_train import distillation_loss
    logits=torch.randn(4,10,requires_grad=True);teacher=torch.randn(4,10)
    loss,details=distillation_loss(logits,teacher,torch.tensor([1,2,3,4]),critical=True)
    loss.backward();assert logits.grad.abs().sum()>0 and details['critical']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_half_cached_prefix_and_real_suffix(llm):
    llm.config.max_position_embeddings=2048
    llm=llm.half().cuda()
    composer=WorldPrefixComposer(32,128,rank=8).half().cuda().eval()
    manager=StableAnchorManager();a,_=manager.update(state(),context(state()))
    ids=torch.tensor([[1,7,8,-200,9]],device='cuda')
    with torch.no_grad():
        native=torch.randn(1,513,32,device='cuda',dtype=torch.float16)
        x=composer(native,a,use_anchor_cache=True)
        y=composer(native+.25,a,use_anchor_cache=True)
        assert torch.equal(x[:,:128],y[:,:128])
        old=expand_input(llm,ids,x,scope=('s',a.anchor_id),signature=('adapter',),
            source=('s','f1'),carrier_layout=composer.layout)
        new=expand_input(llm,ids,y,scope=old.scope,signature=old.signature,
            source=('s','f2'),carrier_layout=composer.layout)
        match=exact_lcp(old,new);assert match==131
        cache=PrefixKVCache(16*1024**2);executor=PrefillExecutor(cache)
        executor.generate(llm,old,0,max_new_tokens=3,do_sample=False)
        actual,diag=executor.generate(llm,new,match,max_new_tokens=3,do_sample=False,profile=True)
        fresh,_=PrefillExecutor(PrefixKVCache(0)).generate(llm,new,0,max_new_tokens=3,do_sample=False)
        assert torch.equal(actual,fresh)
        assert diag['reused_tokens']==131 and diag['forward_prefill_lengths']==[386]


def test_current_commit_candidates_do_not_become_stable_facts():
    from types import SimpleNamespace
    from worldstream.m3.stable_world import freeze_stable_world
    s=state();ctx=context(s)
    candidate=SimpleNamespace(identity_link_qualified=False,history_observations=5)
    store=SimpleNamespace(anchors={'candidate':candidate},scene_id='s',generation=1,
        commit_count=1,execution_signature='exec')
    manager=SimpleNamespace(store=store,_physical_issuer=SimpleNamespace(qualification_sha256=None),
        get_current_certifications=lambda:SimpleNamespace(scene_id='s',frame_id='f1',timestamp=1.,
            certifications=(),rejected=()))
    snapshot=freeze_stable_world(manager,ctx)
    assert not snapshot.facts
    p=json.loads(snapshot.provenance_json)
    assert p['unqualified_geometric_tracks']==p['multi_observation_candidates']==1
    with pytest.raises(ValueError,match='completed M1 commit'):
        freeze_stable_world(manager,dict(ctx,state_version=2))


def test_training_preflight_rejects_acceptance_and_empty_stability(tmp_path):
    from tools.worldstream_m123.wrcps_train import validate_manifest
    from tools.worldstream_m123.benchmark.common import sha256
    from nuscenes.utils.splits import create_splits_scenes
    table=tmp_path/'scene.json'
    table.write_text(json.dumps([dict(token='s',name=create_splits_scenes()['train'][0])]))
    capture=tmp_path/'capture.pt';capture.write_bytes(b'synthetic unit fixture only')
    manifest=dict(scene_splits={'s':'train','v':'validation','a':'independent_acceptance'},
        nuscenes_scene_table=str(table),samples=[dict(scene_id='a',frame_id='f1')])
    with pytest.raises(ValueError,match='acceptance'):
        validate_manifest(manifest)
    manifest['samples']=[dict(scene_id='s',frame_id='f1',capture=str(capture),
        capture_sha256=sha256(capture),stable_snapshot=asdict(state(facts=())))]
    with pytest.raises(ValueError,match='No qualified stable facts'):
        validate_manifest(manifest)
