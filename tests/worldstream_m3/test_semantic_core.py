"""Synthetic invariants only; all research evidence comes from actual captures."""
from dataclasses import asdict,replace
import json
import pytest
import torch
from worldstream.m3.stable_world import StableWorldSnapshot,StableAnchorManager
from worldstream.m3.semantic_core import SemanticQualifier,QUALIFICATION_VERSION,SEMANTIC_CLAIM
from worldstream.m3.world_prefix_composer import WorldPrefixComposer


def snapshot(frame,facts,qualification='q'):
    return StableWorldSnapshot('scene','f'+str(frame),float(frame)*.5,1,frame,'native',
        tuple(sorted(facts)),'{}',qualification_id=qualification)


def context(s):
    return dict(scene_id=s.scene_id,frame_id=s.frame_id,timestamp=s.timestamp,state_epoch=s.state_epoch,
        state_version=s.state_version,state_execution_signature=s.execution_signature)


def test_unrelated_new_object_does_not_replace_immutable_core():
    manager=StableAnchorManager();fact=('a','object','car')
    s=snapshot(1,[fact]);a,_=manager.update(s,context(s))
    s=snapshot(2,[fact,('b','object','pedestrian')]);b,hit=manager.update(s,context(s))
    assert hit and b.anchor_id==a.anchor_id and b.facts==(fact,)
    s=snapshot(3,[('b','object','pedestrian')]);c,hit=manager.update(s,context(s))
    assert not hit and c.anchor_id!=a.anchor_id
    assert c.facts==(('b','object','pedestrian'),)


def test_class_change_and_qualification_revision_revoke_anchor():
    manager=StableAnchorManager();s=snapshot(1,[('a','object','car')]);a,_=manager.update(s,context(s))
    s=snapshot(2,[('a','object','truck')]);b,hit=manager.update(s,context(s))
    assert not hit and b.anchor_id!=a.anchor_id
    s=snapshot(3,b.facts,'new_policy');c,hit=manager.update(s,context(s))
    assert not hit and c.update_reason=='semantic_qualification_changed'


def test_opaque_tracking_id_does_not_enter_stable_semantic_encoder():
    manager=StableAnchorManager();s=snapshot(1,[('random_id_123','object','car')]);a,_=manager.update(s,context(s))
    s=snapshot(2,[('completely_different_uuid','object','car')]);b,_=manager.update(s,context(s))
    assert a.anchor_id!=b.anchor_id
    torch.manual_seed(7);composer=WorldPrefixComposer(32,128,rank=8).eval()
    with torch.no_grad():assert torch.equal(composer.encode_stable(a),composer.encode_stable(b))


def policy():
    metrics=dict(scored=50,scenes=3,wrong=0,unknown_fraction=0.)
    return dict(version=QUALIFICATION_VERSION,claim=SEMANTIC_CLAIM,protocol_signature='native_protocol',
        selector_implementation=SemanticQualifier.implementation_signature(),independent_validation_completed=True,
        calibration_scene_ids=['c1','c2','c3'],validation_scene_ids=['v1','v2','v3'],
        calibration_log_ids=['lc1','lc2','lc3'],validation_log_ids=['lv1','lv2','lv3'],
        qualification_limits=dict(min_scored_edges=30,min_scenes=3,max_observed_wrong_edges=0,max_unknown_fraction=.05),
        rules=dict(car=dict(admitted=True,min_observations=3,min_score=.8,max_residual_m=.3,
            calibration=metrics,validation=metrics)))


def test_qualifier_claim_scope_and_current_observation_only(tmp_path):
    path=tmp_path/'policy.json';path.write_text(json.dumps(policy()));q=SemanticQualifier(path)
    e=dict(anchor_id='a',anchor_type='object',semantic_class='car',source_frame='f1',current_frame='f2',
        source_timestamp=1.,timestamp=1.5,consecutive_observations=3,score=.9,residual_m=.1,
        margin_m=None,no_competing_edge=True,observed_camera_count=1)
    record=dict(anchor_id='a',anchor_type='object',semantic_class='car',score=.9,evidence=e)
    candidates=dict(protocol_signature='native_protocol',context=dict(timestamp=1.5),records=[record])
    admitted=q.select(candidates)
    assert admitted['facts']==(('a','object','car'),) and not admitted['physical_certification']
    record['evidence']=None
    assert not q.select(candidates)['facts']
    record['evidence']=dict(e,source_timestamp=.1)
    assert q.select(candidates)['rejected']['a']=='stale_or_invalid_source'
    candidates['protocol_signature']='changed'
    with pytest.raises(ValueError,match='protocol'):q.select(candidates)


@pytest.mark.parametrize('change', ['overlap','unsupported','wrong','implementation'])
def test_qualification_cannot_be_boolean_relabeling(tmp_path,change):
    p=policy()
    if change=='overlap':p['validation_log_ids'][0]=p['calibration_log_ids'][0]
    elif change=='unsupported':p['rules']['car']['validation']['scored']=1
    elif change=='wrong':p['rules']['car']['validation']['wrong']=1
    else:p['selector_implementation']='other'
    path=tmp_path/'p.json';path.write_text(json.dumps(p))
    with pytest.raises(ValueError):SemanticQualifier(path)


def test_training_assembly_matches_real_fp32_carrier_to_half_llm_boundary(llm):
    from tools.worldstream_m123.wrcps_train import multimodal_embeddings
    llm.half()
    native=torch.randn(1,513,32,requires_grad=True)
    ids=torch.tensor([1,7,-200,9]);answer=torch.tensor([[3,4]])
    assembled,start,labels=multimodal_embeddings(llm,ids,native,answer)
    assert assembled.dtype==torch.float16 and start==515 and labels.tolist()==[3,4]
    assembled.float().square().sum().backward()
    assert native.grad is not None and (native.grad.abs().sum(-1)>0).all()


def test_crossing_critical_review_does_not_treat_waiting_as_crossing():
    from tools.worldstream_m123.wrcps_semantic_quality import critical_motion_claims
    q='What are the pedestrians doing?'
    assert ('pedestrian_crossing',False) in critical_motion_claims('No pedestrians are crossing the road.',q)
    assert ('pedestrian_crossing',True) in critical_motion_claims('The pedestrians are crossing the road.',q)
    assert ('pedestrian_crossing',True) not in critical_motion_claims('The pedestrians are waiting.',q)


def test_revoked_core_preserves_only_real_fixed_text_kv(llm):
    from worldstream.m3.visual_state import expand_input
    from worldstream.m3.prefix_matcher import exact_lcp,prefix_statistics
    from worldstream.m3.prefix_kv_cache import PrefixKVCache
    from worldstream.m3.prefill_executor import PrefillExecutor
    llm.config.max_position_embeddings=2048
    composer=WorldPrefixComposer(32,128,rank=8).eval()
    ids=torch.tensor([[1]+[7]*34+[-200,9]])
    vision=torch.randn(1,513,32)
    kwargs=dict(scope=('scene','camera',1),signature=('same_model',),carrier_layout=composer.layout)
    with torch.no_grad():
        old=expand_input(llm,ids,vision,source=('scene','f1'),stable_anchor=('invalid1',False),**kwargs)
        new=expand_input(llm,ids,vision+.1,source=('scene','f2'),stable_anchor=('invalid2',False),**kwargs)
        assert exact_lcp(old,new)==prefix_statistics(old,new,include_blocks=False)[0]==35
        same_frame=expand_input(llm,ids,vision,source=('scene','f1'),stable_anchor=('invalid1',False),**kwargs)
        assert exact_lcp(old,same_frame)==old.length
        # Even IDENTICAL stable embeddings cannot carry a revoked visual core.
        forged_equal=expand_input(llm,ids,vision,source=('scene','f2'),stable_anchor=('invalid1',False),**kwargs)
        assert exact_lcp(old,forged_equal)==35
        cache=PrefixKVCache(16*1024**2);executor=PrefillExecutor(cache)
        executor.generate(llm,old,0,max_new_tokens=2,do_sample=False)
        actual,diag=executor.generate(llm,new,35,max_new_tokens=2,do_sample=False)
        fresh,_=PrefillExecutor(PrefixKVCache(0)).generate(llm,new,0,max_new_tokens=2,do_sample=False)
        assert torch.equal(actual,fresh) and diag['forward_prefill_lengths']==[new.length-35]
        cache.entry.expanded.stable_anchor=('tampered',True)
        assert cache.validate_storage() is None
