from dataclasses import replace
import pytest
import torch

from worldstream.m3.visual_feature_bridge import RegionFeature
from worldstream.m3.visual_memory import VisualMemoryManager, VisualMemoryProjection


def inputs(frame='f1', timestamp=1., **changes):
    context = dict(scene_id='s',frame_id=frame,timestamp=timestamp,state_epoch=1,
        state_version=int(timestamp*2),model_sha256='model',adapter_sha256='adapter',memory_version='v1')
    context.update(changes)
    region = RegionFeature('objectA','car','front',frame,'geometry-v1',(0,0,128,128),
        tuple(range(80)),torch.arange(80*8).float().reshape(80,8),80,0)
    return context,region


def test_actual_historical_feature_and_embedding_pointer_retention():
    manager=VisualMemoryManager();projection=VisualMemoryProjection(8,16).eval()
    ctx,region=inputs(); memory,event=manager.update(ctx,[region],['objectA'],'qualification')
    assert event=='create:cold' and len(memory.features)==64
    embedding=projection(memory);old=memory.features.clone()
    ctx,current=inputs('f2',1.5);current.features+=300
    kept,event=manager.update(ctx,[current],['objectA'],'qualification')
    assert kept is memory and event=='keep' and kept.age==1
    assert projection(kept) is embedding and torch.equal(kept.features,old)
    assert kept.generating_frame=='f1' and kept.latest_frame=='f2'
    native=torch.randn(1,513,16);composed=projection.compose(native,kept)
    assert composed.shape==(1,577,16) and torch.equal(composed[:,64:],native)
    assert projection.compose(native,None) is native


@pytest.mark.parametrize('field,value', [('scene_id','new'),('state_epoch',2),
    ('model_sha256','new'),('adapter_sha256','new'),('memory_version','new')])
def test_execution_changes_replace_and_revoke_previous_memory(field,value):
    manager=VisualMemoryManager();ctx,region=inputs()
    old,_=manager.update(ctx,[region],['objectA'],'qualification')
    ctx,current=inputs('f2',1.5,**{field:value})
    memory,event=manager.update(ctx,[current],['objectA'],'qualification')
    assert memory is not old and memory.revision>old.revision and old.invalidation_reason
    assert event.startswith('create:')


def test_revocation_cannot_resurrect_identical_visual_kv(llm):
    from worldstream.m3.visual_state import expand_input
    from worldstream.m3.prefix_matcher import exact_lcp
    manager=VisualMemoryManager();ctx,region=inputs();old,_=manager.update(ctx,[region],['objectA'],'qualification')
    ctx2,_=inputs('f2',1.5);assert manager.update(ctx2,[],[],'qualification')[0] is None
    ctx3,current=inputs('f3',2.);new,_=manager.update(ctx3,[current],['objectA'],'qualification')
    assert torch.equal(old.features,new.features) and old.revision!=new.revision
    # Even identical embedding values cannot inherit a revoked memory's KV.
    images=torch.ones(1,64,32);ids=torch.tensor([1,7,-200,9])
    a=expand_input(llm,ids,images,source=('s','f1'),stable_anchor=(str(old.revision),True))
    b=expand_input(llm,ids,images,source=('s','f3'),stable_anchor=(str(new.revision),True))
    a.visual_group_spans=b.visual_group_spans=(('stable',2,66),)
    assert exact_lcp(a,b)==2


def test_source_ids_never_enter_learned_projection():
    ctx,a=inputs();b=replace(a,anchor_id='unrelated-world-ID')
    one,_=VisualMemoryManager().update(ctx,[a],[a.anchor_id],'q')
    two,_=VisualMemoryManager().update(ctx,[b],[b.anchor_id],'q')
    projection=VisualMemoryProjection(8,16).eval()
    assert torch.equal(projection(one),projection(two))


def test_no_repeat_expansion_mutation_guard_and_stale_observation():
    ctx,a=inputs();manager=VisualMemoryManager(128)
    assert manager.update(ctx,[a],[a.anchor_id],'q')[0] is None
    manager=VisualMemoryManager();old,_=manager.update(ctx,[a],[a.anchor_id],'q')
    ctx2,_=inputs('f2',1.5)
    assert manager.update(ctx2,[a],[a.anchor_id],'q')[0] is None
    manager=VisualMemoryManager();old,_=manager.update(ctx,[a],[a.anchor_id],'q')
    old.features.add_(1)
    with pytest.raises(ValueError,match='mutated'):
        manager.update(ctx2,[],[],'q')


def test_horizon_and_projection_parameter_changes():
    manager=VisualMemoryManager(max_age_frames=2);projection=VisualMemoryProjection(8,16).eval()
    ctx,a=inputs();old,_=manager.update(ctx,[a],['objectA'],'q');first=projection(old)
    with torch.no_grad():projection.projection.bias.add_(1)
    assert not torch.equal(first,projection(old))
    ctx,a=inputs('f2',1.5);assert manager.update(ctx,[a],['objectA'],'q')[1]=='keep'
    ctx,a=inputs('f3',2.);new,event=manager.update(ctx,[a],['objectA'],'q')
    assert new.revision!=old.revision and event=='create:memory_horizon_expired'


def test_failed_cost_gate_refuses_training_before_any_artifact(tmp_path):
    import json
    from tools.worldstream_m123.world_visual_memory_report import assert_training_ready
    for name in ('bridge_analysis.json','memory_semantic_probe.json'):
        (tmp_path/name).write_text(json.dumps(dict(status='PASS')))
    (tmp_path/'prefix_cost_feasibility.json').write_text(json.dumps(dict(status='NO_GO')))
    before=set(tmp_path.iterdir())
    with pytest.raises(ValueError,match='training refused'):
        assert_training_ready(tmp_path)
    assert set(tmp_path.iterdir())==before
