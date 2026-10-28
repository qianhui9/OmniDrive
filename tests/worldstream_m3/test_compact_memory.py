from dataclasses import replace
import pytest
import torch

from worldstream.m3.visual_memory import VisualMemoryManager, VisualMemoryProjection
from worldstream.m3.world_prefix_composer import CompactVisualComposer, CompactMemoryLayout
from worldstream.m3.visual_feature_bridge import RegionFeature


def inputs(frame='f1', timestamp=1.):
    context = dict(scene_id='s',frame_id=frame,timestamp=timestamp,state_epoch=1,
        state_version=int(timestamp*2),model_sha256='model',adapter_sha256='adapter',memory_version='v1')
    region = RegionFeature('objectA','car','front',frame,'geometry-v1',(0,0,128,128),
        tuple(range(80)),torch.arange(80*8).float().reshape(80,8),80,0)
    return context,region


def test_eight_unique_observations_from_fifteen_and_sixteen_rejected():
    ctx, region = inputs()
    region = replace(region, grid_indices=tuple(range(15)), features=region.features[:15])
    memory, _ = VisualMemoryManager(8).update(ctx, [region], ['objectA'], 'q')
    assert len(set(memory.grid_indices)) == 8
    for value, grid in zip(memory.features, memory.grid_indices):
        assert torch.equal(value, region.features[grid])
    assert VisualMemoryManager(16).update(ctx, [region], ['objectA'], 'q')[0] is None
    bad = replace(region, grid_indices=(0,)*15)
    assert VisualMemoryManager(8).update(ctx, [bad], ['objectA'], 'q')[0] is None


def test_lazy_keep_does_not_read_features_or_hash_again(monkeypatch):
    ctx, region = inputs()
    manager = VisualMemoryManager(8)
    memory, _ = manager.update(ctx, [region], ['objectA'], 'q')
    ctx2, region2 = inputs('f2', 1.5)
    def forbidden(*args):
        raise AssertionError('Keep must not gather or hash features')
    monkeypatch.setattr(type(region2), 'read_features', forbidden)
    import worldstream.m3.visual_memory as module
    monkeypatch.setattr(module.hashlib, 'sha256', forbidden)
    kept, event = manager.update(ctx2, [region2], ['objectA'], 'q')
    assert kept is memory and event == 'keep'
    assert 'gpu_sync_d2h_ms' not in manager.last_timings
    assert manager.update(ctx2, [], [], 'q')[0] is None


def test_fixed_budget_untouched_slots_and_removed_evidence():
    ctx, region = inputs()
    memory, _ = VisualMemoryManager(8).update(ctx, [region], ['objectA'], 'q')
    composer = CompactVisualComposer(16, 8, 8).eval()
    native = torch.randn(1, 513, 16)
    assert composer(native, None) is native
    compact = composer(native, memory)
    assert compact.shape == native.shape
    assert torch.equal(compact[:, 260:261], native[:, 256:257])
    for group, start, stop, outstart in ((composer.objects, 0, 256, 8), (composer.maps, 257, 513, 261)):
        result = compact[:, outstart:outstart+252]
        original = native[:, start:stop]
        for j, i in enumerate(group.retained.tolist()):
            if i not in group.destination:
                assert torch.equal(result[:, j], original[:, i])
        changed = native.clone(); changed[:, start+int(group.source[0])] += 1
        assert not torch.equal(composer(changed, memory)[:, outstart:outstart+252], result)


def test_actual_compact_prefix_and_revocation(llm):
    from worldstream.m3.visual_state import expand_input
    from worldstream.m3.prefix_matcher import exact_lcp
    ctx, region = inputs(); manager = VisualMemoryManager(8)
    memory, _ = manager.update(ctx, [region], ['objectA'], 'q')
    composer = CompactVisualComposer(32, 8, 8).eval()
    ids = torch.tensor([1, 7, -200, 9])
    with torch.no_grad():
        one = composer(torch.randn(1, 513, 32), memory)
        two = composer(torch.randn(1, 513, 32), memory)
        a = expand_input(llm, ids, one, source=('s','f1'), carrier_layout=composer.layout, stable_anchor=('rev1',True))
        b = expand_input(llm, ids, two, source=('s','f2'), carrier_layout=composer.layout, stable_anchor=('rev1',True))
    assert exact_lcp(a,b) == 10
    b.stable_anchor = ('rev2',True)
    assert exact_lcp(a,b) == 2


def test_projected_tensor_mutation_rejected():
    ctx, region = inputs(); memory, _ = VisualMemoryManager(8).update(ctx,[region],['objectA'],'q')
    projection = VisualMemoryProjection(8,16).eval()
    embedding = projection(memory)
    with torch.no_grad(): embedding.add_(1)
    with pytest.raises(ValueError,match='projected embedding'):
        projection(memory)


def test_untrained_artifact_cannot_deploy(tmp_path):
    path = tmp_path/'untrained.pt'; torch.save(dict(trained_steps=0), path)
    with pytest.raises(ValueError,match='cost/training/quality'):
        CompactVisualComposer.load_adapter(path,32,'cpu')


@pytest.mark.parametrize('field,value', [('content_sha256','forged'),('grid_indices',tuple(range(8))),
    ('scope',('other_scene',)),('semantic_class','pedestrian')])
def test_provenance_tampering_is_rejected(field,value):
    ctx, region=inputs();manager=VisualMemoryManager(8)
    memory,_=manager.update(ctx,[region],['objectA'],'q')
    setattr(memory,field,value)
    with pytest.raises(ValueError,match='provenance mutated'):
        manager.update(ctx,[region],['objectA'],'q')


def test_tensor_replacement_and_failed_cost_gate(tmp_path):
    import json
    from tools.worldstream_m123.compact_visual_memory import assert_training_ready
    ctx,region=inputs();manager=VisualMemoryManager(8)
    memory,_=manager.update(ctx,[region],['objectA'],'q')
    memory.features=memory.features.clone()
    with pytest.raises(ValueError,match='mutated'):
        VisualMemoryProjection(8,16).eval()(memory)
    (tmp_path/'feasibility_decision.json').write_text(json.dumps(dict(status='NO_GO')))
    with pytest.raises(ValueError,match='training refused'):
        assert_training_ready(tmp_path)


def test_no_memory_runtime_is_native(llm):
    from worldstream.m3.runtime import M3Runtime
    from worldstream.m3.config import M3Config
    from worldstream.m3.visual_state import CarrierLayout
    from worldstream.m3.stable_world import StableWorldSnapshot
    from types import SimpleNamespace
    runtime=M3Runtime(M3Config(prefix_layout='world_compact_memory',reuse_strategy='exact_prefix'))
    runtime.semantic_qualifier=SimpleNamespace(sha256='q',implementation_signature=lambda:'q')
    context=dict(scene_id='s',frame_id='f',timestamp=1.,state_epoch=1,state_version=1,
                 stable_state_version=1,state_execution_signature='exec',camera_ids=['front'],model_version='m')
    snapshot=StableWorldSnapshot('s','f',1.,1,1,'exec',(),'{}',qualification_id='q')
    frame=SimpleNamespace(scene_id='s',frame_id='f',timestamp=1.,cameras={'front':None})
    current=torch.randn(1,513,32); ids=torch.tensor([1,7,-200,9])
    kwargs=dict(context=context,questions=['q'],layout=CarrierLayout(256,1,256),max_new_tokens=2,do_sample=False)
    with torch.no_grad():
        native,_=M3Runtime(M3Config(reuse_strategy='exact_prefix')).generate(llm,[ids],current,**kwargs)
        compact,_=runtime.generate(llm,[ids],current,stable_world_state=snapshot,
            visual_memory_inputs=dict(frame=frame,anchors=[],feature_map=torch.zeros(1,1,8,2,2),config=None),**kwargs)
    assert runtime.composer is None
    assert torch.equal(native[0],compact[0])
