import pytest
import torch
from worldstream.m2 import IncrementalVisionRuntime, M2Config
from worldstream.m2.packing import partition, window_mask
from worldstream.m2.window_executor import WindowExecutor


@pytest.mark.parametrize('position', [(0, 0), (3, 4), (9, 9)])
@torch.no_grad()
def test_real_block_padding_rope(eva, position):
    block = eva.blocks[0]
    # Nonzero bias catches norm-before-pad regressions.
    block.norm1.bias.fill_(.3)
    x = torch.randn(2, 10, 10, 32)
    old = block(x)
    cache, _ = partition(old, 4)
    changed = torch.zeros(2, 10, 10, dtype=torch.bool)
    changed[0, position[0], position[1]] = True
    current = x.clone(); current[changed] += .2
    ids = window_mask(changed, 4).nonzero().flatten()
    executor = WindowExecutor()
    actual, _ = executor.execute(block, current, cache, ids)
    assert torch.allclose(actual, block(current), atol=2e-6, rtol=2e-6)
    naive, _ = executor.execute(block, current, cache, ids, naive=True)
    assert torch.allclose(actual, naive, atol=2e-6, rtol=2e-6)


@torch.no_grad()
def test_window_permutation(eva):
    x = torch.randn(2, 10, 10, 32)
    cache, _ = partition(eva.blocks[0](x), 4)
    changed = x.clone(); changed[0, 0, 0] += .1; changed[1, 9, 9] -= .2
    ex = WindowExecutor()
    a, _ = ex.execute(eva.blocks[0], changed, cache, torch.tensor([0, 17]))
    b, _ = ex.execute(eva.blocks[0], changed, cache, torch.tensor([17, 0]))
    assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)
    assert torch.allclose(a, eva.blocks[0](changed), atol=2e-6, rtol=2e-6)


@torch.no_grad()
def test_runtime_sequence_global_barrier(eva, context):
    runtime = IncrementalVisionRuntime(eva, M2Config(mode='exact'))
    x = torch.randn(2, 3, 40, 40)
    for i in range(5):
        if i:
            x = x.clone(); x[0, :, i, i] += .1
        y, d = runtime.forward(x, execution_context=dict(context, frame_id=str(i)))
        assert torch.allclose(y[0], eva(x)[0], atol=3e-6, rtol=3e-6)
        assert d['cache_bytes'] <= runtime.cache.budget_bytes
        if i:
            assert d['actual_skipped_windows'] == 34
            assert d['stages'][2]['mode'] == d['stages'][3]['mode'] == 'full'


@pytest.mark.parametrize('change,reason', [
    ({'scene_id': 'other'}, 'scene_change'),
    ({'camera_ids': ['back', 'front']}, 'execution_signature_mismatch'),
    ({'execution_signature': 'other'}, 'execution_signature_mismatch'),
    ({'source_version': -1}, 'cache_version_mismatch'),
    ({'source_frame_id': 'wrong'}, 'cache_version_mismatch'),
    ({'frame_id': '0'}, 'frame_order_mismatch')])
@torch.no_grad()
def test_cache_invalidates(eva, context, change, reason):
    runtime = IncrementalVisionRuntime(eva, dict(mode='exact'))
    x = torch.randn(2, 3, 40, 40)
    runtime.forward(x, execution_context=context)
    _, d = runtime.forward(x, execution_context=dict(context, frame_id='1', **change)
                           if 'frame_id' not in change else dict(context, **change))
    assert d['stages'][0]['fallback_reason'] == reason
    assert not d['actual_skipped_windows']


@torch.no_grad()
def test_weight_and_buffer_change(eva, context):
    runtime = IncrementalVisionRuntime(eva, dict(mode='exact'))
    x = torch.randn(2, 3, 40, 40)
    runtime.forward(x, execution_context=context)
    eva.blocks[0].attn.q_proj.weight.add_(.01)
    y, d = runtime.forward(x, execution_context=dict(context, frame_id='1'))
    assert d['stages'][0]['fallback_reason'] == 'execution_signature_mismatch'
    assert torch.equal(y[0], eva(x)[0])


@torch.no_grad()
def test_rope_buffer_change(eva, context):
    runtime=IncrementalVisionRuntime(eva,dict(mode='exact'))
    x=torch.randn(2,3,40,40)
    runtime.forward(x,execution_context=context)
    eva.rope_win.freqs_cos.add_(.01)
    y,d=runtime.forward(x,execution_context=dict(context,frame_id='1'))
    assert d['stages'][0]['fallback_reason']=='execution_signature_mismatch'
    assert torch.equal(y[0],eva(x)[0])


@torch.no_grad()
def test_sparse_oom_dense_retry(eva, context, monkeypatch):
    runtime=IncrementalVisionRuntime(eva,dict(mode='exact'))
    x=torch.randn(2,3,40,40)
    runtime.forward(x,execution_context=context)
    def oom(*a,**kw): raise torch.cuda.OutOfMemoryError('simulated allocation exhaustion')
    monkeypatch.setattr(runtime.executor,'execute',oom)
    y,d=runtime.forward(x,execution_context=dict(context,frame_id='1'))
    assert d['fallback_reason']=='memory_budget' and not runtime.cache.bytes
    assert torch.equal(y[0],eva(x)[0])


@torch.no_grad()
def test_bounded_lifetime(eva, context):
    runtime=IncrementalVisionRuntime(eva,dict(mode='exact'))
    x=torch.randn(2,3,40,40)
    sizes=[]
    for i in range(25):
        runtime.forward(x,execution_context=dict(context,frame_id=str(i)))
        sizes.append(runtime.cache.bytes)
        assert set(runtime.cache.outputs)=={0,1}
    assert len(set(sizes))==1
    runtime.reset()
    assert runtime.cache.bytes==0 and runtime.executor.buffer.tensor is None


@torch.no_grad()
def test_disabled_and_memory_budget(eva, context):
    x = torch.randn(2, 3, 40, 40)
    for config, reason in [(dict(enabled=False), 'disabled'), (dict(memory_budget_mb=0), 'memory_budget')]:
        r = IncrementalVisionRuntime(eva, config)
        y, d = r.forward(x, execution_context=context)
        assert torch.equal(y[0], eva(x)[0]) and d['fallback_reason'] == reason
        assert r.cache.bytes == 0


@torch.no_grad()
def test_shape_change(eva, context):
    # Different camera batch size keeps fixed-length global RoPE valid.
    r = IncrementalVisionRuntime(eva, dict(mode='exact'))
    r.forward(torch.randn(2, 3, 40, 40), execution_context=context)
    x = torch.randn(1, 3, 40, 40)
    y, d = r.forward(x, execution_context=dict(context, frame_id='1', camera_ids=['front']))
    assert d['stages'][0]['fallback_reason'] == 'shape_unsupported'
    assert torch.equal(y[0], eva(x)[0])


@torch.no_grad()
def test_auto_without_measurement(eva, context):
    r = IncrementalVisionRuntime(eva)
    x = torch.randn(2, 3, 40, 40)
    r.forward(x, execution_context=context)
    _, d = r.forward(x, execution_context=dict(context, frame_id='1'))
    assert d['fallback_reason'] == 'unprofiled_cost'
    assert not d['cache_saved']
    assert not d['actual_skipped_windows']


@torch.no_grad()
def test_topology_change_to_global(eva, context):
    r=IncrementalVisionRuntime(eva,dict(mode='exact'))
    x=torch.randn(2,3,40,40)
    r.forward(x,execution_context=context)
    eva.blocks[0].window_size=0
    eva.blocks[0].attn.rope=eva.rope_glb
    y,d=r.forward(x,execution_context=dict(context,frame_id='1'))
    assert d['fallback_reason']=='global_dependency'
    assert torch.equal(y[0],eva(x)[0]) and not r.cache.bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@torch.no_grad()
def test_cuda_real_eva(eva, context):
    eva = eva.cuda().half()
    # Real reference attention here; FlashAttention is separately profiled at production size.
    r = IncrementalVisionRuntime(eva, dict(mode='exact'))
    x = torch.randn(2, 3, 40, 40, device='cuda', dtype=torch.float16)
    r.forward(x, execution_context=context)
    x[0, :, 39, 39] += .1
    y, d = r.forward(x, execution_context=dict(context, frame_id='1'))
    assert torch.allclose(y[0], eva(x)[0], atol=.01, rtol=.01)
    assert d['actual_skipped_windows'] == 34
