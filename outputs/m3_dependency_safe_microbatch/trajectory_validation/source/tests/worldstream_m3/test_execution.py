from dataclasses import replace
import pytest
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from worldstream.m3 import M3Runtime, expand_input, CarrierLayout
from worldstream.m3.visual_state import execution_signature
from worldstream.m3.carrier_selector import select_carriers
from worldstream.m3.prefix_matcher import exact_lcp
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from worldstream.m3.prefill_executor import PrefillExecutor


def expanded(llm, carriers, ids=None, **kw):
    return expand_input(llm, torch.tensor([[1, 3, -200, 4, 5]]) if ids is None else ids,
                        carriers, scope=('s',), signature=('weights',), **kw)


def test_native_expansion(llm, carriers):
    x = expanded(llm, carriers)
    assert x.length == 13
    assert torch.equal(x.embeddings[:, 2:11], carriers)
    assert x.position_ids.tolist() == [list(range(13))]
    assert x.visual_mask.sum() == 9


def test_multiple_images_and_missing_image(llm, carriers):
    x = expanded(llm, carriers.repeat(2, 1, 1), torch.tensor([[1, -200, 3, -200, 5]]))
    assert x.length == 21 and x.visual_mask.sum() == 18
    no_image = expanded(llm, carriers, torch.tensor([[1, 3, 4]]))
    assert no_image.length == 3 and no_image.visual_mask.sum() == 0
    with pytest.raises(IndexError):
        expanded(llm, carriers, torch.tensor([[1, -200, -200]]))


def test_padding_native_position_semantics(llm, carriers):
    ids = torch.tensor([[0, 1, -200, 4, 0]])
    x = expanded(llm, carriers, ids, attention_mask=torch.tensor([[0, 1, 1, 1, 0]]),
                 position_ids=torch.tensor([[8, 9, 10, 11, 12]]))
    assert x.length == 11
    assert x.attention_mask.all()
    # Native OmniDrive ignores supplied absolute positions during image expansion.
    assert x.position_ids.tolist() == [list(range(11))]


def test_text_only_padding_and_nonfinite(llm, carriers):
    x = expand_input(llm, torch.tensor([[0, 1, 3]]), None, attention_mask=torch.tensor([[0, 1, 1]]))
    assert x.position_ids.tolist() == [[1, 0, 1]]
    carriers[:, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        expanded(llm, carriers)


@pytest.mark.parametrize('field', ['position_ids', 'attention_mask', 'token_ids', 'visual_mask'])
def test_lcp_context_invalidates(llm, carriers, field):
    a = expanded(llm, carriers)
    b = a.prefix(a.length, clone=True)
    getattr(b, field)[0, 3] = 0 if field in ('attention_mask', 'visual_mask') else 33
    assert exact_lcp(a, b) == 3
    assert exact_lcp(a, replace(b, scope=('other_scene',))) == 0
    assert exact_lcp(a, replace(b, signature=('new_lora',))) == 0


def test_visual_change_only_text_prefix_reusable(llm, carriers):
    a = expanded(llm, carriers)
    changed = carriers.clone(); changed[0, 0, 0] += 1
    assert exact_lcp(a, expanded(llm, changed)) == 2


@pytest.mark.parametrize('sample', [False, True])
def test_greedy_sampling_tokens_logits_and_no_prefill_replay(llm, carriers, context, sample):
    runtime = M3Runtime(dict(mode='exact', offline_experiment=True, min_prefix_tokens=2))
    questions = [torch.tensor([1, 3, -200, 4, 5]), torch.tensor([1, 3, -200, 7, 6, 8])]
    kw = dict(do_sample=sample, max_new_tokens=6, return_dict_in_generate=True, output_scores=True)
    if sample:
        kw.update(temperature=.7, top_p=.75, top_k=12)
    torch.manual_seed(13)
    reference = [llm.generate(inputs=x[None], images=carriers, **kw) for x in questions]
    torch.manual_seed(13)
    outputs, diagnostics = runtime.generate(llm, questions, carriers, context=context, **kw)
    for a, b in zip(reference, outputs):
        assert torch.equal(a.sequences, b.sequences)
        for s, t in zip(a.scores, b.scores):
            torch.testing.assert_close(s, t, rtol=1e-5, atol=1e-6)
    second = diagnostics['requests'][1]
    assert second['reused_tokens'] == 11
    assert second['forward_prefill_lengths'] == [3]


def test_identical_prompt_empty_suffix_and_cache_immutable(llm, carriers):
    cache = PrefixKVCache(1024**2)
    executor = PrefillExecutor(cache)
    x = expanded(llm, carriers)
    answer, _ = executor.generate(llm, x, max_new_tokens=4, do_sample=False)
    snapshot = tuple(tuple(t.clone() for t in layer) for layer in cache.entry.kv)
    other, diag = executor.generate(llm, x, x.length, max_new_tokens=4, do_sample=False)
    assert torch.equal(answer, other)
    assert diag['forward_prefill_lengths'] == []
    for a, b in zip(snapshot, cache.entry.kv):
        for s, t in zip(a, b):
            assert torch.equal(s, t)
    assert cache.entry.computed_source == x.source


def test_shorter_prompt_needs_last_token_logits(llm, carriers):
    cache = PrefixKVCache(1024**2); executor = PrefillExecutor(cache)
    long = expanded(llm, carriers)
    executor.generate(llm, long, max_new_tokens=2)
    short = long.prefix(long.length - 1)
    _, diag = executor.generate(llm, short, short.length, max_new_tokens=2)
    assert diag['forward_prefill_lengths'] == [1]


def test_memory_budget(llm, carriers):
    cache = PrefixKVCache(1)
    _, diag = PrefillExecutor(cache).generate(llm, expanded(llm, carriers), max_new_tokens=2)
    assert not diag['cache_stored'] and cache.entry is None


def test_signature_model_and_lora_change(llm):
    before = execution_signature(llm)
    with torch.no_grad():
        llm.model.embed_tokens.weight.add_(.01)
    assert before != execution_signature(llm)


def test_instance_backend_change_invalidates_signature(llm):
    from functools import wraps
    before=execution_signature(llm)
    original=llm.model.layers[0].self_attn.forward
    @wraps(original)
    def replacement(*a,**kw):return original(*a,**kw)
    llm.model.layers[0].self_attn.forward=replacement
    assert before!=execution_signature(llm)


def test_cross_frame_budget_failure_after_legal_reuse(llm,carriers):
    cache=PrefixKVCache(1024**2);executor=PrefillExecutor(cache)
    first=expanded(llm,carriers,source=('s','frame0'))
    executor.generate(llm,first,max_new_tokens=1)
    cache.budget_bytes=cache.nbytes
    longer=expanded(llm,carriers,torch.tensor([[1,3,-200,4,5,6,7,8]]),source=('s','frame1'))
    answer,diag=executor.generate(llm,longer,cache.match(longer),max_new_tokens=2)
    assert answer.shape[1]>=2 and cache.entry is None
    assert not diag['cache_stored'] and diag['branch_workspace_bytes']>0


def test_real_peft_adapter_isolation(llm, carriers):
    from peft import get_peft_model, LoraConfig
    wrapped = get_peft_model(llm, LoraConfig(r=2, lora_alpha=2,
        target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')).eval()
    signature = execution_signature(wrapped)
    with wrapped.disable_adapter():
        assert execution_signature(wrapped) != signature
    assert execution_signature(wrapped) == signature
    x = expand_input(wrapped, torch.tensor([[1, -200, 3]]), carriers)
    assert x.length == 11


def test_beam_search_stays_native(llm, carriers, context):
    ids = torch.tensor([1, -200, 4])
    expected = llm.generate(inputs=ids[None], images=carriers, num_beams=2, max_new_tokens=3)
    actual, diag = M3Runtime(dict(mode='exact', offline_experiment=True)).generate(
        llm, [ids], carriers, context=context, num_beams=2, max_new_tokens=3)
    assert torch.equal(expected, actual[0])
    assert diag['reason'] == 'unsupported_execution_contract'


def test_text_only_exact_generation_and_right_padding(llm):
    cache = PrefixKVCache(1024**2); executor = PrefillExecutor(cache)
    ids = torch.tensor([[1, 3, 4, 0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    x = expand_input(llm, ids, None, attention_mask=mask)
    expected = llm.generate(inputs=ids, attention_mask=mask, max_new_tokens=3, do_sample=False)
    actual, _ = executor.generate(llm, x, max_new_tokens=3, do_sample=False)
    assert torch.equal(actual, expected)


def test_cache_rejects_unsupported_layout(llm, carriers):
    from worldstream.m3.prefix_kv_cache import clone_kv
    with pytest.raises(ValueError):
        clone_kv(object(), 1)
    with pytest.raises(ValueError):
        clone_kv(((torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)),), 1)


def test_partial_prefix_keeps_cache_owned(llm, carriers):
    cache = PrefixKVCache(1024**2); executor = PrefillExecutor(cache)
    x = expanded(llm, carriers)
    executor.generate(llm, x, max_new_tokens=2)
    original = tuple(tuple(t.clone() for t in layer) for layer in cache.entry.kv)
    changed = expanded(llm, carriers, torch.tensor([[1, 3, -200, 6, 8]]))
    executor.generate(llm, changed, cache.match(changed), max_new_tokens=3)
    assert all(torch.equal(a,b) for x,y in zip(original,cache.entry.kv) for a,b in zip(x,y))


def test_runtime_full_unchanged(llm, carriers):
    ids = torch.tensor([1, -200, 3])
    torch.manual_seed(19); expected = llm.generate(inputs=ids[None], images=carriers, do_sample=True, max_new_tokens=5)
    torch.manual_seed(19); actual, diag = M3Runtime(dict(mode='full')).generate(llm, [ids], carriers, do_sample=True, max_new_tokens=5)
    assert torch.equal(expected, actual[0]) and diag['mode'] == 'full'


def test_scene_camera_lifecycle(llm, carriers, context):
    runtime = M3Runtime(dict(mode='exact', offline_experiment=True, cross_frame=True, min_prefix_tokens=1))
    ids = [torch.tensor([1, -200, 3])]
    for changes in ({}, {'scene_id': 'other'}, {'camera_ids': ['back']}, {'timestamp': 0.}):
        _, diag = runtime.generate(llm, ids, carriers, context=dict(context, **changes), max_new_tokens=2)
        assert diag['requests'][0]['reused_tokens'] == 0


def test_cross_frame_text_then_current_visual_reuse(llm, carriers, context):
    runtime = M3Runtime(dict(mode='exact', offline_experiment=True, cross_frame=True, min_prefix_tokens=1))
    ids = [torch.tensor([1, 3, -200, 4, 5]), torch.tensor([1, 3, -200, 6, 5])]
    runtime.generate(llm, ids, carriers, context=context, max_new_tokens=2)
    changed = carriers.clone(); changed[0, 0, 0] += 1
    _, diag = runtime.generate(llm, ids, changed, context=dict(context, frame_id='next', timestamp=2.), max_new_tokens=2)
    assert diag['requests'][0]['reused_tokens'] == 2
    assert diag['requests'][1]['reused_tokens'] == 11
    assert runtime.cache.entry.prefix_lineage == ((0, 2, ('s', 'f', 1., 0)),)
    assert runtime.cache.entry.computed_source == ('s', 'next', 2., 0)


def test_uncalibrated_auto_native_fallback(llm, carriers, context):
    runtime = M3Runtime()
    _, diag = runtime.generate(llm, [torch.tensor([1, -200, 4])], carriers, context=context, max_new_tokens=2)
    assert diag['reason'] == 'no_calibrated_policy' and runtime.cache.entry is None


def test_invalid_lifecycle_falls_back(llm, carriers, context):
    runtime = M3Runtime(dict(mode='exact', offline_experiment=True))
    for changes in ({'timestamp': float('nan')}, {'camera_ids': ['front', 'front']}, {'scene_id': ''}):
        _, diag = runtime.generate(llm, [torch.tensor([1, -200, 4])], carriers, context=dict(context, **changes), max_new_tokens=2)
        assert diag['reason'] == 'missing_lifecycle'


def test_eos_and_stopping(llm, carriers):
    x = expanded(llm, carriers)
    first = llm(inputs_embeds=x.embeddings).logits[0, -1].argmax().item()
    answer, _ = PrefillExecutor(PrefixKVCache(1024**2)).generate(llm, x, eos_token_id=first, max_new_tokens=8)
    assert answer.shape[1] == 2  # Original HF inputs_embeds convention includes dummy BOS.
    class StopAtThree(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return input_ids.shape[1] >= 3
    answer, _ = PrefillExecutor(PrefixKVCache(1024**2)).generate(llm, x, eos_token_id=None, max_new_tokens=8,
        stopping_criteria=StoppingCriteriaList([StopAtThree()]))
    assert answer.shape[1] <= 3


def test_selection_order_ego_and_quality_gate(carriers, context):
    from types import SimpleNamespace
    layout = CarrierLayout(4, 1, 4)
    state = SimpleNamespace(scene_id='s', last_timestamp=1.)
    for method in ('task', 'topk', 'random'):
        selection = select_carriers(carriers, layout, 'distance of car in front', method, .5,
                                    world_state=state, context=context)
        ids = selection.indices.tolist()
        assert ids == sorted(set(ids)) and 4 in ids and len(ids) < 9
    full = select_carriers(carriers, layout, 'is it safe?', 'task', .5, world_state=state, context=context)
    assert full.indices.numel() == 9
    stale = select_carriers(carriers, layout, 'distance of car', 'task', .5, world_state=state, context=dict(context, timestamp=2.))
    assert stale.indices.numel() == 9
    from worldstream.m3.scheduler import Scheduler
    from worldstream.m3.config import M3Config
    assert not Scheduler(M3Config()).compact_allowed(selection.diagnostics, context)[0]
