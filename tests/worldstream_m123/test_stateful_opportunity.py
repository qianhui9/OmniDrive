"""Audit integrity and real matcher contracts, without authorizing new reuse."""
from dataclasses import replace
from contextlib import ExitStack
import json
from functools import wraps
import inspect
from types import SimpleNamespace

import pytest
import torch

from tools.worldstream_m123.stateful_opportunity.common import (
    Context, counts, diagnostic_json, history_rejection, replace_method, stats, write)
from tools.worldstream_m123.stateful_opportunity.report import candidate_profitability, validate_run
from worldstream.m3.prefix_matcher import exact_lcp
from worldstream.m3.visual_state import ExpandedInput
from tools.worldstream_m123.stateful_opportunity.gpu_audit import TemporalObserver


def context():
    return Context('scene', 'previous', 1., ('A', 'B', 'C', 'D', 'E', 'F'), 1, 'weights/backend')


@pytest.mark.parametrize('changes,reason', [
    ({'scene':'other'}, 'scene_changed'),
    ({'frame':'previous'}, 'not_cross_frame'),
    ({'timestamp':1.}, 'time_not_increasing'),
    ({'timestamp':.5}, 'time_not_increasing'),
    ({'timestamp':float('nan')}, 'time_not_increasing'),
    ({'timestamp':None}, 'time_not_increasing'),
    ({'timestamp':3.}, 'expired'),
    ({'epoch':2}, 'state_epoch_changed'),
    ({'signature':'new LoRA'}, 'execution_changed'),
    ({'cameras':('B','A','C','D','E','F')}, 'camera_order_changed'),
    ({'cameras':()}, 'camera_order_changed'),
])
def test_lifecycle_rejects_before_matching(changes, reason):
    previous = context()
    current = replace(previous, frame='current', timestamp=1.5)
    assert history_rejection(previous, replace(current, **changes), 1.) == reason


def test_valid_history_is_only_a_metadata_check():
    previous = context()
    assert history_rejection(None, previous, 1.) == 'cold_start'
    assert history_rejection(previous, replace(previous, frame='current', timestamp=1.5), 1.) is None
    invalid = replace(previous, epoch=None)
    assert history_rejection(invalid, replace(invalid, frame='current', timestamp=1.5), 1.) == 'state_epoch_changed'


def expanded(source):
    return ExpandedInput(torch.arange(24.).reshape(1,6,4), torch.tensor([[1,2,-200,-200,3,4]]),
        torch.arange(6)[None], torch.ones(1,6,dtype=torch.long),
        torch.tensor([[False,False,True,True,False,False]]),
        scope=('scene',('A','B')), signature=('model',), source=source)


def test_actual_cross_frame_prefix_stops_at_changed_visual_input():
    previous = expanded(('scene','t-1'))
    current = expanded(('scene','t'))
    current.embeddings[:,2:] += .1
    assert exact_lcp(previous, current) == 2
    # A frame-agnostic scope allows comparison, never creates a visual hit.
    assert int(current.visual_mask[:,:exact_lcp(previous,current)].sum()) == 0


@pytest.mark.parametrize('field', ['token_ids','position_ids','attention_mask','visual_mask'])
def test_each_expanded_dependency_can_break_the_prefix(field):
    a, b = expanded(('s','previous')), expanded(('s','current'))
    value = getattr(b,field)
    value[:,1] = 0 if value[:,1].item() else 1
    assert exact_lcp(a,b) == 1


def test_scope_signature_and_nonfinite_reject():
    a,b = expanded(('s','previous')),expanded(('s','current'))
    assert exact_lcp(a,replace(b,signature=('changed',))) == 0
    assert exact_lcp(a,replace(b,scope=('other',))) == 0
    b.embeddings[:,0] = float('nan')
    assert exact_lcp(a,b) == 0


@pytest.mark.parametrize('exc', [RuntimeError, torch.cuda.OutOfMemoryError])
def test_temporary_instrumentation_restored_on_exception(exc):
    class Original:
        def forward(self): return 7
    model = Original()
    with pytest.raises(exc):
        with replace_method(model, 'forward', lambda: 8):
            assert model.forward() == 8
            raise exc('probe failure')
    assert model.forward() == 7
    assert 'forward' not in vars(model)


def test_existing_instance_override_restored():
    model = SimpleNamespace(forward=lambda: 7)
    old = model.forward
    with replace_method(model,'forward',lambda: 8):
        assert model.forward() == 8
    assert model.forward is old


@pytest.mark.parametrize('values', [(1,2,0,0),(3,2,3,0),(3,2,1,2)])
def test_funnel_cannot_invent_execution(values):
    with pytest.raises(ValueError): counts(*values)


def test_shadow_fallback_is_not_active_reuse():
    assert counts(12,10) == dict(eligible=12,attempted=10,admitted=0,actually_skipped=0,fallback=10)


def test_incomplete_runs_never_become_final_evidence(tmp_path):
    write(tmp_path/'status.json',dict(status='failed'))
    with pytest.raises(ValueError,match='Incomplete'):
        validate_run(tmp_path)


def test_nonfinite_measurements_not_serialized_as_success(tmp_path):
    with pytest.raises(ValueError): write(tmp_path/'bad.json',dict(net=float('nan')))
    assert not (tmp_path/'bad.json').exists()
    assert stats([]) is None
    with pytest.raises(ValueError): stats([float('inf')])


def test_native_unbounded_diagnostic_stays_explicitly_unknown(tmp_path):
    native = dict(uncertainty=[float('inf'),float('nan')])
    value = diagnostic_json(native)
    write(tmp_path/'diagnostic.json',value)
    assert value['uncertainty'][0] == dict(status='Unknown',nonfinite_value='inf')
    assert native['uncertainty'][0] == float('inf')


def test_prefix_cost_includes_real_signature_match_and_publication():
    p = dict(scene='s',frame='t',exact_input_prefix_tokens=45,
             optimistic_prefill_wall_delta_ms=2.,last_logits_bitwise_equal=True,last_logits_max_abs=0.)
    q = dict(scene='s',frame='t',signature_wall_ms=3.,match_wall_ms=.1)
    f = dict(scene='s',frame='t',cache_capture_wall_ms=[.5])
    result = candidate_profitability([], [p], [q], [f])
    assert result['prefix_by_scene'][0]['optimistic_net_wall_ms'] == pytest.approx(-1.6)
    assert result['any_profitable_candidate'] is False


def test_profitable_candidate_cannot_be_frozen_as_no_go():
    p = dict(scene='s',branch='map',measured_net_wall_ms=.2,
             measured_gpu_interval_delta_ms=.1,bitwise_equal=True)
    result = candidate_profitability([p], [], [], [])
    assert result['any_profitable_candidate'] is True


def test_native_mha_observer_tracks_fifo_without_changing_forward():
    """Real PyTorch MHA and projection hook; no synthetic task success claims."""
    def head():
        attention = torch.nn.MultiheadAttention(8,2,batch_first=True).eval()
        layer = SimpleNamespace(transformer_layers=[SimpleNamespace(attn=attention)])
        return SimpleNamespace(transformer=SimpleNamespace(query_decoder=SimpleNamespace(_layers=[layer])),
            topk_proposals=2,memory_len=8,num_propagated=0), attention
    det, da = head(); mp, ma = head()
    model = SimpleNamespace(pts_bbox_head=det,map_head=mp)
    current = dict(scene='scene',frame='previous',ordinal=2,context=context())
    original = torch.nn.functional._in_projection_packed
    old_v = torch.arange(80.).reshape(1,10,8)
    q = torch.ones(1,2,8)
    with ExitStack() as stack, torch.no_grad():
        observer = TemporalObserver(model,current,stack)
        # Prevent CUDA cost probes in this CPU instrumentation integration test.
        observer.measured.update({('scene',('map',0)),('scene',('detection',0))})
        da(q,old_v+1,old_v)
        ma(q,old_v+1,old_v)
        assert all(r['retained_source_rows']==0 for r in observer.rows)
        current.update(frame='current',ordinal=3,
                       context=replace(context(),frame='current',timestamp=1.5))
        value = torch.cat((torch.zeros(1,4,8),old_v[:,2:6],torch.zeros(1,2,8)),1)
        # Changed key positions intentionally prevent copying K/attention.
        observed = da(q,value+13,value)[0]
        ma(q,value+13,value)
        assert all(r['identical_value_input_rows']==4 for r in observer.rows[-2:])
        assert all(r['identical_key_input_rows']==0 for r in observer.rows[-2:])
        assert all(r['new_execution_skips']==0 for r in observer.rows)
    assert torch.nn.functional._in_projection_packed is original
    with torch.no_grad():
        native = da(q,value+13,value)[0]
    assert torch.equal(native,observed)


def test_source_epoch_change_revokes_retained_row_population():
    def head():
        attn = torch.nn.MultiheadAttention(8,2,batch_first=True).eval()
        return SimpleNamespace(topk_proposals=2,memory_len=8,num_propagated=0,
            transformer=SimpleNamespace(query_decoder=SimpleNamespace(_layers=[
                SimpleNamespace(transformer_layers=[SimpleNamespace(attn=attn)])]))), attn
    h,a = head(); m,b = head()
    current = dict(scene='scene',frame='previous',ordinal=2,context=context())
    v = torch.ones(1,10,8); q = torch.zeros(1,2,8)
    with ExitStack() as stack, torch.no_grad():
        observer = TemporalObserver(SimpleNamespace(pts_bbox_head=h,map_head=m),current,stack)
        a(q,v,v)
        current.update(frame='current',ordinal=3,context=replace(context(),frame='current',timestamp=1.5,epoch=2))
        a(q,v,v)
        assert observer.rows[-1]['lifecycle_rejection']=='state_epoch_changed'
        assert observer.rows[-1]['retained_source_rows']==0


def test_hf_generation_signature_and_sampling_survive_forward_observer():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,bos_token_id=1,eos_token_id=2,pad_token_id=0)).eval()
    ids = torch.tensor([[1,3,4,5]])
    embeds = model.get_input_embeddings()(ids)
    native = model.forward
    calls = []
    @wraps(native)
    def observer(*a,**kw):
        result = native(*a,**kw)
        calls.append(result.logits.detach().clone())
        return result
    options = dict(inputs_embeds=embeds,attention_mask=None,do_sample=True,
                   temperature=.1,top_p=.75,max_new_tokens=5,use_cache=True)
    with torch.no_grad():
        torch.manual_seed(42)
        reference = model.generate(**options)
        torch.manual_seed(42)
        with replace_method(model,'forward',observer):
            assert inspect.signature(model.forward)==inspect.signature(native)
            actual = model.generate(**options)
    assert calls
    assert torch.equal(reference,actual)


@pytest.mark.parametrize('changed', [False,True])
def test_detection_source_can_move_from_propagated_query_into_memory(changed):
    def head():
        a = torch.nn.MultiheadAttention(8,2,batch_first=True).eval()
        return SimpleNamespace(topk_proposals=2,memory_len=4,num_propagated=2,
            transformer=SimpleNamespace(query_decoder=SimpleNamespace(_layers=[
                SimpleNamespace(transformer_layers=[SimpleNamespace(attn=a)])]))), a
    h,a = head(); m,_ = head()
    current = dict(scene='scene',frame='previous',ordinal=1,context=context())
    v = torch.arange(48.).reshape(1,6,8); q = torch.ones(1,4,8)
    with ExitStack() as stack, torch.no_grad():
        observer = TemporalObserver(SimpleNamespace(pts_bbox_head=h,map_head=m),current,stack)
        observer.measured.add(('scene',('detection',0)))
        a(q,v+1,v)
        current.update(frame='current',ordinal=2,context=replace(context(),frame='current',timestamp=1.5))
        new_v = torch.cat((torch.zeros(1,4,8),v[:,2:4]),1)
        if changed:
            new_v[:,4] += 1
        a(q,new_v+11,new_v)
        row = observer.rows[-1]
        assert row['retained_source_rows']==2
        assert row['identical_value_input_rows']==(1 if changed else 2)
        assert row['identical_key_input_rows']==0
