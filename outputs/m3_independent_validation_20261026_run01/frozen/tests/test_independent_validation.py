"""Evidence accounting and counterfactuals must not manufacture experiment gains."""
from dataclasses import asdict
import json

import pytest

from tools.worldstream_m123.benchmark.decision_replay import replay, compare_trace
from tools.worldstream_m123.benchmark.independent_analysis import (
    cluster_interval, unique_index, validate_actual, quality_statistics)
from tools.worldstream_m123.benchmark.independent_validation import historical_audit
from worldstream.m3.config import M3Config
from worldstream.m3.scheduler import POLICY_VERSION


def trace_fixture(tmp_path, m1=True):
    binding=dict(model_version='m',weights_sha256='w',execution_key='e',
        generation_key='g',layout_signature='l',dtype='torch.float16')
    policy=dict(version=POLICY_VERSION, binding=binding, block_size=32,
        max_cache_age_frames=2, max_approx_depth=1, profiles=dict(map_relations=dict(approx=dict(
            quality_loss_upper=0., calibration_samples=2, validation_samples=2,
            risk_limit=.32, max_rms=2., max_token_change=2., max_visual_tokens=128,
            cost=dict(samples=2,net_saved_ms_per_token=1.,min_input_tokens=570,max_input_tokens=580,
                min_reuse_tokens=32,max_reuse_tokens=100)))))
    path=tmp_path/'policy.json';path.write_text(json.dumps(policy))
    context=dict(binding,scene_id='s',frame_id='f1',timestamp=1.,frame_index=1)
    t=dict(config=asdict(M3Config()),question='Describe the road layout.',context=context,
        history=dict(state_valid=m1),m2=dict(execution_completed=True,
            cache_context=dict(scene_id='s',frame_id='f1',timestamp=1.),
            stages=[dict(total_windows=10,executed_windows=10,cache_age_frames=0)]),
        input_tokens=573,dtype='torch.float16',source=['s','f1',1.,0],
        layout_signature='l',visual_group_spans=[['object',35,291],['ego',291,292],['map',292,548]],
        cache=dict(source=['s','f0',.5,0],computed_source=['s','f0',.5,0],input_tokens=573,
            root_frame_index=0,approx_reuse_depth=0,quality_risk=0.,risk_profile='map_relations'),
        cache_validated=True,overhead_ms=2.,
        details=dict(task_profile='map_relations',matched_prefix_tokens=35,structural_prefix_tokens=548,
            block_changes=[dict(start=35,end=67,group='object',rms=.4,max_token=0.)]))
    t['decision']=replay(t,path)['decision']
    return t,path


def test_m2_counterfactual_uses_production_risk_with_fixed_overhead(tmp_path):
    trace,path=trace_fixture(tmp_path)
    before=path.read_bytes()
    result=compare_trace(trace,path)
    assert result['baseline_reproduced']
    assert result['actual']['decision']['reuse_length']==0
    assert result['without_m2']['decision']['reuse_length']==67
    assert result['without_m1']['decision']['reuse_length']==0
    assert path.read_bytes()==before


def test_m1_neutralization_reveals_candidates_even_when_cost_still_rejects(tmp_path):
    trace,path=trace_fixture(tmp_path,m1=False)
    result=compare_trace(trace,path)
    assert result['actual']['candidate_prefix_tokens']==[35]
    assert result['without_m1']['candidate_prefix_tokens']==[35,67]
    assert result['without_m1']['decision']['reuse_length']==0


def test_cluster_bootstrap_does_not_count_repeats_as_independent_scenes():
    records=[dict(scene_id='a',native=100.,wsdr=90.),dict(scene_id='b',native=100.,wsdr=110.)]
    a=cluster_interval(records);b=cluster_interval(records*5)
    assert a['scenes']==b['scenes']==2
    assert a['ci95_delta']==b['ci95_delta']
    assert a['delta']==b['delta']==0.
    assert cluster_interval(records[:1])['ci95_delta'] is None


def test_accounting_uses_actual_executor_forward_lengths():
    q=dict(reuse_diagnostics=dict(input_tokens=100,reused_tokens=20,effective_prefill_tokens=80,
        forward_prefill_lengths=[80],prefill_completed=True,decode_completed=True,reuse_type='bounded_approx'))
    validate_actual(q)
    q['reuse_diagnostics']['forward_prefill_lengths']=[100]
    with pytest.raises(ValueError,match='forward lengths'):validate_actual(q)
    with pytest.raises(ValueError,match='Duplicate'):unique_index([{'id':1},{'id':1}],['id'])


def test_history_audit_derives_scene_from_frame_and_retains_questions(tmp_path, monkeypatch):
    import tools.worldstream_m123.benchmark.independent_validation as mod
    policy=tmp_path/'policy.json';policy.write_text(json.dumps(dict(scene_splits={'policy_scene':'calibration'})))
    monkeypatch.setattr(mod,'POLICY',policy)
    old=tmp_path/'m3_old';old.mkdir()
    (old/'sequence.json').write_text(json.dumps(dict(tokens=['f'])))
    (old/'service_evidence.jsonl').write_text(json.dumps(dict(frame_id='f',question='Where?'))+'\n')
    current=tmp_path/'new';current.mkdir()
    (current/'sequence.json').write_text(json.dumps(dict(tokens=['new'])))
    audit=historical_audit(tmp_path,current,[dict(token='f',scene_token='s'),dict(token='new',scene_token='new_scene')])
    assert set(audit['excluded_scenes'])=={'s','policy_scene'}
    assert audit['excluded_frame_questions']==[('f','Where?')]
    assert 'new' not in audit['excluded_frames']


def test_quality_correct_to_incorrect_separate_from_existing_errors():
    from tools.worldstream_m123.benchmark.quality_calibration import paired_quality
    items=[dict(scene_id='s1',token_equal=False,quality=paired_quality('2 cars','1 car','1 car')),
        dict(scene_id='s2',token_equal=True,quality=paired_quality('3 cars','3 cars','1 car'))]
    result=quality_statistics(items)
    assert result['unique_questions']==2
    assert result['task_metrics']['numeric_exact']['native_errors']==1
    assert result['task_metrics']['numeric_exact']['new_correct_to_incorrect']==1
    assert result['critical_regression_requests']==1
