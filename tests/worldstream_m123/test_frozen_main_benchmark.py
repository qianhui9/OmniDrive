"""Fairness and denominator checks for the fixed R0 Native comparison."""
import json

import pytest

from tools.worldstream_m123.frozen_main_benchmark import historical_m3_config, content_hash
from tools.worldstream_m123.frozen_main_report import cluster_bootstrap, quality_summary
from tools.worldstream_m123.continuous_stream import question_schedule
from tools.worldstream_m123.frozen_main_archive import split_file
from tools.worldstream_m123.frozen_main_report import rows


def test_complete_historical_m3_config_preserves_non_summary_flags(tmp_path):
    cfg=historical_m3_config(tmp_path/'reference_policy.json')
    assert cfg['target_aware_exact_only'] is True
    assert cfg['current_detection_support'] is True
    assert cfg['draft_length']==8 and cfg['min_prefix_tokens']==32
    assert cfg['question_rng_enabled'] and cfg['experimental_speculative_sampling']
    assert cfg['reuse_strategy']=='exact_prefix' and cfg['decode_strategy']=='world_guided'
    assert cfg['selective_answer_mode']=='online'


def test_unchanged_five_schedules_include_idle_visual_frames():
    natural=['natural0','natural1','natural2']
    for period,freq in ((4,'every4'),(2,'every2')):
        assert sum(bool(question_schedule('addressable',freq,i,natural,natural)) for i in range(8))==8//period
    assert len(question_schedule('addressable','three',0,natural,natural))==3
    assert question_schedule('mixed','three',4,natural,natural)[1:]==natural[:2]


def test_bootstrap_keeps_whole_log_pairs_and_identical_speedup():
    pairs=[dict(log_id=log,A_ms=a,C_ms=a/2) for log in ('L1','L2','L3') for a in (100,200)]
    result=cluster_bootstrap(pairs,100)
    assert result['unit']=='Log' and result['logs']==3
    assert result['speedup_CI95']==[2.,2.]
    assert cluster_bootstrap(pairs[:2],100)['CI95'] is None


def test_unknown_answers_never_inflate_scored_accuracy():
    events=[dict(quality=dict(known=True,score=0.,severe_error=True),answer='wrong',llm_skipped=True),
        dict(quality=dict(known=False,score=None,severe_error=None,reason='ambiguous'),answer='unknown',llm_skipped=False)]
    result=quality_summary(events)
    assert result['requests']==2 and result['scored']==1 and result['UNKNOWN']==1
    assert result['scored_answer_accuracy']==0 and result['absolute_severe']==1
    assert result['FAST_coverage']==.5


def test_configuration_hash_includes_non_summary_parameter_changes():
    cfg=historical_m3_config('policy.json');before=content_hash(cfg)
    cfg['current_detection_support']=False
    assert content_hash(cfg)!=before


def test_raw_paired_receipts_roundtrip_through_lossless_index(tmp_path):
    path=tmp_path/'paired_gpu_receipts.jsonl'
    expected=[dict(pair_id=p,workload_id='S1',frame_id=f,log_id='L',A=dict(answer='native'),C=dict(answer='ours'))
        for p,f in [('P1','f0'),('P1','f1'),('P2','f2')]]
    path.write_text(''.join(json.dumps(r)+'\n' for r in expected))
    result=split_file(path,tmp_path/'chunks',paired=True)
    assert result['roundtrip_verified'] and len(result['files'])==2
    assert list(rows(path))==expected


def test_corrupt_index_cannot_supply_unverified_receipts(tmp_path):
    path=tmp_path/'paired_gpu_receipts.jsonl'
    path.write_text(json.dumps(dict(pair_id='P',workload_id='S1',frame_id='f',log_id='L',A={},C={}))+'\n')
    split_file(path,tmp_path/'chunks',paired=True)
    index=json.loads(path.read_text());index['record_sha256']='wrong';path.write_text(json.dumps(index)+'\n')
    with pytest.raises(AssertionError):list(rows(path))
