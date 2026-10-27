"""Reporting must separate text, visual, same-frame and actual execution."""
import pytest
from tools.worldstream_m123.crossframe_report import reuse, traffic_claims
from tools.worldstream_m123.benchmark.quality_calibration import cross_frame_visual


def request(length,source='old',kind='exact'):
    d=dict(input_tokens=600,effective_prefill_tokens=600-length,forward_prefill_lengths=[600-length],
        reuse_length=length,reused_tokens=length,cache_source_frame=source,reuse_type=kind,cache_age_frames=1,approx_reuse_depth=int(kind=='bounded_approx'),
        visual_group_spans=(('object',35,291),('ego',291,292),('map',292,548)),root_frame_index=0,
        computed_source=['s',source,1.,0],prefix_lineage=[],task_profile='map_relations',estimated_quality_risk=.1)
    return dict(scene_id='s',frame_id='new',question='Describe the road.',input_tokens=600,
        reused_prefix_length=length,reuse_diagnostics=d)


def test_text35_does_not_become_cross_frame_visual_reuse():
    r=reuse([request(35)])
    assert r['actually_reused_kv_tokens']==r['cross_frame_exact_tokens']==35
    assert r['cross_frame_visual_tokens']==0 and r['cross_frame_requests']==1
    assert r['cross_frame_evidence'][0]['forward_lengths']==[565]


def test_same_frame_inheritance_is_distinct_from_cross_frame_visual_reuse():
    r=reuse([request(163,kind='bounded_approx'),request(558,source='new',kind='bounded_approx')])
    assert r['cross_frame_bounded_approx_tokens']==163 and r['cross_frame_visual_tokens']==128
    assert r['same_frame_inherited_approx_tokens']==558 and r['cross_frame_requests']==1
    assert r['effective_prefill_tokens']==1200-163-558


@pytest.mark.parametrize('fault',['forward','planned','provenance'])
def test_corrupt_execution_evidence_cannot_produce_successful_report(fault):
    q=request(163);d=q['reuse_diagnostics']
    if fault=='forward':d['forward_prefill_lengths']=[600]
    if fault=='planned':d['reuse_length']=35
    if fault=='provenance':d.pop('visual_group_spans')
    with pytest.raises(ValueError):reuse([q])


def test_missing_or_same_frame_metadata_cannot_certify_approximate_policy():
    q=request(163,kind='bounded_approx')
    row=dict(frame_id='new',diagnostics=q['reuse_diagnostics'])
    assert cross_frame_visual(row)
    row['diagnostics']['cache_source_frame']='new'
    assert not cross_frame_visual(row)
    row['diagnostics'].pop('cache_source_frame')
    assert not cross_frame_visual(row)
    row['diagnostics']['cache_source_frame']='old'
    row.pop('frame_id')
    assert not cross_frame_visual(row)


def test_traffic_claim_flags_cover_the_historical_failure_without_f1_override():
    claims=traffic_claims('A double yellow line indicates a no-passing zone and two-way traffic with one lane each way.')
    assert set(claims)=={'double_yellow','no_passing','two_way','lane_count'}
