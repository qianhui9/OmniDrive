"""Meaningful split, temporal coverage and report checks; synthetic data only."""
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from tools.environment_stability_eval.common import EVENT_TYPES, dump_jsonl, read, write
from tools.environment_stability_eval.fq3_expanded import gt_audit, split_scenes, report
from tools.environment_stability_eval.tests.test_fq3 import classify, GROUPS


def test_split_is_disjoint_exhaustive_and_keeps_original_scenes():
    scenes=[dict(scene_id=f's{i:03}',tokens=[str(i)]) for i in range(150)]
    original={s['scene_id'] for s in scenes[:18]}
    evaluation,calibration=split_scenes(scenes,original)
    a={s['scene_id'] for s in evaluation}; b={s['scene_id'] for s in calibration}
    assert len(a)==120 and len(b)==30
    assert not a & b and a | b == {s['scene_id'] for s in scenes}
    assert original <= a
    # Iteration order and attached outcomes must not affect selection.
    permuted=[dict(s,model_accuracy=1,GT_events=999) for s in reversed(scenes)]
    assert [s['scene_id'] for s in split_scenes(permuted,original)[0]]==[s['scene_id'] for s in evaluation]


def test_split_rejects_missing_original_duplicate_and_no_expansion():
    scenes=[dict(scene_id=str(i),tokens=[]) for i in range(5)]
    for data,original,count in [(scenes,{'missing'},1),(scenes+[scenes[0]],set(),1),
                                 (scenes,{'0'},4),(scenes,set(),0)]:
        with pytest.raises(ValueError): split_scenes(data,original,count)


def audit(states,times=None):
    rows=classify(states,times)
    manifest=dict(config=dict(response_horizon_seconds=5.,max_transition_gap_seconds=1.))
    # Place onset in S1 and all its later samples in S4 to verify no filtering.
    windows=[dict(stability_level='S1',frame_ids=['0','1']),
             dict(stability_level='S4',frame_ids=[str(i) for i in range(2,len(states))])]
    return gt_audit(manifest,rows,windows)


def test_coverage_tracks_across_levels_and_never_treats_GT_as_response():
    rows=audit(['old','new','new','new'])
    s1=rows[0]
    assert s1['changed_events']==3
    assert s1['changed_scenes']==1
    assert s1['events_with_followup']==3 and s1['events_with_1s_followup']==3
    assert 'response_delay_mean_ms' not in s1
    assert not s1['planning_target_met']


def test_coverage_stops_at_unknown_and_does_not_bridge_it():
    rows=audit(['old','new',None,'new','new'])
    s1=rows[0]
    assert s1['events_with_followup']==0
    assert s1['gt_followup_stop_reasons']=={'UNKNOWN_GT':3}
    assert rows[-1]['changed_events']==3


def test_coverage_stops_before_a_second_change_or_time_gap():
    rows=audit(['a','b','c'])
    assert rows[0]['gt_followup_stop_reasons']=={'SUPERSEDED':3}
    rows=audit(['a','b','b'],[0,.5,2.])
    assert rows[0]['gt_followup_stop_reasons']=={'OBSERVATION_GAP':3}


def test_report_full_serialization_is_dynamic_and_preserves_missing_levels():
    # Mock only upstream receipt loading/scoring; exercise real aggregation,
    # event tracking, tables, confidence handling and completion publication.
    from tools.environment_stability_eval.tests.test_change_response import fixture, grade
    from tools.environment_stability_eval.change_response import change_events
    base,frames,objects=fixture(states=('old','new','new'),predictions=('old','old','new'))
    scored=[]
    for method,part in frames.items():
        for f in part:
            f.update(method=method,forward_ms=10.,stability_level='S1',requests=[{} for _ in GROUPS])
        for q in [r for r in base if r['method']==method]:
            for qi,g in enumerate(GROUPS):
                e=deepcopy(q)
                e.update(question='TEST_ONLY_'+g,query_index=qi,query_id='q'+str(qi),group=g,
                         stale_eligible=e['fact_status']=='Changed',
                         stale_answer=e['fact_status']=='Changed' and e['prediction']=='old')
                scored.append(e)
    thresholds=[.015,.02,.025]
    manifest=dict(smoke=False,frames=3,scenes=[dict(scene_id='test_scene',tokens=['0','1','2'])],
        thresholds=dict(thresholds=thresholds,calibration_scene_ids=['cal'],evaluation_scene_ids=['test_scene']),
        config=dict(bootstrap_samples=0,seed=1,response_horizon_seconds=5.,max_transition_gap_seconds=1.,
                    threshold_quantiles=[.25,.5,.75]),forward_definition='SYNTHETIC_TEST_ONLY')
    windows=[dict(stability_level='S1',frame_ids=['0','1','2'],complete=True,change_rate=.01,
        frames=3,events={k:0 for k in EVENT_TYPES},valid_fact_transitions=1,changed_facts=0)]
    with tempfile.TemporaryDirectory(prefix='expanded_SYNTHETIC_TEST_') as temp:
        root=Path(temp); out=root/'evaluation'; out.mkdir()
        write(out/'manifest.json',manifest)
        dump_jsonl(out/'calibration_windows.jsonl',[dict(complete=True,change_rate=v) for v in [.01,.02,.03]])
        dump_jsonl(out/'evaluation_objects.jsonl',[dict(frame_id=t,objects=o) for t,o in objects.items()])
        write(out/'data_coverage.json',dict(SYNTHETIC_TEST_ONLY=True))
        write(root/'gt_audit.json',dict(plan=dict(planning_target_events=50,planning_target_scenes=10),by_level=[]))
        for method in frames:
            (out/method).mkdir()
            for name in ('receipts.jsonl','inference_complete.json','processing_segments.json','effective_runtime.json'):
                (out/method/name).write_text('SYNTHETIC_TEST_ONLY')
        def local_score(rows,*args): return [q for q in scored if q['method']==rows[0]['method']]
        def local_events(*args,**kwargs): return change_events(*args,grade_fn=grade,**kwargs)
        with patch('tools.environment_stability_eval.fq3_expanded.load_completed',return_value=(manifest,windows,[],frames)), \
             patch('tools.environment_stability_eval.metrics.score_rows',side_effect=local_score), \
             patch('tools.environment_stability_eval.change_response.change_events',side_effect=local_events):
            result=report(root)
        assert result['frames_per_method']==3
        assert len(result['metrics'])==12
        assert next(r for r in result['metrics'] if r['stability_level']=='S2')['response_delay_mean_ms'] is None
        completion=read(root/'completion.json')
        folder=Path(completion['report_directory'])
        text=(folder/'结果.md').read_text()
        assert '1 场景、3 帧、9 条问题' in text
        assert '723' not in text and '62帧' not in text
        assert 'N/A' in text and (folder/'accuracy_by_task.csv').exists()
