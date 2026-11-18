"""FQ-3 protocol: GT transitions, workload integrity, and synthetic report gates."""
from copy import deepcopy
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

from tools.environment_stability_eval.fq3 import GROUPS, check_core, strict_transitions


def gt_rows(states, times=None):
    result = []
    for i, state in enumerate(states):
        for qi, group in enumerate(GROUPS):
            result.append(dict(scene_id='scene',frame_id=str(i),timestamp=times[i] if times else i*.5,
                group=group,query_index=qi,question='TEST_ONLY_'+group,
                ground_truth=dict(known=state is not None,state={'value':state} if state is not None else None,
                                  fact_type='relative_position')))
    return result


def classify(states,times=None):
    rows=gt_rows(states,times)
    scenes=[dict(scene_id='scene',tokens=[str(i) for i in range(len(states))])]
    return strict_transitions(rows,scenes,1.)


class TransitionTests(unittest.TestCase):
    def test_all_three_tasks_are_retained_in_each_state(self):
        rows=classify(['a','a','b'])
        for frame,expected in [('0','Unclassified'),('1','Stable'),('2','Changed')]:
            selected=[r for r in rows if r['frame_id']==frame]
            self.assertEqual([r['group'] for r in selected],list(GROUPS))
            self.assertEqual({r['fact_status'] for r in selected},{expected})

    def test_unknown_never_bridged(self):
        rows=classify(['old',None,'new','new'])
        self.assertEqual({r['fact_status'] for r in rows if r['frame_id']=='2'},{'Unclassified'})
        self.assertEqual({r['previous_frame_id'] for r in rows if r['frame_id']=='2'},{None})
        self.assertEqual({r['fact_status'] for r in rows if r['frame_id']=='3'},{'Stable'})

    def test_time_gap_is_not_labeled_changed(self):
        rows=classify(['a','b'],[0,1.01])
        self.assertTrue(all(r['fact_status']=='Unclassified' for r in rows))

    def test_changed_remembers_exact_immediate_previous_GT(self):
        rows=classify(['a','b','c'])
        for r in rows[-3:]:
            self.assertEqual(r['previous_frame_id'],'1')
            self.assertEqual(r['previous_ground_truth']['state'],{'value':'b'})
            self.assertEqual(r['change_type'],'relative_position')

    def test_original_QA_rejected(self):
        rows=gt_rows(['a'])
        rows[2]['group']='other'
        with self.assertRaisesRegex(ValueError,'exclude all original QA'):
            strict_transitions(rows,[dict(scene_id='scene',tokens=['0'])],1.)

    def test_schema_mismatch_unclassified(self):
        rows=gt_rows(['a','b'])
        rows[3]['ground_truth']['state']={'different':'b'}
        result=strict_transitions(rows,[dict(scene_id='scene',tokens=['0','1'])],1.)
        self.assertEqual(result[3]['fact_status'],'Unclassified')

    def test_missing_query_cannot_create_nonadjacent_change(self):
        rows=gt_rows(['a','b','c'])
        rows=[r for r in rows if not(r['frame_id']=='1' and r['query_index']==0)]
        result=strict_transitions(rows,[dict(scene_id='scene',tokens=['0','1','2'])],1.)
        self.assertEqual(next(r for r in result if r['frame_id']=='2' and r['query_index']==0)['fact_status'],
                         'Unclassified')

    def test_scene_boundary_is_not_linked(self):
        rows=gt_rows(['a','b'])
        for r in rows[3:]: r['scene_id']='other_scene'
        result=strict_transitions(rows,[dict(scene_id='scene',tokens=['0']),
            dict(scene_id='other_scene',tokens=['1'])],1.)
        self.assertTrue(all(r['fact_status']=='Unclassified' for r in result))


class WorkloadTests(unittest.TestCase):
    def row(self):
        return dict(groups=list(GROUPS),questions=['q0','q1','q2'],answers=['a']*3,requests=[{}]*3,
            forward_ms=10.,forward_raw_ms=11.,forward_excluded_ms=1.,
            forward_exclusion_calls={'answer_text_decode':3,'answer_file_write':1})

    def test_exact_core_workload_and_measured_forward(self):
        check_core([self.row()])

    def test_q0_q1_q3_is_not_FQ3(self):
        row=self.row(); row['groups'][2]='other'
        with self.assertRaises(ValueError): check_core([row])

    def test_five_question_export_accounting_is_rejected(self):
        row=self.row(); row['forward_exclusion_calls']['answer_text_decode']=5
        with self.assertRaises(ValueError): check_core([row])

    def test_bad_timing_is_rejected(self):
        for value in [float('nan'),-1,0]:
            row=self.row(); row['forward_ms']=value
            with self.assertRaises(ValueError): check_core([row])

    def test_default_config_excludes_original_QA(self):
        from tools.environment_stability_eval.run import load_config
        path=Path(__file__).parents[1]/'fq3.yaml'
        self.assertEqual(load_config(path)['other_questions'],0)


class ReportTests(unittest.TestCase):
    def test_real_report_serialization_with_synthetic_three_task_receipts(self):
        # These fabricated values are confined to an automatically removed temp
        # directory and test report logic, never actual experimental performance.
        from tools.environment_stability_eval.fq3 import report, sha
        from tools.environment_stability_eval.common import dump_jsonl, write, read
        from tools.environment_stability_eval.change_response import change_events as actual_events
        from tools.environment_stability_eval.tests.test_change_response import fixture, grade
        base_events, base_frames, objects = fixture(states=('old','new','new'),predictions=('old','old','new'))
        frames={}
        scored=[]
        for method in ('native','worldstream'):
            frames[method]=[]
            for original in base_frames[method]:
                row=dict(original,method=method,questions=['TEST_ONLY_'+g for g in GROUPS],groups=list(GROUPS),
                    answers=['old']*3,requests=[{} for _ in GROUPS],fairness={'SYNTHETIC_TEST':True},
                    execution_status='COMPLETED',e2e_ms=12.,forward_ms=10.,forward_raw_ms=11.,forward_excluded_ms=1.,
                    forward_exclusion_calls={'answer_text_decode':3,'answer_file_write':1},
                    stability_level='S1',window_id='w')
                frames[method].append(row)
            for q in [e for e in base_events if e['method']==method]:
                for qi,g in enumerate(GROUPS):
                    e=deepcopy(q)
                    e.update(question='TEST_ONLY_'+g,query_index=qi,query_id='q'+str(qi),group=g,
                        stale_eligible=e['fact_status']=='Changed',
                        stale_answer=e['fact_status']=='Changed' and e['prediction']=='old')
                    scored.append(e)
        with tempfile.TemporaryDirectory(prefix='fq3_SYNTHETIC_TEST_') as temp:
            out=Path(temp)
            manifest=dict(smoke=True,frames=3,scenes=[dict(scene_id='test_scene',tokens=['0','1','2'])],
                config=dict(repeats=1,bootstrap_samples=0,seed=1,response_horizon_seconds=5.,max_transition_gap_seconds=1.),
                transition_definition='SYNTHETIC_TEST',query_definition='SYNTHETIC_TEST',forward_definition='SYNTHETIC_TEST')
            write(out/'manifest.json',manifest)
            dump_jsonl(out/'query_ground_truth.jsonl',[])
            dump_jsonl(out/'test_windows.jsonl',[dict(frame_ids=['0','1','2'],window_id='w',stability_level='S1',
                                                    start_timestamp=0.,end_timestamp=5.)])
            dump_jsonl(out/'evaluation_objects.jsonl',[dict(frame_id=t,objects=o) for t,o in objects.items()])
            write(out/'data_coverage.json',dict(known_transition_counts={'Changed':3},SYNTHETIC_TEST_ONLY=True))
            for method in frames:
                folder=out/method; folder.mkdir()
                dump_jsonl(folder/'receipts.jsonl',frames[method])
                write(folder/'processing_segments.json',[])
                write(folder/'inference_complete.json',dict(status='COMPLETED',frames=3,
                    receipts_sha256=sha(folder/'receipts.jsonl'),segments_sha256=sha(folder/'processing_segments.json')))
                write(folder/'effective_runtime.json',dict(checkpoint_sha256='TEST',device='TEST',gpu_uuid=None,
                    fp16=None,batch_size=1,generation={},precision_by_dtype=[]))
            def local_score(rows,*args):
                return [e for e in scored if e['method']==rows[0]['method']]
            def local_events(*args,**kwargs):
                return actual_events(*args,grade_fn=grade,**kwargs)
            with patch('tools.environment_stability_eval.evaluator.verify_manifest',return_value=manifest), \
                 patch('tools.environment_stability_eval.metrics.score_rows',side_effect=local_score), \
                 patch('tools.environment_stability_eval.change_response.change_events',side_effect=local_events):
                result=report(out)
            self.assertEqual(read(out/'completion.json')['queries_per_method'],9)
            self.assertEqual(len(result['primary']),8)
            self.assertEqual(result['paired_response']['paired_events'],3)
            self.assertIn('论文核心表',(out/'结果.md').read_text())
            self.assertTrue((out/'forward_latency_summary.csv').is_file())


if __name__=='__main__':
    unittest.main()
