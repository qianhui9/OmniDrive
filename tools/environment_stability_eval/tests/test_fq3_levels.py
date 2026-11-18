"""Offline level aggregation semantics; fabricated data never enters outputs."""
from copy import deepcopy
import unittest

from tools.environment_stability_eval.fq3_levels import aggregate, verify_level_assignments


def query(method,level,status,score,scene='test_scene'):
    return dict(method=method,scene_id=scene,stability_level=level,fact_status=status,
        scorable=score is not None,score=score,correct=score==1 if score is not None else None,
        stale_eligible=status=='Changed' and score is not None,
        stale_answer=status=='Changed' and score==0)


def event(method,level,delay,scene='test_scene'):
    return dict(method=method,scene_id=scene,stability_level=level,repeat=0,query_id=level,
        frame_id=level,responded=delay is not None,response_delay_ms=delay,
        censored=delay is None,superseded=False,stop_reason='RESPONDED' if delay is not None else 'SCENE_END',
        wrong_outputs_before_response=0,max_consecutive_stale_outputs=0,
        max_stale_observed_span_ms=0,stale_outputs=0,stale_direct_outputs=0,historical_unscorable_outputs=0)


class LevelAggregationTests(unittest.TestCase):
    def setup_data(self):
        scored,events,frames=[],[],{}
        for method in ('native','worldstream'):
            scored += [query(method,'S1','Stable',1),query(method,'S1','Changed',0),
                query(method,'S2','Stable',0),query(method,'S4','Changed',1),
                query(method,'Unclassified','Stable',1),query(method,'S3','Unclassified',None)]
            events += [event(method,'S1',None if method=='native' else 500),event(method,'S4',0)]
            frames[method]=[dict(stability_level=level) for level in ('S1','S2','S3','S4','Unclassified')]
        return scored,events,frames

    def test_four_levels_and_tail_and_all_remain_separate(self):
        rows,_=aggregate(*self.setup_data(),samples=0)
        native={r['stability_level']:r for r in rows if r['method']=='native'}
        self.assertEqual(set(native),{'S1','S2','S3','S4','Unclassified','ALL'})
        self.assertEqual(native['S1']['stable_queries'],1)
        self.assertEqual(native['Unclassified']['stable_queries'],1)
        self.assertEqual(native['ALL']['stable_queries'],3)
        self.assertEqual(native['ALL']['frames'],5)

    def test_unobserved_response_is_null_not_zero(self):
        rows,_=aggregate(*self.setup_data(),samples=0)
        n=next(r for r in rows if r['method']=='native' and r['stability_level']=='S1')
        w=next(r for r in rows if r['method']=='worldstream' and r['stability_level']=='S4')
        self.assertIsNone(n['response_delay_mean_ms'])
        self.assertEqual(n['observed_response_rate_percent'],0.)
        self.assertEqual(w['response_delay_mean_ms'],0.)
        self.assertEqual(w['observed_response_rate_percent'],100.)

    def test_unknown_GT_does_not_enter_fact_accuracy(self):
        rows,_=aggregate(*self.setup_data(),samples=0)
        r=next(r for r in rows if r['method']=='native' and r['stability_level']=='S3')
        self.assertEqual(r['unknown_queries'],1)
        self.assertEqual(r['stable_queries'],0)
        self.assertIsNone(r['stable_fact_accuracy_percent'])
        self.assertIsNone(r['changed_fact_accuracy_percent'])
        self.assertIsNone(r['stale_answer_rate_percent'])

    def test_response_keeps_change_onset_level_even_if_completion_crossed_window(self):
        scored,events,frames=self.setup_data()
        for e in events:
            if e['stability_level']=='S1':
                e.update(responded=True,response_delay_ms=1000.,stop_reason='RESPONDED',
                         censored=False,completion_level='S4')
        rows,_=aggregate(scored,events,frames,samples=0)
        n=next(r for r in rows if r['method']=='native' and r['stability_level']=='S1')
        self.assertEqual(n['responded_events'],1)
        self.assertEqual(n['response_delay_mean_ms'],1000.)

    def test_changed_events_missing_is_rejected(self):
        scored,events,frames=self.setup_data()
        with self.assertRaisesRegex(ValueError,'count mismatch'):
            aggregate(scored,[e for e in events if e['stability_level']!='S4'],frames,samples=0)

    def test_only_one_changed_scene_has_no_bootstrap_CI_despite_other_stable_scenes(self):
        scored,events,frames=self.setup_data()
        for method in frames:
            scored += [query(method,'S1','Stable',1,'other_scene')]
        rows,_=aggregate(scored,events,frames,samples=10)
        for r in rows:
            if r['stability_level']=='S1':
                self.assertEqual(r['stable_scenes'],2)
                self.assertEqual(r['changed_scenes'],1)
                self.assertIsNone(r['changed_fact_accuracy_percent_ci95_low'])
                self.assertIsNone(r['response_delay_mean_ms_ci95_low'])

    def test_all_accuracy_is_weighted_not_mean_of_levels(self):
        scored,events,frames=self.setup_data()
        for method in frames:
            scored += [query(method,'S1','Stable',1) for _ in range(8)]
        rows,_=aggregate(scored,events,frames,samples=0)
        total=next(r for r in rows if r['method']=='native' and r['stability_level']=='ALL')
        self.assertAlmostEqual(total['stable_fact_accuracy_percent'],100*10/11)

    def test_changed_pairing_is_required(self):
        scored,events,frames=self.setup_data()
        with self.assertRaisesRegex(ValueError,'Missing paired'):
            aggregate(scored,[e for e in events if e['method']=='native'],frames,samples=0)


class FrozenLevelsTests(unittest.TestCase):
    def manifest(self):
        return dict(thresholds=dict(thresholds=[.1,.2,.3],calibration_scene_ids=['cal']),
                    scenes=[dict(scene_id='eval')])

    def test_ties_go_to_lower_bin_and_incomplete_tail_not_forced_to_S4(self):
        windows=[dict(change_rate=.1,complete=True,stability_level='S1'),
                 dict(change_rate=.21,complete=True,stability_level='S3'),
                 dict(change_rate=.9,complete=False,stability_level='Unclassified')]
        self.assertEqual(verify_level_assignments(windows,self.manifest()),[.1,.2,.3])

    def test_performance_based_relabel_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'differs'):
            verify_level_assignments([dict(change_rate=.1,complete=True,stability_level='S4')],self.manifest())

    def test_calibration_must_be_disjoint(self):
        m=self.manifest(); m['thresholds']['calibration_scene_ids']=['eval']
        with self.assertRaisesRegex(ValueError,'overlap'):
            verify_level_assignments([],m)


if __name__=='__main__':
    unittest.main()
