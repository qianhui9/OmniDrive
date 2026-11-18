"""Synthetic unit fixtures, never experimental results."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.environment_stability_eval import data_adapter as data
from tools.environment_stability_eval.metrics import (accuracy, cluster_intervals, compute,
    latency, response_delays, score_rows, validate_pair)
from tools.environment_stability_eval.stability_classifier import (assign_levels, build_windows,
    distribution, fit_thresholds, semantic_transition)


def obj(key='a', pos=(10., 0., 0.), cls='car', visible=True, attrs=('vehicle.moving',)):
    return dict(instance_id=key, annotation_id=key, center_ego=list(pos), semantic_class=cls,
                visible=visible, cameras=['CAM_FRONT'], attributes=list(attrs), source_frame='frame')


class SemanticTests(unittest.TestCase):
    def test_stable_with_continuous_motion(self):
        # Coordinate/image movement inside the same semantic sector is stable.
        t = semantic_transition([obj()], [obj(pos=(15., .5, 0.))])
        self.assertEqual(t['changed_facts'], 0)
        self.assertGreater(t['valid_fact_transitions'], 0)

    def test_appearance_disappearance(self):
        enter = semantic_transition([], [obj()])
        leave = semantic_transition([obj()], [])
        self.assertEqual(enter['events']['appearance'], 1)
        self.assertEqual(leave['events']['disappearance'], 1)
        self.assertEqual(enter['valid_fact_transitions'], 1)
        self.assertEqual(leave['change_rate'], 1.)

    def test_position_and_attributes(self):
        t = semantic_transition([obj()], [obj(pos=(-10., 0., 0.), attrs=('vehicle.stopped',))])
        self.assertEqual(t['events']['relative_position'], 1)
        self.assertEqual(t['events']['attribute'], 1)

    def test_missing_attributes_not_known_empty(self):
        t = semantic_transition([obj(attrs=())], [obj(attrs=('vehicle.moving',))])
        self.assertNotIn('attribute', t['valid_by_type'])
        self.assertEqual(t['events']['attribute'], 0)

    def test_pairwise_relation(self):
        t = semantic_transition([obj('a'), obj('b', (10., 4., 0.))],
                                [obj('a'), obj('b', (10., -4., 0.))])
        self.assertEqual(t['events']['spatial_relation'], 1)
        self.assertEqual(t['valid_by_type']['spatial_relation'], 1)

    def test_visibility_unknown_for_position(self):
        t = semantic_transition([obj()], [obj(pos=(-10., 0., 0.), visible=False)])
        self.assertNotIn('relative_position', t['valid_by_type'])
        self.assertEqual(t['events']['disappearance'], 0)

    def test_empty_transition(self):
        self.assertIsNone(semantic_transition([], [])['change_rate'])

    def test_threshold_ties_not_fabricated_groups(self):
        ws = [dict(complete=True, change_rate=.1) for _ in range(8)]
        fit = fit_thresholds(ws)
        self.assertTrue(fit['repeated_thresholds'])
        assign_levels(ws, fit['thresholds'])
        self.assertEqual({w['stability_level'] for w in ws}, {'S1'})

    def test_threshold_assignment(self):
        ws = [dict(complete=True, change_rate=x) for x in (0., .25, .26, .51, .9)]
        assign_levels(ws, [.25, .5, .75])
        self.assertEqual([w['stability_level'] for w in ws], ['S1', 'S1', 'S2', 'S3', 'S4'])

    def test_no_valid_calibration(self):
        with self.assertRaises(ValueError):
            fit_thresholds([dict(complete=True, change_rate=None)])

    def test_windows_fixed_time_and_partial_not_dropped(self):
        by = {str(i): dict(token=str(i), timestamp=i * 500000) for i in range(13)}
        scenes = [dict(scene_id='s', tokens=list(by))]
        objects = {str(i): [obj(pos=(10., 0., 0.))] for i in range(13)}
        ws = build_windows(scenes, by, objects, 5., 1.)
        assign_levels(ws, [.1, .2, .3])
        self.assertEqual([w['frames'] for w in ws], [10, 3])
        self.assertEqual([w['stability_level'] for w in ws], ['S1', 'Unclassified'])
        self.assertEqual(ws[1]['valid_fact_transitions'], 12)  # 3 transitions × 4 facts
        self.assertEqual(sum(r['frames'] for r in distribution(ws, [.1, .2, .3])), 13)

    def test_gt_history_skips_missing_and_never_crosses_sequences(self):
        scenes = [dict(scene_id='s1', tokens=['a', 'b', 'c', 'd']), dict(scene_id='s2', tokens=['e'])]
        by = {t: dict(timestamp=i * 500000) for i, t in enumerate('abcde')}
        states = {'a': 'car', 'b': None, 'c': 'car', 'd': 'truck', 'e': 'truck'}
        def truth(q, t, objects):
            return dict(known=states[t] is not None, state=dict(cls=states[t]) if states[t] else None,
                        fact_type='category')
        with patch('tools.worldstream_m123.r0_task_evaluation.facts_questions', return_value=(['Q'], ['vehicle_category'])), \
                patch.object(data, 'query_gt', side_effect=truth):
            result = data.build_query_records(dict(data_root='data/nuscenes', other_questions=2), scenes, by, {})
        self.assertEqual([r['fact_status'] for r in result],
                         ['Unclassified', 'Unclassified', 'Stable', 'Changed', 'Unclassified'])
        self.assertEqual(result[2]['previous_frame_id'], 'a')

    def test_query_state_ignores_target_identity(self):
        q = 'What type of vehicle is directly ahead?'
        self.assertEqual(data.query_gt(q, 'a', {'a': [obj('x') ]})['state'],
                         data.query_gt(q, 'b', {'b': [obj('y') ]})['state'])

    def test_no_visible_target_is_unknown_in_main_evaluator(self):
        self.assertFalse(data.query_gt('Is there a car in front?', 'a', {'a': []})['known'])

    def test_duplicate_frames_rejected(self):
        with self.assertRaises(ValueError):
            data.validate_frames([dict(scene_id='s', tokens=['a', 'a'])], {'a': dict(timestamp=0)}, False)


class ScoringTests(unittest.TestCase):
    def rows_and_gt(self, method, answers):
        states = ['car', 'car', 'truck']
        labels = ['Unclassified', 'Stable', 'Changed']
        rows, gt, objects = [], [], {}
        for i, answer in enumerate(answers):
            t = str(i)
            objects[t] = [obj(cls=states[i])]
            rows.append(dict(method=method, repeat=0, scene_id='s', frame_id=t, timestamp=i * .5,
                             questions=['What type of vehicle is directly ahead?'], answers=[answer],
                             window_id='w', stability_level='S1'))
            gt.append(dict(query_id='q0', group='vehicle_category', fact_status=labels[i], timestamp=i * .5,
                           change_type='category' if i == 2 else None, question=rows[-1]['questions'][0], frame_id=t,
                           ground_truth=dict(known=True, state=dict(cls=states[i])),
                           previous_frame_id=str(i - 1) if i else None,
                           previous_timestamp=(i - 1) * .5 if i else None,
                           previous_ground_truth=dict(known=True, state=dict(cls=states[i - 1])) if i else None))
        return rows, gt, objects

    def test_stale_exact_conditions_for_both_methods(self):
        for method in ('native', 'worldstream'):
            rows, gt, objects = self.rows_and_gt(method, ['A car.', 'A car.', 'A car.'])
            result = score_rows(rows, gt, objects)
            summary = accuracy(result)
            self.assertEqual(summary['stable_fact_accuracy_percent'], 100.)
            self.assertEqual(summary['changed_fact_accuracy_percent'], 0.)
            self.assertEqual(summary['stale_answer_rate_percent'], 100.)
            self.assertEqual(summary['queries_after_changes'], 1)
            self.assertIsNone(result[0]['stale_answer'])

    def test_changed_but_correct_not_stale(self):
        rows, gt, objects = self.rows_and_gt('native', ['A car.', 'A car.', 'A truck.'])
        result = score_rows(rows, gt, objects)
        self.assertEqual(accuracy(result)['changed_fact_accuracy_percent'], 100.)
        self.assertFalse(result[2]['stale_answer'])

    def test_unrelated_wrong_answer_not_stale(self):
        rows, gt, objects = self.rows_and_gt('worldstream', ['A car.', 'A car.', 'A bus.'])
        self.assertFalse(score_rows(rows, gt, objects)[2]['stale_answer'])

    def test_missing_gt_excluded_from_accuracy(self):
        rows, gt, objects = self.rows_and_gt('native', ['A car.', 'A car.', 'A truck.'])
        gt[0]['ground_truth'] = dict(known=False, state=None)
        result = score_rows(rows, gt, objects)
        self.assertEqual(accuracy(result)['scorable_queries'], 2)
        self.assertIsNone(result[0]['correct'])

    def test_zero_denominators(self):
        summary = accuracy([])
        for k in ('overall_accuracy_percent', 'stable_fact_accuracy_percent',
                  'changed_fact_accuracy_percent', 'stale_answer_rate_percent'):
            self.assertIsNone(summary[k])
        self.assertIsNone(latency([], [])['sustained_fps'])
        self.assertIsNone(compute([], [])['visual_compute_reduction_percent'])

    def test_throughput_uses_real_segment_wall_time(self):
        result = latency([dict(e2e_ms=100), dict(e2e_ms=100)], [dict(frames=2, wall_seconds=.5)])
        self.assertEqual(result['sustained_fps'], 4.)
        self.assertEqual(result['average_e2e_ms'], 100.)

    def test_invalid_throughput_denominator(self):
        with self.assertRaises(ValueError):
            latency([dict(e2e_ms=1)], [])

    def test_single_sequence_ci_is_unavailable(self):
        ci = cluster_intervals([dict(scene_id='s')], lambda s: {'value': 1}, ['value'])
        self.assertIsNone(ci['value_ci95_low'])

    def test_sampled_response_delay_and_censoring(self):
        rows, gt, objects = self.rows_and_gt('native', ['A car.', 'A car.', 'A car.'])
        result = score_rows(rows, gt, objects)
        delay = response_delays(result, 5.)[0]
        self.assertFalse(delay['responded'])
        self.assertTrue(delay['censored'])

    def test_paired_receipts_reject_query_mismatch(self):
        window = dict(window_id='w', frame_ids=['a'], stability_level='S1', start_timestamp=0, end_timestamp=5)
        row = dict(repeat=0, scene_id='s', frame_id='a', execution_status='COMPLETED', e2e_ms=1,
                   questions=['q'], groups=['g'], answers=['a'], requests=[{}],
                   window_id='w', stability_level='S1', timestamp=0, fairness={})
        manifest = dict(config=dict(repeats=1), scenes=[dict(scene_id='s', tokens=['a'])])
        rows = dict(native=[deepcopy(row)], worldstream=[deepcopy(row)])
        validate_pair(manifest, [window], rows)
        rows['worldstream'][0]['questions'] = ['changed']
        with self.assertRaises(ValueError):
            validate_pair(manifest, [window], rows)

    def test_timestamp_must_match_authoritative_gt(self):
        rows, gt, objects = self.rows_and_gt('native', ['A car.', 'A car.', 'A truck.'])
        rows[1]['timestamp'] = .6
        with self.assertRaises(ValueError):
            score_rows(rows, gt, objects)


if __name__ == '__main__':
    unittest.main()
