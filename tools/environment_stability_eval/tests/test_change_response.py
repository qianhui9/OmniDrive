"""Scientific metric edge cases. All fabricated values stay in test fixtures."""
from copy import deepcopy
import unittest

from tools.environment_stability_eval.change_response import (
    change_events, paired_statistics, statistics)


def fixture(states=('old', 'new', 'new', 'new'), predictions=('old', 'old', 'old', 'new'),
            timestamps=None, missing=(), unknown=()):
    """SYNTHETIC TEST ONLY: exact-state toy grader, two identical methods."""
    times = list(timestamps) if timestamps is not None else [i * .5 for i in range(len(states))]
    scored, frames, objects = [], {}, {}
    for method in ('native', 'worldstream'):
        rows, previous = [], None
        for i, (state, prediction, timestamp) in enumerate(zip(states, predictions, times)):
            token = str(i)
            objects[token] = state
            rows.append(dict(frame_id=token, repeat=0, scene_id='test_scene', timestamp=timestamp,
                             requests=[dict(llm_skipped=method == 'worldstream')]))
            if i in missing:
                continue
            gt = dict(known=i not in unknown, state=state)
            event = dict(method=method, repeat=0, scene_id='test_scene', frame_id=token,
                timestamp=timestamp, query_id='q', query_index=0, question='TEST_ONLY_STATE',
                group='test', stability_level='S1', ground_truth=gt, prediction=prediction,
                scorable=i not in unknown, score=float(state == prediction) if i not in unknown else None,
                correct=state == prediction if i not in unknown else None,
                fact_status=('Changed' if state != previous['ground_truth']['state'] else 'Stable')
                            if previous and gt['known'] else 'Unclassified',
                previous_ground_truth=deepcopy(previous['ground_truth']) if previous else None,
                previous_frame_id=previous['frame_id'] if previous else None,
                previous_timestamp=previous['timestamp'] if previous else None,
                change_type='relative_position')
            scored.append(event)
            if gt['known']:
                previous = event
        frames[method] = rows
    return scored, frames, objects


def grade(question, prediction, frame, objects):
    return dict(known=True, score=float(prediction == objects[frame]))


def evaluate(*args, **kwargs):
    values = fixture(*args, **kwargs)
    return change_events(*values, grade_fn=grade)


class ChangeResponseTests(unittest.TestCase):
    def test_stale_followup_stable_rows_are_graded_against_frozen_old_GT(self):
        events = evaluate()
        for event in events:
            self.assertEqual(event['response_delay_ms'], 1000.)
            self.assertEqual(event['max_consecutive_stale_outputs'], 2)
            self.assertEqual(event['initial_consecutive_stale_outputs'], 2)
            self.assertEqual(event['wrong_outputs_before_response'], 2)
            self.assertEqual(event['max_stale_observed_span_ms'], 500.)
            self.assertEqual(event['stale_runs'][0]['end_bracket_ms'], 1000.)
            self.assertTrue(event['contiguous_onset'])
            self.assertEqual(event['stale_direct_outputs'], 2 if event['method'] == 'worldstream' else 0)

    def test_immediate_correct_has_zero_delay_and_no_stale_run(self):
        for event in evaluate(predictions=('old', 'new', 'new', 'new')):
            self.assertEqual(event['response_delay_ms'], 0.)
            self.assertEqual(event['stale_outputs'], 0)
            self.assertFalse(event['censored'])

    def test_wrong_unrelated_answer_is_not_stale(self):
        for event in evaluate(predictions=('old', 'unrelated', 'unrelated', 'new')):
            self.assertEqual(event['wrong_outputs_before_response'], 2)
            self.assertEqual(event['stale_outputs'], 0)
            self.assertEqual(event['max_consecutive_stale_outputs'], 0)

    def test_scene_end_censors_and_keeps_run_lower_bound(self):
        for event in evaluate(predictions=('old', 'old', 'old', 'old')):
            self.assertIsNone(event['response_delay_ms'])
            self.assertEqual(event['stop_reason'], 'SCENE_END')
            self.assertTrue(event['censored'])
            self.assertEqual(event['max_consecutive_stale_outputs'], 3)
            self.assertTrue(event['stale_runs'][0]['right_censored'])

    def test_unknown_GT_stops_followup_and_does_not_bridge_to_correct(self):
        for event in evaluate(unknown=(2,)):
            self.assertFalse(event['responded'])
            self.assertEqual(event['stop_reason'], 'UNKNOWN_GT')
            self.assertEqual(event['stale_outputs'], 1)

    def test_query_absence_breaks_consecutive_observations(self):
        for event in evaluate(missing=(2,)):
            self.assertEqual(event['stop_reason'], 'QUERY_NOT_ASKED')
            self.assertFalse(event['responded'])
            self.assertEqual(event['stale_outputs'], 1)

    def test_temporal_gap_censors_instead_of_connecting(self):
        for event in evaluate(timestamps=(0, .5, 2., 2.5)):
            self.assertEqual(event['stop_reason'], 'OBSERVATION_GAP')
            self.assertEqual(event['stale_outputs'], 1)

    def test_noncontiguous_change_onset_retained_but_flagged(self):
        events = evaluate(timestamps=(0, 2., 2.5, 3.))
        self.assertTrue(all(not e['contiguous_onset'] for e in events))
        self.assertTrue(all(e['response_delay_ms'] == 1000. for e in events))

    def test_next_GT_change_supersedes_event_even_when_old_state_returns(self):
        events = evaluate(states=('old', 'new', 'old', 'old'),
                          predictions=('old', 'old', 'old', 'old'))
        first = [e for e in events if e['frame_id'] == '1']
        self.assertTrue(all(e['stop_reason'] == 'SUPERSEDED' for e in first))
        self.assertTrue(all(not e['responded'] for e in first))

    def test_horizon_limit_does_not_count_later_success(self):
        values = fixture()
        events = change_events(*values, horizon=.5, grade_fn=grade)
        self.assertTrue(all(e['stop_reason'] == 'HORIZON' for e in events))
        self.assertTrue(all(not e['censored'] and not e['responded'] for e in events))

    def test_separate_stale_runs_and_single_output_span(self):
        events = evaluate(states=('old', 'new', 'new', 'new', 'new'),
                          predictions=('old', 'old', 'unrelated', 'old', 'new'))
        for event in events:
            self.assertEqual([r['frames'] for r in event['stale_runs']], [1, 1])
            self.assertEqual(event['max_stale_observed_span_ms'], 0.)
            self.assertEqual(event['wrong_outputs_before_response'], 3)

    def test_unknown_historical_score_cannot_close_a_run_as_known_nonstale(self):
        values = fixture(predictions=('old', 'old', 'unknown', 'new'))
        def uncertain(q, pred, frame, objects):
            return dict(known=False) if pred == 'unknown' else grade(q, pred, frame, objects)
        for event in change_events(*values, grade_fn=uncertain):
            self.assertTrue(event['stale_runs'][0]['right_censored'])
            self.assertEqual(event['historical_unscorable_outputs'], 1)

    def test_unpaired_events_are_rejected(self):
        scored, frames, objects = fixture()
        with self.assertRaisesRegex(ValueError, 'not paired'):
            change_events([q for q in scored if q['method'] == 'native'], frames, objects, grade_fn=grade)

    def test_repeats_and_scenes_never_join(self):
        scored, frames, objects = fixture()
        extra = deepcopy(scored)
        for q in extra:
            q['repeat'] = 1
        for method in frames:
            more = deepcopy(frames[method])
            for f in more:
                f['repeat'] = 1
            frames[method] += more
        events = change_events(scored + extra, frames, objects, grade_fn=grade)
        self.assertEqual(len(events), 4)
        self.assertTrue(all(e['max_consecutive_stale_outputs'] == 2 for e in events))

    def test_paired_responders_and_empty_denominators(self):
        events = evaluate()
        events[1].update(responded=False, response_delay_ms=None)
        pair = paired_statistics(events)
        self.assertEqual(pair['native_only_responded'], 1)
        self.assertEqual(pair['both_responded'], 0)
        self.assertIsNone(pair['common_worldstream_mean_ms'])
        empty = statistics([])
        self.assertIsNone(empty['observed_response_rate_percent'])
        self.assertIsNone(empty['response_delay_mean_ms'])


if __name__ == '__main__':
    unittest.main()
