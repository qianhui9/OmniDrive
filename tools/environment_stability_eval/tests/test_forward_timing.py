"""Toy times exercise interval boundaries only; never written as experiment data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tools.environment_stability_eval.forward_timing import ForwardClock, observe_forward


class FakeTime:
    def __init__(self):
        self.now = 0.
        self.pending = 0.
        self.sync_calls = 0

    def advance(self, seconds):
        self.now += seconds

    def synchronize(self):
        self.sync_calls += 1
        self.now += self.pending
        self.pending = 0.

    def clock(self):
        return self.now


class ClockTests(unittest.TestCase):
    def test_pending_preprocessing_is_excluded_but_model_work_is_included(self):
        time = FakeTime()
        time.pending = 3.  # preprocessing/H2D work must finish before t_start
        timer = ForwardClock(time.synchronize, time.clock)
        timer.start()
        time.advance(.1)
        time.pending = .9
        with timer.exclude('export'):
            time.advance(.2)
        result = timer.stop()
        self.assertAlmostEqual(result['forward_ms'], 1000.)
        self.assertAlmostEqual(result['forward_raw_ms'], 1200.)
        self.assertAlmostEqual(result['forward_excluded_ms'], 200.)

    def test_nested_exclusions_are_not_double_counted(self):
        time = FakeTime()
        timer = ForwardClock(time.synchronize, time.clock)
        timer.start()
        time.advance(.1)
        with timer.exclude('outer'):
            time.advance(.2)
            with timer.exclude('inner'):
                time.advance(.1)
        result = timer.stop()
        self.assertAlmostEqual(result['forward_ms'], 100.)
        self.assertEqual(result['forward_exclusion_calls'], {'outer': 1})
        self.assertNotIn('inner', result['forward_excluded_by_label_ms'])

    def test_exclusion_outside_model_call_does_not_leak_into_next_frame(self):
        time = FakeTime()
        timer = ForwardClock(time.synchronize, time.clock)
        with timer.exclude('not_forward'):
            time.advance(5.)
        timer.start()
        time.advance(.1)
        result = timer.stop()
        self.assertAlmostEqual(result['forward_ms'], 100.)
        self.assertEqual(result['forward_excluded_ms'], 0.)

    def test_each_frame_gets_new_accounting(self):
        time = FakeTime()
        timer = ForwardClock(time.synchronize, time.clock)
        for seconds in (.1, .2):
            timer.start()
            self.assertIsNone(timer.last)
            time.advance(seconds)
            with timer.exclude('export'):
                time.advance(.01)
            self.assertAlmostEqual(timer.stop()['forward_ms'], seconds * 1000)

    def test_incomplete_or_nested_root_interval_rejected(self):
        time = FakeTime()
        timer = ForwardClock(time.synchronize, time.clock)
        with self.assertRaises(RuntimeError):
            timer.stop()
        timer.start()
        with self.assertRaises(RuntimeError):
            timer.start()
        timer.abort()
        self.assertIsNone(timer.last)

    def test_exception_restores_exclusion_depth(self):
        time = FakeTime()
        timer = ForwardClock(time.synchronize, time.clock)
        timer.start()
        time.advance(.1)
        with self.assertRaises(ValueError):
            with timer.exclude('failed_export'):
                time.advance(.01)
                raise ValueError('test-only')
        self.assertEqual(timer.depth, 0)
        self.assertAlmostEqual(timer.stop()['forward_ms'], 100.)


class ObserverTests(unittest.TestCase):
    def test_final_export_only_excluded_and_answers_and_files_preserved(self):
        import torch
        time = FakeTime()

        class Tokenizer:
            def batch_decode(self, values, **kwargs):
                time.advance(.02)
                return ['test-only answer']

        class Model(torch.nn.Module):
            def __init__(self, path):
                super().__init__()
                self.tokenizer = Tokenizer()
                self.save_path = str(path) + '/'

            def forward(self):
                time.advance(.1)
                # Internal semantic decoding must remain part of inference.
                self.tokenizer.batch_decode([0])
                return self.simple_test_pts()

            def simple_test_pts(self):
                values = self.tokenizer.batch_decode([0])
                with open(self.save_path + 'answer.json', 'w') as handle:
                    time.advance(.03)
                    json.dump(values, handle)
                return values

        with tempfile.TemporaryDirectory(prefix='forward_timing_SYNTHETIC_TEST_') as temp:
            model = Model(temp)
            timer = ForwardClock(time.synchronize, time.clock)
            module = sys.modules[model.__class__.__module__]
            had_open = 'open' in vars(module)
            original_forward = model.forward
            with observe_forward(model, timer):
                self.assertEqual(model(), ['test-only answer'])
                self.assertAlmostEqual(timer.last['forward_ms'], 120.)
                self.assertAlmostEqual(timer.last['forward_raw_ms'], 170.)
                self.assertAlmostEqual(timer.last['forward_excluded_ms'], 50.)
                self.assertEqual(timer.last['forward_exclusion_calls'],
                    {'answer_text_decode': 1, 'answer_file_write': 1})
                self.assertEqual(json.loads((Path(temp) / 'answer.json').read_text()), ['test-only answer'])
            self.assertEqual('open' in vars(module), had_open)
            self.assertNotIn('batch_decode', vars(model.tokenizer))
            self.assertEqual(model.forward, original_forward)
            self.assertEqual(len(model._forward_hooks), 0)
            self.assertEqual(len(model._forward_pre_hooks), 0)

    def test_model_exception_removes_instrumentation(self):
        import torch
        class Tokenizer:
            def batch_decode(self, *args):
                return []
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tokenizer = Tokenizer()
                self.save_path = '/tmp/unused_TEST_ONLY/'
            def forward(self):
                raise ValueError('test-only model error')
        model, time = Model(), FakeTime()
        timer = ForwardClock(time.synchronize, time.clock)
        with self.assertRaises(ValueError):
            with observe_forward(model, timer):
                model()
        self.assertFalse(timer.active)
        self.assertIsNone(timer.last)
        self.assertFalse(model._forward_hooks)
        self.assertNotIn('batch_decode', vars(model.tokenizer))


if __name__ == '__main__':
    unittest.main()
