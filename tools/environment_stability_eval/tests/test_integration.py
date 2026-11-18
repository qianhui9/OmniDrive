"""CPU configuration checks and synthetic report contracts (no model inference)."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.environment_stability_eval.common import dump_jsonl, read, write
from tools.environment_stability_eval.metrics import report
from tools.environment_stability_eval.run import load_config
from tools.environment_stability_eval.tests import test_semantics as fixtures


class ConfigurationTests(unittest.TestCase):
    def test_real_existing_native_and_r0_configs(self):
        from tools.worldstream_m123.benchmark.worker import effective_config
        from worldstream.r0 import model_settings
        with tempfile.TemporaryDirectory(prefix='environment_stability_config_') as tmp:
            native_out, world_out = Path(tmp) / 'native', Path(tmp) / 'worldstream'
            native_out.mkdir(); world_out.mkdir()
            native = effective_config('B0', 'planning', native_out)
            world = effective_config('B3', 'planning', world_out)
            for key, value in model_settings().items():
                self.assertEqual(world.model[key], value)
            self.assertEqual(native.data.test.pipeline, world.data.test.pipeline)
            for key in ('m1_enable', 'm2_enable', 'm3_enable', 'worldstream_require_m123'):
                self.assertFalse(native.model.get(key, False))
            self.assertEqual(native.get('fp16'), world.get('fp16'))
            self.assertEqual(native.model.img_backbone, world.model.img_backbone)
            self.assertEqual(native.model.lm_head, world.model.lm_head)

    def test_shipped_paths_exist(self):
        from tools.environment_stability_eval.common import ROOT
        config = load_config(Path(__file__).parents[1] / 'config.yaml')
        for key in ('infos', 'data_root', 'main_manifest', 'checkpoint', 'r0_config'):
            self.assertTrue((ROOT / config[key]).exists(), key)


class ReportContractTests(unittest.TestCase):
    def test_complete_report_and_figures_with_test_only_receipts(self):
        # SYNTHETIC TEST ONLY: fixed toy timings exist solely to test serialization
        # and aggregation; TemporaryDirectory removes them. Never paper results.
        from tools.worldstream_m123.r0_task_evaluation import sha
        from tools.environment_stability_eval.common import ROOT
        with tempfile.TemporaryDirectory(prefix='environment_stability_SYNTHETIC_TEST_') as tmp:
            out = Path(tmp)
            fixture = fixtures.ScoringTests()
            native, gt, objects = fixture.rows_and_gt('native', ['A car.', 'A car.', 'A car.'])
            world, _, _ = fixture.rows_and_gt('worldstream', ['A car.', 'A car.', 'A truck.'])
            window = dict(window_id='w', scene_id='s', frame_ids=['0', '1', '2'], frames=3,
                          stability_level='S1', change_rate=.1, start_timestamp=0, end_timestamp=5)
            config = dict(repeats=1, bootstrap_samples=0, seed=1, response_horizon_seconds=5.)
            manifest = dict(config=config, scenes=[dict(scene_id='s', tokens=['0', '1', '2'])],
                            thresholds=dict(thresholds=[.25, .5, .75]), smoke=True)
            dump_jsonl(out / 'test_windows.jsonl', [window])
            dump_jsonl(out / 'query_ground_truth.jsonl', gt)
            dump_jsonl(out / 'evaluation_objects.jsonl', [dict(frame_id=t, objects=values) for t, values in objects.items()])
            write(out / 'data_coverage.json', dict(SYNTHETIC_TEST_ONLY=True))
            for method, rows in (('native', native), ('worldstream', world)):
                directory = out / method
                directory.mkdir()
                for i, row in enumerate(rows):
                    row.update(frame_index=i, groups=['vehicle_category'], execution_status='COMPLETED',
                        e2e_ms=100., requests=[dict(llm_skipped=False)], fairness=dict(SYNTHETIC_TEST_ONLY=True),
                        executed_macs=dict(visual=10, llm=10, prefill=5, decode=5),
                        visual_diagnostics=None, llm_forward_lengths=[])
                dump_jsonl(directory / 'receipts.jsonl', rows)
                write(directory / 'processing_segments.json', [dict(scene_id='s', stability_level='S1',
                      window_id='w', repeat=0, frames=3, wall_seconds=.5)])
                write(directory / 'effective_runtime.json', dict(checkpoint_sha256='SYNTHETIC_TEST_ONLY',
                    device='SYNTHETIC_TEST_ONLY', gpu_uuid=None, fp16={}, batch_size=1,
                    generation={}, precision_by_dtype=['test']))
                write(directory / 'inference_complete.json', dict(status='COMPLETED', frames=3,
                    receipts_sha256=sha(directory / 'receipts.jsonl'),
                    segments_sha256=sha(directory / 'processing_segments.json')))
            with patch('tools.environment_stability_eval.evaluator.verify_manifest', return_value=manifest):
                report(out)
                with self.assertRaises(FileExistsError):
                    report(out)
            summary = read(out / 'experiment_summary.json')
            self.assertTrue(summary['smoke'])
            self.assertTrue(summary['fair_inputs_verified'])
            self.assertEqual(len(summary['figures']), 10)
            for name in ('raw_frame_results.csv', 'raw_qa_results.csv', 'latency_summary.csv',
                         'accuracy_summary.csv', 'compute_summary.csv', 'response_delay_summary.csv'):
                self.assertTrue((out / name).is_file())
            native_s1 = next(r for r in summary['accuracy'] if r['method'] == 'native' and r['stability_level'] == 'S1')
            self.assertEqual(native_s1['stale_answer_rate_percent'], 100.)
            self.assertIsNone(next(r for r in summary['accuracy'] if r['stability_level'] == 'S4')['overall_accuracy_percent'])


if __name__ == '__main__':
    unittest.main()
