"""CPU-only integrity checks. These are tests, never experiment measurements."""
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import hashlib
from unittest.mock import patch

import numpy as np
import torch

from .memory import ProcessMonitor, SizeLedger, module_snapshot
from .report import overall_by_repeat, stats
from .run import now, write


class AccountingTests(unittest.TestCase):
    def test_expanded_selection_short_logs(self):
        from .selection import choose_scenes
        candidates = [dict(scene_id=f'{log}{i}', log_id=log,
            selection_features=dict(object_mean=i, map_mean=i*i, night=False, rain=False))
            for log, count in [('a', 1), ('b', 5), ('c', 5)] for i in range(count)]
        selected = choose_scenes(candidates, ['a0', 'b0', 'c0'], 8)
        self.assertEqual(len({s['scene_id'] for s in selected}), 8)
        self.assertTrue({'a0', 'b0', 'c0'} <= {s['scene_id'] for s in selected})
        self.assertEqual(sorted(sum(s['log_id'] == log for s in selected) for log in ('a', 'b', 'c')), [1, 3, 4])
        self.assertTrue(any(s['selection_reason'] == 'short_log_slot_redistribution' for s in selected))
        self.assertEqual(selected, choose_scenes(list(reversed(candidates)), ['a0', 'b0', 'c0'], 8))
        with self.assertRaises(ValueError):
            choose_scenes(candidates, ['a0', 'b0', 'c0'], 20)

    def test_fresh_uss_and_scene_sampler(self):
        monitor = ProcessMonitor(gpu=False)
        monitor.uss = 1000000  # An old load-phase reading must never leak.
        monitor.last_uss_at = time.perf_counter()
        with patch.object(monitor.process, 'memory_info', return_value=SimpleNamespace(rss=100)), \
             patch.object(monitor.process, 'memory_full_info', return_value=SimpleNamespace(rss=120, uss=90)):
            row = monitor.sample()
            self.assertIsNone(row['cpu_uss_bytes'])
            monitor.set_phase('measurement')
            monitor.set_scene('scene1')
            row = monitor.sample(full=True)
            self.assertEqual((row['cpu_rss_bytes'], row['cpu_uss_bytes']), (120, 90))
            self.assertTrue(row['rss_uss_same_read'])
            self.assertEqual(monitor.phases['scene:scene1']['rss_peak'], 120)
            self.assertEqual(monitor.phases['measurement']['uss_peak'], 90)

    def test_cpu_plateau_pass_and_timeout(self):
        from .memory import stable_cpu_baseline
        config = dict(min_samples=3, interval_seconds=.001, timeout_seconds=.008, relative_tolerance=0.)
        monitor = SimpleNamespace(sample=lambda full: dict(cpu_rss_bytes=100, cpu_uss_bytes=80))
        self.assertTrue(stable_cpu_baseline(monitor, config)['stable'])
        counter = [0]
        def oscillate(full):
            counter[0] += 1
            return dict(cpu_rss_bytes=100+100*(counter[0] % 2), cpu_uss_bytes=80)
        result = stable_cpu_baseline(SimpleNamespace(sample=oscillate), config)
        self.assertFalse(result['stable'])
        self.assertFalse(result['cache_flush_performed'])

    def test_scene_ci_averages_repeats(self):
        from .extended_statistics import paired_log_ci, distribution
        rows = []
        for repeat in range(3):
            for sid in range(4):
                for method, delta in [('native', 0), ('worldstream', 1)]:
                    rows.append(dict(method=method, workload='S4', repeat=repeat, scene_id=str(sid),
                        log_id=str(sid//2), peak_gpu_allocated_GB=10+delta,
                        peak_cpu_rss_GB=5+delta, incremental_peak_cpu_rss_GB=1+delta))
        ci = paired_log_ci(rows, 3, samples=30)
        for r in ci:
            self.assertEqual((r['distinct_scenes'], r['log_clusters']), (4, 2))
            self.assertEqual(r['mean_scene_peak_delta'], 1.)
            self.assertEqual((r['ci95_low'], r['ci95_high']), (1., 1.))
        d = distribution([0, 10, 20, 30, 40])
        self.assertEqual(d['p95'], 38.)
        self.assertEqual(d['p99'], 39.6)

    def test_extended_outputs_cpu_fixtures(self):
        # Unit-test fixtures only: no charts or benchmark results are produced.
        from .extended_statistics import extend
        import json
        with tempfile.TemporaryDirectory(prefix='resource-extended-test-') as folder:
            root = Path(folder)
            scenes = [dict(scene_id=str(i), log_id=str(i)) for i in range(2)]
            manifest = dict(scenes=scenes, config=dict(repeats=1, workloads=['S4'], seed=1, bootstrap_samples=10))
            base = dict(cpu_rss_bytes=100, cpu_uss_bytes=80, gpu_allocated_bytes=0,
                        rss_uss_same_read=True, sample_duration_ms=.1)
            runs, frames = [], []
            for method in ('native', 'worldstream'):
                runs.append(dict(method=method, workload='S4', repeat=0, baseline=base,
                    warmup_end=base, cpu_baseline_stable=True,
                    metrics=dict(incremental_peak_cpu_rss_bytes=10, incremental_peak_cpu_rss_from_warmup_bytes=10)))
                records = []
                for scene in scenes:
                    records.append(dict(**scene, frames=2, start=base,
                        end=dict(gpu_peak_allocated_bytes=0, gpu_peak_reserved_bytes=0),
                        sampling=dict(rss_peak=110, rss_time_integral=210, sampled_seconds=2,
                                      rss_sum=210, samples=2, uss_peak=90, nvml_peak=None)))
                    for fi in range(2):
                        row = dict(method=method, workload='S4', repeat=0, scene_id=scene['scene_id'],
                                   gpu_allocated_bytes=0, gpu_reserved_bytes=0, cpu_rss_bytes=100+fi,
                                   cpu_uss_bytes=80+fi)
                        row.update({c+'_'+k: 0 for c in ('OGSM', 'DAVP', 'QCSE')
                                    for k in ('cpu_bytes', 'gpu_bytes', 'entries', 'elements')})
                        if method == 'worldstream':
                            # Opposing peaks: true same-frame Total max=100,
                            # whereas summing independent maxima would be 200.
                            row['OGSM_cpu_bytes'] = 100 if fi == 0 else 0
                            row['DAVP_cpu_bytes'] = 0 if fi == 0 else 100
                        frames.append(row)
                write(root/'runs'/f'S4_r00_{method}'/'scene_resources.json', records)
            extend(root, manifest, runs, frames)
            result = json.loads((root/'extended_statistics.json').read_text())
            self.assertEqual(len(result['scene_resources']), 4)
            self.assertEqual(len(result['scene_module_distributions']), 4*4*4)
            self.assertTrue((root/'tail_memory.csv').exists())
            self.assertIn('95% CI', (root/'extended_report.md').read_text())
            total = [r for r in result['scene_module_distributions'] if r['method'] == 'worldstream'
                     and r['component'] == 'Total' and r['metric'] == 'cpu_MB']
            self.assertTrue(all(r['maximum'] == .0001 for r in total))

    def test_relocated_policy_is_content_checked(self):
        from .run import validate_runtime_config
        with tempfile.TemporaryDirectory(prefix='resource-policy-') as folder:
            a, b = Path(folder)/'a.json', Path(folder)/'b.json'
            a.write_text('{"policy":1}')
            b.write_text(a.read_text())
            actual = dict(selective_answer_policy=str(a), draft_length=8)
            frozen = dict(selective_answer_policy=str(b), draft_length=8)
            result = validate_runtime_config('M3', actual, frozen)
            self.assertTrue(result['relocated_policy']['byte_identical'])
            with self.assertRaisesRegex(ValueError, 'differs from main'):
                validate_runtime_config('M3', dict(actual, draft_length=7), frozen)
            b.write_text('{"policy":2}')
            with self.assertRaisesRegex(ValueError, 'contents differ'):
                validate_runtime_config('M3', actual, frozen)

    def test_worker_error_shows_traceback(self):
        from .run import worker_error_tail
        with tempfile.TemporaryDirectory(prefix='resource-log-') as folder:
            path = Path(folder)/'worker.log'
            path.write_text('warning:' + 'x'*10000 + '\nTraceback (most recent call last):\n  File "test.py", line 1\nAssertionError: actual error\n')
            message = worker_error_tail(path)
            self.assertIn('AssertionError: actual error', message)
            self.assertNotIn('warning:', message)

    def test_numeric_gpu_visibility(self):
        import os
        from .run import configure_gpu_visibility
        inventory = '0, GPU-test0, 00000000:81:00.0\n1, GPU-test1, 00000000:01:00.0\n'
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-incompatible'}), \
                patch('tools.experiments.resource_overhead.run.subprocess.check_output', return_value=inventory):
            binding = configure_gpu_visibility(0)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '1')
            self.assertEqual(os.environ['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
            self.assertEqual(binding['uuid'], 'GPU-test0')
            self.assertEqual(torch.cuda._parse_visible_devices(), {1})
            with self.assertRaisesRegex(ValueError, 'absent'):
                configure_gpu_visibility(2)

    def test_storage_views_and_cycles(self):
        tensor = torch.arange(1024, dtype=torch.float32)
        values = [tensor[:1], tensor[2:], tensor]
        values.append(values)
        measured = SizeLedger().measure(values)
        self.assertEqual(measured['cpu_storage_bytes'], tensor.storage().nbytes())
        self.assertEqual(measured['unique_storages'], 1)

    def test_numpy_shared_owner(self):
        array = np.arange(1000, dtype=np.float64)
        measured = SizeLedger().measure([array[::2], array[1:], array, torch.from_numpy(array)])
        self.assertEqual(measured['cpu_storage_bytes'], array.nbytes)
        self.assertEqual(measured['unique_storages'], 1)

    def test_weights_excluded(self):
        model = torch.nn.Linear(4, 8)
        model.register_buffer('cached_buffer', torch.arange(9))
        measured = SizeLedger(model).measure([model, model.weight.detach(), model.cached_buffer])
        self.assertEqual(measured['cpu_storage_bytes'], 0)

    def test_real_runtime_cache_inventory(self):
        from worldstream.m1.state_manager import WorldStateManager
        from worldstream.m2.state_cache import StateCache
        from worldstream.m3.runtime import M3Runtime
        from worldstream.r0 import m3_settings
        # Small test fixtures exercise the real cache commit/remember APIs.
        # They are not VLM inference and are not exported to experiment tables.
        model = torch.nn.Linear(1, 1)
        model.m1_manager = WorldStateManager()
        cache = StateCache(1024**2)
        images = torch.zeros(1, 3, 4, 4)
        block = torch.ones(1, 2, 2, 4)
        cache.commit(images, {0: block}, dict(scene_id='test', frame_id='test0',
            camera_ids=['front'], timestamp=0.), ('test',), block0_input=block,
            sources={0: []})
        model.m2_runtime = SimpleNamespace(cache=cache)
        model.m3_runtime = M3Runtime(m3_settings())
        model.m3_runtime.answer_memory.remember('Is there a car in front?', (1, 2), (3, 4, 5),
            dict(scene_id='test', frame_id='test0', frame_index=0, timestamp=0.),
            ('model', 'tokenizer', 'generation'), None)
        result = module_snapshot(model)
        davp = next(m for m in result if m['component'] == 'DAVP')
        qcse = next(m for m in result if m['component'] == 'QCSE')
        self.assertEqual(davp['cached_windows'], 1)
        self.assertEqual(davp['cpu_storage_bytes'], cache.bytes)
        self.assertEqual(qcse['entries'], 1)
        self.assertEqual(qcse['historical_candidate_tokens'], 3)
        self.assertEqual(qcse['persistent_kv_bytes'], 0)
        self.assertGreater(qcse['candidate_cache_bytes'], 0)

    def test_global_module_dedup(self):
        from worldstream.m1.state_manager import WorldStateManager
        model = torch.nn.Linear(1, 1)
        shared = torch.ones(20)
        model.m1_manager = WorldStateManager()
        model.m1_manager.last_masks = {'test': shared}
        model.m2_runtime = SimpleNamespace(cache=SimpleNamespace(outputs={}, global_anchor=None), alias=shared)
        model.m3_runtime = None
        result = module_snapshot(model)
        self.assertEqual(sum(m['cpu_storage_bytes'] for m in result), shared.storage().nbytes())

    def test_repeat_statistics(self):
        value = stats([1., 2., 3.])
        self.assertEqual(value['std'], 1.)
        self.assertFalse(stats([1.])['std_defined'])
        self.assertIsNone(stats([None])['mean'])
        rows = [dict(method='m', repeat=0, metrics=dict(peak_gpu_GB=x, baseline_gpu_GB=x)) for x in (1., 3.)]
        merged = overall_by_repeat(rows, ['method', 'repeat'])[0]['metrics']
        self.assertEqual(merged['peak_gpu_GB'], 3.)
        self.assertEqual(merged['baseline_gpu_GB'], 2.)

    def test_streamed_input_digest(self):
        from .run import input_fingerprint
        for dtype in (torch.float16, torch.float32):
            x = torch.arange(400000, dtype=torch.float32).to(dtype)
            actual = input_fingerprint([x])[0]
            self.assertEqual(actual['sha256'], hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())

    def test_s1_s5_schedule(self):
        from tools.worldstream_m123.frozen_main_benchmark import WORKLOADS
        from tools.worldstream_m123.continuous_stream import question_schedule
        counts = {}
        for key, _, workload, frequency in WORKLOADS:
            counts[key] = [len(question_schedule(workload, frequency, i, ['a', 'b', 'c'], [])) for i in range(4)]
        self.assertEqual(counts, dict(S1=[1, 0, 0, 0], S2=[1, 0, 1, 0], S3=[1]*4, S4=[3]*4, S5=[3]*4))

    def test_incomplete_results_rejected(self):
        from .report import summarize
        with tempfile.TemporaryDirectory(prefix='resource-smoke-') as folder:
            root = Path(folder)
            write(root/'manifest.json', dict(config=dict(methods=['native', 'worldstream'], workloads=['S4'], repeats=3)))
            with self.assertRaisesRegex(ValueError, 'Incomplete experiment'):
                summarize(root)
            self.assertFalse((root/'summary.json').exists())

    def test_rss_monitor(self):
        monitor = ProcessMonitor(gpu=False, interval=.01, uss_interval=.01)
        monitor.start()
        monitor.set_phase('measurement')
        payload = bytearray(4 * 1024**2)
        time.sleep(.06)
        record = monitor.sample()
        result = monitor.finish()
        self.assertGreater(record['cpu_rss_bytes'], len(payload))
        self.assertGreater(result['phases']['measurement']['samples'], 1)
        self.assertGreater(result['process_lifetime_peak_rss_bytes'], 0)
        self.assertFalse(result['errors'])


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AccountingTests)
    with (out/'tests.log').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    import os
    write(out/'cpu_smoke.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        finished_at=now(), tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        cpu_affinity=sorted(os.sched_getaffinity(0)), torch=torch.__version__,
        model_inference_performed=False, benchmark_data=False,
        note='Actual CPU allocator/RSS sampler and real runtime cache API tests with small test fixtures. No GPU/cache benchmark result claimed.'))
    print((out/'tests.log').read_text())
    if not result.wasSuccessful():
        raise RuntimeError('CPU smoke failed')
