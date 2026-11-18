"""Driver-startup regression: no system changes and no silent GPU bypass."""
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools.environment_stability_eval import gpu_environment as gpu
from tools.worldstream_ablation import driver_compat as compat


def result(code=0, stdout='580.173.02\n', stderr=''):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


MISMATCH = result(18, 'Failed to initialize NVML: Driver/library version mismatch\n')


class Reexecuted(Exception):
    pass


class DriverEnvironmentTests(unittest.TestCase):
    def test_healthy_system_keeps_existing_environment(self):
        with patch.dict(os.environ, {'OMNIDRIVE_DRIVER_COMPAT_MANIFEST': ''}), \
             patch.object(gpu, 'nvml_probe', return_value=result()), \
             patch.object(gpu.os, 'execve') as execute:
            self.assertEqual(gpu.ensure_gpu_environment(['--run-full'])['mode'], 'system')
            execute.assert_not_called()

    def test_mismatch_reexecutes_same_arguments_with_existing_bundle(self):
        with tempfile.TemporaryDirectory(prefix='driver_startup_test_') as tmp:
            bundle = Path(tmp) / '580.173.02'
            bundle.mkdir()
            (bundle / 'manifest.json').write_text('{}')  # TEST ONLY; real verifier is mocked below.
            env = {'LD_LIBRARY_PATH': str(bundle), 'PATH': '/existing'}
            arguments = ['--run-full', '--output', '/tmp/new-run', '--_lock-fd', '7']
            with patch.dict(os.environ, {'OMNIDRIVE_DRIVER_COMPAT_MANIFEST': ''}), \
                 patch.object(gpu, 'nvml_probe', side_effect=[MISMATCH, result()]) as probe, \
                 patch.object(compat, 'DEFAULT_ROOT', Path(tmp)), \
                 patch.object(compat, 'kernel_version', return_value='580.173.02'), \
                 patch.object(compat, 'environment', return_value=env) as select, \
                 patch.object(gpu.os, 'execve', side_effect=Reexecuted) as execute:
                with self.assertRaises(Reexecuted):
                    gpu.ensure_gpu_environment(arguments)
                select.assert_called_once_with(bundle)
                self.assertEqual(execute.call_args.args[1][-len(arguments):], arguments)
                self.assertEqual(execute.call_args.args[2], env)
                self.assertEqual(probe.call_args.args, (env,))

    def test_selected_bundle_is_verified_and_recorded(self):
        with patch.dict(os.environ, {'OMNIDRIVE_DRIVER_COMPAT_MANIFEST': '/tmp/test-driver/manifest.json'}), \
             patch.object(gpu, 'nvml_probe', return_value=result()), \
             patch.object(compat, 'verify', return_value=dict(kernel_version='580.173.02', library_directory='/tmp/test-driver/lib')), \
             patch('tools.worldstream_ablation.run.sha', return_value='TEST_ONLY'):
            provenance = gpu.ensure_gpu_environment(['--run-full'])
            self.assertEqual(provenance['mode'], 'process_local_matching_driver')
            self.assertEqual(provenance['manifest_sha256'], 'TEST_ONLY')

    def test_mismatch_cannot_reexec_forever(self):
        with patch.dict(os.environ, {'OMNIDRIVE_DRIVER_COMPAT_MANIFEST': '/tmp/already-selected/manifest.json'}), \
             patch.object(gpu, 'nvml_probe', return_value=MISMATCH), \
             patch.object(gpu.os, 'execve') as execute:
            with self.assertRaisesRegex(RuntimeError, 'refusing an exec loop'):
                gpu.ensure_gpu_environment(['--run-full'])
            execute.assert_not_called()

    def test_missing_bundle_reports_real_cause_without_download(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {'OMNIDRIVE_DRIVER_COMPAT_MANIFEST': ''}), \
             patch.object(gpu, 'nvml_probe', return_value=MISMATCH), \
             patch.object(compat, 'DEFAULT_ROOT', Path(tmp)), \
             patch.object(compat, 'kernel_version', return_value='580.173.02'):
            with self.assertRaisesRegex(RuntimeError, 'Matching bundle missing'):
                gpu.ensure_gpu_environment(['--run-full'])

    def test_other_driver_errors_are_not_treated_as_idle(self):
        with patch.object(gpu, 'nvml_probe', return_value=result(9, '', 'Driver not loaded')):
            with self.assertRaisesRegex(RuntimeError, 'exit 9.*Driver not loaded'):
                gpu.ensure_gpu_environment(['--run-full'])

    def test_unusable_matching_bundle_fails_before_expensive_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / '580.173.02'
            bundle.mkdir()
            (bundle / 'manifest.json').write_text('{}')  # TEST ONLY.
            with patch.dict(os.environ, {'OMNIDRIVE_DRIVER_COMPAT_MANIFEST': ''}), \
                 patch.object(gpu, 'nvml_probe', side_effect=[MISMATCH, result(18, 'still mismatched', '')]), \
                 patch.object(compat, 'DEFAULT_ROOT', Path(tmp)), \
                 patch.object(compat, 'kernel_version', return_value='580.173.02'), \
                 patch.object(compat, 'environment', return_value={}), \
                 patch.object(gpu.os, 'execve') as execute:
                with self.assertRaisesRegex(RuntimeError, 'cannot initialize NVML'):
                    gpu.ensure_gpu_environment(['--run-full'])
                execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
