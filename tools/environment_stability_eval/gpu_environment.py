"""Select the existing, kernel-matched driver bundle before loading CUDA.

Only the new experiment process is re-executed. System drivers, kernel modules,
other processes, and existing results are never changed.
"""
import os
from pathlib import Path
import subprocess

from .common import PYTHON


def nvml_probe(env=None):
    command = ['nvidia-smi', '--id=0', '--query-gpu=driver_version',
               '--format=csv,noheader,nounits']
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=15, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('GPU preflight could not run nvidia-smi: ' + str(exc)) from exc


def ensure_gpu_environment(arguments):
    """Return provenance or exec a fresh interpreter with matching cached libs."""
    result = nvml_probe()
    detail = '\n'.join(v.strip() for v in (result.stdout, result.stderr) if v and v.strip())
    selected = os.environ.get('OMNIDRIVE_DRIVER_COMPAT_MANIFEST')
    if result.returncode == 0:
        provenance = dict(mode='system', nvml_driver_version=result.stdout.strip())
        if selected:
            from tools.worldstream_ablation.driver_compat import verify
            from tools.worldstream_ablation.run import sha
            manifest = Path(selected).resolve()
            bundle = verify(manifest.parent)
            provenance.update(mode='process_local_matching_driver', manifest=str(manifest),
                manifest_sha256=sha(manifest), kernel_version=bundle['kernel_version'],
                library_directory=bundle['library_directory'])
        return provenance
    if result.returncode != 18 or 'driver/library version mismatch' not in detail.lower():
        raise RuntimeError(f'GPU preflight failed (nvidia-smi exit {result.returncode}): {detail}')
    if selected:
        raise RuntimeError('Selected process-local driver still fails; refusing an exec loop: ' + detail)
    from tools.worldstream_ablation.driver_compat import DEFAULT_ROOT, environment, kernel_version
    version = kernel_version()
    bundle = DEFAULT_ROOT / version
    if not (bundle / 'manifest.json').is_file():
        raise RuntimeError(f'{detail}\nLoaded kernel driver: {version}. Matching bundle missing: {bundle}. '
            'Prepare it with: taskset -c 0-3,6-31 ' + PYTHON +
            ' -B -m tools.worldstream_ablation.driver_compat prepare')
    env = environment(bundle)
    probe = nvml_probe(env)
    if probe.returncode:
        raise RuntimeError(f'Matching driver bundle cannot initialize NVML (exit {probe.returncode}): '
                           + probe.stdout + probe.stderr)
    print('PROCESS_DRIVER_COMPAT_REEXEC', bundle, 'kernel=' + version, flush=True)
    # CUDA must load the matched library from process startup, not via late CDLL.
    command = [PYTHON, '-B', '-u', '-m', 'tools.environment_stability_eval.run', *arguments]
    os.execve(PYTHON, command, env)
