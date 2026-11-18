"""Use existing process-only driver compatibility; never install/change drivers."""
import os
import sys
from tools.environment_stability_eval.common import PYTHON, require_affinity
from tools.environment_stability_eval.gpu_environment import nvml_probe


def main():
    require_affinity()
    # Offline preparation/reporting must also work with no GPU access.
    if '--prepare-only' in sys.argv or '--report-only' in sys.argv:
        from .run import main as run
        run()
        return
    result = nvml_probe()
    env = dict(os.environ)
    if result.returncode == 18 and 'driver/library version mismatch' in (result.stdout+result.stderr).lower():
        from tools.worldstream_ablation.driver_compat import DEFAULT_ROOT, kernel_version, environment
        env = environment(DEFAULT_ROOT / kernel_version())
        if nvml_probe(env).returncode:
            raise RuntimeError('Existing process-local driver compatibility failed')
    elif result.returncode:
        raise RuntimeError('GPU unavailable: '+result.stdout+result.stderr)
    os.execve(PYTHON, [PYTHON, '-B', '-u', '-m', 'tools.component_latency_eval.run', *sys.argv[1:]], env)


if __name__ == '__main__':
    main()
