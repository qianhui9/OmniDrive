"""Run regression tests through a spawn-safe module entry and existing GPU lock."""
import argparse
from collections import Counter
from contextlib import nullcontext, redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
from time import perf_counter


class Results:
    def __init__(self):
        self.outcomes = Counter()
        self.cuda_tests = []

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or (report.when == 'setup' and report.skipped):
            self.outcomes[report.outcome] += 1
            if 'cuda' in report.nodeid.lower() and report.when == 'call':
                self.cuda_tests.append(dict(nodeid=report.nodeid, outcome=report.outcome))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if {4, 5} & os.sched_getaffinity(0):
        raise RuntimeError('Launch with taskset -c 0-3,6-31')
    if not args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import pytest
    import torch
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    args.output.mkdir(parents=True, exist_ok=False)
    with gpu_job_lock() if args.gpu else nullcontext():
        if args.gpu:
            wait_gpu()
            if not torch.cuda.is_available():
                raise RuntimeError('GPU validation requested, CUDA unavailable')
        result = Results()
        tick = perf_counter()
        with (args.output / 'pytest.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            code = pytest.main(['-q', 'tests', '--disable-warnings'], plugins=[result])
        summary = dict(exit_code=int(code), counts=dict(result.outcomes),
                       CUDA_available=torch.cuda.is_available(),
                       device=torch.cuda.get_device_name(0) if args.gpu else None,
                       cpu_affinity=sorted(os.sched_getaffinity(0)), elapsed_s=perf_counter() - tick,
                       cuda_named_tests=result.cuda_tests, full_M123_GPU_experiments=0)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
        return int(code)


if __name__ == '__main__':
    raise SystemExit(main())
