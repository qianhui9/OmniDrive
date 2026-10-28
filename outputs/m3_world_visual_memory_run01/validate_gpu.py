"""Final regression and known-value witness; existing environment only."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
from tools.worldstream_m2.global_residual_benchmark import wait_gpu

root = Path(__file__).resolve().parent
assert not ({4, 5} & set(os.sched_getaffinity(0)))
with gpu_job_lock():
    wait_gpu()
    import torch
    import transformers
    torch.manual_seed(0)
    x = torch.arange(64, dtype=torch.float32, device='cuda').reshape(8, 8)
    value = float((x @ x.T).sum())
    assert value == 510720.
    environment = dict(interpreter=sys.executable, python=sys.version,
        torch=torch.__version__, transformers=transformers.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        affinity=sorted(os.sched_getaffinity(0)), witness=value,
        environment_rebuilt=False, packages_installed=False)
    (root/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    del x
    torch.cuda.empty_cache()
    command = ['taskset', '-c', '0-3,6-31', sys.executable, '-m', 'pytest', '-q',
        'tests/worldstream_m1', 'tests/worldstream_m2', 'tests/worldstream_m3', 'tests/worldstream_m123']
    with (root/'tests_gpu_final.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    (root/'final_test_result.json').write_text(json.dumps(dict(
        returncode=result.returncode, command=command, cuda_available=torch.cuda.is_available()), indent=2)+'\n')
    raise SystemExit(result.returncode)
