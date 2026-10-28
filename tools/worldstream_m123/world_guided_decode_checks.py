"""Serialize final actual-GPU regression and environment witness."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import torch
from tools.worldstream_m123.benchmark.common import write,sha256
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
from tools.worldstream_m2.global_residual_benchmark import wait_gpu


def checks(root):
    with gpu_job_lock():
        wait_gpu()
        assert not ({4,5}&os.sched_getaffinity(0))
        torch.manual_seed(0)
        x=torch.arange(64,dtype=torch.float32,device='cuda').reshape(8,8)
        witness=float((x@x.T).sum());assert witness==510720.
        command=['taskset','-c','0-3,6-31',sys.executable,'-m','pytest',
            'tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','tests/worldstream_m123','-q']
        with (root/'tests_gpu.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        write(root/'gpu_regression.json',dict(status='PASS' if result.returncode==0 else 'FAIL',
            returncode=result.returncode,command=command,kernel_witness=witness,
            gpu=torch.cuda.get_device_name(0),affinity=sorted(os.sched_getaffinity(0)),
            torch_version=torch.__version__,weights_sha256=sha256('ckpts/iter_10548.pth')))
        if result.returncode:raise RuntimeError('GPU regression failed; see tests_gpu.log')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs/m3_world_guided_decode_run01'))
    checks(parser.parse_args().root)
