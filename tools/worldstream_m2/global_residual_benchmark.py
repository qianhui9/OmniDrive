"""R0-required utilities; historical alternative runners have been retired."""

import subprocess

import time

def wait_gpu():
    while True:
        status=subprocess.check_output(['taskset','-c','0-3,6-31','nvidia-smi',
            '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if int(status.strip().splitlines()[0])<500:break
        print('WAIT GPU occupied; leave other processes untouched',flush=True)
        time.sleep(30)
