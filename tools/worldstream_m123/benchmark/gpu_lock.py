"""Serialize this experiment's model-loading jobs before checking GPU idleness."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def gpu_job_lock():
    # Waiting on a file lock also covers CPU checkpoint loading, when GPU memory
    # alone cannot reveal another job that is about to allocate the same GPU.
    path = Path('/tmp/omnidrive_m123_latency_gpu.lock')
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
