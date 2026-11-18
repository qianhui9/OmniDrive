"""Small CPU-only I/O helpers; importing these never initializes CUDA."""
import csv
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/home/qh/miniconda3/envs/OmniDrive/bin/python'
METHODS = ('native', 'worldstream')
LEVELS = ('S1', 'S2', 'S3', 'S4')
EVENT_TYPES = ('appearance', 'disappearance', 'relative_position', 'category',
               'attribute', 'spatial_relation')


@contextmanager
def gpu_lease(inherited_fd=None):
    """Use the existing project GPU lock for the entire paired short/full run.

    The existing helper does not expose its descriptor. Here the same lock path
    is opened so fresh workers can inherit one lease without relocking/deadlock.
    A child never releases its parent's lease between the paired methods.
    """
    if inherited_fd is not None:
        os.fstat(inherited_fd)
        fcntl.flock(inherited_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield inherited_fd
        return
    with Path('/tmp/omnidrive_m123_latency_gpu.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield handle.fileno()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def require_affinity():
    if {4, 5} & os.sched_getaffinity(0):
        raise RuntimeError('Use taskset -c 0-3,6-31; CPUs 4/5 are faulty')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    tmp.replace(path)


def jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dump_jsonl(path, rows):
    with Path(path).open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def csv_write(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple))
                             else v for k, v in row.items()})
