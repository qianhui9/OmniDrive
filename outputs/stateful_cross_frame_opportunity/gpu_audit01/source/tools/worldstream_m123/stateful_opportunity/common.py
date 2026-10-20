"""Fail-closed audit bookkeeping. These records never authorize execution."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

PYTHON = '/home/qh/miniconda3/envs/OmniDrive/bin/python'
AFFINITY = '0-3,6-31'
SAMPLING = dict(do_sample=True, temperature=0.1, top_p=0.75,
                max_new_tokens=320, num_beams=1, use_cache=True)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError('Unsupported audit value: '+str(type(item)))
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=convert)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(payload + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def stats(values):
    values = list(values)
    if not values:
        return None
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite measurements must be reported as missing')
    return dict(count=len(values), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), min=float(min(values)), max=float(max(values)))


@dataclass(frozen=True)
class Context:
    scene: str
    frame: str
    timestamp: float
    cameras: tuple
    epoch: int
    signature: str


def history_rejection(previous, current, max_age):
    """Audit isolation only; tensor equality and task sufficiency are separate."""
    if previous is None:
        return 'cold_start'
    if not previous.scene or previous.scene != current.scene:
        return 'scene_changed'
    if not previous.frame or not current.frame or previous.frame == current.frame:
        return 'not_cross_frame'
    if (not isinstance(previous.timestamp, Real) or not isinstance(current.timestamp, Real)
            or not math.isfinite(previous.timestamp) or not math.isfinite(current.timestamp)
            or current.timestamp <= previous.timestamp):
        return 'time_not_increasing'
    if current.timestamp - previous.timestamp > max_age:
        return 'expired'
    if (not previous.cameras or len(set(previous.cameras)) != len(previous.cameras)
            or previous.cameras != current.cameras):
        return 'camera_order_changed'
    if (type(previous.epoch) is not int or previous.epoch < 1 or previous.epoch != current.epoch):
        return 'state_epoch_changed'
    if not previous.signature or previous.signature != current.signature:
        return 'execution_changed'
    return None


def counts(eligible, attempted, admitted=0, actually_skipped=0, fallback=None):
    if not 0 <= actually_skipped <= admitted <= attempted <= eligible:
        raise ValueError('Invalid opportunity funnel')
    if fallback is None:
        fallback = attempted - admitted
    if fallback < 0 or admitted + fallback != attempted:
        raise ValueError('Admissions and fallbacks must partition attempts')
    return dict(eligible=eligible, attempted=attempted, admitted=admitted,
                actually_skipped=actually_skipped, fallback=fallback)


@contextmanager
def replace_method(obj, name, replacement):
    """Restore even after OOM/exception; avoid leaving bound-method cycles."""
    owned = name in vars(obj)
    old = getattr(obj, name)
    setattr(obj, name, replacement)
    try:
        yield old
    finally:
        if owned:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


def protected_files():
    return sorted([*Path('worldstream').rglob('*.py'),
                   *Path('projects').rglob('*.py'), *Path('configs').glob('worldstream*')])


def manifest(paths):
    return {str(p): sha256(p) for p in paths}


def source_evidence(path):
    path = Path(path)
    if not path.is_file():
        return dict(path=str(path), status='Missing', sha256=None, value=None)
    return dict(path=str(path), status='Available', sha256=sha256(path),
                value=json.loads(path.read_text()))
