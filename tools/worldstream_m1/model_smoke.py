"""Bounded real PyTorch inference; optional disabled/enabled tensor-digest comparison."""
import argparse
import hashlib
from pathlib import Path
import random
from time import perf_counter,process_time
import numpy as np
from .io_utils import write_json

_status_path = None


def digest(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    if hasattr(value, 'tensor'): return digest(value.tensor)
    if isinstance(value, dict): return {k: digest(v) for k, v in value.items() if k != 'text_out'}
    if isinstance(value, (list, tuple)): return [digest(v) for v in value]
    return value
