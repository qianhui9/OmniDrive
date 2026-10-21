"""Rebuild a source-anchored audit before changing any execution algorithm."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def stats(values):
    return dict(count=len(values), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95))) if values else None
