"""Shared bookkeeping for the sole mainline benchmark; no strategy decisions."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch


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

def diagnostic_json(value):
    """Native diagnostics can contain unbounded uncertainty; never call it zero.

    Only raw external diagnostics use this encoding. New cost measurements still
    go through strict finite stats()/write() and fail on nonfinite values.
    """
    if isinstance(value, np.ndarray):
        return diagnostic_json(value.tolist())
    if isinstance(value, np.generic):
        return diagnostic_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return dict(status='Unknown', nonfinite_value=repr(value))
    if isinstance(value, dict):
        return {k: diagnostic_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic_json(v) for v in value]
    return value

def ground_truth(token):
    result = {}
    for folder in ('conv/val', 'vqa/val'):
        path = Path('data/nuscenes') / folder / (token+'.json')
        if path.exists():
            for pair in json.loads(path.read_text()):
                result[pair['question']] = pair['answer']
    return result

def typed_digest(value):
    if hasattr(value,'tensor'):value=value.tensor
    if isinstance(value,torch.Tensor):
        tensor=value.detach().cpu().contiguous()
        return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value,np.ndarray):
        return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,dict):return {k:typed_digest(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [typed_digest(v) for v in value]
    return value
