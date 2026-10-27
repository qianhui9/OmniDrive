"""Complete live state binding in one traversal, without a second JSON tree copy."""
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=64)
def schema_names(cls):
    return tuple(f.name for f in fields(cls))


def normalized(value):
    if isinstance(value, np.ndarray):
        payload = normalized(value.tolist()) if value.dtype.hasobject else hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        return ('ndarray', str(value.dtype), value.shape, payload)
    if isinstance(value, np.generic):
        return normalized(value.item())
    if is_dataclass(value):
        return {name: normalized(getattr(value,name)) for name in schema_names(type(value))}
    if isinstance(value, dict):
        return {str(k): normalized(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return ('nonfinite', repr(value))
    return value


def source_digest(store):
    # Dataclass metadata already includes geometry, descriptors, histories and
    # certifications. Hash each once, rather than adding duplicate summaries,
    # geometry hashes and certificate reprs before walking everything again.
    features = {}
    for fid, record in store.features.items():
        row = {name: normalized(getattr(record,name)) for name in schema_names(type(record)) if name != 'tensor'}
        tensor = record.tensor
        if isinstance(tensor, np.ndarray):
            row['tensor'] = normalized(tensor)
        elif tensor is not None:
            row['tensor'] = (tensor.data_ptr(), tensor._version, tuple(tensor.shape),
                             tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))
        else:
            row['tensor'] = None
        features[fid] = row
    value = ('complete_live_state_v3', store.generation, store.scene_id,
        store.execution_signature, store.next_anchor_id, sorted(store.committed_frame_ids),
        store.last_timestamp, store.commit_count, normalized(store.anchors), features,
        normalized(store.provenance))
    # normalized() creates an owned acyclic JSON tree. Circular bookkeeping is
    # redundant here; the canonical bytes and complete live-state binding stay identical.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,check_circular=False).encode()).hexdigest()
