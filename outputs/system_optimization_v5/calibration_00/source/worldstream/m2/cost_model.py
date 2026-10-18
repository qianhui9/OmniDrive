"""Versioned, per-block measured costs. Unknown or mismatched records fail closed."""
import hashlib
import json
import weakref
import torch

SCHEMA_VERSION = 2
_fingerprints = weakref.WeakKeyDictionary()


def block_fingerprint(block):
    tensors = list(block.named_parameters())+list(block.named_buffers())
    revision = tuple((n,t.data_ptr(),t._version,str(t.dtype),str(t.device)) for n,t in tensors)
    old = _fingerprints.get(block)
    if old is not None and old[0] == revision:
        return old[1]
    digest = hashlib.sha256()
    for name,t in tensors:
        digest.update(name.encode());digest.update(str(t.dtype).encode())
        digest.update(t.detach().cpu().contiguous().numpy().tobytes())
    result = digest.hexdigest()
    _fingerprints[block] = (revision,result)
    return result


def cost_key(block, x, block_index=None):
    """Content fingerprint prevents reuse across same-shape different checkpoints."""
    props = torch.cuda.get_device_properties(x.device) if x.is_cuda else None
    payload = dict(schema=SCHEMA_VERSION, block_index=block_index,
        torch=torch.__version__,cuda=torch.version.cuda,
        gpu=(props.name,props.major,props.minor,props.total_memory) if props else 'cpu',
        input_shape=list(x.shape), input_stride=list(x.stride()),dtype=str(x.dtype),
        window=block.window_size, heads=block.attn.num_heads, flash=block.attn.flash_attn,
        scale=block.attn.scale, norm_eps=[block.norm1.eps,block.norm2.eps],
        model_version=block_fingerprint(block), autocast=torch.is_autocast_enabled(),
        autocast_dtype=str(torch.get_autocast_gpu_dtype()),
        tf32=[torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32])
    return json.dumps(payload,sort_keys=True)


class CostModel:
    def __init__(self, path=''):
        self.records=[]
        self.status='unprofiled_cost'
        if path:
            try:
                with open(path) as handle: data=json.load(handle)
                if data.get('schema_version') != SCHEMA_VERSION:
                    self.status='cost_schema_mismatch'
                    return
                records=data['records']
                for record in records:
                    key=json.loads(record['key'])
                    if key['schema']!=SCHEMA_VERSION or not isinstance(key['block_index'],int):
                        raise ValueError('Missing per-block identity')
                policy=data.get('formal_quality_policy')
                approved=set(policy.get('approved_counts',[])) if isinstance(policy,dict) else set()
                self.records=[]
                for record in records:
                    record=dict(record)
                    if record['updated'] not in approved:
                        record['quality_certified']=False
                        record.setdefault('quality_rejection_reason','downstream_quality_unverified')
                    self.records.append(record)
                self.status='ready'
            except (OSError,ValueError,KeyError,TypeError,AttributeError):
                self.status='invalid_cost_table'

    def lookup(self,key,count):
        return next((r for r in self.records if r['key']==key and r['updated']==count),None)

    def beneficial(self,key,count):
        record=self.lookup(key,count)
        if record is None:
            return False,'unprofiled_cost'
        if not record.get('quality_certified',False):
            return False,record.get('quality_rejection_reason','numerical_quality_unverified')
        good=(record['incremental_gpu_ms']<record['full_gpu_ms']
              and record['incremental_wall_ms']<record['full_wall_ms'])
        return good,None if good else 'estimated_cost_not_beneficial'
