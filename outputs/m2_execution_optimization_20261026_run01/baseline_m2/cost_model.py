# Offline profiling/archived promotion utilities only. Serving M2 has no cost dependency.
"""Versioned, per-block measured costs. Unknown or mismatched records fail closed."""
import hashlib
import json
import weakref
import math
from pathlib import Path
import torch

SCHEMA_VERSION = 3
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


def cost_key(block, x, block_index=None, model=None, refresh_policy=None):
    """Content fingerprint prevents reuse across same-shape different checkpoints."""
    props = torch.cuda.get_device_properties(x.device) if x.is_cuda else None
    from .window_executor import NUMERICAL_POLICY
    payload = dict(schema=SCHEMA_VERSION,numerical_policy=NUMERICAL_POLICY,block_index=block_index,
        torch=torch.__version__,cuda=torch.version.cuda,
        gpu=(props.name,props.major,props.minor,props.total_memory) if props else 'cpu',
        input_shape=list(x.shape), input_stride=list(x.stride()),dtype=str(x.dtype),
        window=block.window_size, heads=block.attn.num_heads, flash=block.attn.flash_attn,
        scale=block.attn.scale, norm_eps=[block.norm1.eps,block.norm2.eps],
        attention_softmax_scale=getattr(getattr(block.attn,'inner_attn',None),'softmax_scale',None),
        attention_fp16_enabled=getattr(getattr(block.attn,'inner_attn',None),'fp16_enabled',None),
        inner_norm_eps=getattr(block.attn.inner_attn_ln,'eps',None),
        mlp_norm_eps=getattr(block.mlp.ffn_ln,'eps',None),
        model_version=block_fingerprint(block),
        backbone_version=block_fingerprint(model) if model is not None else None,
        backbone_geometry=(dict(kernel=model.patch_embed.proj.kernel_size,stride=model.patch_embed.proj.stride,
            padding=model.patch_embed.proj.padding,dilation=model.patch_embed.proj.dilation,
            cls_token=model.pretrain_use_cls_token,
            topology=[(b.window_size,b.use_residual_block) for b in model.blocks]) if model is not None else None),
        change_detection=refresh_policy[0] if refresh_policy else 'wsgrr_v1', refresh_policy=refresh_policy,
        autocast=torch.is_autocast_enabled(),
        autocast_dtype=str(torch.get_autocast_gpu_dtype()),
        tf32=[torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32],
        cudnn=[torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic],
        reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    return json.dumps(payload,sort_keys=True)


class CostModel:
    def __init__(self, path=''):
        self.records=[]
        self.production_ready=False
        self._evidence_hashes={}
        self.status='unprofiled_cost'
        if path:
            try:
                with open(path) as handle: data=json.load(handle)
                if data.get('schema_version') != SCHEMA_VERSION:
                    self.status='cost_schema_mismatch'
                    return
                records=data['records']
                if not isinstance(records,list):raise ValueError('records must be a list')
                identities=set()
                for record in records:
                    key=json.loads(record['key'])
                    if key['schema']!=SCHEMA_VERSION or not isinstance(key['block_index'],int):
                        raise ValueError('Missing per-block identity')
                    if type(record['updated']) is not int or record['updated'] < 0:
                        raise ValueError('Invalid updated window count')
                    if record.get('block') != key['block_index']:
                        raise ValueError('Block identity mismatch')
                    identity=(record['key'],record['updated'])
                    if identity in identities:raise ValueError('Duplicate window cost identity')
                    identities.add(identity)
                    for name in ('incremental_gpu_ms','full_gpu_ms','incremental_wall_ms','full_wall_ms'):
                        value=record[name]
                        if type(value) not in (float,int) or not math.isfinite(value) or value<0:
                            raise ValueError('Invalid measured window cost')
                policy=data.get('formal_quality_policy')
                counts=policy.get('approved_counts',[]) if isinstance(policy,dict) else []
                if any(type(c) is not int or c<0 for c in counts):raise ValueError('Invalid approvals')
                approved=set(counts)
                self.production_ready=data.get('production_ready') is True
                self.records=[]
                for record in records:
                    record=dict(record)
                    if record['updated'] not in approved or not self.production_ready:
                        record['quality_certified']=False
                        record.setdefault('quality_rejection_reason','downstream_quality_unverified')
                    self.records.append(record)
                self.status='ready' if self.records else 'empty_cost_table'
            except (OSError,ValueError,KeyError,TypeError,AttributeError):
                self.status='invalid_cost_table'

    def _approval_valid(self,record):
        approval=record.get('approval',{})
        if (not isinstance(approval,dict) or not isinstance(approval.get('approved_by'),str) or
                not approval['approved_by'].strip() or type(approval.get('updated')) is not int or
                approval.get('key_sha256')!=hashlib.sha256(record['key'].encode()).hexdigest() or
                approval.get('updated')!=record['updated']):
            return False
        for name in ('candidate','quality_report','numerical_evidence'):
            path=approval.get(name);expected=approval.get(name+'_sha256')
            if not isinstance(path,str) or not isinstance(expected,str):return False
            cache_key=(path,expected)
            if cache_key not in self._evidence_hashes:
                try:self._evidence_hashes[cache_key]=hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
                except OSError:self._evidence_hashes[cache_key]=False
            if not self._evidence_hashes[cache_key]:return False
        return True

    def lookup(self,key,count):
        return next((r for r in self.records if r['key']==key and r['updated']==count),None)

    def beneficial(self,key,count):
        record=self.lookup(key,count)
        if record is None:
            return False,self.status if self.status in ('empty_cost_table','invalid_cost_table','cost_schema_mismatch') else 'unprofiled_cost'
        if record.get('numerical_verified') is not True:
            return False,'window_numerical_evidence_missing'
        if record.get('quality_certified') is not True:
            return False,record.get('quality_rejection_reason','numerical_quality_unverified')
        if not self._approval_valid(record):
            return False,'production_approval_evidence_missing'
        good=(record['incremental_gpu_ms']<record['full_gpu_ms']
              and record['incremental_wall_ms']<record['full_wall_ms'])
        return good,None if good else 'estimated_cost_not_beneficial'

    def has_beneficial(self,key):
        """Bootstrap capture only for this model/layout with an admitted scale."""
        return any(r['key']==key and self.beneficial(key,r['updated'])[0] for r in self.records)

    def assessment(self,key,count):
        """Small audit of the exact proposed count; never authorizes another scale."""
        record=self.lookup(key,count)
        allowed,reason=self.beneficial(key,count)
        return dict(cost_record_found=record is not None,
            cost_key_matched=any(r['key']==key for r in self.records),
            numerical_verified=bool(record and record.get('numerical_verified') is True),
            quality_certified=bool(record and record.get('quality_certified') is True),
            approval_valid=bool(record and self._approval_valid(record)),
            beneficial=allowed,cost_rejection_reason=reason)
