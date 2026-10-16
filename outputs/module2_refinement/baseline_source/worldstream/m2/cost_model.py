"""Measured total CUDA event and wall costs, with no extrapolation to unknown shapes."""
import json


class CostModel:
    def __init__(self, path=''):
        self.records = []
        if path:
            try:
                with open(path) as handle:
                    self.records = json.load(handle)['records']
            except (OSError, ValueError, KeyError):
                # A missing/offline table must never prevent original inference.
                self.records = []

    def beneficial(self, key, count):
        record = next((r for r in self.records if r['key'] == key and r['updated'] == count), None)
        if record is None:
            return False, 'unprofiled_cost'
        if not record.get('quality_certified', False):
            return False, 'numerical_quality_unverified'
        # Both total measurements include partition/pack/scatter/cache copy and launch gaps.
        good = (record['incremental_gpu_ms'] < record['full_gpu_ms']
                and record['incremental_wall_ms'] < record['full_wall_ms'])
        return good, None if good else 'estimated_cost_not_beneficial'


def cost_key(block, x):
    import torch
    return repr((torch.__version__, torch.version.cuda,
                 torch.cuda.get_device_name(x.device) if x.is_cuda else 'cpu',
                 tuple(x.shape), str(x.dtype), block.window_size,
                 block.attn.num_heads, block.attn.flash_attn,
                 tuple((n, tuple(p.shape)) for n, p in block.named_parameters()),
                 torch.is_autocast_enabled(), str(torch.get_autocast_gpu_dtype()),
                 torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
