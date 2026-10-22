"""Execute original EVA attention/MLP weights on selected independent windows only."""
import torch
from .packing import partition, unpartition, PackBuffer, SpatialWindowGather


def packed_forward(block, windows, valid):
    if block.training or block.use_residual_block or not block.window_size:
        raise ValueError('Sparse execution needs eval-mode independent EVA windows')
    # EVA normalizes BEFORE padding. LayerNorm bias at padded slots must remain zero.
    normalized = block.norm1(windows) * valid
    x = windows + block.drop_path(block.attn(normalized))
    # Native EVA crops attention padding BEFORE the residual/MLP. These
    # synthetic positions must not become nonzero cached activations.
    return (x + block.drop_path(block.mlp(block.norm2(x)))).masked_fill(~valid.bool(), 0)


class WindowExecutor:
    def __init__(self):
        self.buffer = PackBuffer()
        self.spatial = SpatialWindowGather()

    def execute(self, block, x, cached_windows, ids, naive=False, strategy='spatial'):
        if torch.is_grad_enabled() or any(m.training for m in block.modules()):
            raise ValueError('Window execution is inference-only')
        if not block.window_size or block.use_residual_block:
            raise ValueError('Window execution requires independent non-shifted windows')
        if x.ndim != 4 or ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError('Expected BHWC features and one-dimensional int64 window IDs')
        size = block.window_size
        h,w = x.shape[1:3]
        padded = ((h+size-1)//size*size,(w+size-1)//size*size)
        expected = (x.shape[0]*(padded[0]//size)*(padded[1]//size), size, size, x.shape[-1])
        if (tuple(cached_windows.shape) != expected or cached_windows.device != x.device
                or cached_windows.dtype != x.dtype):
            raise ValueError('Cached windows do not match the current feature grid')
        if len(ids) == 0:
            output = unpartition(cached_windows, size, padded, x.shape[1:3])
            # For a single window unpartition can return an alias. A caller
            # modifying this frame's result must not corrupt retained state.
            if output.data_ptr() == cached_windows.data_ptr():
                output = output.clone()
            return output, cached_windows
        if strategy=='original':
            windows,_=partition(x,size)
            valid,_=partition(torch.ones_like(x[...,:1]),size)
            selected=self.buffer.gather(windows,ids)
            selected_valid=valid.index_select(0,ids)
        elif strategy=='spatial':
            selected,selected_valid = self.spatial.gather(x,size,ids)
        else:
            raise ValueError('Unknown packing strategy')
        if naive:
            updates = torch.cat([packed_forward(block, selected[i:i+1], selected_valid[i:i+1])
                                 for i in range(len(ids))], dim=0)
        else:
            updates = packed_forward(block, selected, selected_valid)
        output = cached_windows.clone()
        output.index_copy_(0, ids, updates)
        return unpartition(output, size, padded, x.shape[1:3]), output

    def clear(self):
        self.buffer.clear()
        self.spatial.clear()
