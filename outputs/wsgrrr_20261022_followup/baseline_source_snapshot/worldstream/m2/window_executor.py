"""One window executor, preserving native pointwise arithmetic geometry.

Native EVA FP32 GEMMs change their reduction algorithm with batch shape; casting
Q/K to FP16 then amplifies the difference. QKV/projection and MLP therefore retain
the original shapes. Only selected independent attention windows execute. This is
reported explicitly: cached output windows do not imply skipped pointwise GEMMs.
"""
import torch
import torch.nn.functional as F
from .packing import partition, unpartition

NUMERICAL_POLICY = 'native_pointwise_shapes_selected_attention_v1'


class WindowExecutor:
    def __init__(self):
        self.attention_observer = None

    @staticmethod
    def _grid(windows,size,padded,x):
        output=unpartition(windows,size,padded,x.shape[1:3])
        # Native residual additions preserve the shortcut's dense permutation.
        # Preserve it for downstream EVA and task-head GEMM dispatch as well.
        if output.stride()!=x.stride():
            aligned=torch.empty_strided(x.shape,x.stride(),dtype=x.dtype,device=x.device)
            aligned.copy_(output)
            output=aligned
        if output.data_ptr()==windows.data_ptr():output=output.clone()
        return output

    def execute(self, block, x, cached_windows, ids):
        if torch.is_grad_enabled() or any(m.training for m in block.modules()):
            raise ValueError('Window execution is inference-only')
        if not block.window_size or block.use_residual_block:
            raise ValueError('Window execution requires independent non-shifted windows')
        if x.ndim != 4 or ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError('Expected BHWC features and one-dimensional int64 window IDs')
        size=block.window_size; h,w=x.shape[1:3]
        padded=((h+size-1)//size*size,(w+size-1)//size*size)
        count=x.shape[0]*(padded[0]//size)*(padded[1]//size)
        expected=(count,size,size,x.shape[-1])
        if cached_windows is None and len(ids)!=count:
            raise ValueError('Partial window execution requires a valid cache')
        if cached_windows is not None and (tuple(cached_windows.shape)!=expected or cached_windows.device!=x.device
                or cached_windows.dtype!=x.dtype):
            raise ValueError('Cached windows do not match the current feature grid')
        if ids.device!=x.device:
            ids=ids.to(x.device)
        if len(ids) and (int(ids.min())<0 or int(ids.max())>=count or len(ids.unique())!=len(ids)):
            raise ValueError('Window IDs must be unique and within the feature grid')
        if not len(ids):
            return self._grid(cached_windows,size,padded,x),cached_windows
        a=block.attn
        normalized,_=partition(block.norm1(x),size)
        tokens=normalized.view(count,-1,x.shape[-1]);n=size*size
        # Exactly the original F.linear dimensions, bias handling and RoPE order.
        q=F.linear(tokens,a.q_proj.weight,a.q_bias)
        k=F.linear(tokens,a.k_proj.weight,None)
        v=F.linear(tokens,a.v_proj.weight,a.v_bias)
        q=q.reshape(count,n,a.num_heads,-1).permute(0,2,1,3)
        k=k.reshape(count,n,a.num_heads,-1).permute(0,2,1,3)
        v=v.reshape(count,n,a.num_heads,-1).permute(0,2,1,3)
        q=a.rope(q).type_as(v);k=a.rope(k).type_as(v)
        if a.flash_attn:
            q=q.permute(0,2,1,3).index_select(0,ids)
            k=k.permute(0,2,1,3).index_select(0,ids)
            v=v.permute(0,2,1,3).index_select(0,ids)
            selected,_=a.inner_attn(q,torch.stack([k,v],dim=2),key_padding_mask=None,causal=False)
            selected=selected.reshape(len(ids),n,-1)
        else:
            q=(q*a.scale).index_select(0,ids)
            k=k.index_select(0,ids);v=v.index_select(0,ids)
            weights=(q@k.transpose(-2,-1)).softmax(dim=-1).type_as(tokens)
            selected=(weights@v).transpose(1,2).reshape(len(ids),n,-1)
        if self.attention_observer is not None:self.attention_observer(len(ids))
        attention=selected.new_zeros(count,n,selected.shape[-1])
        attention.index_copy_(0,ids,selected)
        attention=a.proj(a.inner_attn_ln(attention)).view(count,size,size,x.shape[-1])
        attention=unpartition(attention,size,padded,(h,w))
        updated=x+block.drop_path(attention)
        updated=updated+block.drop_path(block.mlp(block.norm2(updated)))
        packed,_=partition(updated,size)
        if len(ids)==count:
            return updated,packed
        output=cached_windows.clone()
        output.index_copy_(0,ids,packed.index_select(0,ids))
        return self._grid(output,size,padded,x),output

    def clear(self):
        pass
