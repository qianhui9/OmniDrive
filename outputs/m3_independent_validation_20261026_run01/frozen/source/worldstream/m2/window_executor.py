"""One independent-window executor. Native full blocks, compact partial work.

Partial GEMM shapes may change FP32 rounding before native FlashAttention casts;
this is a versioned numerical policy, validated quantitatively against native EVA.
No new dtype conversion, attention approximation, or cross-window reuse is used.
"""
import torch
import torch.nn.functional as F
from .packing import partition, unpartition

NUMERICAL_POLICY = 'selected_window_qkv_projection_valid_token_mlp_v2'


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
        if len(ids)==count:
            # Avoid gather/zero-fill/scatter overhead when there is no reuse.
            updated=block(x)
            if self.attention_observer is not None:self.attention_observer(count)
            return updated,partition(updated,size)[0]
        a=block.attn
        shortcut,_=partition(x,size)
        shortcut=shortcut.index_select(0,ids)
        valid,_=partition(torch.ones_like(x[...,:1],dtype=torch.bool),size)
        valid=valid.index_select(0,ids)
        # Native normalizes real tokens BEFORE window padding. Learned norm
        # biases must not create nonzero normalized virtual padding tokens.
        normalized=block.norm1(shortcut).masked_fill(~valid,0)
        tokens=normalized.view(len(ids),-1,x.shape[-1]);n=size*size
        q=F.linear(tokens,a.q_proj.weight,a.q_bias)
        k=F.linear(tokens,a.k_proj.weight,None)
        v=F.linear(tokens,a.v_proj.weight,a.v_bias)
        q=q.reshape(len(ids),n,a.num_heads,-1).permute(0,2,1,3)
        k=k.reshape(len(ids),n,a.num_heads,-1).permute(0,2,1,3)
        v=v.reshape(len(ids),n,a.num_heads,-1).permute(0,2,1,3)
        q=a.rope(q).type_as(v);k=a.rope(k).type_as(v)
        if a.flash_attn:
            q=q.permute(0,2,1,3)
            k=k.permute(0,2,1,3)
            v=v.permute(0,2,1,3)
            selected,_=a.inner_attn(q,torch.stack([k,v],dim=2),key_padding_mask=None,causal=False)
            selected=selected.reshape(len(ids),n,-1)
        else:
            q=q*a.scale
            weights=(q@k.transpose(-2,-1)).softmax(dim=-1).type_as(tokens)
            selected=(weights@v).transpose(1,2).reshape(len(ids),n,-1)
        if self.attention_observer is not None:self.attention_observer(len(ids))
        attention=a.proj(a.inner_attn_ln(selected)).view(len(ids),size,size,x.shape[-1])
        updated=shortcut+block.drop_path(attention)
        # MLP has no spatial dependence. Exclude virtual padding entirely rather
        # than replacing the 40x40 native grid with a 48x48 packed MLP grid.
        valid_tokens=valid[...,0]
        values=updated[valid_tokens]
        values=values+block.drop_path(block.mlp(block.norm2(values)))
        packed=torch.zeros_like(updated)
        packed[valid_tokens]=values
        output=cached_windows.clone()
        output.index_copy_(0,ids,packed)
        return self._grid(output,size,padded,x),output

    def clear(self):
        pass
