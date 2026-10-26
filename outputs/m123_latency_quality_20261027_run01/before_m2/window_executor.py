"""One independent-window executor. Native full blocks, compact partial work.

Partial GEMM shapes may change FP32 rounding before native FlashAttention casts;
this is a versioned numerical policy, validated quantitatively against native EVA.
No new dtype conversion, attention approximation, or cross-window reuse is used.
"""
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from .packing import partition, unpartition, window_geometry

NUMERICAL_POLICY = 'selected_window_qkv_projection_valid_token_mlp_v2'


@dataclass(frozen=True)
class _WindowIdProof:
    pointer: int
    version: int
    total: int
    values: tuple


class WindowExecutor:
    def __init__(self):
        self.attention_observer = None
        self._geometry = OrderedDict()
        self._selections = OrderedDict()

    def _layout(self,x,size):
        key=(tuple(x.shape[:3]),size,str(x.device))
        if key not in self._geometry:
            coords,mask,grid=window_geometry(*x.shape[:3],size)
            self._geometry[key]=(tuple(torch.tensor(a,device=x.device) for a in coords),
                torch.tensor(mask[...,None],device=x.device),torch.tensor(grid,device=x.device))
            if len(self._geometry)>2:self._geometry.popitem(last=False)
        self._geometry.move_to_end(key)
        return self._geometry[key]

    def _valid_positions(self,x,size,values):
        key=(tuple(x.shape[:3]),size,str(x.device),tuple(values))
        if key not in self._selections:
            _,mask,_=window_geometry(*x.shape[:3],size)
            positions=np.flatnonzero(mask[list(values)].reshape(-1)).copy()
            self._selections[key]=torch.tensor(positions,device=x.device,dtype=torch.long)
            if len(self._selections)>8:self._selections.popitem(last=False)
        self._selections.move_to_end(key)
        return self._selections[key]

    @property
    def metadata_bytes(self):
        values=[t for coords,mask,grid in self._geometry.values() for t in (*coords,mask,grid)]
        values+=list(self._selections.values())
        return sum(t.numel()*t.element_size() for t in values)

    def _grid(self,windows,size,padded,x):
        # A single gather writes native strides directly, avoiding padded-grid
        # materialization, crop.contiguous(), then a second layout copy.
        _,_,grid=self._layout(x,size)
        output=torch.empty_strided(x.shape,x.stride(),dtype=x.dtype,device=x.device)
        if x.stride(1)==x.shape[2]*x.stride(2):
            source=windows.reshape(1,-1,x.shape[-1]).expand(len(x),-1,-1)
            torch.gather(source,1,grid[...,None].expand(-1,-1,x.shape[-1]),
                out=output.view(len(x),-1,x.shape[-1]))
        else:
            output.copy_(unpartition(windows,size,padded,x.shape[1:3]))
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
        proof=getattr(ids,'_m2_window_proof',None)
        if proof is not None:
            if (not isinstance(proof,_WindowIdProof) or proof.pointer!=ids.data_ptr()
                    or proof.version!=ids._version or proof.total!=count or len(proof.values)!=len(ids)):
                raise ValueError('Planned window IDs modified or geometry mismatch')
            values=proof.values
        else:
            # Independent public calls retain the full validation contract.
            values=tuple(ids.cpu().tolist())
            if len(set(values))!=len(ids) or any(i<0 or i>=count for i in values):
                raise ValueError('Window IDs must be unique and within the feature grid')
        if not len(ids):
            return self._grid(cached_windows,size,padded,x),cached_windows
        if len(ids)==count:
            # Avoid gather/zero-fill/scatter overhead when there is no reuse.
            updated=block(x)
            if self.attention_observer is not None:self.attention_observer(count)
            packed=partition(updated,size)[0]
            if packed.storage().data_ptr()==updated.storage().data_ptr():packed=packed.clone()
            return updated,packed
        a=block.attn
        coordinates,mask,_=self._layout(x,size)
        cc,yy,xx=(t.index_select(0,ids) for t in coordinates)
        shortcut=x[cc,yy,xx]
        valid=mask.index_select(0,ids)
        positions=self._valid_positions(x,size,values)
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
        values=updated.reshape(-1,x.shape[-1]).index_select(0,positions)
        values=values+block.drop_path(block.mlp(block.norm2(values)))
        packed=torch.zeros_like(updated)
        packed.view(-1,x.shape[-1]).index_copy_(0,positions,values)
        output=cached_windows.clone()
        output.index_copy_(0,ids,packed)
        return self._grid(output,size,padded,x),output

    def clear(self):
        self._geometry.clear()
        self._selections.clear()
