"""Owned legacy KV packing for independent requests; no prefix cache access."""
import torch


def pack_independent_kv(states, device=None):
    if not states:raise ValueError('No request KV')
    layers=len(states[0]);lengths=[s[0][0].shape[-2] for s in states];width=max(lengths)
    first=states[0][0][0];device=device or first.device
    if any(len(s)!=layers for s in states):raise ValueError('Mismatched cache layers')
    output=[]
    for li in range(layers):
        pair=[]
        for ki in range(2):
            shape=states[0][li][ki].shape
            dst=torch.zeros((len(states),shape[1],width,shape[3]),device=device,dtype=first.dtype)
            for i,(state,length) in enumerate(zip(states,lengths)):
                t=state[li][ki]
                if (t.ndim!=4 or t.shape[0]!=1 or t.shape[1]!=shape[1] or t.shape[3]!=shape[3]
                        or t.shape[-2]!=length or t.dtype!=first.dtype):raise ValueError('Incompatible request KV')
                dst[i:i+1,:,width-length:,:].copy_(t)
            pair.append(dst)
        output.append(tuple(pair))
    mask=torch.zeros((len(states),width),dtype=torch.long,device=device)
    for i,length in enumerate(lengths):mask[i,width-length:]=1
    positions=torch.tensor(lengths,dtype=torch.long,device=device)[:,None]
    return tuple(output),mask,positions


def select_active_kv(kv, indices):
    return tuple(tuple(t.index_select(0,indices) for t in layer) for layer in kv)
