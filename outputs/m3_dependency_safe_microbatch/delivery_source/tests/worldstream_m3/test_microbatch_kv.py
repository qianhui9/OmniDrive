import pytest
import torch
from worldstream.m3.microbatch_kv import pack_independent_kv,select_active_kv


def state(n):return tuple(tuple(torch.randn(1,2,n,4) for _ in range(2)) for _ in range(2))


def test_independent_left_padding_positions_owned_storage_and_compaction():
    states=[state(2),state(5),state(3)]
    packed,mask,pos=pack_independent_kv(states)
    assert mask.tolist()==[[0,0,0,1,1],[1,1,1,1,1],[0,0,1,1,1]]
    assert pos.tolist()==[[2],[5],[3]]
    for row,n in enumerate((2,5,3)):
        for layer in range(2):
            for part in range(2):
                assert torch.equal(packed[layer][part][row:row+1,:,-n:,:],states[row][layer][part])
    compact=select_active_kv(packed,torch.tensor([2,0]))
    assert compact[0][0].shape[0]==2
    compact[0][0].zero_();assert packed[0][0].abs().sum()>0
    packed[0][0].zero_();assert states[0][0][0].abs().sum()>0


def test_invalid_request_shape_and_layer_count_rejects():
    with pytest.raises(ValueError):pack_independent_kv([])
    with pytest.raises(ValueError):pack_independent_kv([state(2),state(3)[:1]])
    s=state(3);s=((s[0][0].expand(2,-1,-1,-1),s[0][1]),s[1])
    with pytest.raises(ValueError):pack_independent_kv([state(2),s])
