import numpy as np
import torch
from worldstream.m2.approximate_probe import align_patch_features


def test_offline_probe_keeps_current_positions_separate():
    feature=torch.randn(2,3,4);yy,xx=np.mgrid[:8,:12]
    source=np.stack((xx,yy),-1).astype(np.float32)
    candidate,report=align_patch_features(feature,feature,source,np.ones((8,12),bool),4)
    assert torch.allclose(candidate,feature,atol=1e-6)
    assert report['transformer_dependency_certified'] is False
    assert report['accepted_patches']==6


def test_incomplete_support_does_not_approve_patch():
    x=torch.ones(2,2,3);mask=np.ones((8,8),bool);mask[::4,::4]=False
    result,d=align_patch_features(x,x,None,mask,4)
    assert result is x and d['accepted_patches']==0
