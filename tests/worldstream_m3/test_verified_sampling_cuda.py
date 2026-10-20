import pytest
import torch
from worldstream.m3.verified_sampling import verify_deterministic_proposal


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('p,d', [([0.,1.,0.],0),([0.,1.,0.],1),
    ([.1,.3,.6],1),([1e-15,.2,.8-1e-15],0),([1-1e-12,1e-12,0.],0)])
def test_cuda_accept_reject_support_extremes_and_distribution(p,d):
    n=60000
    target=torch.tensor(p,device='cuda',dtype=torch.float64)
    outputs,accepted=verify_deterministic_proposal(target.expand(n,-1),
        torch.full((n,),d,device='cuda'),torch.Generator(device='cuda').manual_seed(42))
    observed=torch.bincount(outputs,minlength=len(p)).double()/n
    bound=6*torch.sqrt(target*(1-target)/n)+6/n
    assert bool(((observed-target).abs()<=bound).all())
    assert bool((accepted==(outputs==d)).all())
    assert bool((target[outputs]>0).all())
