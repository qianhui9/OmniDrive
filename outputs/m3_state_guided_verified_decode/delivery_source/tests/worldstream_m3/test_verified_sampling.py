import itertools
import pytest
import torch
from worldstream.m3.verified_sampling import verify_deterministic_proposal


@pytest.mark.parametrize('p,d', [([0.,1.,0.],1),([0.,1.,0.],0),
    ([.2,.3,.5],1),([1e-15,.25,.75-1e-15],0),([1.-1e-12,1e-12,0.],0)])
def test_monte_carlo_matches_target_with_confidence_bound(p,d):
    n=60000;gen=torch.Generator().manual_seed(42)
    probs=torch.tensor(p,dtype=torch.float64).expand(n,-1)
    out,accepted=verify_deterministic_proposal(probs,torch.full((n,),d),gen)
    observed=torch.bincount(out,minlength=len(p)).double()/n
    target=torch.tensor(p,dtype=torch.float64)
    tolerance=6*torch.sqrt(target*(1-target)/n)+6/n
    assert bool(((observed-target).abs()<=tolerance).all())
    assert int(accepted.sum())==int((out==d).sum())


def test_joint_law_for_every_draft_length_and_eos():
    # Enumerate all branches of the accept/reject law, with prefix-dependent
    # targets. EOS=2 terminates immediately, including inside an accepted block.
    def target(prefix):
        return ([.2,.5,.3] if not prefix else [.4,.1,.5] if prefix[-1]==0 else [.1,.3,.6])
    def law(draft):
        results={}
        def step(prefix,mass):
            if prefix and prefix[-1]==2 or len(prefix)==4:
                results[tuple(prefix)]=mass;return
            p=target(prefix);d=draft[len(prefix)%len(draft)];pd=p[d]
            for y in range(3):
                chance=pd if y==d else (1-pd)*p[y]/sum(p[z] for z in range(3) if z!=d)
                step(prefix+[y],mass*chance)
        step([],1.);return results
    baseline=law([0])
    for k in (1,2,4,8,16):
        for d in (0,1,2):
            actual=law([d]*k)
            assert sum(actual.values())==pytest.approx(1.)
            assert actual==pytest.approx(baseline)


def test_native_top_p_support_and_processor_order():
    from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopPLogitsWarper
    scores=torch.tensor([[0.,3.,2.,1.]])
    ids=torch.tensor([[1,1]])
    proc=LogitsProcessorList([RepetitionPenaltyLogitsProcessor(1.2)])
    warp=LogitsProcessorList([TemperatureLogitsWarper(.1),TopPLogitsWarper(.75)])
    p=torch.softmax(warp(ids,proc(ids,scores)),dim=-1)[0]
    out,accepted=verify_deterministic_proposal(p,0,torch.Generator().manual_seed(5))
    assert not accepted and p[out]>0 and p[0]==0


@pytest.mark.parametrize('p,d', [([0.,0.],0),([float('nan'),1.],0),([-1.,2.],0),([.2,.8],2)])
def test_invalid_distributions_fail_closed(p,d):
    with pytest.raises(ValueError):verify_deterministic_proposal(torch.tensor(p),d)
