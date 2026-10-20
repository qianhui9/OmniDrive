"""Save reproducible small-vocabulary distribution evidence independently of LLM."""
import argparse
from pathlib import Path
import torch
from worldstream.m3.verified_sampling import verify_deterministic_proposal
from tools.worldstream_m1.io_utils import write_json


def run(root):
    rows=[];n=200000
    for name,p,d in [('categorical',[.2,.3,.5],1),('all_accept',[0.,1.,0.],1),
                     ('all_reject',[0.,1.,0.],0),('rare_proposal',[1e-12,.25,.75-1e-12],0),
                     ('near_one',[1-1e-12,1e-12,0.],0),('truncated_support',[.25,0.,.75],1)]:
        target=torch.tensor(p,dtype=torch.float64)
        output,accepted=verify_deterministic_proposal(target.expand(n,-1),torch.full((n,),d),
                                                     torch.Generator().manual_seed(20261020))
        counts=torch.bincount(output,minlength=len(p)).double()
        freq=counts/n
        # Wilson 95% CIs are reported; deterministic acceptance is judged with
        # a predeclared 6-sigma bound + 6/N, not random inclusion of every 95% CI.
        z=1.959963984540054
        center=(freq+z*z/(2*n))/(1+z*z/n)
        radius=z*torch.sqrt(freq*(1-freq)/n+z*z/(4*n*n))/(1+z*z/n)
        bound=6*torch.sqrt(target*(1-target)/n)+6/n
        rows.append(dict(case=name,n=n,target=p,draft=d,observed=freq.tolist(),
            counts=counts.int().tolist(),accepted=int(accepted.sum()),
            wilson95=list(zip((center-radius).tolist(),(center+radius).tolist())),
            max_abs_error=float((freq-target).abs().max()),passed=bool(((freq-target).abs()<=bound).all())))
    artifact=dict(seed=20261020,rows=rows,passed=all(r['passed'] for r in rows),
        proof='P(Y=d)=p(d); P(Y=y!=d)=(1-p(d))*p(y)/(1-p(d))=p(y). '
              'Conditioning on every valid emitted prefix gives the joint law. '
              'At p(d)=0 always reject; at p(d)=1 always accept. EOS terminates the law.',
        joint_enumeration='tests/worldstream_m3/test_verified_sampling.py: prefix-dependent three-token vocabulary, '
                          'EOS and budgets, lengths 1/2/4/8/16',
        limitation='Primitive distribution test only; not a parallel FP16 numerical or complete-model task certificate')
    write_json(root/'sampling_distribution_tests.json',artifact)
    return artifact


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/m3_state_guided_verified_decode')
    args=p.parse_args();print(run(Path(args.output))['passed'])


if __name__=='__main__':main()
