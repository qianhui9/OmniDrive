"""Distribution primitive for offline deterministic-proposal verification.

For a fixed candidate d, P(output=d)=p(d), and for y!=d,
P(output=y)=(1-p(d))*p(y)/(1-p(d))=p(y). Applying this identity
conditionally at every legal prefix proves the joint autoregressive law.
This is not a numerical equivalence certificate for parallel FP16 forwards.
"""
import torch


def verify_deterministic_proposal(probabilities, draft, generator=None):
    """Return (token, accepted), supporting a vector or a batch of vectors.

    Input is the *already processed/warped native* target distribution. Double
    precision normalization avoids subtractive cancellation near p(d)=1. The
    rejection distribution is computed from its positive remaining mass.
    A separate RNG stream is permitted; same-seed token identity is not promised.
    """
    if probabilities.ndim not in (1, 2) or probabilities.shape[-1] == 0:
        raise ValueError('Expected a nonempty vocabulary vector or batch')
    single = probabilities.ndim == 1
    p = probabilities.reshape(-1, probabilities.shape[-1]).double()
    d = torch.as_tensor(draft, dtype=torch.long, device=p.device).reshape(-1)
    if d.numel() != p.shape[0] or bool(((d < 0) | (d >= p.shape[1])).any()):
        raise ValueError('One in-vocabulary deterministic proposal per row required')
    if not bool(torch.isfinite(p).all()) or bool((p < 0).any()) or bool((p.sum(-1) <= 0).any()):
        raise ValueError('Invalid target probability mass')
    p = p / p.sum(-1, keepdim=True)
    pd = p.gather(1, d[:, None]).squeeze(1)
    accepted = torch.rand(pd.shape, dtype=torch.float64, device=p.device, generator=generator) < pd
    answer = d.clone()
    rejected = ~accepted
    if bool(rejected.any()):
        residual = p[rejected].clone()
        residual.scatter_(1, d[rejected, None], 0)
        # No 1-p subtraction. If p(d)=1 the row can never enter this branch.
        residual /= residual.sum(-1, keepdim=True)
        answer[rejected] = torch.multinomial(residual, 1, generator=generator).squeeze(1)
    return (answer[0], accepted[0]) if single else (answer, accepted)
