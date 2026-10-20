# Dependency-safe sampling microbatch research

The native sampler, frozen M1/M2 paths and existing greedy `BatchedExecutor`
sampling rejection remain unchanged. Use the isolated executor only in a local,
single-owner **offline** experiment:

```python
M3Config(
    mode='full', offline_experiment=True,
    experimental_rng_isolated_microbatch=True,
    sampling_microbatch_size=2,  # fixed 2 or 4
    sampling_microbatch_seed=123,
)
```

It requires six-camera context, exact native single-turn question binding,
completed shared carriers, and no answer references or state-writing/order
constraints. Uninspected prompts/configurations, single questions and failed
transactions fall back to original Native generation with attempt cost charged.
Original generation overrides, including the 320-token budget, are unchanged.

Every request receives its own `torch.Generator` derived from seed, scene, frame
and original question index. The reference experiment also runs original Native
generation serially with those private seeds. This is distinct from the original
model's shared global RNG stream. Neither private RNG nor a few equal warped
probability vectors proves equivalence after changing FP16 GEMM batch shapes.

Prefill retains each entire native visual prompt independently. The executor
left-pads owned KV rows, assigns explicit native positions, masks all padding,
and samples/stops each row independently. Finished requests leave the next
batch; output order is restored before any results are submitted. All KV is
request-local and never enters the persistent Prefix Cache. No current or
historical visual fact is replaced by a text answer.

The first GPU probe uses different real same-frame questions at Batch=1/2/4/8.
Only a positive measured opportunity admits the fixed 2/4 prototype. Complete
service experiments keep all native questions and full state maintenance. Cold
state, per-question completion/publication tails and all single-question or
failed-attempt fallbacks are separately recorded. Greedy is a labeled engineering
control and cannot certify Sampling.

Reproduce from the repository root in a fresh result directory:

```bash
taskset -c 0-3,6-31 bash outputs/m3_dependency_safe_microbatch/commands.sh
```

The commands use the existing OmniDrive interpreter, inherit CPU affinity that
excludes faulty CPU 4/5, and wait for idle GPU resources without terminating
other jobs. The SGVD attribution is limited to the previously archived failing
samples. CUDA profile, raw results, loaded source snapshots, checkpoint hashes
and formal Go/No-Go fields live in the result directory.
