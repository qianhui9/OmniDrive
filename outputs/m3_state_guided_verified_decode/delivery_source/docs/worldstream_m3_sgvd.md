# SGVD offline feasibility

Run the archived command script from the repository root with a fresh namespace:

```bash
taskset -c 0-3,6-31 bash outputs/m3_state_guided_verified_decode/commands.sh
```

The interpreter is `/home/qh/miniconda3/envs/OmniDrive/bin/python`. The model
runner waits for an idle GPU and never terminates another process. The default
replay directory is `outputs/m3_state_guided_verified_decode_replay`; set
`OMNIDRIVE_SGVD_OUTPUT` to a different **new** directory for another run.

`tools.worldstream_m3.sgvd.probe` completes original OmniDrive requests before
analyzing historical answers. It records full six-camera prompt provenance,
native tokens/stopping, task and protocol contracts, per-request timings, and
raw complete-model outputs. Native planning, expanded original multi-question
and controlled same-question populations are kept separate. The fixed scenes
are previously exposed and cannot confer heldout production qualification.

`historical_draft.HistoricalDraftCache` accepts only completed, parsed, submitted
answers. Exact question and text-prompt signatures prevent semantic task mixing.
Model/tokenizer/generation, camera and scene changes invalidate history. Current
visual activations, target logits and ground truth never enter this cache.

`verified_sampling.verify_deterministic_proposal` implements the categorical
accept/reject identity. Its proof and small-vocabulary Monte Carlo evidence do
not certify a differently shaped FP16 model forward. The numerical runner
compares native serial and parallel processed target probabilities at real
prefixes and checks owned KV truncation at rejected positions.

No decoder is installed by these offline tools. Native Generation, M3 Runtime,
default configuration, frozen M1/M2, direct evidence admission and conservative
prefix policy retain their existing execution paths. The `validate` runner
archives real target-distribution failures and then evaluates a bounded KV
fallback and full-model frozen-service quality. `finalize` refuses to manufacture
a numerical No-Go if no target-distribution difference was measured; that case
requires subsequent prototype work and its own task/cost qualification.

The final JSON separates hypothetical native-prefix savings, true forward
counts, unchanged native-service performance, categorical correctness and
real-model target qualification. Unexecuted SGVD quality and speedups are
explicitly unavailable. Existing CUDA Graph negative evidence is linked with a
source SHA rather than captured again.
