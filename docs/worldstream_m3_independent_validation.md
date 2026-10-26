# Frozen WSDR-KV independent validation

The benchmark compares only native full prefill and the existing `wsdr_kv`
runtime, retaining the deployed M1/M2 computation, checkpoint, preprocessing,
token order and sampling. It never invokes calibration or writes the production
policy. The run directory contains an immutable copy of the policy and a hash
manifest for production and benchmark sources.

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.benchmark.independent_validation prepare --run-dir outputs/NEW_UNUSED_RUN
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.benchmark.independent_validation run --run-dir outputs/NEW_UNUSED_RUN
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.benchmark.independent_validation analyze --run-dir outputs/NEW_UNUSED_RUN
```

`prepare` audits historical language results, calibration/service records and
M3/M123 sequence files. It excludes entire observed scenes and the policy's
split membership. It selects the first linked consecutive segment of scenes at
fixed quantiles of metadata ego motion, then selects real labelled questions by
rotating task profiles. Selection never uses carrier similarity, model outputs,
reuse rate or new quality measurements. A generic question may share its text
with a historical question; that is disclosed separately from frame-question
overlap. Prior perception-only full-split experiments are disclosed and do not
make this a new perception dataset. Defaults are four scenes, six frames each,
three questions per frame: contiguous short segments, not entire scenes.

`run` rehashes the actual checkpoint and frozen sources, waits for an idle GPU
without disturbing other processes, and runs three paired rounds in AB/BA/AB
order. Both paths use the same warmup/reset rule and hash-derived per-frame seed.
Native uses the original `generate`; WSDR uses the original Planner and
PrefillExecutor. Runtime input IDs, complete carrier tensors and generation
arguments are hashed outside measured service time to check paired equality.
The worker retains all failed invocations and refuses to overwrite incomplete
results. Complete runs can be resumed only under the same frozen plan.

`DecisionTrace` records scalar candidate statistics and lineage without copying
KV or embeddings. Offline `decision_replay` calls the production Planner and
Scheduler on those measured statistics. It holds observed scheduler overhead
fixed, checks that baseline replay reproduces the actual length/type/reason,
then neutralizes the M1 validity gate or the M2 risk factor. The variants are
planning diagnostics, never model executions or quality/performance evidence.
They do not model downstream cache evolution under a changed policy.

`analyze` requires every expected frame/question and completed worker. It checks
actual executor forward lengths, prefill/decode completion, native/WSDR input
equality and seed pairing. Candidate lengths do not count as skipped work.
Performance uses all rounds and scene-cluster bootstrap intervals. Primary
quality uses only round zero's unique questions; later rounds check output and
latency stability. Numeric/spatial/motion/safety scores are explicit extraction
proxies, with native errors separated from new regressions. Road/driving subset
scores are proxies too; semantic safety is not automatically certified.

The report retains means, P50/P95, throughput, memory, all-round raw outputs,
per-profile/per-scene quality losses and concrete regression cases. Positive
latency changes mean slower WSDR. A small observed quality loss alone cannot
establish success: actual cross-frame approximation and positive end-to-end
paired evidence are required, and few scene clusters remain a stated limit.
Cache update timing is CPU enqueue time; its GPU cost is included in prefill,
not presented as a separately measured kernel duration.
