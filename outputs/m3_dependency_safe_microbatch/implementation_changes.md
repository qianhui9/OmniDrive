# Implementation changes

- `question_dependencies.py`: exact native single-turn prompt binding, answer-reference and state/ordering barriers, ready groups in source order.
- `microbatch_kv.py`: owned independent left-padded legacy KV rows, explicit per-row positions, active-row compaction; no prefix sharing.
- `sampling_microbatch.py`: offline fixed 2/4 full-input sampled decode, original native serial prefill, private request RNG, EOS/budget/processor handling, request-local temporary state, atomic ordered return, failure/OOM cleanup and Native fallback.
- `M3Config`: append-only default-off research switch and mutually exclusive offline guards. `M3Runtime`: the new branch is reached only with that switch; original Native Decode and original Sampling rejection in BatchedExecutor remain unchanged.
- New numerical/dependency/KV/RNG/EOS/compaction/exception tests, CUDA test, real-model probe and complete-service tools under `tools/worldstream_m3/microbatch/`.
- Frozen M1/M2, original network/weights, generation budget and task-quality gate are not modified. No production certificate is issued.
