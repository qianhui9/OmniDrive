# Delivery

Two minimal M1 hashing efficiency edits, no serving M2/M3 algorithm edit. External profiling/benchmark code reuses the existing full model, state manager, M2 WindowExecutor and M3Runtime. The executor was frozen before staged hyperparameter DEV. All experiments use the existing environment, GPU lock and CPU affinity excluding4/5. No training/download/installation.

Executor regression/equivalence validation passed, but the final selected policy risk gate is FAILED: 1 new severe primary error(s). The configuration remains a frozen retrospective research candidate and is not enabled for production.

Actual full-model frame calls=2466, real prefill/decode skips=1041, unique frames=104. Main B→C=1.471560×; pure Native A→C=1.927564×.

Instrumentation files: critical_path_probe.py (CPU/CUDA observers), critical_path.py (existing-worker adapter and explicit comparison controls), critical_path_analysis.py (offline scheduling/statistics/reporting). Production changes: source_digest.py and input_contract.py only. Complete live mutation and byte-compatible hash tests pass. Profiler traces/tensors are not versioned; byte-identical raw request receipts are gzip archived with hashes.
