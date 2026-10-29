# Reproduce

Existing OmniDrive environment, RTX3090, original checkpoint; no installs, download or training. GPU jobs acquire /tmp/omnidrive_m123_latency_gpu.lock and wait for occupancy. All invocations exclude CPU4/5.

Verify retained results (restores byte-identical compressed receipts if needed, no inference):

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.critical_path_analysis verify
```

For a fixed-input/configuration inference repetition, copy the frozen inputs/configuration records to a NEW output root: baseline_sources/, optimized_sources/, benchmark_v2_sources/, selection/protocol, GT annotation cache for OFFLINE analysis, policies, parameter-result/selection/freezes, historical manifest, candidate/job manifests and receipt-reuse mappings. Omit completed stage directories, execution logs, validation directories, receipt_archive.json and generated presentation reports. Replace policy path prefixes in copied job manifests with the new root and rebind rewritten manifests before execution; keep the original artifacts unchanged. Repetition of these104 already-used frames is regression/performance repetition, never new independent evaluation. Full must precede Incremental to regenerate temporary comparison tensors.

Stages in order: profile, baseline, executor_check, executor_clean, tune_m1, tune_m2, tune_m3, joint, selected_m2_ablation, final, feature_audit. Set baseline_m1=true in the initial profile/baseline replay jobs to invoke the original archived digest implementation; later before/after controls already carry this flag. For the original run tune_m3 had one changed candidate and32 new calls. Each stage uses its retained <stage>_jobs.json; execute with:

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.critical_path <stage> --root /tmp/omnidrive-critical-replay
```

Worker source versions are retained: baseline_sources/.../critical_path.py for original profile/baseline; optimized_sources/.../critical_path.py for executor/M1 DEV; benchmark_v2_sources/.../critical_path.py for M2/M3 DEV, joint confirmation, selected M2 profiling and final measurements. Current worker corrects feature capture for all-mode returned tensors and was used in the48-call feature_audit. Replaying current worker reproduces inference/configuration with corrected observation, not the invalid legacy feature-witness fields. For historical controller reconstruction use the archived source versions in an isolated checkout and their execution_sources.json fingerprints. Serving M1/M2/M3 semantics and frozen policy are unchanged. Legacy digest functions remain external benchmark controls, never deployment fallbacks. See feature_audit_report.md and the explicit548-row legacy invalidity ledger.

For a fresh retrospective parameter search, retain the predeclared protocol and run select_m1/select_m2, prepare_m3/select_m3, prepare_joint/select_joint only after the corresponding DEV stage completes. Candidate0 reference B/C receipts already executed with identical configuration are explicitly reused rather than rerun. Do not choose new thresholds using final results. The report records the actual frozen policy and all unknowns.

Regression invocations:

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --output /tmp/omnidrive-critical-replay/validation_cpu
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --gpu --output /tmp/omnidrive-critical-replay/validation_gpu
```

After all stages, run analysis profile, executor, executor_clean, selected_m2_ablation and feature_audit, followed by report, archive, verify. Existing result reanalysis uses the same retained raw data and grader; inference replay timings vary with clocks/order/system load. Native A performs no M1/M2/M3 computation; the3Q A_with_M1 companion retains old accounting. Process peak memory can include inactive harness buffers and is not a minimal standalone footprint. Timing is compute service latency, not a real-time arrival/queue simulation. The final selected candidate FAILED its new-severe-error gate; do not retune from that case or enable production. Historical conclusions remain unchanged.
