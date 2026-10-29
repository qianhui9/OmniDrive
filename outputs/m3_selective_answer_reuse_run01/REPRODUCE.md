# WRSAR reproduction

Use existing local data, checkpoints and OmniDrive Python. No downloads. Every command excludes faulty CPUs4/5. GPU stages acquire /tmp/omnidrive_m123_latency_gpu.lock and wait for idle GPU memory, never stop another process. Use a NEW output root for each reproduction; stages refuse to overwrite their directories. The fixed native generation parameters remain max_new_tokens320, sampling .1/.75/num_beams1/top_k50 and full six-camera inputs.

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.selective_answer_reuse --root outputs/wrsar_reproduce01 --stage prepare
```

Repeat the same command prefix/root with stages `shadow`, `calibrate`, `pareto`, `run`, `report` in that order. Shadow and pareto/run need host CUDA access. Only a shadow-qualified DEV policy enables online stages. Final TEST never tunes a threshold. The three predeclared detector floors provide the measured DEV Pareto curve; the budgets can select the same policy, which must be reported as the same operating point.

Per-stage execution_sources.json and executed_sources/ bind exact GPU source bytes. The initial attempt is retained in initial_prompt_context/ and excluded: the fixed first-frame native-token assertion caught a city-vs-site context mismatch before any fixed/addressable TEST call. Corrected encoding matches the native loader on Boston and Singapore tests; all Shadow/calibration/Pareto/TEST stages were rerun. No initial TEST labels were evaluated. The original metadata-selected DEV-disjoint TEST cohort was retained unchanged because additional image-complete independent logs were unavailable; no TEST answers/GT informed threshold selection. Calibration source and selected-policy hashes bind the freeze before the replacement test. Raw logs contain answers/tokens, current M1 evidence, forward counts and synchronized frame timings.

For tests, use the spawn-safe existing `tools.worldstream_m123.answer_validity_validate --gpu --output <NEW validation directory>` with the same taskset/Python prefix. CPU testing omits --gpu. Regression CUDA checks are reported separately from full model GPU experiments. Raw duplicate model answer exports remain local; no large tensor captures are duplicated.

After `report`, run the post-inference source/policy/GT-grounding audit with the same taskset/interpreter prefix:

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.selective_answer_reuse_audit --root outputs/wrsar_reproduce01 --verify-checkpoint
```

The audit emits class/identity evidence, same-hit-mask A/B/C quality, per-task errors, causal history coverage and conservative unknown-label bounds. None of these audit modules is called by online decisions. Original checkpoint SHA256 was verified in the delivered run. See execution_notes.md for excluded prompt-context debug calls and per-stage source snapshots.
