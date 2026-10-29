# Reproduce

Use the existing OmniDrive environment; no install/download/training. Every command excludes faulty CPU4/5. This delivery contains no fresh formal TEST and no claim of new full-model speedup.

Read-only historical integrity verification:

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_fact_bypass verify
```

To regenerate diagnostics, use a NEW root. The original historical outputs must remain in place:

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_fact_bypass prepare --root /tmp/omnidrive-final-fact-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_fact_bypass diagnose --root /tmp/omnidrive-final-fact-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --output /tmp/omnidrive-final-fact-replay/validation_cpu
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --gpu --output /tmp/omnidrive-final-fact-replay/validation_gpu
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_fact_bypass report --root /tmp/omnidrive-final-fact-replay
```

CUDA regression requires host GPU access, acquires /tmp/omnidrive_m123_latency_gpu.lock, waits for other users, and never kills their processes. A dataset audit is metadata/annotation-only. Historical DEV threshold fitting is diagnostic; report CPU timings vary by system load. Fresh independent DEV/TEST and a pre-TEST measured policy freeze are required before any new formal model run. Historical v4 exact repetition still uses its frozen_sources and original REPRODUCE.md; changing current code does not overwrite those sources.
