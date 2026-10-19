#!/usr/bin/env bash
# Run from /data/qh/phdCode/work4/OmniDrive in the existing OmniDrive environment.
# Fresh output directory required. Preserve completed outputs before reproducing.
# GPU commands require actual host GPU access. Runners wait for an idle GPU;
# they do not terminate or modify another process.
# Actual run logs: /tmp/m2_state_feasibility.log, /tmp/m2_state_probe.log,
# /tmp/m2_state_cuda_batch.log (preserved failed diagnostic attempt),
# /tmp/m2_state_cuda_batch2.log and gpu_validation/*.log.
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m2.state_aware_feasibility
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m2.state_aware_probe
# Executes fresh pytest M1/M2/M3, real VLM A/B/C and CUDA executor profiling.
# First integration attempt exposed an observer layer-key mismatch; its complete
# regression and failed trace remain in gpu_validation_attempt1/. Corrected
# integration restricts both observers to patch/block 0/1/2/23. Fresh batch rerun:
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m2.state_aware_cuda_batch
# After GPU timing, repeat regression against final source (includes cache-never
# guard test), then generate/audit all artifacts. May be started concurrently;
# it waits for the primary batch and performs no competing timing workload.
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m2.state_aware_finish_batch
# Finalizer also collects final-source per-token details and an 8-scene
# stride-2 (actual doubled intervals archived) probe after all E2E timings,
# preserving the first probe.
# Its interval run is equivalent to the following command in a fresh directory:
# taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m2.state_aware_probe --output outputs/module2_state_aware_incremental_reuse/temporal_gap_probe --frame-stride 2
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.state_aware_finalize
# Production: projects/configs/OmniDrive/worldstream_m2_state_aware.py
# Research-only opt-in: projects/configs/OmniDrive/worldstream_m2_conditional_research.py
