#!/usr/bin/env bash
set -euo pipefail
# Actual executed modules. Reproduction must choose NEW output destinations.
OMNIDRIVE_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
# Run before implementation edits to capture the frozen original baseline.
taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/final_mpl "$OMNIDRIVE_PYTHON" -u -m tools.worldstream_m1.final_validation_diagnose
# Freeze sampling protocol before inspecting scores; sampling constants are in the module.
taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$OMNIDRIVE_PYTHON" -m pytest tests/worldstream_m1/test_final_validation.py -q
# Final full paired 96-frame replay (five arms).
taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/final_mpl "$OMNIDRIVE_PYTHON" -u -m tools.worldstream_m1.final_validation_replay
# Approved HOST GPU context: waits for idle resources, never stops other jobs.
# The coordinator runs pytest all M1/M2/M3 with CUDA and the full original model.
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.worldstream_m2.final_validation_cuda
# Offline artifacts only.
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/final_mpl OPENBLAS_NUM_THREADS=1 "$OMNIDRIVE_PYTHON" -m tools.worldstream_m1.final_validation_overlays
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/final_mpl OPENBLAS_NUM_THREADS=1 "$OMNIDRIVE_PYTHON" -m tools.worldstream_m1.final_validation_report
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m tools.worldstream_m1.final_validation_audit
taskset -c 0-3,6-31 git diff --check
