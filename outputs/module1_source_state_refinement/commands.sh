#!/usr/bin/env bash
set -euo pipefail
# Executed invocations; choose NEW output paths for reproduction.
# Every Python/test/experiment process excludes faulty CPU 4/5.
OMNIDRIVE_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
taskset -c 0-3,6-31 git rev-parse HEAD
taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/state_mpl "$OMNIDRIVE_PYTHON" -m cProfile -o outputs/module1_source_state_refinement/profile_output/baseline.prof -m tools.worldstream_m1.state_interface_replay --scenes 2 --frames 4 --output outputs/module1_source_state_refinement/profile_output/baseline_replay
taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$OMNIDRIVE_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_source_state_refinement/regression_cpu.xml -o junit_family=legacy
taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/state_mpl "$OMNIDRIVE_PYTHON" -u -m tools.worldstream_m1.source_state_refinement_replay --scenes 8 --frames 12 --output outputs/module1_source_state_refinement/replay
taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/state_mpl "$OMNIDRIVE_PYTHON" -m tools.worldstream_m1.source_state_refinement_spatial
taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/state_mpl "$OMNIDRIVE_PYTHON" -m tools.worldstream_m1.source_state_refinement_centre_diagnostics
# Run in approved host CUDA context. Wait for idle GPU; never stop other jobs.
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.worldstream_m2.source_state_refinement_cuda_batch --repeats 1 --output outputs/module1_source_state_refinement/gpu_validation_v2
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/state_mpl OPENBLAS_NUM_THREADS=1 "$OMNIDRIVE_PYTHON" -m tools.worldstream_m1.source_state_refinement_report

# Final-source replay + CUDA regression + natural/repeat complete-model rechecks.
# The coordinator owns replay_final/ and waits for GPU availability before tests.
taskset -c 0-3,6-31 \
  "$OMNIDRIVE_PYTHON" -u -m tools.worldstream_m2.source_state_refinement_finalize
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/state_mpl OPENBLAS_NUM_THREADS=1 \
  "$OMNIDRIVE_PYTHON" -m tools.worldstream_m1.source_state_refinement_report
taskset -c 0-3,6-31 \
  "$OMNIDRIVE_PYTHON" -m tools.worldstream_m1.source_state_refinement_audit
taskset -c 0-3,6-31 git diff --check
