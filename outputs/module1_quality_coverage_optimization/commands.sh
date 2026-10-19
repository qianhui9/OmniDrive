#!/bin/sh
# Run from /data/qh/phdCode/work4/OmniDrive in a fresh checkout/output directory.
# Completed output is protected by the experiment tools; never delete old runs.
# Every child inherits CPU affinity excluding faulty CPUs 4 and 5.
# Host CUDA invocations below required sandbox escalation in this session.

taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=/tmp/m1_quality_baseline.xml
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.quality_coverage_audit
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.quality_coverage_probe

# Following implementation and controlled tests, final real Development replay:
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/m1_quality_mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.quality_coverage_experiments --stage development --workers 4

# Actual CUDA tests and unchanged-checkpoint perception comparison on idle GPU.
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/m1_quality_test_mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_quality_coverage_optimization/regression_gpu_corrected.xml
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/m1_quality_gpu_mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.model_smoke --compare --perception-only --frames 8 --m1-config configs/worldstream_m1_quality_coverage.yaml --output outputs/module1_quality_coverage_optimization/gpu_parity
taskset -c 0-3,6-31 sha256sum ckpts/iter_10548.pth

# No fresh Calibration logs remain. Run empty stage honestly, freeze null selection.
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.quality_coverage_experiments --stage calibration
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.finalize_quality_coverage --freeze

# Heldout and Final summary files are Not Run; no exposed data are relabeled.
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.finalize_quality_coverage
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_quality_mpl /home/qh/miniconda3/envs/OmniDrive/bin/python outputs/module1_quality_coverage_optimization/make_figures.py
taskset -c 0-3,6-31 git diff --check

# Attempts/ preserves interrupted diagnostics, the first Development pilot,
# the pre-final-binding replay, and the mixed object/map CUDA failure.
# None contributes to the final metrics or independent sample count.
