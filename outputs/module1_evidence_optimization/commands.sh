#!/bin/sh
# Actual command inventory, run from /data/qh/phdCode/work4/OmniDrive.
# Archived results already exist: do not execute this whole file to overwrite them.
# Replay measured quality source from measured_prototype_source when auditing
# original protocol digests; final code changes only qualification fingerprint.
# Every executable inherits CPU affinity excluding faulty CPUs 4/5.
set -eu
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.prepare_evidence_data
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.collect_evidence_observations'
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.evidence_experiments --stage development'
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.evidence_experiments --stage calibration'
# Freeze formal_validation_protocol.json before starting heldout/final scoring.
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.evidence_experiments --stage heldout'
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.evidence_experiments --stage final_test'
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.model_smoke --compare --perception-only --frames 8 --m1-config configs/worldstream_m1_evidence.yaml --output outputs/module1_evidence_optimization/gpu_parity'
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.evidence_geometry_diagnostics
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.visualize_evidence
# Isolated final-source GPU parity: wait until GPU is idle, leave other jobs alone.
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.model_smoke --compare --perception-only --frames 8 --m1-config configs/worldstream_m1_evidence.yaml --output outputs/module1_evidence_optimization/gpu_parity_isolated'
# Run after all other analysis/GPU jobs above have completed.
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.profile_evidence'
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_evidence_optimization/regression.xml
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.finalize_evidence
taskset -c 0-3,6-31 git diff --check
