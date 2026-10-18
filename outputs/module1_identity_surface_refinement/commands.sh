#!/bin/sh
# Actual command inventory. Results already exist; do not overwrite archived runs.
# Working directory: /data/qh/phdCode/work4/OmniDrive
# Environment: /home/qh/miniconda3/envs/OmniDrive/bin/python
# Exclude faulty CPUs 4/5. GPU commands were run only after an idle-GPU check.
set -eu
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.identity_surface_audit'
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.prepare_identity_surface
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.collect_identity_surface_observations'
# root_cause_audit.md and experiment_protocol.json saved before core changes.
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.identity_surface_experiments --stage development'
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.identity_surface_experiments --stage calibration'
# Calibration writes algorithm_selection.json and frozen_validation_protocol.json.
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.identity_surface_experiments --stage heldout'
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.identity_surface_experiments --stage final_test'
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_identity_surface_refinement/regression.xml'
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.visualize_identity_surface
taskset -c 0-3,6-31 sh -c 'sha256sum ckpts/iter_10548.pth > outputs/module1_identity_surface_refinement/checkpoint.sha256'
# Run subsequent timings without overlapping quality/analysis jobs.
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.model_smoke --compare --perception-only --frames 8 --m1-config configs/worldstream_m1_identity_surface.yaml --output outputs/module1_identity_surface_refinement/gpu_parity_isolated'
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.profile_identity_surface'
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.finalize_identity_surface'
taskset -c 0-3,6-31 git diff --check
