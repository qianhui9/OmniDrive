#!/bin/sh
# Actual invocation inventory, 2026-10-19. Preserve these archived outputs.
# Reproduction tools require a fresh --output directory.
# Working directory: /data/qh/phdCode/work4/OmniDrive
# Python: /home/qh/miniconda3/envs/OmniDrive/bin/python
# Every process and its children exclude the faulty CPUs 4/5.

# Initial baseline: 194 M1 tests passed.
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 -q'

# Before the fixes: 13 failed / 5 passed (controlled contracts only).
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1/test_identity_surface_bindings.py -q > outputs/module1_identity_surface_refinement/revalidation_20261019/logs/binding_before_fix.log 2>&1'

# After the fixes: 74 selected evidence/identity/surface tests passed.
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1/test_identity_surface.py tests/worldstream_m1/test_identity_surface_bindings.py tests/worldstream_m1/test_evidence_contract.py -q --junitxml=outputs/module1_identity_surface_refinement/revalidation_20261019/binding_regression.xml > outputs/module1_identity_surface_refinement/revalidation_20261019/logs/binding_after_fix.log 2>&1'

# Sandbox CPU suite: 420 passed / 1 CUDA-only skipped.
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_identity_surface_refinement/revalidation_20261019/regression.xml > outputs/module1_identity_surface_refinement/revalidation_20261019/logs/regression.log 2>&1'

# True RGB/detection/LiDAR/annotation reproduction; no selection or new qualification.
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.revalidate_identity_surface --stage development --output outputs/module1_identity_surface_refinement/revalidation_20261019/replay_development > outputs/module1_identity_surface_refinement/revalidation_20261019/logs/replay_development.log 2>&1'
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.revalidate_identity_surface --stage calibration --output outputs/module1_identity_surface_refinement/revalidation_20261019/replay_calibration > outputs/module1_identity_surface_refinement/revalidation_20261019/logs/replay_calibration.log 2>&1'
taskset -c 0-3,6-31 sh -c 'sha256sum ckpts/iter_10548.pth > outputs/module1_identity_surface_refinement/revalidation_20261019/checkpoint.sha256'

# Host execution with approved sandbox escalation, after confirming idle GPU.
# The sandbox's initial nvidia-smi probe failed because it could not see devices.
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.model_smoke --compare --perception-only --frames 8 --m1-config configs/worldstream_m1_identity_surface.yaml --output outputs/module1_identity_surface_refinement/revalidation_20261019/gpu_parity > outputs/module1_identity_surface_refinement/revalidation_20261019/logs/gpu_parity.log 2>&1'

# Host CUDA suite: 421 passed / 0 skipped / 0 failed.
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_identity_surface_refinement/revalidation_20261019/regression_gpu.xml > outputs/module1_identity_surface_refinement/revalidation_20261019/logs/regression_gpu.log 2>&1'

# No other experiment overlapped this preloaded CPU timing comparison.
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.profile_identity_surface_bindings --output outputs/module1_identity_surface_refinement/revalidation_20261019/isolated_timing > outputs/module1_identity_surface_refinement/revalidation_20261019/logs/isolated_timing.log 2>&1'
taskset -c 0-3,6-31 git diff --check
