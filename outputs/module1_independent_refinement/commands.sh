#!/bin/sh
# Commands used for final verification. Run from /data/qh/phdCode/work4/OmniDrive.
# Experiment outputs refuse accidental overwrite; archive an earlier run first.
# GPU commands require host device access and an idle GPU; never stop other jobs.
set -eu
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_independent_refinement/regression_final.xml
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.independent_refinement'
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
taskset -c 0-3,6-31 sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.model_smoke --compare --perception-only --frames 8 --m1-config configs/worldstream_m1_independent.yaml --output outputs/module1_independent_refinement/gpu_perception_profiled'
taskset -c 0-3,6-31 sha256sum ckpts/iter_10548.pth
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.visualize_independent_cases
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.finalize_independent
taskset -c 0-3,6-31 git diff --check
