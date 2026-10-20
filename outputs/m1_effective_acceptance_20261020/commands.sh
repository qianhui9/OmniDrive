#!/usr/bin/env bash
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
# Choose a fresh directory for every reproduction; never overwrite old runs.
run_dir="${1:-outputs/m1_effective_acceptance_reproduction}"
mkdir "$run_dir"
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.qualification_inventory --output "$run_dir/qualification_inventory.json"
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123 -q -r s --junitxml="$run_dir/regression.xml" > "$run_dir/regression.log" 2>&1
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.effective_acceptance --replay-output "$run_dir/rgb_replay" > "$run_dir/rgb_replay.log" 2>&1
# Run only when GPU is idle. In this session CUDA required authorized host access.
taskset -c 0-3,6-31 nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.effective_gpu_validation --frames 8 --output "$run_dir/full_model" > "$run_dir/full_model.log" 2>&1
