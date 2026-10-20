#!/usr/bin/env bash
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
run_dir="${1:-outputs/m11_scene_reproduction}"
mkdir "$run_dir"
# 0 means all available target scenes / all keyframes. This delivered run uses
# six CPU scenes with historical detections and two native validation scenes.
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123 -q --disable-warnings -r s --junitxml="$run_dir/regression.xml" > "$run_dir/regression.log" 2>&1
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.effective_acceptance --scenes 6 --frames 12 --repeats 2 --config configs/worldstream_m11_diagnostics.yaml --replay-output "$run_dir/cpu_scenes" > "$run_dir/cpu.log" 2>&1
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.offline_identity_scoring --root "$run_dir/cpu_scenes" --output "$run_dir/offline_cpu"
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.scene_diagnosis --root "$run_dir/cpu_scenes" --output "$run_dir/diagnosis_cpu"
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.identity_cases --root "$run_dir/cpu_scenes" --output "$run_dir/cases_cpu"
# Only proceed when GPU is idle; do not stop anyone else's jobs.
taskset -c 0-3,6-31 nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.effective_gpu_validation --scenes 2 --frames 12 --config projects/configs/OmniDrive/worldstream_m11_diagnostics.py --m1-config configs/worldstream_m11_diagnostics.yaml --output "$run_dir/gpu_scenes" > "$run_dir/gpu.log" 2>&1
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.offline_identity_scoring --root "$run_dir/gpu_scenes" --output "$run_dir/offline_gpu"
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.scene_diagnosis --root "$run_dir/gpu_scenes" --output "$run_dir/diagnosis_gpu"
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.identity_cases --root "$run_dir/gpu_scenes" --output "$run_dir/cases_gpu"
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.qualification_inventory --output "$run_dir/qualification_inventory.json"

taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.scene_revalidate --root "$run_dir/cpu_scenes" --output "$run_dir/version_ledger_cpu.json"
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.scene_revalidate --root "$run_dir/gpu_scenes" --output "$run_dir/version_ledger_gpu.json"
