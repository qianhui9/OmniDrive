#!/usr/bin/env bash
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/omnidrive-v4-mpl
OMNIDRIVE_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
V4_ROOT=outputs/system_optimization_v4
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q > "$V4_ROOT/logs/preflight_gpu_tests.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m tools.system_optimization.wait_gpu --minimum-mib 24000 --output "$V4_ROOT/logs/gpu_availability.jsonl"
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.v4 > "$V4_ROOT/logs/gpu_execution.log" 2>&1
