#!/usr/bin/env bash
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/omnidrive-v3-mpl
OMNIDRIVE_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
V3_ROOT=outputs/system_optimization_v3
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q > "$V3_ROOT/logs/preflight_gpu_tests.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m tools.system_optimization.wait_gpu --output "$V3_ROOT/logs/gpu_availability.jsonl"
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.v3 > "$V3_ROOT/logs/gpu_execution.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.v3_physical > "$V3_ROOT/logs/independent_physical.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.v3_report > "$V3_ROOT/logs/report.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q > "$V3_ROOT/logs/final_gpu_tests.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.v3_delivery > "$V3_ROOT/logs/delivery.log" 2>&1
