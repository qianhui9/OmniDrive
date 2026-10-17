#!/usr/bin/env bash
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/omnidrive-v4-mpl
OMNIDRIVE_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
V4_ROOT=outputs/system_optimization_v4
# Keep GPU measurements isolated from heavy Oracle/report processing.
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -c 'import json,time,pathlib
p=pathlib.Path("outputs/system_optimization_v4/gpu_status.json")
while True:
 s=json.loads(p.read_text())["status"]
 if s=="completed":break
 if s=="failed":raise SystemExit("GPU runner failed; refusing success report")
 time.sleep(10)'
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m tools.system_optimization.task_dependency_audit > "$V4_ROOT/logs/final_task_audit.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.v4_facts > "$V4_ROOT/logs/task_fact_diagnostic.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.v4_report > "$V4_ROOT/logs/report.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m tools.system_optimization.wait_gpu --minimum-mib 24000 --output "$V4_ROOT/logs/final_test_gpu_availability.jsonl"
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q > "$V4_ROOT/logs/final_gpu_tests.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.v4_delivery > "$V4_ROOT/logs/delivery.log" 2>&1
