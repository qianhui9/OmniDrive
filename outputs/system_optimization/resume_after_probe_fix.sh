#!/usr/bin/env bash
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
SYSTEM_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/omnidrive-system-mpl
ready() { "$SYSTEM_PYTHON" -u -m tools.system_optimization.wait_gpu --output outputs/system_optimization/logs/gpu_availability.jsonl; }
ready
"$SYSTEM_PYTHON" -u -m tools.system_optimization.experiment --output outputs/system_optimization/lifecycle_validation --workload single --scene-start 98 --scenes 2 --frames 3 --warmup 0 --repeats 1 --modes original --lifecycle-probe --profile-stages > outputs/system_optimization/logs/lifecycle_validation.log 2>&1
ready
"$SYSTEM_PYTHON" -u -m tools.system_optimization.experiment --output outputs/system_optimization/long_stream_retry --workload single --scene-start 98 --scenes 2 --frames 32 --warmup 0 --repeats 1 --modes original m123 unified_persistent unified_output --no-lifecycle-probe --profile-stages > outputs/system_optimization/logs/long_stream_retry.log 2>&1
for budget in 0 128; do
 ready
 "$SYSTEM_PYTHON" -u -m tools.system_optimization.experiment --output "outputs/system_optimization/budget_$budget" --workload multi --scenes 2 --frames 2 --warmup 1 --repeats 1 --cache-budget-mb "$budget" --modes original m123 batch adaptive unified_persistent --profile-stages > "outputs/system_optimization/logs/budget_$budget.log" 2>&1
done
ready
"$SYSTEM_PYTHON" -u -m tools.system_optimization.experiment --output outputs/system_optimization/stride_refresh --workload single --scenes 2 --frames 4 --frame-stride 2 --refresh-interval 2 --warmup 1 --repeats 1 --modes original m1 m123 unified_persistent --profile-stages > outputs/system_optimization/logs/stride_refresh.log 2>&1
"$SYSTEM_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q > outputs/system_optimization/logs/delivery_gpu_tests.log 2>&1
"$SYSTEM_PYTHON" -m tools.system_optimization.report --root outputs/system_optimization
