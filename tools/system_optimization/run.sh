#!/usr/bin/env bash
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
SYSTEM_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
SYSTEM_ROOT=${SYSTEM_ROOT:-outputs/system_optimization}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
export MPLCONFIGDIR=/tmp/omnidrive-system-mpl
mkdir -p "$SYSTEM_ROOT/logs"
ready() { "$SYSTEM_PYTHON" -u -m tools.system_optimization.wait_gpu --output "$SYSTEM_ROOT/logs/gpu_availability.jsonl"; }
# The audit is frozen before algorithm changes and intentionally not overwritten.
if [[ ! -f "$SYSTEM_ROOT/audit_manifest.json" ]]; then
  "$SYSTEM_PYTHON" -m tools.system_optimization.audit --output "$SYSTEM_ROOT"
fi
ready
"$SYSTEM_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q \
  > "$SYSTEM_ROOT/logs/final_gpu_tests.log" 2>&1
"$SYSTEM_PYTHON" -m tools.system_optimization.gate_benchmark --output "$SYSTEM_ROOT" --cuda \
  > "$SYSTEM_ROOT/logs/gate_gpu.log" 2>&1
for workload in single multi; do
  ready
  SYSTEM_EXTRA=()
  if [[ "$workload" == single ]]; then SYSTEM_EXTRA=(--decode-probe); fi
  "$SYSTEM_PYTHON" -u -m tools.system_optimization.experiment \
    --output "$SYSTEM_ROOT/$workload" --workload "$workload" --profile-stages --trace "${SYSTEM_EXTRA[@]}" \
    > "$SYSTEM_ROOT/logs/$workload.log" 2>&1
done
ready
"$SYSTEM_PYTHON" -u -m tools.system_optimization.experiment \
  --output "$SYSTEM_ROOT/language_candidates" --workload multi --scenes 2 --frames 4 \
  --warmup 1 --repeats 2 --modes original prefix batch adaptive compact state_only state_full --profile-stages \
  > "$SYSTEM_ROOT/logs/language_candidates.log" 2>&1
ready
"$SYSTEM_PYTHON" -u -m tools.system_optimization.experiment \
  --output "$SYSTEM_ROOT/long_stream" --workload single --scene-start 98 --scenes 2 \
  --frames 32 --warmup 0 --repeats 1 --modes original m123 unified_persistent unified_output --profile-stages \
  > "$SYSTEM_ROOT/logs/long_stream.log" 2>&1
for budget in 0 128; do
  ready
  "$SYSTEM_PYTHON" -u -m tools.system_optimization.experiment \
    --output "$SYSTEM_ROOT/budget_$budget" --workload multi --scenes 2 --frames 2 \
    --warmup 1 --repeats 1 --cache-budget-mb "$budget" \
    --modes original m123 batch adaptive unified_persistent --profile-stages \
    > "$SYSTEM_ROOT/logs/budget_$budget.log" 2>&1
done
ready
"$SYSTEM_PYTHON" -u -m tools.system_optimization.experiment \
  --output "$SYSTEM_ROOT/stride_refresh" --workload single --scenes 2 --frames 4 \
  --frame-stride 2 --refresh-interval 2 --warmup 1 --repeats 1 \
  --modes original m1 m123 unified_persistent --profile-stages \
  > "$SYSTEM_ROOT/logs/stride_refresh.log" 2>&1
"$SYSTEM_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q \
  > "$SYSTEM_ROOT/logs/delivery_gpu_tests.log" 2>&1
"$SYSTEM_PYTHON" -m tools.system_optimization.report --root "$SYSTEM_ROOT"
"$SYSTEM_PYTHON" -c 'import sys; from tools.system_optimization.delivery import main; main(sys.argv[1])' "$SYSTEM_ROOT"
