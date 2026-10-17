#!/usr/bin/env bash
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/omnidrive-system-v2-mpl
OMNIDRIVE_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
V2_ROOT=outputs/system_optimization_v2
mkdir -p "$V2_ROOT/logs"
case "${1:-all}" in
  all)
    taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m tools.system_optimization.wait_gpu --output "$V2_ROOT/logs/gpu_availability.jsonl"
    taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.v2 --stage smoke > "$V2_ROOT/logs/smoke.log" 2>&1
    ;;
  after-smoke)
    # Resume an already authorized smoke launcher without starting another.
    taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" - <<'PY'
import json,time
from pathlib import Path
p=Path('outputs/system_optimization_v2/smoke_status.json')
while True:
    if p.exists():
        status=json.loads(p.read_text())
        if status['status']=='failed':raise SystemExit('Smoke failed: '+str(status))
        if status['status']=='completed':break
    time.sleep(10)
PY
    ;;
  *) echo 'Usage: run_v2.sh [all|after-smoke]' >&2; exit 2 ;;
esac
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m tools.system_optimization.wait_gpu --output "$V2_ROOT/logs/gpu_availability.jsonl"
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.v2 --stage formal > "$V2_ROOT/logs/formal.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.physical_oracle > "$V2_ROOT/logs/physical_oracle.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.v2_report > "$V2_ROOT/logs/report.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q > "$V2_ROOT/logs/final_gpu_regression.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.delivery_v2 > "$V2_ROOT/logs/delivery.log" 2>&1
