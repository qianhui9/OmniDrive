#!/usr/bin/env bash
# Continue the authorized task after the immutable initial formal program.
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/omnidrive-system-v2-mpl
OMNIDRIVE_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
V2_ROOT=outputs/system_optimization_v2
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" - <<'PY'
import json,time
from pathlib import Path
p=Path('outputs/system_optimization_v2/formal_status.json')
while True:
    state=json.loads(p.read_text()) if p.exists() else {}
    if state.get('status')=='failed':raise SystemExit('Original formal stage failed: '+str(state))
    if state.get('status')=='completed':break
    time.sleep(10)
PY
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q > "$V2_ROOT/logs/pinned_gpu_regression.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m tools.system_optimization.wait_gpu --output "$V2_ROOT/logs/gpu_availability.jsonl"
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.v2_cost_refresh > "$V2_ROOT/logs/cost_refresh_pinned.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.physical_oracle > "$V2_ROOT/logs/physical_oracle_refreshed.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.v2_report > "$V2_ROOT/logs/report_refreshed.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q > "$V2_ROOT/logs/final_current_gpu_regression.log" 2>&1
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.system_optimization.delivery_v2 > "$V2_ROOT/logs/final_delivery.log" 2>&1
