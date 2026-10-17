#!/usr/bin/env bash
# Reproduction commands; full inference requires unused output subdirectories.
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/omnidrive-system-v2-mpl
OMNIDRIVE_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m tools.system_optimization.identity_audit
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q
taskset -c 0-3,6-31 bash tools/system_optimization/run_v2.sh
taskset -c 0-3,6-31 bash tools/system_optimization/finalize_v2.sh
# Analysis-only (after actual GPU execution):
# taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m tools.system_optimization.physical_oracle
# taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m tools.system_optimization.v2_report

# Actual recovery after the user identified faulty CPU cores 4/5:
# taskset -c 0-3,6-31 bash tools/system_optimization/finalize_v2.sh > outputs/system_optimization_v2/logs/finalize_pinned_launch.log 2>&1
# Gate-only failed cost-refresh attempt is preserved; no measurement output is overwritten.
