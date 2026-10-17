#!/usr/bin/env bash
# Actual third-round invocation; all subcommands inherit healthy CPU affinity.
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
taskset -c 0-3,6-31 bash tools/system_optimization/run_v3.sh
# CPU-only exploratory diagnosis, already executed separately:
# taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/omnidrive-v3-mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.system_optimization.v3_physical --diagnose
# Existing measurement directories are immutable; for a fresh run use a new ROOT and freeze a new protocol.
