#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
# Default only prepares real GT and validates inputs; full GPU work is opt-in.
exec taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u \
  -m tools.experiments.long_run_state_aging.run "$@"
