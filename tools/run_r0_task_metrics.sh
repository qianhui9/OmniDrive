#!/usr/bin/env bash
set -euo pipefail
# Enforce the server's CPU exclusion even when invoked as plain `bash ...`.
if [[ "${OMNIDRIVE_TASK_AFFINITY_READY:-0}" != 1 ]]; then
    exec taskset -c 0-3,6-31 env OMNIDRIVE_TASK_AFFINITY_READY=1 bash "$0" "$@"
fi
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/omnidrive-r0-task-metrics-mpl}"
exec /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -X faulthandler \
    -m tools.worldstream_m123.r0_task_evaluation "$@"
