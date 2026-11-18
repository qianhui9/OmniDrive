#!/usr/bin/env bash
set -euo pipefail
if [[ "${OMNIDRIVE_COMPONENT_AFFINITY:-0}" != 1 ]]; then
    exec taskset -c 0-3,6-31 env OMNIDRIVE_COMPONENT_AFFINITY=1 bash "$0" "$@"
fi
COMPONENT_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$COMPONENT_PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false TZ=Asia/Shanghai
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/omnidrive-component-mpl}"
# Reuse the already verified process-local matching driver bundle if necessary.
exec /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -m tools.component_latency_eval.driver "$@"
