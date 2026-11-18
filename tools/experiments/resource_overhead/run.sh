#!/usr/bin/env bash
set -euo pipefail
if [[ "${RESOURCE_OVERHEAD_AFFINITY:-0}" != 1 ]]; then
    exec taskset -c 0-3,6-31 env RESOURCE_OVERHEAD_AFFINITY=1 bash "$0" "$@"
fi
RESOURCE_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$RESOURCE_PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TZ=Asia/Shanghai
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/omnidrive-resource-mpl}"
exec /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -X faulthandler \
    -m tools.experiments.resource_overhead.run "$@"
