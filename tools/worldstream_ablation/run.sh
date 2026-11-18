#!/usr/bin/env bash
set -euo pipefail
if [[ "${OMNIDRIVE_ABLATION_AFFINITY:-0}" != 1 ]]; then
    exec taskset -c 0-3,6-31 env OMNIDRIVE_ABLATION_AFFINITY=1 bash "$0" "$@"
fi
ABLATION_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ABLATION_PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export TZ=Asia/Shanghai TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/omnidrive-ablation-mpl}"
exec /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -X faulthandler \
    -m tools.worldstream_ablation.run "$@"
