#!/usr/bin/env bash
set -euo pipefail
exec taskset -c 0-3,6-31 bash -c '
    cd -- "$1"
    shift
    exec /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u \
        -m tools.worldstream_ablation.driver_compat exec -- \
        bash tools/worldstream_ablation/recover.sh "$@"
' ablation-driver-compat "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)" "$@"
