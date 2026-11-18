#!/usr/bin/env bash
set -euo pipefail

# CPU 4/5 are faulty. Re-enter before running any external commands.
if [[ "${FQ3_EXPANDED_AFFINITY:-}" != "1" ]]; then
    exec taskset -c 0-3,6-31 env FQ3_EXPANDED_AFFINITY=1 bash "${BASH_SOURCE[0]}" "$@"
fi
cd /data/qh/phdCode/work4/OmniDrive
export MPLCONFIGDIR=/tmp/environment-stability-mpl
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# With no arguments, run the complete expanded experiment. Pass --check-only
# for CPU-only GT preparation, or the documented resume arguments after a stop.
if [[ "$#" -eq 0 ]]; then
    set -- --run-full
fi
exec /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u \
    -m tools.environment_stability_eval.fq3_expanded "$@"
