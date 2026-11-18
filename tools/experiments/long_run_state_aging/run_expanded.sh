#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
# run.sh supplies the OmniDrive interpreter, CPU affinity and thread limits.
exec taskset -c 0-3,6-31 bash tools/experiments/long_run_state_aging/run.sh \
  --config tools/experiments/long_run_state_aging/expanded_config.yaml "$@"
