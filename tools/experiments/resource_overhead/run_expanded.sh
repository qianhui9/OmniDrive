#!/usr/bin/env bash
set -euo pipefail
# run.sh applies the approved CPU affinity and the OmniDrive interpreter.
RESOURCE_EXPANDED_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$RESOURCE_EXPANDED_DIR/run.sh" --config "$RESOURCE_EXPANDED_DIR/config_54_s4_s5.json" "$@"
