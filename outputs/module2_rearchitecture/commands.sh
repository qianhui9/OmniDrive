#!/usr/bin/env bash
# Reproduction record; run from /data/qh/phdCode/work4/OmniDrive.
# The full integration runner requires a fresh output directory. Preserve the
# supplied outputs before a new run. Previous M1/M2 directories are read-only inputs.
set -euo pipefail
PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
"$PYTHON" -m tools.worldstream_m2.rearchitecture.wait_gpu
"$PYTHON" -m tools.worldstream_m2.rearchitecture.phase_ab --repeats 10
"$PYTHON" -m tools.worldstream_m2.rearchitecture.feasibility
"$PYTHON" -m tools.worldstream_m2.rearchitecture.integration \
  --output outputs/module2_rearchitecture/integration_final --warmup 1 --repeats 2
"$PYTHON" -m tools.worldstream_m2.rearchitecture.postflight
"$PYTHON" -m compileall -q worldstream/m2 tools/worldstream_m2/rearchitecture tests/worldstream_m2
bash -n outputs/module2_rearchitecture/commands.sh
git diff --check
