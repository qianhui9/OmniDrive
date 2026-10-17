#!/usr/bin/env bash
# Current delivery command ledger; original module results are never overwritten.
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
SYSTEM_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Audit was frozen before algorithm changes (audit_manifest.json).
# python -m tools.system_optimization.audit
"$SYSTEM_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q
# For fresh independent reproduction:
# SYSTEM_ROOT=outputs/system_optimization_repeat bash tools/system_optimization/run.sh
"$SYSTEM_PYTHON" -m tools.system_optimization.report --root outputs/system_optimization
"$SYSTEM_PYTHON" -m tools.system_optimization.delivery
