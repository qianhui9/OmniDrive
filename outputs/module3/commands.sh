#!/usr/bin/env bash
# Regenerate the delivered analysis from existing real GPU records (no inference).
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
M3_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/m3-matplotlib
M3_MULTI=outputs/module3/clean_multi_20261016_220113
M3_SINGLE=outputs/module3/clean_single_20261016_220113
M3_SUPPLEMENT=outputs/module3/quantity_supplement_01

"$M3_PYTHON" -m tools.worldstream_m3.qa_analysis --run "$M3_MULTI"
"$M3_PYTHON" -m tools.worldstream_m3.qa_analysis --run "$M3_SINGLE"
"$M3_PYTHON" -m tools.worldstream_m3.report \
  --run "$M3_MULTI" --single-run "$M3_SINGLE" --supplement "$M3_SUPPLEMENT" \
  --test-log outputs/module3/logs/clean_tests_20261016_220113.log

# Full fresh GPU reproduction, including the 197-test suite, six methods, single
# question deployment, quantity coverage and both genuine profiler traces:
# bash tools/worldstream_m3/resume_after_access.sh
# The script waits for >=18500 MiB free GPU memory, never stops other workloads,
# and chooses new timestamped directories. Failed run01/run02/run03 are retained.
