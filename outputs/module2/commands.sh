#!/usr/bin/env bash
# Executed in /data/qh/phdCode/work4/OmniDrive with authorized host CUDA access.
# Integration refuses an existing run directory; choose a fresh --output for replay.
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
"$PYTHON" -m tools.worldstream_m2.inspect_pipeline
"$PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 -q --junitxml=outputs/module2/test_logs/unit_tests.xml
"$PYTHON" -m tools.worldstream_m2.benchmark --repeats 20
"$PYTHON" -m tools.worldstream_m2.integration --output outputs/module2/integration_final --frames 4 --warmup 1 --repeats 2
"$PYTHON" -m tools.worldstream_m2.finalize
"$PYTHON" -m tools.worldstream_m2.plot_results
