#!/usr/bin/env bash
# Executed refinement workflow, from the repository root, in environment OmniDrive.
# GPU commands need host CUDA access (RTX 3090). No downloads or weight changes.
# This is a reproduction record: integration refuses to overwrite runs.json.
# Choose fresh --output paths for a new run and archive aggregate JSON first.
set -euo pipefail
PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python

# Baseline source was copied before edits to baseline_source/; its SHA256 manifest
# and original_artifacts_sha256.json preserve the original code/results oracle.
"$PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 -q \
  --junitxml=outputs/module2_refinement/test_logs/unit_tests.xml

# Per-block Full / original packed / spatial packed / naive, profiler traces,
# two independent full-EVA seeds checked against the unchanged original envelope.
"$PYTHON" -m tools.worldstream_m2.refine_gpu --repeats 20

# Four independent scenes, Latin-square mode rotation, real online M1, full VLM.
"$PYTHON" -m tools.worldstream_m2.refine_integration \
  --output outputs/module2_refinement/integration_final --frames 4 --warmup 1 --repeats 2
"$PYTHON" -m tools.worldstream_m2.diagnose_exact_gate
"$PYTHON" -m tools.worldstream_m2.analyze_archived_masks

# Native-dtype Memory and final output-layout audit: changed and identical inputs.
# Publishes conservative downstream quality admission, then measures Auto/full EVA.
"$PYTHON" -m tools.worldstream_m2.refine_integration \
  --output outputs/module2_refinement/quality_audit_layout --frames 4 --warmup 0 \
  --repeats 1 --controlled-only --run-repeat-after --run-auto-after

# Final cost-layout correction. Pre-correction measurements are preserved in
# pre_cost_layout/. Whole-EVA quality evidence is retained; actual block stride
# is now preserved and independently timed. Re-publish policy and remeasure Auto.
"$PYTHON" -m tools.worldstream_m2.refine_gpu --blocks-only --repeats 20
"$PYTHON" -m tools.worldstream_m2.refine_auto

"$PYTHON" -m tools.worldstream_m2.report_refinement
"$PYTHON" -m tools.worldstream_m2.plot_refinement
"$PYTHON" -m compileall -q worldstream/m2 tools/worldstream_m2 tests/worldstream_m2
git diff --check

# Retained failed candidates are not aggregate inputs:
# integration/: full-mask CPU admission regressed, stopped and replaced.
# quality_audit/: stopped before inference to retain Memory's original dtype.
# quality_audit_final/: native-dtype audit before final output-stride correction.
# pre_output_layout/: Auto, EVA, and quality aggregates before that correction.
