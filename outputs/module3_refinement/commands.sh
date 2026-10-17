#!/usr/bin/env bash
# Full reproduction: bash outputs/module3_refinement/commands.sh
# Existing artifacts are never overwritten; each invocation creates a new root.
# Recorded primary run: multi_20261017_005514; numeric_e2e_20261017_005514;
# single_20261017_005514. Controlled probes were copied byte-for-byte from
# multi_20261017_002956/controlled_probes (provenance.json records every hash).
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
M3_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
M3_REPRO_ROOT="outputs/module3_refinement/reproduction_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$M3_REPRO_ROOT/logs"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/m3-refinement-matplotlib
while true; do
  M3_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  if [[ "$M3_FREE" =~ ^[0-9]+$ ]] && (( M3_FREE >= 18500 )); then break; fi
  sleep 30
done
"$M3_PYTHON" -u -m tools.worldstream_m3.numerical_refinement --output "$M3_REPRO_ROOT/numerical" > "$M3_REPRO_ROOT/logs/numerical.log" 2>&1
"$M3_PYTHON" -u -m tools.worldstream_m3.common_prefix_probe --output "$M3_REPRO_ROOT/common_prefix" > "$M3_REPRO_ROOT/logs/common_prefix.log" 2>&1
"$M3_PYTHON" -u -m tools.worldstream_m3.refinement --output "$M3_REPRO_ROOT/multi" --probes > "$M3_REPRO_ROOT/logs/multi.log" 2>&1
"$M3_PYTHON" -u -m tools.worldstream_m3.refinement --output "$M3_REPRO_ROOT/numeric_e2e" --numeric-only --modes baseline precision_deterministic precision_no_fp16_reduction precision_fp32_lora common_prefix_seq > "$M3_REPRO_ROOT/logs/numeric_e2e.log" 2>&1
"$M3_PYTHON" -u -m tools.worldstream_m3.refinement --output "$M3_REPRO_ROOT/single" --single --sampling --modes baseline legacy_exact refined_exact batched m123 > "$M3_REPRO_ROOT/logs/single.log" 2>&1
"$M3_PYTHON" -u -m tools.worldstream_m3.refinement --output "$M3_REPRO_ROOT/projection_e2e" --numeric-only --precision fp32_qkv --modes baseline fp32_full refined_exact batched refined_combined m123 > "$M3_REPRO_ROOT/logs/projection_e2e.log" 2>&1
"$M3_PYTHON" -u -m tools.worldstream_m3.world_state_sweep --output "$M3_REPRO_ROOT/state_sweep" --primary "$M3_REPRO_ROOT/multi" > "$M3_REPRO_ROOT/logs/state_sweep.log" 2>&1
"$M3_PYTHON" -m pytest -q tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 > "$M3_REPRO_ROOT/logs/tests.log" 2>&1
for M3_RUN in multi numeric_e2e single projection_e2e; do
  "$M3_PYTHON" -m tools.worldstream_m3.qa_analysis --run "$M3_REPRO_ROOT/$M3_RUN"
done
"$M3_PYTHON" -m tools.worldstream_m3.refinement_report --run "$M3_REPRO_ROOT/multi" --single-run "$M3_REPRO_ROOT/single" --numerical "$M3_REPRO_ROOT/numerical" --common-prefix "$M3_REPRO_ROOT/common_prefix" --numeric-e2e "$M3_REPRO_ROOT/numeric_e2e" --state-sweep "$M3_REPRO_ROOT/state_sweep" --projection-e2e "$M3_REPRO_ROOT/projection_e2e" --test-log "$M3_REPRO_ROOT/logs/tests.log" --output "$M3_REPRO_ROOT"
