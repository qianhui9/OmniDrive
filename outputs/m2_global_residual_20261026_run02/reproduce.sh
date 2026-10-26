#!/usr/bin/env bash
set -euo pipefail
# Invoke from the repository root. Each run requires a NEW output directory.
task_run_root="${1:-outputs/m2_global_residual_reproduction_budget_0p05}"
task_source=outputs/m2_global_residual_20261026_run02
task_prior=outputs/m2_global_residual_20261026_run01
task_python=/home/qh/miniconda3/envs/OmniDrive/bin/python
taskset -c 0-3,6-31 mkdir "$task_run_root"
taskset -c 0-3,6-31 cp -r "$task_source/baseline_m2" "$task_run_root/baseline_m2"
taskset -c 0-3,6-31 cp "$task_source/baseline_state_manager.py" "$task_run_root/baseline_state_manager.py"
# Reuse the declared development-only drift evidence, with its recorded SHA256.
# This does not reuse task outputs/timings for the new parameter candidates.
taskset -c 0-3,6-31 cp -rL "$task_source/drift_calibration40" "$task_run_root/drift_calibration40"
taskset -c 0-3,6-31 cp "$task_source/validation_sequence24.json" "$task_run_root/validation_sequence24.json"
taskset -c 0-3,6-31 cp "$task_source/environment.json" "$task_run_root/environment.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/omnidrive_mpl
taskset -c 0-3,6-31 "$task_python" -m tools.worldstream_m2.global_residual_analysis --root "$task_run_root/previous_budget_reassessment" --stage reassess --evidence-root "$task_prior" --quality-budget 0.05
taskset -c 0-3,6-31 "$task_python" -u -m tools.worldstream_m2.global_residual_benchmark --root "$task_run_root" --stage calibration --quality-budget 0.05 --drift-quantile 0.75 > "$task_run_root/calibration_driver.log" 2>&1
taskset -c 0-3,6-31 "$task_python" -m tools.worldstream_m2.global_residual_analysis --root "$task_run_root" --stage calibration --quality-budget 0.05
# Fix the selected interval/threshold before fresh heldout validation, same-state
# visual controls, GPU regression, and three alternating complete service pairs.
if taskset -c 0-3,6-31 test -f "$task_run_root/selection.json"; then
    taskset -c 0-3,6-31 "$task_python" -u -m tools.worldstream_m2.global_residual_benchmark --root "$task_run_root" --stage verify --quality-budget 0.05 > "$task_run_root/verification_driver.log" 2>&1
else
    taskset -c 0-3,6-31 "$task_python" -u -m tools.worldstream_m2.global_residual_benchmark --root "$task_run_root" --stage tests --quality-budget 0.05 > "$task_run_root/tests_driver.log" 2>&1
fi
