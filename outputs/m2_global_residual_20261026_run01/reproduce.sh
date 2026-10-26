#!/usr/bin/env bash
set -euo pipefail
task_run_root="${1:-outputs/m2_global_residual_reproduction}"
taskset -c 0-3,6-31 mkdir "$task_run_root"
taskset -c 0-3,6-31 cp -r outputs/m2_global_residual_20261026_run01/baseline_m2 "$task_run_root/baseline_m2"
taskset -c 0-3,6-31 cp outputs/m2_global_residual_20261026_run01/baseline_state_manager.py "$task_run_root/baseline_state_manager.py"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/omnidrive_mpl
task_python=/home/qh/miniconda3/envs/OmniDrive/bin/python
task_wait_gpu() {
taskset -c 0-3,6-31 "$task_python" - <<'PY'
import subprocess,time
while True:
    used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
    if used<500:break
    print('WAIT GPU occupied; do not stop other processes',flush=True)
    time.sleep(30)
PY
}
task_wait_gpu
taskset -c 0-3,6-31 "$task_python" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123 -q > "$task_run_root/tests_gpu.log" 2>&1
task_wait_gpu
taskset -c 0-3,6-31 "$task_python" -u -m tools.worldstream_m2.window_validate --wsgrr --main-only --formal --snapshot-boundary --frames 40 --repeats 3 --warmup 1 --anchor-intervals 2 3 4 --global-drift-threshold 2 --checkpoint ckpts/iter_10548.pth --m1-predictions outputs/wsgrr_mainfix_20261023/references/wsgrrr_20261022_followup/Native_full/official_eval/detection/pts_bbox/results_nusc.json --output "$task_run_root/drift_calibration40" > "$task_run_root/drift_calibration40.log" 2>&1
taskset -c 0-3,6-31 "$task_python" -u -m tools.worldstream_m2.global_residual_benchmark --root "$task_run_root" --stage calibration > "$task_run_root/calibration_pipeline.log" 2>&1
taskset -c 0-3,6-31 "$task_python" -m tools.worldstream_m2.global_residual_analysis --root "$task_run_root" --stage calibration
taskset -c 0-3,6-31 "$task_python" -u -m tools.worldstream_m2.global_residual_benchmark --root "$task_run_root" --stage service > "$task_run_root/service_pipeline.log" 2>&1
taskset -c 0-3,6-31 "$task_python" -u -m tools.worldstream_m2.global_residual_benchmark --root "$task_run_root" --stage validation > "$task_run_root/validation_pipeline.log" 2>&1
# The measured global drift threshold is deliberately supplied explicitly; it
# never enables another executor. Frozen legacy code is only a paired control.
task_threshold=$(taskset -c 0-3,6-31 "$task_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' "$task_run_root/selection.json")
task_interval=$(taskset -c 0-3,6-31 "$task_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["interval"])' "$task_run_root/selection.json")
task_wait_gpu
taskset -c 0-3,6-31 "$task_python" -u -m tools.worldstream_m2.window_validate --wsgrr --main-only --formal --snapshot-boundary --frames 40 --repeats 3 --warmup 1 --anchor-intervals "$task_interval" --global-drift-threshold "$task_threshold" --reference-runtime "$task_run_root/baseline_m2" --reference-state-manager "$task_run_root/baseline_state_manager.py" --followup-profile --m1-predictions outputs/wsgrr_mainfix_20261023/references/wsgrrr_20261022_followup/Native_full/official_eval/detection/pts_bbox/results_nusc.json --output "$task_run_root/paired40" > "$task_run_root/paired40.log" 2>&1
taskset -c 0-3,6-31 "$task_python" -u -m tools.worldstream_m2.task_quality --run outputs/m3_independent_validation_20261026_run01/runs/r0_wsdr_kv --output "$task_run_root/historical_wsgrr_validation_official" > "$task_run_root/historical_wsgrr_validation_official.log" 2>&1
taskset -c 0-3,6-31 "$task_python" -m tools.worldstream_m2.global_residual_analysis --root "$task_run_root" --stage final
