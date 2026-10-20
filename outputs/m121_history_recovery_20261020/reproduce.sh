#!/usr/bin/env bash
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/qh/miniconda3/envs/OmniDrive/bin/python
ROOT=${1:?Provide a NEW nonexistent output directory}
taskset -c 0-3,6-31 "$PY" -m tools.worldstream_m1.history_recovery_experiment freeze --output "$ROOT"
for variant in baseline old_b a b c aligned; do
  taskset -c 0-3,6-31 "$PY" -m tools.worldstream_m1.history_recovery_experiment run --output "$ROOT" --cohort fixed --variant "$variant"
done
taskset -c 0-3,6-31 "$PY" -m tools.worldstream_m1.postprocess_history_recovery --root "$ROOT" --cohort fixed
for variant in baseline c aligned; do
  taskset -c 0-3,6-31 "$PY" -m tools.worldstream_m1.history_recovery_experiment run --output "$ROOT" --cohort expanded --variant "$variant"
done
taskset -c 0-3,6-31 "$PY" -m tools.worldstream_m1.postprocess_history_recovery --root "$ROOT" --cohort expanded
taskset -c 0-3,6-31 "$PY" -m tools.worldstream_m1.depth_oracle_diagnosis --root "$ROOT/fixed/aligned" --output "$ROOT/oracle_strict"
# Wait for other jobs; never stop another user's process.
while true; do
  M121_GPU_METRICS=$(taskset -c 0-3,6-31 nvidia-smi --id=0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) || exit 2
  if taskset -c 0-3,6-31 "$PY" -c 'import sys; m,u=map(int,sys.argv[1].split(","));sys.exit(0 if m<=100 and u==0 else 1)' "$M121_GPU_METRICS"; then break; fi
  taskset -c 0-3,6-31 sleep 10
done
taskset -c 0-3,6-31 "$PY" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123 -q --junitxml="$ROOT/tests.xml"
taskset -c 0-3,6-31 "$PY" -m tools.worldstream_m1.effective_gpu_validation --output "$ROOT/gpu" --frames 12 --scene-ids 01452fbfbf4543af8acdfd3e8a1ee806,01e4fcbe6e49483293ce45727152b36e --m1-config configs/worldstream_m121_aligned.yaml
