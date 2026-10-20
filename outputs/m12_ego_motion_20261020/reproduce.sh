#!/usr/bin/env bash
# Invoke from the repository: taskset -c 0-3,6-31 bash <this-file> <NEW-output-directory>
set -euo pipefail
OMNI_PY=/home/qh/miniconda3/envs/OmniDrive/bin/python
M12_OUT=${1:?Provide a new, nonexistent output directory}
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OMNI_PY" -m tools.worldstream_m1.motion_ablation freeze --output "$M12_OUT" --frames 12 --scenes 6
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OMNI_PY" -m tools.worldstream_m1.motion_ablation run --output "$M12_OUT" --repeats 2
for variant in baseline a b c; do
  taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OMNI_PY" -m tools.worldstream_m1.motion_scoring --root "$M12_OUT/$variant" --output "$M12_OUT/scoring_$variant"
  taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OMNI_PY" -m tools.worldstream_m1.offline_identity_scoring --root "$M12_OUT/$variant" --output "$M12_OUT/identity_$variant"
  taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OMNI_PY" -m tools.worldstream_m1.motion_revalidate --root "$M12_OUT/$variant" --output "$M12_OUT/ledger_$variant.json"
  taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OMNI_PY" -m tools.worldstream_m1.scene_diagnosis --root "$M12_OUT/$variant" --output "$M12_OUT/diagnosis_$variant" --config "configs/worldstream_m12_$variant.yaml"
done
taskset -c 0-3,6-31 "$OMNI_PY" -m tools.worldstream_m1.qualification_inventory --output "$M12_OUT/qualification_inventory.json"
# Monitor only; never kill or preempt another GPU job.
taskset -c 0-3,6-31 "$OMNI_PY" - <<'PY'
import subprocess,time
while True:
    row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]
    memory,utilization=[int(x.strip()) for x in row.split(',')]
    if memory<200 and utilization<5:break
    print('Waiting for idle GPU:',row,flush=True)
    time.sleep(10)
PY
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OMNI_PY" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123 -q --junitxml="$M12_OUT/regression.xml"
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OMNI_PY" -m tools.worldstream_m1.effective_gpu_validation --output "$M12_OUT/gpu" --m1-config configs/worldstream_m12_c.yaml --frames 12 --scene-ids 01452fbfbf4543af8acdfd3e8a1ee806,01e4fcbe6e49483293ce45727152b36e
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OMNI_PY" -m tools.worldstream_m1.offline_identity_scoring --root "$M12_OUT/gpu" --output "$M12_OUT/identity_gpu"
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OMNI_PY" -m tools.worldstream_m1.motion_scoring --root "$M12_OUT/gpu" --output "$M12_OUT/scoring_gpu" --runtime-config projects/configs/OmniDrive/worldstream_m11_diagnostics.py --m1-config configs/worldstream_m12_c.yaml
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OMNI_PY" -m tools.worldstream_m1.scene_revalidate --root "$M12_OUT/gpu" --output "$M12_OUT/ledger_gpu.json"
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OMNI_PY" -m tools.worldstream_m1.motion_cases --root "$M12_OUT/c" --scoring "$M12_OUT/scoring_c" --output "$M12_OUT/cases_c"
