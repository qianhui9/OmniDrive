#!/usr/bin/env bash
# Existing OmniDrive environment, fresh outputs, observe other GPU workloads only.
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
M3_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/m3-refinement-matplotlib
export CUBLAS_WORKSPACE_CONFIG=:4096:8
M3_ROOT=outputs/module3_refinement
mkdir -p "$M3_ROOT/logs"
M3_STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
while true; do
  M3_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  if [[ "$M3_FREE" =~ ^[0-9]+$ ]] && (( M3_FREE >= 18500 )); then break; fi
  echo "Waiting for >=18500 MiB free GPU memory; observed $M3_FREE MiB"
  sleep 30
done
M3_MULTI="$M3_ROOT/multi_$M3_STAMP"
M3_SINGLE="$M3_ROOT/single_$M3_STAMP"
M3_NUMERIC="$M3_ROOT/numeric_e2e_$M3_STAMP"
M3_PROBE_ARGS=(--probes)
if [[ -n "${M3_CONTROLLED_SOURCE:-}" ]]; then M3_PROBE_ARGS=(); fi
"$M3_PYTHON" -u -m tools.worldstream_m3.refinement --output "$M3_MULTI" \
  --precision native "${M3_PROBE_ARGS[@]}" 2>&1 | tee "$M3_ROOT/logs/multi_$M3_STAMP.log"
if [[ -n "${M3_CONTROLLED_SOURCE:-}" ]]; then
  "$M3_PYTHON" - "$M3_CONTROLLED_SOURCE" "$M3_MULTI" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
source=Path(sys.argv[1]);target=Path(sys.argv[2])/'controlled_probes'
required=['batch_proof.json','sampling_proof.json','overhead.json','probe_source_manifest.json']
assert all((source/name).is_file() for name in required)
shutil.copytree(source,target)
hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
assert all(hashlib.sha256((target/p).read_bytes()).hexdigest()==sha for p,sha in hashes.items())
(target/'provenance.json').write_text(json.dumps(dict(source=str(source),sha256=hashes,
    scope='Earlier completed real controlled probes, separate from continuous timings; copied without changing bytes'),indent=2))
PY
fi
"$M3_PYTHON" -u -m tools.worldstream_m3.refinement --output "$M3_NUMERIC" --numeric-only \
  --modes baseline precision_deterministic precision_no_fp16_reduction precision_fp32_lora common_prefix_seq \
  2>&1 | tee "$M3_ROOT/logs/numeric_e2e_$M3_STAMP.log"
"$M3_PYTHON" -u -m tools.worldstream_m3.refinement --output "$M3_SINGLE" \
  --precision native --single --sampling \
  --modes baseline legacy_exact refined_exact batched m123 \
  2>&1 | tee "$M3_ROOT/logs/single_$M3_STAMP.log"
"$M3_PYTHON" -m pytest -q tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 \
  2>&1 | tee "$M3_ROOT/logs/tests_$M3_STAMP.log"
"$M3_PYTHON" -m tools.worldstream_m3.qa_analysis --run "$M3_MULTI"
"$M3_PYTHON" -m tools.worldstream_m3.qa_analysis --run "$M3_SINGLE"
"$M3_PYTHON" -m tools.worldstream_m3.qa_analysis --run "$M3_NUMERIC"
# Refinement report is generated only after completion and source/artifact checks.
"$M3_PYTHON" -m tools.worldstream_m3.refinement_report --run "$M3_MULTI" \
  --single-run "$M3_SINGLE" --numerical "$M3_ROOT/numerical_02" \
  --numeric-e2e "$M3_NUMERIC" --common-prefix "$M3_ROOT/common_prefix_01" \
  --test-log "$M3_ROOT/logs/tests_$M3_STAMP.log"
