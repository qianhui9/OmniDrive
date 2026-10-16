#!/usr/bin/env bash
# Run from an authorized host terminal, or from this agent after host access returns.
set -euo pipefail
cd /data/qh/phdCode/work4/OmniDrive
M3_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/m3-matplotlib
M3_STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
M3_MULTI="outputs/module3/clean_multi_${M3_STAMP}"
M3_SINGLE="outputs/module3/clean_single_${M3_STAMP}"
M3_SUPPLEMENT="outputs/module3/quantity_${M3_STAMP}"
mkdir -p outputs/module3/logs

# Observe only. Never terminate or suspend someone else's workload.
while true; do
  M3_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  if [[ "$M3_FREE" =~ ^[0-9]+$ ]] && (( M3_FREE >= 18500 )); then break; fi
  echo "$(TZ=Asia/Shanghai date '+%F %T') waiting for >=18500 MiB free; observed ${M3_FREE} MiB"
  sleep 30
done

"$M3_PYTHON" -u -m tools.worldstream_m3.experiment \
  --output "$M3_MULTI" --scenes 6 --frames 2 --questions 5 --warmup 1 --repeats 2 \
  2>&1 | tee "outputs/module3/logs/clean_multi_${M3_STAMP}.log"
"$M3_PYTHON" -u -m tools.worldstream_m3.experiment \
  --output "$M3_SINGLE" --scenes 2 --frames 2 --questions 1 --warmup 1 --repeats 2 \
  --deployment-single --sampling --perception --skip-probes \
  2>&1 | tee "outputs/module3/logs/clean_single_${M3_STAMP}.log"
"$M3_PYTHON" -u -m tools.worldstream_m3.supplement --output "$M3_SUPPLEMENT" \
  2>&1 | tee "outputs/module3/logs/quantity_${M3_STAMP}.log"
"$M3_PYTHON" -m pytest -q tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 \
  2>&1 | tee "outputs/module3/logs/clean_tests_${M3_STAMP}.log"
"$M3_PYTHON" -m tools.worldstream_m3.qa_analysis --run "$M3_MULTI"
"$M3_PYTHON" -m tools.worldstream_m3.qa_analysis --run "$M3_SINGLE"
"$M3_PYTHON" -m tools.worldstream_m3.report --run "$M3_MULTI" --single-run "$M3_SINGLE" \
  --supplement "$M3_SUPPLEMENT" --test-log "outputs/module3/logs/clean_tests_${M3_STAMP}.log"
echo "Completed multi=${M3_MULTI} single=${M3_SINGLE} supplement=${M3_SUPPLEMENT}"
