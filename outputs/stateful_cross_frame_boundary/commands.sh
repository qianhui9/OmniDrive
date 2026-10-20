#!/usr/bin/env bash
set -euo pipefail
# Run from /data/qh/phdCode/work4/OmniDrive. Never overwrite prior successful runs.
study_root="${BOUNDARY_OUTPUT_ROOT:-outputs/stateful_cross_frame_boundary/reproduction01}"
taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.large_boundary.run \
  --output "$study_root/gpu_run" --scenes 8 --frames 12
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -c \
  'from tools.worldstream_m123.stateful_opportunity.gpu_audit import wait_idle; wait_idle()'
taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python -c '
import sys, pathlib, torch, cv2, pytest
torch.set_num_threads(1); cv2.setNumThreads(1)
raise SystemExit(pytest.main(["tests/worldstream_m1","tests/worldstream_m2","tests/worldstream_m3",
    "tests/worldstream_m123","-q","--junitxml="+str(pathlib.Path(sys.argv[1])/"regression.xml")]))
' "$study_root/gpu_run"
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -c '
import json, pathlib, sys, xml.etree.ElementTree as ET
run=pathlib.Path(sys.argv[1]); suites=ET.parse(run/"regression.xml").getroot().findall("testsuite")
r={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ("tests","failures","errors","skipped")}
r["status"]="passed" if not (r["errors"] or r["failures"]) else "failed"
r["scope"]="Frozen M1/M2/M3 and audit regression; Active quality Not Run"
(run/"regression.json").write_text(json.dumps(r,indent=2)+"\n")
' "$study_root/gpu_run"
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.large_boundary.report \
  --run "$study_root/gpu_run" --output "$study_root" --regression "$study_root/gpu_run/regression.json"

# This delivery's exact full GPU invocation (log archived alongside raw run):
# taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#   /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.large_boundary.run \
#   --output outputs/stateful_cross_frame_boundary/gpu_run01 --scenes 8 --frames 12 \
#   > outputs/stateful_cross_frame_boundary/gpu_run01.log 2>&1
#
# Read-only reconstruction to a fresh analysis directory:
# taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.large_boundary.report \
#   --run outputs/stateful_cross_frame_boundary/gpu_run01 \
#   --output outputs/stateful_cross_frame_boundary/analysis_rebuild01 \
#   --regression outputs/stateful_cross_frame_boundary/gpu_run01/regression.json
