#!/usr/bin/env bash
set -euo pipefail
# Invoke from /data/qh/phdCode/work4/OmniDrive. Existing OmniDrive env only.
# Every code/test command excludes hardware-fault CPUs 4/5; subprocesses inherit.
# A fresh run namespace is mandatory, including native language answer files.
# Override OPPORTUNITY_RUN for another P0 replay. GPU runner waits if GPU 0 is busy.
audit_run="${OPPORTUNITY_RUN:-outputs/stateful_cross_frame_opportunity/reproduction01}"
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.stateful_opportunity.gpu_audit \
  --output "$audit_run" --scenes 8 --frames 12 --seed 0
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -c \
  'from tools.worldstream_m123.stateful_opportunity.gpu_audit import wait_idle; wait_idle()'
taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python -c '
import pathlib, sys, torch, cv2, pytest
torch.set_num_threads(1)
cv2.setNumThreads(1)
raise SystemExit(pytest.main(["tests/worldstream_m1", "tests/worldstream_m2", "tests/worldstream_m3",
    "tests/worldstream_m123", "-q", "--junitxml="+str(pathlib.Path(sys.argv[1])/"regression.xml")]))
' "$audit_run"
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -c '
import json, pathlib, sys, xml.etree.ElementTree as ET
run = pathlib.Path(sys.argv[1])
suites = ET.parse(run / "regression.xml").getroot().findall("testsuite")
result = {k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ("tests","failures","errors","skipped")}
result["status"] = "passed" if not (result["failures"] or result["errors"]) else "failed"
result["scope"] = "existing M1/M2/M3 plus independent opportunity audit; not P2 active-mechanism qualification"
(run / "regression.json").write_text(json.dumps(result, indent=2)+"\n")
' "$audit_run"
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.stateful_opportunity.report \
  --run "$audit_run" --output outputs/stateful_cross_frame_opportunity \
  --regression "$audit_run/regression.json"

# Analysis-only rebuild of this delivery (no GPU rerun):
# taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.stateful_opportunity.report \
#   --run outputs/stateful_cross_frame_opportunity/gpu_audit02 \
#   --regression outputs/stateful_cross_frame_opportunity/gpu_audit02/regression.json
#
# Actual initial GPU invocation (stdout/stderr archived, no background service changes):
# taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.stateful_opportunity.gpu_audit \
#   --output outputs/stateful_cross_frame_opportunity/gpu_audit02 --scenes 8 --frames 12 \
#   > outputs/stateful_cross_frame_opportunity/gpu_audit02.log 2>&1
