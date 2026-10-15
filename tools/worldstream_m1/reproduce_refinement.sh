#!/usr/bin/env bash
# Run from the repository root on the host with the OmniDrive environment and GPU access.
set -euo pipefail
M1_PYTHON="${M1_PYTHON:-/home/qh/miniconda3/envs/OmniDrive/bin/python}"
M1_OUTPUT="${1:-outputs/module1_refinement_reproduced}"
M1_ARCHIVE="$PWD/outputs/module1_refinement/reproducibility/before_optimization"
if [ -e "$M1_OUTPUT" ]; then
    echo 'Choose a fresh output directory to preserve completed/partial experiments.' >&2
    exit 1
fi
mkdir -p "$M1_OUTPUT"
export MPLCONFIGDIR=/tmp/omnidrive-mpl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"$M1_PYTHON" -m tools.worldstream_m1.diagnose_validity --output "$M1_OUTPUT"
"$M1_PYTHON" -m tools.worldstream_m1.diagnose_constraints --output "$M1_OUTPUT"
"$M1_PYTHON" -u -m tools.worldstream_m1.run_refinement --output "$M1_OUTPUT"
"$M1_PYTHON" - "$M1_ARCHIVE" "$M1_OUTPUT/latency_before_optimization.json" <<'PY'
import runpy,sys
archive,output=sys.argv[1:]
sys.path.insert(0,archive)
sys.argv=['benchmark_refinement','--output',output]
runpy.run_module('tools.worldstream_m1.benchmark_refinement',run_name='__main__')
PY
"$M1_PYTHON" -u -m tools.worldstream_m1.benchmark_refinement --output "$M1_OUTPUT/latency_profile.json"
"$M1_PYTHON" -u -m tools.worldstream_m1.map_demo --config configs/worldstream_m1_refinement.yaml --output "$M1_OUTPUT/map_annotation_demo"
"$M1_PYTHON" - "$M1_ARCHIVE" "$M1_OUTPUT/gpu" <<'PY'
import runpy,sys
archive,output=sys.argv[1:]
sys.path.insert(0,archive)
sys.argv=['gpu_refinement','--output',output]
runpy.run_module('tools.worldstream_m1.gpu_refinement',run_name='__main__')
PY
"$M1_PYTHON" -u -m tools.worldstream_m1.gpu_refinement --output "$M1_OUTPUT/gpu_optimized" --skip-feature-oracle
"$M1_PYTHON" -u -m tools.worldstream_m1.run_refinement --output "$M1_OUTPUT/optimized_replay" --methods new_full
"$M1_PYTHON" -m pytest tests/worldstream_m1 -q --junitxml="$M1_OUTPUT/unit_tests.xml"
"$M1_PYTHON" -m tools.worldstream_m1.summarize_refinement --output "$M1_OUTPUT"
