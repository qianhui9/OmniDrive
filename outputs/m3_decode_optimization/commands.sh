#!/usr/bin/env bash
# Run from /data/qh/phdCode/work4/OmniDrive. Choose a NEW namespace each replay.
# Each model runner waits for idle GPU; never stop another user's process.
set -euo pipefail
decode_output="${OMNIDRIVE_DECODE_REPLAY_OUTPUT:-outputs/m3_decode_replay_$(taskset -c 0-3,6-31 date +%Y%m%d_%H%M%S)}"
decode_python=/home/qh/miniconda3/envs/OmniDrive/bin/python
taskset -c 0-3,6-31 mkdir "$decode_output"
taskset -c 0-3,6-31 sha256sum ckpts/iter_10548.pth > "$decode_output/checkpoint.sha256"
taskset -c 0-3,6-31 "$decode_python" -m tools.worldstream_m3.decode_optimization.replay --phase profile --output "$decode_output" > "$decode_output/profiling.log" 2>&1
taskset -c 0-3,6-31 "$decode_python" -m tools.worldstream_m3.decode_optimization.replay --phase attribution --output "$decode_output"
taskset -c 0-3,6-31 "$decode_python" -m tools.worldstream_m3.decode_optimization.replay --phase service --output "$decode_output" > "$decode_output/service.log" 2>&1
taskset -c 0-3,6-31 "$decode_python" -m tools.worldstream_m3.decode_optimization.replay --phase micro --output "$decode_output" > "$decode_output/micro.log" 2>&1
taskset -c 0-3,6-31 "$decode_python" -m tools.worldstream_m3.decode_optimization.replay --phase prefix --output "$decode_output" > "$decode_output/prefix.log" 2>&1
taskset -c 0-3,6-31 "$decode_python" -m tools.worldstream_m3.decode_optimization.replay --phase confirmation --output "$decode_output" > "$decode_output/confirmation.log" 2>&1
taskset -c 0-3,6-31 "$decode_python" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml="$decode_output/regression_cuda.xml" > "$decode_output/regression.log" 2>&1
taskset -c 0-3,6-31 "$decode_python" -m tools.worldstream_m3.decode_optimization.replay --phase finalize --output "$decode_output"
