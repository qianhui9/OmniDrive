#!/usr/bin/env bash
# Run from /data/qh/phdCode/work4/OmniDrive; select a NEW output namespace.
# All Python and subprocess work inherits the CPU affinity excluding CPU 4/5.
set -euo pipefail
sgvd_output="${OMNIDRIVE_SGVD_OUTPUT:-outputs/m3_state_guided_verified_decode_replay}"
sgvd_python=/home/qh/miniconda3/envs/OmniDrive/bin/python
taskset -c 0-3,6-31 "$sgvd_python" -m tools.worldstream_m3.sgvd.probe --output "$sgvd_output" > "${sgvd_output}.phase1.log" 2>&1
taskset -c 0-3,6-31 cp "${sgvd_output}.phase1.log" "$sgvd_output/phase1.log"
taskset -c 0-3,6-31 sha256sum ckpts/iter_10548.pth ckpts/pretrain_qformer/config.json ckpts/pretrain_qformer/generation_config.json ckpts/pretrain_qformer/tokenizer.model ckpts/pretrain_qformer/pytorch_model-00001-of-00002.bin ckpts/pretrain_qformer/pytorch_model-00002-of-00002.bin > "$sgvd_output/model_signatures.sha256"
taskset -c 0-3,6-31 "$sgvd_python" -m tools.worldstream_m3.sgvd.distribution --output "$sgvd_output"
taskset -c 0-3,6-31 "$sgvd_python" -m tools.worldstream_m3.sgvd.summarize_probe --output "$sgvd_output"
taskset -c 0-3,6-31 "$sgvd_python" -m tools.worldstream_m3.sgvd.validate --output "$sgvd_output" --phase all > "$sgvd_output/validation.log" 2>&1
taskset -c 0-3,6-31 "$sgvd_python" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml="$sgvd_output/regression_cuda.xml" > "$sgvd_output/regression.log" 2>&1
taskset -c 0-3,6-31 "$sgvd_python" -m tools.worldstream_m3.sgvd.finalize --output "$sgvd_output"
