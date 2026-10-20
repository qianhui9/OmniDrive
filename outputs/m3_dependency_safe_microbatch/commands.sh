#!/usr/bin/env bash
# Repository root; use a fresh namespace. Existing results are never overwritten.
set -euo pipefail
microbatch_output="${OMNIDRIVE_MICROBATCH_OUTPUT:-outputs/m3_dependency_safe_microbatch_replay}"
microbatch_python=/home/qh/miniconda3/envs/OmniDrive/bin/python
taskset -c 0-3,6-31 "$microbatch_python" -m tools.worldstream_m3.microbatch.probe --output "$microbatch_output" > "${microbatch_output}.probe.log" 2>&1
taskset -c 0-3,6-31 cp "${microbatch_output}.probe.log" "$microbatch_output/probe.log"
taskset -c 0-3,6-31 nvidia-smi > "$microbatch_output/nvidia_environment.txt"
taskset -c 0-3,6-31 sha256sum ckpts/iter_10548.pth ckpts/pretrain_qformer/config.json ckpts/pretrain_qformer/generation_config.json ckpts/pretrain_qformer/tokenizer.model ckpts/pretrain_qformer/pytorch_model-00001-of-00002.bin ckpts/pretrain_qformer/pytorch_model-00002-of-00002.bin > "$microbatch_output/model_signatures.sha256"
taskset -c 0-3,6-31 "$microbatch_python" -m pip freeze > "$microbatch_output/requirements.txt"
taskset -c 0-3,6-31 "$microbatch_python" -m tools.worldstream_m3.microbatch.service --output "$microbatch_output" > "$microbatch_output/service.log" 2>&1
taskset -c 0-3,6-31 "$microbatch_python" -m tools.worldstream_m3.microbatch.regression --output "$microbatch_output" > "$microbatch_output/finalization.log" 2>&1
# This session added the following bounded follow-up after the model run and
# regression; current service code already includes its trajectory checks.
# taskset -c 0-3,6-31 "$microbatch_python" -m tools.worldstream_m3.microbatch.supplement --output "$microbatch_output" > "$microbatch_output/trajectory_validation.log" 2>&1
