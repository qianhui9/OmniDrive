#!/usr/bin/env bash
set -euo pipefail
# Run from /data/qh/phdCode/work4/OmniDrive. All subprocesses inherit affinity.
# Existing frozen outputs intentionally reject overwrite. For a NEW reproduction
# export OMNIDRIVE_V5_OUTPUT=outputs/system_optimization_v5_reproduction
# Final source reproduces the frozen short format. Historical failed calibration
# attempts remain preserved in the original v5 directory.
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.system_optimization.v5 --freeze
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.system_optimization.v5_semantics
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.system_optimization.v5_semantic_review
taskset -c 0-3,6-31 nvidia-smi
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.system_optimization.v5
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.system_optimization.v5_pipeline
