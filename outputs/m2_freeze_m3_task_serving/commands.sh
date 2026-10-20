#!/usr/bin/env zsh
# Run from /data/qh/phdCode/work4/OmniDrive. CPU 4/5 must never be used.
# Actual first-run commands (already executed, directories refuse overwrites):
# taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m3.task_serving_profile --phase initial_profile --scenes 2 --frames 2 --questions 3
# taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m3.task_serving_batch
# taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m3.task_serving_final_validation
# Final immutable confirmation: taskset -c 0-3,6-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m3.task_serving_confirm
# Initial runner OOM and Python 3.9 annotation failure are retained as attempts.
# Reproduce final code into a fresh directory; change the suffix for later runs.
set -eu

taskset -c 0-3,6-31 env OMNIDRIVE_TASK_SERVING_OUTPUT=outputs/m2_freeze_m3_task_serving_reproduction /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m3.task_serving_bootstrap
taskset -c 0-3,6-31 env OMNIDRIVE_TASK_SERVING_OUTPUT=outputs/m2_freeze_m3_task_serving_reproduction OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m3.task_serving_profile --phase initial_profile --scenes 2 --frames 2 --questions 3
taskset -c 0-3,6-31 env OMNIDRIVE_TASK_SERVING_OUTPUT=outputs/m2_freeze_m3_task_serving_reproduction OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m3.task_serving_batch
taskset -c 0-3,6-31 env OMNIDRIVE_TASK_SERVING_OUTPUT=outputs/m2_freeze_m3_task_serving_reproduction OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m3.task_serving_final_validation
taskset -c 0-3,6-31 env OMNIDRIVE_TASK_SERVING_OUTPUT=outputs/m2_freeze_m3_task_serving_reproduction OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m3.task_serving_confirm
taskset -c 0-3,6-31 env OMNIDRIVE_TASK_SERVING_OUTPUT=outputs/m2_freeze_m3_task_serving_reproduction OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m3.task_serving_finalize
