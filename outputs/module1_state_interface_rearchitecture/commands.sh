#!/usr/bin/env bash
set -euo pipefail
# Executed commands, from /data/qh/phdCode/work4/OmniDrive.
# Replay and CUDA runners reject existing output directories: choose NEW paths
# when reproducing; retain the original run directories as evidence.
OMNIDRIVE_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
taskset -c 0-3,6-31 git rev-parse HEAD
taskset -c 0-3,6-31 git status --short
# CPU replay was repeated after the producer-version correction.
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/state_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 "$OMNIDRIVE_PYTHON" -u -m tools.worldstream_m1.state_interface_replay --scenes 2 --frames 12 --output outputs/module1_state_interface_rearchitecture/replay
# GPU commands execute in the approved host context; sandbox device visibility
# alone does not indicate host GPU availability. Wait for idle; never kill jobs.
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
taskset -c 0-3,6-31 /home/qh/miniconda3/bin/nvitop --once
# Earlier diagnostic run in gpu_validation/ was intentionally interrupted after
# reproducing the producer-version issue; all raw records remain preserved.
# Completed full regression, 8-frame perception parity, 12-frame natural and
# 16-frame repeat five-arm experiments; warmup=1, measured repeats=2.
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.worldstream_m2.state_interface_cuda_batch --output outputs/module1_state_interface_rearchitecture/gpu_validation_v2
# Supplement: new signature-specific ZERO-update costs plus full-task cache-hit
# qualification, unchanged quality gates; warmup=1, measured repeats=1.
# Historical camera_costs.json is neither overwritten nor granted a new key.
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -u -m tools.worldstream_m2.state_interface_cuda_batch --zero-cost-followup --repeats 1 --output outputs/module1_state_interface_rearchitecture/gpu_validation_zero_cost
# Regenerate CPU replay summaries, then merge actual CUDA results.
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/state_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$OMNIDRIVE_PYTHON" -m tools.worldstream_m1.state_interface_report
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/state_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$OMNIDRIVE_PYTHON" -m tools.worldstream_m1.state_interface_gpu_report --gpu-dir outputs/module1_state_interface_rearchitecture/gpu_validation_v2 --supplement-dir outputs/module1_state_interface_rearchitecture/gpu_validation_zero_cost
taskset -c 0-3,6-31 "$OMNIDRIVE_PYTHON" -m tools.worldstream_m1.state_interface_delivery_audit
