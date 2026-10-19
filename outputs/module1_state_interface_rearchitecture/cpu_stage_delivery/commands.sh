#!/usr/bin/env bash
set -euo pipefail
# Run from /data/qh/phdCode/work4/OmniDrive. Every command excludes faulty CPUs 4/5.
# Existing outputs are intentionally not overwritten by replay/CUDA runners.
# Use a fresh --output directory when reproducing.
OMNIDRIVE_PYTHON=/home/qh/miniconda3/envs/OmniDrive/bin/python
taskset -c 0-3,6-31 git rev-parse HEAD
taskset -c 0-3,6-31 git status --short
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/state_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 "$OMNIDRIVE_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_state_interface_rearchitecture/regression_final.xml
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/state_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 "$OMNIDRIVE_PYTHON" -u -m tools.worldstream_m1.state_interface_replay --scenes 2 --frames 12 --output outputs/module1_state_interface_rearchitecture/replay
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/state_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$OMNIDRIVE_PYTHON" -m tools.worldstream_m1.state_interface_report
# GPU query was attempted inside sandbox. Outside-sandbox read-only retries
# were not executed because automatic permission review timed out twice.
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
# Only execute CUDA jobs when a GPU is free; never stop another user's process.
# The integration entry was actually attempted inside sandbox and wrote Not Run.
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/state_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$OMNIDRIVE_PYTHON" -m tools.worldstream_m2.state_interface_integration --case natural --scenes 2 --frames 12 --output outputs/module1_state_interface_rearchitecture/cuda_integration
# Pending CUDA follow-ups, NOT claimed as executed:
# taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/state_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$OMNIDRIVE_PYTHON" -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_state_interface_rearchitecture/regression_cuda.xml
# taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/state_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$OMNIDRIVE_PYTHON" -m tools.worldstream_m2.state_interface_integration --case repeat --scenes 2 --frames 16 --output outputs/module1_state_interface_rearchitecture/cuda_repeat
# taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/state_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$OMNIDRIVE_PYTHON" -m tools.worldstream_m1.model_smoke --compare --perception-only --frames 8 --m1-config configs/worldstream_m1_source_linked_state.yaml --output outputs/module1_state_interface_rearchitecture/gpu_parity

# New automatic GPU follow-up entry. Requires a process with visible GPU devices.
# Outside-sandbox launches were attempted three times this follow-up and not executed (approval timeout).
# taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m2.state_interface_cuda_batch --output outputs/module1_state_interface_rearchitecture/gpu_validation
