#!/usr/bin/env bash
# 从仓库根目录运行；初始化/冻结/场景输出拒绝覆盖，请用全新副本复现。
# 所有进程及子进程避开硬件故障CPU 4/5。
set -euo pipefail

taskset -c 0-3,6-31 git rev-parse HEAD
taskset -c 0-3,6-31 git status --short

# 必须在算法修改前执行。真实诊断与源码基线先冻结。
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_recovery_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.correspondence_diagnostics --initialize
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_recovery_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.correspondence_verify_inputs

# 安装本轮代码后验证并冻结。没有阈值扫描或GT筛选在线人口。
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1/test_surface_observability.py tests/worldstream_m1/test_correspondence_recovery.py -q
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_recovery_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.correspondence_experiments --freeze-source --workers 2
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_recovery_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.correspondence_validation --workers 2
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_recovery_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.correspondence_mask_audit --workers 4
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_recovery_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.correspondence_patch_archive

# GPU须在空闲时运行；沙箱内不可见时，已授权在沙箱外执行同一命令。
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_recovery_mpl OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_correspondence_recovery_optimization/regression_cuda.xml
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_recovery_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.model_smoke --compare --perception-only --frames 8 --m1-config configs/worldstream_m1_correspondence_recovery.yaml --output outputs/module1_correspondence_recovery_optimization/gpu_parity

# 仅派生报告/图，可由固定原始轨迹重建；禁止改写历史实验目录。
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_recovery_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.correspondence_results
