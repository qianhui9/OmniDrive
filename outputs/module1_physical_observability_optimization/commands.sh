#!/usr/bin/env bash
# 实际执行命令清单；从仓库根目录运行，Python环境OmniDrive。
# 已保存冻结/场景会拒绝覆盖。先保留整轮产物再在全新工作副本复现。
# 分析和报告命令可在当前产物上重建派生结果。

taskset -c 0-3,6-31 git rev-parse HEAD
taskset -c 0-3,6-31 git status --short

# 改动前完整CPU回归（当时沙箱CUDA不可见：505通过，1跳过）。
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_observability_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_physical_observability_optimization/baseline_regression_cpu.xml

# 修改方法前的实际nuScenes可辨识性测量；已冻结输出禁止覆盖。
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_observability_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.observability_diagnostics --workers 4

# 新机制受控合约验证。
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1/test_surface_observability.py -q

# 冻结新代码后20场景Development、消融、身份逐边核对。
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_observability_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.observability_experiments --freeze-source --workers 2

# nuScenes-only全量可用性/曝光/缓存审计。
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.observability_nuscenes_audit

# GPU先确认空闲；占用时等待，不停止其他进程。
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_observability_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.model_smoke --compare --perception-only --frames 8 --m1-config configs/worldstream_m1_physical_observability.yaml --output outputs/module1_physical_observability_optimization/gpu_parity

taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_observability_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_physical_observability_optimization/regression_cuda.xml

# 独立manager重放消融和四个失败ROI，保存真实前后掩码与RGB。
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_observability_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.observability_replay_checks

# 从原始JSONL生成所有派生分析、图和报告。
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_observability_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.observability_results
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_observability_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.observability_report
