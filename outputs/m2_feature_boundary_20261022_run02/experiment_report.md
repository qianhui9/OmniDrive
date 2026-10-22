# M2 精确特征变化与离线准入实现报告

本次代码、回归测试和真实权重窗口校准已完成。生产仍保持 Auto/window/source_linked/maintain/adaptive，批准为空。没有签发下游质量证书，没有启用生产复用，也没有测得自然帧或完整服务端到端收益。

## 实现与文件

- `dependency_planner.py`：共享原生 Block 0 输入准备；精确 BHWC Token 比较；M1 像素提示经过真实卷积感受野传播；修复膨胀卷积的大 padding 限制。
- `runtime.py`：每次只算一次 embedding/位置嵌入，结果直接进入唯一执行器；先检查缓存完整性；统一成本准入；只在相符且获批规模存在时 adaptive 捕获；区分整 Block 数和输出窗口数；不把视觉输出精确一致自动标成任务认证。
- `state_cache.py`：新增 Block 0 输入的克隆、预算、有限值、版本和布局检查，发布前完成所有可失败工作；预算／非有限值／分配失败清除缓存，接口错误抛出。
- `m1_adapter.py`：保持 M1 接口，缩窄掩码解码异常捕获，保留 dirty/局部来源过期传播及 unknown 独立比较语义。
- `cost_model.py`：增加完整视觉骨干权重／缓冲区和 Patch/拓扑键；候选或无正式证据的记录不触发缓存；显式批准、质量和证据哈希均必需。
- `window_validate.py`：独立离线执行，覆盖全部可缓存 Block、自然帧和多种受控规模；配对交替计时、预热、中位数和原始样本；发布成本单列。
- `promote_costs.py`：显式规模／批准者，重新加载正式模型键，逐条检查数值场景、独立 Detection/Map/VLM 评测、双时间收益和文件哈希后原子写出；拒绝覆盖非空批准表。
- 正式配置仅显式写出原有默认 `cache_capture=adaptive`。M1/M3 核心、EVA 模型、模型权重及正式成本表不变。`WindowExecutor` 原有数值实现经检查后直接沿用。

修改／新增文件完整列表：

- `docs/worldstream_m2_window.md`
- `projects/configs/OmniDrive/worldstream_m123_auto.py`
- `tests/worldstream_m2/conftest.py`
- `tests/worldstream_m2/test_computation_source.py`
- `tests/worldstream_m2/test_cost_promotion.py`
- `tests/worldstream_m2/test_execution.py`
- `tests/worldstream_m2/test_feature_boundary.py`
- `tests/worldstream_m2/test_window_completion.py`
- `tools/worldstream_m2/promote_costs.py`
- `tools/worldstream_m2/window_validate.py`
- `worldstream/m2/cost_model.py`
- `worldstream/m2/dependency_planner.py`
- `worldstream/m2/m1_adapter.py`
- `worldstream/m2/runtime.py`
- `worldstream/m2/state_cache.py`

文件哈希和保持不变的关键文件检查见 `implementation_manifest.json`。用户原有未跟踪目录 `outputs/m3_state_guided_verified_decode/` 未修改。

## 真实权重与自然连续帧

RTX 3090，OmniDrive Python，FP32 权重配原生 FlashAttention；真实 checkpoint `ckpts/iter_10548.pth` strict load。当前正式预处理、六相机 640×640，Block 0、1 各 54 窗口。模型指纹及 Torch/CUDA/亲和性等环境见 `results.json`。每个测量预热 3 次、配对重复 10 次，以中位数报告。

下面三行是同场景严格递增的真实相邻帧；首帧只建立离线历史。变化率均为百分比。像素变化按任一输入通道不同；特征变化／Token 更新按任一 Block 0 输入通道不同。

| 帧对 | 像素变化 | 旧像素判定 Token 更新 | 新特征变化／Token 更新 | Block 0/1 窗口更新 | 增量 wall/GPU ms | 原 EVA wall/GPU ms |
|---|---:|---:|---:|---:|---:|---:|
| natural_1 | 97.5312% | 99.6667% | 99.6667% | 100% / 100% | 478.739 / 478.515 | 476.082 / 476.064 |
| natural_2 | 97.6668% | 99.6667% | 99.6667% | 100% / 100% | 483.868 / 483.854 | 480.951 / 480.932 |
| natural_3 | 97.6813% | 99.6771% | 99.6771% | 100% / 100% | 486.034 / 486.021 | 483.546 / 483.533 |

本次精确特征比较没有减少自然帧 Token 更新比例；窗口依赖闭包后全部更新，实际复用 0/108。三对输出均位级相同。比较前后均不引入阈值或采样。

## 受控验证和成本

所有 10 个受控案例与原 EVA 的最终输出均位级相同，最大绝对误差均为 0；18 条按键／规模汇总的候选记录也均通过每个覆盖场景的 Block 精确对照。受控输入不计入自然命中率。

| 案例 | 每 Block 更新窗口 | 复用输出窗口（两 Block） | 增量 wall/GPU ms | 原 EVA wall/GPU ms |
|---|---:|---:|---:|---:|
| identical | 0 / 54 | 108 | 443.973 / 443.957 | 484.409 / 484.390 |
| one_window | 1 / 54 | 106 | 484.859 / 484.840 | 484.176 / 484.163 |
| padded_edge_window | 1 / 54 | 106 | 484.680 / 484.665 | 484.083 / 484.062 |
| one_window_per_camera | 6 / 54 | 96 | 485.369 / 485.352 | 484.674 / 484.656 |
| window_scale_2 | 2 / 54 | 104 | 485.634 / 485.621 | 484.701 / 484.684 |
| window_scale_4 | 4 / 54 | 100 | 486.005 / 485.985 | 485.454 / 485.431 |
| window_scale_8 | 8 / 54 | 92 | 486.446 / 486.430 | 485.429 / 485.414 |
| window_scale_27 | 27 / 54 | 54 | 488.037 / 488.020 | 485.695 / 485.684 |
| window_scale_53 | 53 / 54 | 2 | 490.014 / 489.997 | 486.105 / 486.083 |
| window_scale_54 | 54 / 54 | 0 | 488.563 / 488.548 | 486.286 / 486.272 |

完全相同输入的视觉管线 wall 中位数降低 8.35%，GPU event span 同时下降；两 Block 零更新，实际跳过 108 个输出窗口对应的 2 个完整 Block。局部更新仍完整执行 QKV/投影/MLP；虽然可复用很多输出窗口，本次所有非零部分规模均未测得双时间收益。

| Block | 更新窗口 | 增量 GPU / 原生 GPU ms | 增量 wall / 原生 wall ms | 双收益 |
|---|---:|---:|---:|---|
| 0 | 0 | 0.443 / 20.552 | 0.451 / 20.560 | 是 |
| 1 | 0 | 0.447 / 20.552 | 0.455 / 20.561 | 是 |
| 0 | 1 | 20.738 / 20.544 | 20.747 / 20.553 | 否 |
| 1 | 1 | 20.870 / 20.615 | 20.878 / 20.624 | 否 |
| 0 | 2 | 20.815 / 20.538 | 20.824 / 20.547 | 否 |
| 1 | 2 | 20.968 / 20.704 | 20.977 / 20.713 | 否 |
| 0 | 4 | 20.919 / 20.553 | 20.929 / 20.562 | 否 |
| 1 | 4 | 20.999 / 20.687 | 21.010 / 20.696 | 否 |
| 0 | 6 | 20.923 / 20.540 | 20.933 / 20.549 | 否 |
| 1 | 6 | 21.103 / 20.712 | 21.112 / 20.720 | 否 |
| 0 | 8 | 20.997 / 20.549 | 21.007 / 20.558 | 否 |
| 1 | 8 | 21.302 / 20.778 | 21.311 / 20.787 | 否 |
| 0 | 27 | 21.679 / 20.520 | 21.691 / 20.529 | 否 |
| 1 | 27 | 21.993 / 20.758 | 22.002 / 20.767 | 否 |
| 0 | 53 | 22.686 / 20.502 | 22.696 / 20.511 | 否 |
| 1 | 53 | 22.864 / 20.814 | 22.874 / 20.822 | 否 |
| 0 | 54 | 21.907 / 20.671 | 21.915 / 20.680 | 否 |
| 1 | 54 | 21.435 / 20.133 | 21.445 / 20.142 | 否 |

每个非零更新 Block 的 QKV／投影处理全部 54 个补齐窗口，MLP 处理全部 9,600 个原网格 Token。零更新 Block 点积工作为 0。Attention 实际窗口数、输出复用量、点积窗口／Token 工作量分别记录在逐案例 stages 和候选 measurements。

视觉管线计时覆盖 embedding、精确比较、依赖闭包、唯一执行器及完整后缀。纯 Block 配对计时不含缓存发布、数据加载或 embedding。缓存发布单列中位数约 1.71–1.72 ms，持久张量缓存为 182,059,008 字节（含图像、Block 0 输入、两个前缀输出）；没有把该独立中位数与视觉管线相加冒充实际请求延迟。M1、数据加载、任务头和完整 VLM 不在视觉管线范围，GPU event 仅为同步流跨度。

## 正式准入状态与未执行项目

- `outputs/final_route/window_costs.json` 原样保留，`approved_counts=[]`。本次生产批准规模：无。
- 测得规模 0、1、2、4、6、8、27、53、54（两个 Block）全部仍未批准，候选 `quality_certified=false`、`production_ready=false`。0 只有数值和成本证据，缺独立下游质量与研究人员明确批准；非零局部规模另缺成本收益。其余未测规模也未批准。
- 已检查现有 `tools/worldstream_m123/benchmark` 和 `outputs/single_route_convergence_20261022/service/report.md`：历史子集报告的检测官方 GT 状态为 NOT_AVAILABLE，地图为 NOT_RUN，语言只覆盖 12 个独立问题，M2 实际复用为 0，不能认证本次键／规模。没有找到可复用的、覆盖本次局部复用模式的完整独立任务质量报告。
- 本次完整 Detection/Map/VLM 质量及配对服务端到端实验未运行；当前没有对应规模的完整任务评测和认证产物可接入，窗口工具只验证视觉算子。不能把历史未复用子集或当前 EVA 精确一致视为全任务通过。晋升契约和命令已提供，缺证据时拒绝晋升，不生成假的质量清单。
- 当前确实发生的复用是离线受控验证中的窗口复用；生产空表和自然帧均无窗口复用。只有相同输入测得视觉 GPU／wall 收益；无自然场景加速、无完整服务端到端加速结论。

## 命令和实际结果

全部命令使用 `taskset -c 0-3,6-31`，解释器 `/home/qh/miniconda3/envs/OmniDrive/bin/python`。GPU 运行因沙箱不可见 CUDA，在获准的沙箱外执行；没有停止其他进程。

```bash
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests -q
# tests_cpu_final.log: 262 passed, 1 skipped, 19 warnings in 5.08s
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m2 tests/worldstream_m123 -q
# tests_gpu_final.log: 116 passed, 19 warnings in 3.11s
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.window_validate --output outputs/m2_feature_boundary_20261022_run02 --frames 4 --repeats 10 --warmup 3
# completed; 3 natural transitions + 10 controlled cases; all exact
taskset -c 0-3,6-31 git diff --check
# PASS
```

CPU 唯一 skip 是 CUDA 测试，已在 GPU 回归中运行；19 条 warning 为已有依赖弃用提示。测试中的成本／质量记录仅为 pytest 临时目录内的合成夹具，未写入正式表。首个沙箱内真实权重尝试报 CUDA 不可用，空目录 run01 不含测量；有效结果为本 run02，不把失败尝试记成通过。

证据：`results.json`（完整逐帧、环境、计时样本）；`window_costs_provisional.json`（18 条候选及全部块级测量）；`tests_cpu_final.log`、`tests_gpu_final.log`；`implementation_manifest.json`。完整晋升契约和使用步骤见 `docs/worldstream_m2_window.md`。
