# 正式路线可复现评测

入口为 `tools.worldstream_m123.benchmark.run`。算法仍使用唯一正式配置
`projects/configs/OmniDrive/worldstream_m123_auto.py`。B1/B2 仅为进程内配置覆盖。
不会修改 checkpoint、原配置、成本表、质量证书或数据集，不提交或推送代码。

```bash
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.worldstream_m123.benchmark.run \
  --checkpoint ckpts/iter_10548.pth \
  --subset 320 --timing-frames 48 --repeats 3 --warmup 3
```

需要 GPU 设备访问权限；受限执行环境内的 CUDA 不可用不意味着宿主 GPU 故障。
每个实验启动前检查 GPU；有其他计算任务时只等待，不结束其他用户的进程。
输出写入新的 `outputs/final_route/benchmark/run_时间戳/`。

默认实验包括：

- 若官方 GT 完整性通过：B0/B3 全量 validation，保留生产 planning 生成路径。
- B0/B1/B2/B3 各 320 帧模块归因：真实 Conv/VQA，每帧一个确定选择的有效标注问题。
- 四个条件在固定前 48 帧上各重复计时三次，每次重新加载同一 checkpoint 并预热。

每个序列按 scene token、场景内时间排序；场景边界清除检测/地图 temporal memory
及 M2/M3 缓存，M1 在原生 precheck 中执行场景重置。B0 与其余条件使用同一测试
预处理，补齐正式 M1 所需的只读相机元数据。完整有效配置保存为 JSON，未另存永久配置。
随机种子由帧 token 确定，相同问题在不同条件与重复中使用相同生成参数和随机数种子。

评测先检查所有验证图像、问答文件和地图资源，记录文件 SHA256；检查 nuScenes
原始 `sample.anns` 与 `sample_annotation.sample_token` 反向链接，并进一步实例化官方
NuScenes SDK，验证其自动重建后的链接、validation token 集和 `load_gt` 实际结果。
原始 JSON 链接差异仅是诊断信息；只有官方 SDK 实际 GT 不完整时，全量主实验记为
`NOT_RUN`、官方指标记为 `NOT_AVAILABLE`，仍执行真实连续数据上的模块和效率诊断。
不会自动修复或复制重写 GT，不会用历史结果补齐缺失数字。

权重载入要求所有模型 state_dict 键和形状完整匹配；仅允许忽略**实际注册为非持久
buffer** 的 RoPE `cos_cached`/`sin_cached`。这些是当前 Transformers 重新计算的缓存，
原生 mmcv 加载也会忽略。所有例外键均记录，缺失参数或其他多余键使实验失败。

`tools/test.py` 当前单卡非分布式入口及 `--out` 有断言，独立 worker 直接调用相同
dataset、模型 forward、原生生成、M1/M2/M3 runtime；没有重写算法。
地图使用项目现有 OpenLaneV2 评估器，修复仅限驱动采集其返回值，未改 dataset.evaluate。

计时范围：

- `model_ms`：模型 forward 起止 CUDA 同步，包含生产路径内语言答案写盘。
- `request_ms`：同步后开始，涵盖 dataset 读取与预处理、collate、H2D、model_ms 和结果 CPU 化。
- 审计日志序列化、GT 离线评分、实际复用反事实检查不计入 request_ms。
- CUDA event 字段为默认流事件跨度，包含 host 提交间隙；不能称为活跃 kernel 总时间。
  prefill/decode、M2、M3 等存在包含关系，不能直接相加。

M1 使用原有 `strict_labels`、`ContinuityMetrics`、`summarize_scene` 和 `validate_rows`。
GT 仅在真实模型提交结束后用于离线评分；不再次运行 M1 来替代在线结果。
这属于唯一类别/尺寸/1m 匹配的部分 Oracle 评估，不是标准 MOT。
所有比率保留分子、分母，零分母写 `null`。

M2 相机复用数必须与真实 EVA 相机执行数相加等于输入相机数。
只有发生实际复用才运行同输入的完整 EVA 正确性检查，保存容差和误差。
无真实复用时不发布“复用质量通过”。历史成本表只作为准入证据保存，不能充当本次加速。

M3 使用生产 Auto；空成本/质量批准列表会产生真实原生回退。
问题数来自实际输出，原生生成数来自真实 `lm_head.generate` 调用，不能从空 diagnostics
列表推断没有语言请求。GT 缺失的 planning 输出不计回答正确率。
当前正式表下无优化路径，因此可直接保存所有原生 token 并和 B0 配对比较。

生成或更新已有运行报告：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.worldstream_m123.benchmark.report \
  --run-dir outputs/final_route/benchmark/run_YYYYMMDD_HHMMSS
```

运行同一目录会保留已完成或失败的实验，不覆盖其预测。失败尝试也保存在运行目录。
需要重试时使用新的运行目录，避免将不同代码版本的结果混在一个配对集合。
报告包含完整范围、attribution/efficiency 区分、原始计数、延迟分布、配对差异和场景
bootstrap 95% 区间；只有一个配对场景时区间为 `null`。
